"""Analysis of a run: per-arm metrics, matched-pair deltas with bootstrap CIs, label-swap sensitivity, a routing
simulation, a dumb (no-API) baseline, charts and results/REPORT.md.

  python -m src.analyze [--run-id RUN_ID[,RUN_ID...]] [--labelswap-run-id RUN_ID] [--out results] [--db data/runs.sqlite]

A comma-separated --run-id analyses the union of those runs as one dataset (HANDOFF-2.md: `main-01,followup-01`);
an arm's config `scope` limits which tiers it is expected to cover.

Reads the `calls` table written by src.run and data/questions.jsonl. The pure metric functions at the top of the module
(brier, ece, reliability_bins, macro_f1, auc, flip_rate, bootstrap_ci) are unit-tested in tests/test_metrics.py.
Metrics pool all run indices (n = questions x runs) over base calls (label_set == "") with status ok or invalid;
a (question, arm, run_index) triple with only failed rows counts as missing.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .common import DB_PATH, FORMATS, QUESTIONS_PATH, RESULTS_DIR, TIERS, Question, load_config, load_questions  # noqa: E402
from .run import arm_scope  # noqa: E402

NAN = float("nan")
CONTROL_ARM = "qwen3.5-9b"   # the one-token control whose system prompt the prompt-variant arms change
INVALID = "invalid"          # pseudo-class for invalid / missing predictions
BASELINE = "baseline"        # name of the dumb no-API arm
PAIR_METRICS = ["accuracy", "brier", "ece", "flip_rate", "invalid_rate", "latency_p50_ms", "cost_per_1k_actual",
                "cost_per_1k_list"]
TAUS = [round(t, 2) for t in np.arange(0.50, 1.0001, 0.02)]
BOOT_REPS = 1000

# ---- chart tokens (dataviz reference palette: fixed categorical order, hairline chrome, text in ink tokens) ----
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
EXTRA_GRAYS = ["#898781", "#52514e", "#c3c2b7", "#0b0b0b"]   # past slot 8: no new hues, grays + marker shape
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
KIND_MARKERS = {"decision": "o", "instant": "s", "reasoning": "^"}

# ---- dumb baseline: words ignored when extracting key terms from a tier-1 question ----
STOPWORDS = set("""does this ticket note memo email invoice log mention mentions mentioned list lists include includes
contain contains enabled field fields line item marked exactly status flag feature which what how many is the of a an
in to for with its true and or are as at be by did do from has have if it on that these there was were who whom any
given according value values non-empty required five present events event level assigned header give gave did""".split())
_WORD = re.compile(r"[A-Za-z0-9_'-]+")


# ======================================================================================
# Pure metric functions
# ======================================================================================
def brier(probs_list, truth_keys) -> float:
    """Multi-class Brier score: mean over items of sum_k (p_k - y_k)^2, k over the keys of the probs dict plus the truth
    key (a truth key missing from the dict counts as p=0). Items whose probs is None are skipped. For yesno questions
    this is twice the binary Brier score. nan when no item has probs."""
    total, n = 0.0, 0
    for p, t in zip(probs_list, truth_keys):
        if p is None:
            continue
        t = str(t)
        keys = set(p) | {t}
        total += sum((float(p.get(k, 0.0)) - (1.0 if k == t else 0.0)) ** 2 for k in keys)
        n += 1
    return total / n if n else NAN


def reliability_bins(conf, correct, bins: int = 10, weights=None) -> list[dict]:
    """Equal-width confidence bins on [0, 1]: one dict per non-empty bin with bin, lo, hi, conf_mean, acc, count.
    `weights` (per item) lets the bootstrap reuse the same code; the default is one per item."""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    w = np.ones_like(conf) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(conf) & np.isfinite(correct) & (w > 0)
    conf, correct, w = conf[ok], correct[ok], w[ok]
    b = np.clip((conf * bins).astype(int), 0, bins - 1)
    out = []
    for i in range(bins):
        m = b == i
        n = float(w[m].sum())
        if n <= 0:
            continue
        out.append({"bin": i, "lo": i / bins, "hi": (i + 1) / bins, "conf_mean": float(np.average(conf[m], weights=w[m])),
                    "acc": float(np.average(correct[m], weights=w[m])), "count": n})
    return out


def ece(conf, correct, bins: int = 10, weights=None) -> float:
    """Expected calibration error: sum_b (n_b / N) * |acc_b - conf_b| over equal-width bins. nan when empty."""
    rows = reliability_bins(conf, correct, bins, weights)
    n = sum(r["count"] for r in rows)
    if n <= 0:
        return NAN
    return float(sum(r["count"] / n * abs(r["acc"] - r["conf_mean"]) for r in rows))


def macro_f1(y_true, y_pred) -> float:
    """Macro-averaged F1 over the classes present in y_true. A prediction outside those classes (e.g. "invalid") is a
    false negative for its true class and a false positive for no class, so it only lowers recall. nan when empty."""
    y_true = [str(y) for y in y_true]
    y_pred = [str(y) for y in y_pred]
    classes = sorted(set(y_true))
    if not classes:
        return NAN
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def auc(scores, labels) -> float:
    """Rank-based (Mann-Whitney) AUC of scores for the positive class; tied scores get average ranks (so all-ties = 0.5).
    nan unless both classes are present."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return NAN
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0     # average of 1-based ranks i+1 .. j+1
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def flip_rate(groups) -> float:
    """Fraction of groups (one per question: the parsed answers of its runs, None being a value of its own) with
    >= 2 entries whose entries are not all equal. nan when no group has >= 2 entries."""
    elig = [list(g) for g in groups if len(g) >= 2]
    if not elig:
        return NAN
    return sum(1 for g in elig if len(set(g)) > 1) / len(elig)


def bootstrap_ci(stat: Callable, n_ids: int, reps: int = BOOT_REPS, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI of stat(weights). Ids 0..n_ids-1 are resampled with replacement; weights[i] is how
    often id i was drawn, so stat recomputes its statistic on the resampled set. stat may return a float or a 1-D
    array (one CI per element). Returns (lo, hi); nan where every replicate is nan."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        draw = rng.integers(0, n_ids, n_ids) if n_ids > 0 else np.zeros(0, int)
        vals.append(np.asarray(stat(np.bincount(draw, minlength=n_ids).astype(float)), float))
    arr = np.stack(vals)
    scalar = arr.ndim == 1
    if scalar:
        arr = arr[:, None]
    lo, hi = np.full(arr.shape[1], NAN), np.full(arr.shape[1], NAN)
    for j in range(arr.shape[1]):
        col = arr[:, j][np.isfinite(arr[:, j])]
        if len(col):
            lo[j], hi[j] = np.percentile(col, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo[0]), float(hi[0])) if scalar else (lo, hi)


# ======================================================================================
# Small helpers
# ======================================================================================
def _nanmean(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else NAN


def _wmean(x, w) -> float:
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & (w > 0)
    return float(np.average(x[m], weights=w[m])) if m.any() and w[m].sum() > 0 else NAN


def _wpercentile(x, w, q: float) -> float:
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & (w > 0)
    if not m.any():
        return NAN
    return float(np.percentile(np.repeat(x[m], np.rint(w[m]).astype(int)), q))


def _mode(values, default="not available") -> str:
    vals = [v for v in values if isinstance(v, str) and v]
    return Counter(vals).most_common(1)[0][0] if vals else default


def f3(x) -> str:
    try:
        return "not available" if x is None or not np.isfinite(float(x)) else f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "not available"


def fs3(x) -> str:
    s = f3(x)
    return s if s == "not available" or s.startswith("-") else "+" + s


def fusd(x) -> str:
    """USD with 4 significant digits in fixed notation."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "not available"
    if not np.isfinite(v):
        return "not available"
    if v == 0:
        return "$0"
    decimals = max(0, 3 - int(np.floor(np.log10(abs(v)))))
    return f"{'-' if v < 0 else ''}${abs(v):.{decimals}f}"


def fint(x) -> str:
    try:
        return "not available" if x is None or not np.isfinite(float(x)) else f"{int(round(float(x)))}"
    except (TypeError, ValueError):
        return "not available"


def md_table(df: pd.DataFrame, cols: list[tuple[str, str, Callable]]) -> str:
    """cols: (column, header, formatter). Returns a markdown table or a 'not available' line."""
    if df is None or df.empty:
        return "_not available_\n"
    head = "| " + " | ".join(h for _, h, _ in cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for _, r in df.iterrows():
        body += "| " + " | ".join(fmt(r.get(c)) for c, _, fmt in cols) + " |\n"
    return head + body


# ======================================================================================
# Loading calls
# ======================================================================================
CALL_COLS = ["call_id", "run_id", "phase", "question_id", "tier", "format", "arm", "model_slug", "kind", "run_index",
             "label_set", "endpoint_path", "status", "latency_ms", "prompt_tokens", "completion_tokens", "reasoning_tokens",
             "cost_usage", "gen_total_cost", "gen_provider", "provider", "probs", "prob_source", "parsed_answer",
             "ground_truth", "is_correct", "invalid_reason", "confidence", "started_at"]


def _connect(db_path) -> sqlite3.Connection | None:
    p = Path(db_path)
    if not p.exists():
        return None
    conn = sqlite3.connect(p)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "calls" not in names:
        conn.close()
        return None
    return conn


def split_run_ids(run_id: str | None) -> list[str]:
    return [r.strip() for r in str(run_id).split(",") if r.strip()] if run_id else []


def pick_run(conn: sqlite3.Connection, run_id: str | None) -> tuple[str | None, dict, list[str]]:
    """The run(s) to analyse. Default: the most recent run (by started_at) whose id does not start with 'dry'; else the
    most recent run of any kind (noted). A comma-separated run_id is kept as one string (the union is analysed; the
    first id's args give the runs per question). Returns (run_id, {'started_at', 'args'}, notes)."""
    notes: list[str] = []
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    runs = []
    if "runs" in names:
        runs = conn.execute("SELECT run_id, started_at, args FROM runs ORDER BY started_at DESC").fetchall()
    seen = {r[0] for r in runs}
    for r in conn.execute("SELECT run_id, MAX(started_at), NULL FROM calls GROUP BY run_id ORDER BY 2 DESC").fetchall():
        if r[0] not in seen:
            runs.append(r)
    if run_id is None:
        cand = [r for r in runs if not str(r[0]).startswith("dry")]
        if cand:
            run_id = cand[0][0]
        elif runs:
            run_id = runs[0][0]
            notes.append(f"No non-dry run exists in the database; the most recent run `{run_id}` is a dry run and was used.")
    ids = split_run_ids(run_id)
    missing = [r for r in ids if r not in seen and r not in {x[0] for x in runs}]
    if missing:
        notes.append(f"Run id(s) not found in the database: {', '.join(missing)}.")
    meta = next((r for r in runs if ids and r[0] == ids[0]), None)
    args = {}
    if meta and meta[2]:
        try:
            args = json.loads(meta[2])
        except ValueError:
            args = {}
    return (",".join(ids) if ids else None), {"started_at": meta[1] if meta else None, "args": args, "run_ids": ids}, notes


def load_calls(conn: sqlite3.Connection | None, run_id: str | None, cfg: dict) -> pd.DataFrame:
    ids = split_run_ids(run_id)
    if conn is None or not ids:
        return prepare_calls(pd.DataFrame(columns=CALL_COLS), cfg)
    qs = ",".join("?" for _ in ids)
    df = pd.read_sql_query(f"SELECT {', '.join(CALL_COLS)} FROM calls WHERE run_id IN ({qs}) ORDER BY rowid", conn, params=tuple(ids))
    return prepare_calls(df, cfg)


def prepare_calls(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Derived columns: parsed probs, confidence, correctness (invalid = wrong), canonical answer, costs, provider."""
    df = df.copy()
    for c in CALL_COLS:
        if c not in df.columns:
            df[c] = None
    df["label_set"] = df["label_set"].fillna("").astype(str)
    df["tier"] = pd.to_numeric(df["tier"], errors="coerce").fillna(0).astype(int)
    df["run_index"] = pd.to_numeric(df["run_index"], errors="coerce").fillna(0).astype(int)
    df["ground_truth"] = df["ground_truth"].astype(object).where(df["ground_truth"].notna(), "").astype(str)
    for c in ["latency_ms", "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_usage", "gen_total_cost",
              "is_correct", "confidence"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    probs = []
    for p in df["probs"]:
        try:
            probs.append(json.loads(p) if isinstance(p, str) and p else None)
        except ValueError:
            probs.append(None)
    df["probs_dict"] = probs
    df["conf"] = [max(float(v) for v in p.values()) if p else NAN for p in probs]
    df["argmax"] = [max(p, key=lambda k: float(p[k])) if p else None for p in probs]
    df["invalid"] = (df["status"] == "invalid").astype(float)
    df["correct"] = ((df["status"] == "ok") & (df["is_correct"].fillna(0) == 1)).astype(float)
    df["answer"] = [a if (s == "ok" and isinstance(a, str) and a) else INVALID
                    for a, s in zip(df["parsed_answer"], df["status"])]
    df["cost_actual"] = df["gen_total_cost"].where(df["gen_total_cost"].notna(), df["cost_usage"]).astype(float)
    prices = {a: (float(c["price_per_million"]["prompt"]) / 1e6, float(c["price_per_million"]["completion"]) / 1e6)
              for a, c in cfg["arms"].items() if "price_per_million" in c}
    pp = df["arm"].map(lambda a: prices.get(a, (NAN, NAN))[0]).astype(float)
    pc = df["arm"].map(lambda a: prices.get(a, (NAN, NAN))[1]).astype(float)
    df["cost_list"] = df["prompt_tokens"] * pp + df["completion_tokens"].fillna(0) * pc
    df["prov"] = df["gen_provider"].where(df["gen_provider"].notna(), df["provider"])
    return df


def dedup_calls(df: pd.DataFrame, systemone_path: str | None) -> pd.DataFrame:
    """One row per (question, arm, run_index, label_set): ok/invalid beats failed, the default endpoint beats the dry-run
    System One check, earliest row otherwise. Failed-only triples are kept (as failed) so they can be counted as missing."""
    if df.empty:
        return df
    pri = df["status"].map({"ok": 0, "invalid": 1}).fillna(2)
    alt = (df["endpoint_path"] == systemone_path).astype(int) if systemone_path else 0
    d = df.assign(_pri=pri, _alt=alt).sort_values(["_pri", "_alt"], kind="stable")
    d = d.drop_duplicates(["question_id", "arm", "run_index", "label_set"], keep="first")
    return d.drop(columns=["_pri", "_alt"]).sort_index()


# ======================================================================================
# Per-slice metrics (weighted so the bootstrap can reuse them)
# ======================================================================================
class ArmSlice:
    """Item-level arrays of one arm over one slice, indexed by question so metrics can be recomputed under
    per-question bootstrap weights."""

    def __init__(self, rows: pd.DataFrame, qids: list[str] | None = None):
        self.qids = list(qids) if qids is not None else sorted(rows["question_id"].unique())
        idx = {q: i for i, q in enumerate(self.qids)}
        rows = rows[rows["question_id"].isin(idx)]
        self.n_q = len(self.qids)
        self.qi = rows["question_id"].map(idx).to_numpy(int) if len(rows) else np.zeros(0, int)
        self.correct = rows["correct"].to_numpy(float)
        self.invalid = rows["invalid"].to_numpy(float)
        self.latency = rows["latency_ms"].to_numpy(float)
        self.cost_actual = rows["cost_actual"].to_numpy(float)
        self.cost_list = rows["cost_list"].to_numpy(float)
        probs, gts, amax = rows["probs_dict"].tolist(), rows["ground_truth"].tolist(), rows["argmax"].tolist()
        self.has_probs = np.array([p is not None for p in probs], bool)
        self.item_brier = np.array([brier([p], [t]) if p is not None else NAN for p, t in zip(probs, gts)], float)
        self.conf = rows["conf"].to_numpy(float)
        self.conf_correct = np.array([float(a == t) if a is not None else NAN for a, t in zip(amax, gts)], float)
        flags = np.full(self.n_q, NAN)
        for q, g in rows.groupby("question_id"):
            answers = g["answer"].tolist()
            if len(answers) >= 2:
                flags[idx[q]] = flip_rate([answers])
        self.flip_flag = flags

    def metrics(self, wq: np.ndarray | None = None) -> dict:
        wq = np.ones(self.n_q) if wq is None else np.asarray(wq, float)
        w = wq[self.qi] if len(self.qi) else np.zeros(0)
        hp = self.has_probs
        return {
            "accuracy": _wmean(self.correct, w),
            "brier": _wmean(self.item_brier[hp], w[hp]) if hp.any() else NAN,
            "ece": ece(self.conf[hp], self.conf_correct[hp], 10, w[hp]) if hp.any() else NAN,
            "flip_rate": _wmean(self.flip_flag, wq),
            "invalid_rate": _wmean(self.invalid, w),
            "latency_p50_ms": _wpercentile(self.latency, w, 50),
            "cost_per_1k_actual": _wmean(self.cost_actual, w) * 1000.0,
            "cost_per_1k_list": _wmean(self.cost_list, w) * 1000.0,
        }


BY_COLS = ["n", "n_questions", "accuracy", "macro_f1", "brier", "n_with_probs", "ece", "flip_rate", "invalid_rate",
           "missing_rate", "latency_p50_ms", "latency_p95_ms", "cost_per_1k_actual", "cost_per_1k_list",
           "mean_prompt_tokens", "mean_completion_tokens", "prob_source", "provider"]


def slice_metrics(ok: pd.DataFrame, expected: int) -> dict:
    """All BY_COLS for one (arm, slice): `ok` holds the ok/invalid rows, `expected` the question x run triples."""
    m = {c: NAN for c in BY_COLS}
    m.update(n=len(ok), n_questions=int(ok["question_id"].nunique()) if len(ok) else 0, n_with_probs=0,
             prob_source="not available", provider="not available",
             missing_rate=(max(0.0, 1.0 - len(ok) / expected) if expected else NAN))
    if ok.empty:
        return m
    s = ArmSlice(ok)
    m.update(s.metrics())
    m["n_with_probs"] = int(s.has_probs.sum())
    m["macro_f1"] = macro_f1(ok["ground_truth"], ok["answer"])
    lat = ok["latency_ms"].to_numpy(float)
    m["latency_p95_ms"] = float(np.nanpercentile(lat, 95)) if np.isfinite(lat).any() else NAN
    m["mean_prompt_tokens"] = _nanmean(ok["prompt_tokens"])
    m["mean_completion_tokens"] = _nanmean(ok["completion_tokens"])
    m["prob_source"] = _mode(ok["prob_source"])
    m["provider"] = _mode(ok["prov"])
    return m


def metrics_table(base: pd.DataFrame, seen: pd.DataFrame, key: str, keys: list, n_runs: int, arms: list[str],
                  cfg: dict | None = None) -> pd.DataFrame:
    """Per (arm, key) metrics. A row exists for every arm with any row in the run and every key with seen questions
    inside the arm's config `scope` (an arm limited to tier 2 gets no tier-1/3 rows and is not counted as missing there)."""
    rows = []
    ok_all = base[base["status"].isin(["ok", "invalid"])]
    for arm in arms:
        if not (base["arm"] == arm).any():
            continue
        tiers = arm_scope(cfg, arm)[0] if cfg and arm in cfg["arms"] else list(TIERS)
        seen_arm = seen[seen["tier"].isin(tiers)]
        for k in keys:
            n_seen = int(seen_arm[seen_arm[key] == k]["question_id"].nunique())
            if n_seen == 0:
                continue
            ok = ok_all[(ok_all["arm"] == arm) & (ok_all[key] == k)]
            rows.append({"arm": arm, key: k, **slice_metrics(ok, n_seen * n_runs)})
    return pd.DataFrame(rows, columns=["arm", key] + BY_COLS)


def reliability_table(base_ok: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    rows = []
    for arm in arms:
        for tier in TIERS:
            d = base_ok[(base_ok["arm"] == arm) & (base_ok["tier"] == tier) & base_ok["probs_dict"].notna()]
            if d.empty:
                continue
            s = ArmSlice(d)
            for b in reliability_bins(s.conf, s.conf_correct):
                rows.append({"arm": arm, "tier": tier, **b})
    return pd.DataFrame(rows, columns=["arm", "tier", "bin", "lo", "hi", "conf_mean", "acc", "count"])


# ======================================================================================
# Matched pairs, label swap, routing
# ======================================================================================
def matched_pairs(cfg: dict) -> list[tuple[str, str]]:
    """(pair arm, base arm) pairs. A decision arm with `pairs_with: X` pairs with X. A non-decision arm with
    `pairs_with: X` (HANDOFF-2.md: the reasoning control built on the same weights) is an alternate base: every head
    that pairs with X also gets a (head, alternate base) pair, listed after the primary pairs."""
    arms = cfg["arms"]
    heads = [(a, c["pairs_with"]) for a, c in arms.items() if c.get("pairs_with") and c.get("kind") == "decision"]
    alts = [(a, c["pairs_with"]) for a, c in arms.items() if c.get("pairs_with") and c.get("kind") != "decision"]
    pairs = list(heads)
    for alt, base in alts:
        pairs += [(head, alt) for head, hb in heads if hb == base]
    return pairs


def alternate_bases(cfg: dict) -> set[str]:
    return {a for a, c in cfg["arms"].items() if c.get("pairs_with") and c.get("kind") != "decision"}


def prompt_variant_arms(cfg: dict, control: str = CONTROL_ARM) -> list[tuple[str, str]]:
    """(arm, prompt variant) for the control and every selected instant arm that shares its slug (HANDOFF-2.md)."""
    if control not in cfg["arms"]:
        return []
    slug = cfg["arms"][control]["slug"]
    out = [(a, str(c.get("prompt_variant", "a"))) for a, c in cfg["arms"].items()
           if c.get("kind") == "instant" and c.get("slug") == slug and c.get("status") == "selected"]
    return sorted(out, key=lambda x: x[1])


def matched_pair_deltas(base_ok: pd.DataFrame, pairs: list[tuple[str, str]], seed: int, reps: int = BOOT_REPS) -> pd.DataFrame:
    """delta = pair arm minus base arm per tier (and pooled 'all'), with a percentile bootstrap 95% CI over question
    ids (both arms recomputed on the same resampled question set)."""
    cols = ["pair_arm", "base_arm", "tier", "metric", "pair_value", "base_value", "delta", "ci_lo", "ci_hi", "n_questions"]
    rows = []
    for pair_arm, base_arm in pairs:
        for tier in [*TIERS, "all"]:
            sl = base_ok if tier == "all" else base_ok[base_ok["tier"] == tier]
            a, b = sl[sl["arm"] == pair_arm], sl[sl["arm"] == base_arm]
            if a.empty or b.empty:
                rows += [dict(pair_arm=pair_arm, base_arm=base_arm, tier=str(tier), metric=m, pair_value=NAN, base_value=NAN,
                              delta=NAN, ci_lo=NAN, ci_hi=NAN, n_questions=0) for m in PAIR_METRICS]
                continue
            qids = sorted(set(a["question_id"]) | set(b["question_id"]))
            A, B = ArmSlice(a, qids), ArmSlice(b, qids)

            def stat(w, A=A, B=B):
                ma, mb = A.metrics(w), B.metrics(w)
                return np.array([ma[m] - mb[m] for m in PAIR_METRICS])

            ma, mb = A.metrics(), B.metrics()
            lo, hi = bootstrap_ci(stat, len(qids), reps=reps, seed=seed)
            for i, m in enumerate(PAIR_METRICS):
                rows.append(dict(pair_arm=pair_arm, base_arm=base_arm, tier=str(tier), metric=m, pair_value=ma[m],
                                 base_value=mb[m], delta=ma[m] - mb[m], ci_lo=lo[i], ci_hi=hi[i], n_questions=len(qids)))
    return pd.DataFrame(rows, columns=cols)


def labelswap_tables(ls_ok: pd.DataFrame, cfg: dict, arms: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per arm x label_set: accuracy, AUC of P(yes), n. Per arm: sensitivity (fraction of (question, run) items with
    all label sets present whose canonical answers differ), n_pairs, mean |P(yes) difference| between label sets."""
    names = [s["name"] for s in cfg["label_sets"]]
    per_set, per_arm = [], []
    for arm in arms:
        d = ls_ok[ls_ok["arm"] == arm]
        if d.empty:
            continue
        for name in names:
            e = d[d["label_set"] == name]
            pyes = np.array([float(p.get("yes", NAN)) if p else NAN for p in e["probs_dict"]], float)
            lab = (e["ground_truth"] == "yes").to_numpy(bool)
            m = np.isfinite(pyes)
            per_set.append(dict(arm=arm, label_set=name, n=len(e), accuracy=_nanmean(e["correct"]) if len(e) else NAN,
                                auc=auc(pyes[m], lab[m]) if m.any() else NAN, n_with_probs=int(m.sum())))
        flagged, complete, diffs = 0, 0, []
        for _, g in d.groupby(["question_id", "run_index"]):
            present = dict(zip(g["label_set"], g["answer"]))
            if not all(s in present for s in names):
                continue
            complete += 1
            flagged += int(len({present[s] for s in names}) > 1)
            py = {s: float(p["yes"]) for s, p in zip(g["label_set"], g["probs_dict"]) if p and "yes" in p}
            if all(s in py for s in names):
                diffs += [abs(py[x] - py[y]) for x, y in combinations(names, 2)]
        per_arm.append(dict(arm=arm, sensitivity=flagged / complete if complete else NAN, n_pairs=complete,
                            mean_abs_pyes_diff=float(np.mean(diffs)) if diffs else NAN))
    return (pd.DataFrame(per_set, columns=["arm", "label_set", "n", "accuracy", "auc", "n_with_probs"]),
            pd.DataFrame(per_arm, columns=["arm", "sensitivity", "n_pairs", "mean_abs_pyes_diff"]))


PS_COLS = ["scope", "arm", "prompt_variant", "n", "n_questions", "accuracy", "ci_lo", "ci_hi", "brier", "ece",
           "invalid_rate", "flip_rate"]


def prompt_sensitivity(base_ok: pd.DataFrame, variants: list[tuple[str, str]], tier: int, seed: int,
                       reps: int = BOOT_REPS) -> pd.DataFrame:
    """HANDOFF-2.md: on one tier, for the control and its prompt-variant arms, accuracy (with a bootstrap 95% CI over
    questions), Brier, ECE, invalid rate and flip rate, plus a `range` row: max - min accuracy across the prompts
    with its own bootstrap CI (every arm recomputed on the same resampled question set). scope = 'all' and per format."""
    rows = []
    sl = base_ok[base_ok["tier"] == tier]
    for scope in ["all", *FORMATS]:
        d = sl if scope == "all" else sl[sl["format"] == scope]
        present = [(a, v) for a, v in variants if (d["arm"] == a).any()]
        if len(present) < 2:
            continue
        qids = sorted(set().union(*[set(d[d["arm"] == a]["question_id"]) for a, _ in present]))
        slices = {a: ArmSlice(d[d["arm"] == a], qids) for a, _ in present}

        def stat(w, slices=slices, present=present):
            accs = [slices[a].metrics(w)["accuracy"] for a, _ in present]
            return np.array(accs + [max(accs) - min(accs)])

        lo, hi = bootstrap_ci(stat, len(qids), reps=reps, seed=seed)
        accs = []
        for i, (a, v) in enumerate(present):
            m = slices[a].metrics()
            accs.append(m["accuracy"])
            rows.append(dict(scope=scope, arm=a, prompt_variant=v, n=int(len(slices[a].qi)), n_questions=len(qids),
                             accuracy=m["accuracy"], ci_lo=lo[i], ci_hi=hi[i], brier=m["brier"], ece=m["ece"],
                             invalid_rate=m["invalid_rate"], flip_rate=m["flip_rate"]))
        rows.append(dict(scope=scope, arm="range", prompt_variant="max-min", n=NAN, n_questions=len(qids),
                         accuracy=max(accs) - min(accs), ci_lo=lo[-1], ci_hi=hi[-1], brier=NAN, ece=NAN, invalid_rate=NAN,
                         flip_rate=NAN))
    return pd.DataFrame(rows, columns=PS_COLS)


ROUTING_COLS = ["arm", "tier", "tau", "frac_routed", "accuracy", "cost_per_1k"]


def routing_simulation(base_ok: pd.DataFrame, dec_arms: list[str], reas_arm: str) -> pd.DataFrame:
    """Items with max(probs) < tau go to the reasoning arm (its result for the same question and run_index; else its
    per-question mean; else its per-tier mean). Cost = decision cost + reasoning cost for routed items (actual USD)."""
    r = base_ok[base_ok["arm"] == reas_arm]
    if r.empty:
        return pd.DataFrame(columns=ROUTING_COLS)
    r_item = {(q, k): (c, cost) for q, k, c, cost in zip(r["question_id"], r["run_index"], r["correct"], r["cost_actual"])}
    r_q = {q: (_nanmean(g["correct"]), _nanmean(g["cost_actual"])) for q, g in r.groupby("question_id")}
    r_tier = {t: (_nanmean(g["correct"]), _nanmean(g["cost_actual"])) for t, g in r.groupby("tier")}
    rows = []
    for arm in dec_arms:
        for tier in TIERS:
            d = base_ok[(base_ok["arm"] == arm) & (base_ok["tier"] == tier)]
            if d.empty:
                continue
            conf = d["conf"].fillna(0.0).to_numpy(float)          # no probability vector -> never trusted
            d_correct, d_cost = d["correct"].to_numpy(float), d["cost_actual"].to_numpy(float)
            rc, rcost = np.full(len(d), NAN), np.full(len(d), NAN)
            for i, (q, k) in enumerate(zip(d["question_id"], d["run_index"])):
                rc[i], rcost[i] = r_item.get((q, k), r_q.get(q, r_tier.get(tier, (NAN, NAN))))
            for tau in TAUS:
                routed = conf < tau
                rows.append(dict(arm=arm, tier=tier, tau=tau, frac_routed=float(routed.mean()),
                                 accuracy=_nanmean(np.where(routed, rc, d_correct)),
                                 cost_per_1k=_nanmean(d_cost + np.where(routed, rcost, 0.0)) * 1000.0))
    return pd.DataFrame(rows, columns=ROUTING_COLS)


# ======================================================================================
# Dumb baseline (no API): tier 1 keyword/regex matcher, tiers 2-3 majority class
# ======================================================================================
def key_terms(question: str) -> list[str]:
    """Quoted terms if any; else the content words (>= 3 chars, not stopwords/generic words), lower-cased, in order."""
    quoted = [q.strip() for q in re.findall(r"['\"]([^'\"]+)['\"]", question) if q.strip()]
    if quoted:
        return quoted
    terms: list[str] = []
    for w in _WORD.findall(question):
        lw = w.strip("'-").lower()
        if len(lw) >= 3 and lw not in STOPWORDS and lw not in terms:
            terms.append(lw)
    return terms


def _term_present(term: str, state: str) -> bool:
    """Case-insensitive match: multi-word or short (< 4 chars) terms as whole words/substrings, others on the first
    four characters of a word (so 'refund' matches 'refunded', 'shipping' matches 'shipment')."""
    t, s = term.lower(), state.lower()
    if " " in t or len(t) < 4:
        return re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", s) is not None
    return re.search(r"(?<!\w)" + re.escape(t[:4]), s) is not None


def baseline_tier1(q: Question) -> str:
    """Canonical answer key of the keyword matcher for one tier-1 question."""
    state = q.state_text()
    opts = q.options or []
    if q.format == "yesno":
        terms = key_terms(q.question)
        return "yes" if terms and all(_term_present(t, state) for t in terms) else "no"
    if q.format == "choice":
        low = state.lower()
        counts = [low.count(o.lower()) if o else 0 for o in opts]
        return str(int(np.argmax(counts))) if max(counts, default=0) > 0 else "0"
    # score
    if [o.strip() for o in opts] == ["1", "2", "3", "4", "5"]:
        terms = key_terms(q.question)
        caps = [t for t in terms if t.isupper() and len(t) >= 3] or [w for w in _WORD.findall(q.question) if w.isupper() and len(w) >= 3]
        term = caps[-1] if caps else (terms[-1] if terms else "")
        if not term:
            return "2"
        flags = 0 if term.isupper() else re.IGNORECASE
        count = len(re.findall(r"(?<!\w)" + re.escape(term) + r"(?!\w)", state, flags))
        return str(min(5, max(1, count)) - 1)
    best, last = None, -1
    for i, lab in enumerate(opts):
        for cand in dict.fromkeys([lab, lab.split()[0] if lab.split() else lab]):
            for m in re.finditer(r"(?<!\w)" + re.escape(cand) + r"(?!\w)", state):
                if m.start() > last:
                    last, best = m.start(), i
    return str(best) if best is not None else "2"


def baseline_predictions(questions: list[Question]) -> pd.DataFrame:
    """One row per question: tier, format, ground truth, baseline prediction, rule used."""
    rows = []
    majority: dict[tuple, str] = {}
    for (tier, fmt), grp in pd.DataFrame([{"tier": q.tier, "format": q.format, "gt": str(q.ground_truth)} for q in questions]).groupby(["tier", "format"]) if questions else []:
        counts = Counter(grp["gt"])
        top = max(counts.values())
        majority[(tier, fmt)] = sorted(k for k, v in counts.items() if v == top)[0]
    for q in questions:
        if q.tier == 1:
            pred, rule = baseline_tier1(q), "keyword-regex"
        else:
            pred, rule = majority[(q.tier, q.format)], "majority-class"
        rows.append({"question_id": q.id, "tier": q.tier, "format": q.format, "ground_truth": str(q.ground_truth),
                     "prediction": pred, "correct": float(pred == str(q.ground_truth)), "rule": rule})
    return pd.DataFrame(rows, columns=["question_id", "tier", "format", "ground_truth", "prediction", "correct", "rule"])


def baseline_table(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if preds.empty:
        return pd.DataFrame(columns=["tier", "format", "n", "accuracy", "macro_f1", "rule"])
    for tier, g in preds.groupby("tier"):
        rows.append(dict(tier=tier, format="all", n=len(g), accuracy=g["correct"].mean(),
                         macro_f1=macro_f1(g["ground_truth"], g["prediction"]), rule=_mode(g["rule"])))
        for fmt, h in g.groupby("format"):
            rows.append(dict(tier=tier, format=fmt, n=len(h), accuracy=h["correct"].mean(),
                             macro_f1=macro_f1(h["ground_truth"], h["prediction"]), rule=_mode(h["rule"])))
    return pd.DataFrame(rows)


def baseline_rows(preds: pd.DataFrame, key: str, keys: list) -> list[dict]:
    """Rows for the metrics tables (arm = 'baseline', cost 0, no probabilities)."""
    out = []
    for k in keys:
        g = preds[preds[key] == k]
        if g.empty:
            continue
        m = {c: NAN for c in BY_COLS}
        m.update(n=len(g), n_questions=len(g), accuracy=g["correct"].mean(), macro_f1=macro_f1(g["ground_truth"], g["prediction"]),
                 n_with_probs=0, invalid_rate=0.0, missing_rate=0.0, cost_per_1k_actual=0.0, cost_per_1k_list=0.0,
                 prob_source="none", provider="none (no API)")
        out.append({"arm": BASELINE, key: k, **m})
    return out


# ======================================================================================
# Charts (matplotlib, Agg). Colors follow the arm (fixed slot per arm across every chart); marker shape follows kind.
# ======================================================================================
def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": SURFACE, "legend.frameon": False, "savefig.dpi": 150,
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def arm_styles(arms: list[str], cfg: dict) -> dict[str, tuple[str, str]]:
    """(color, marker) per arm: selected arms first in config order take the 8 categorical slots; later arms get grays."""
    acfg = cfg["arms"]
    order = sorted(set(arms), key=lambda a: (acfg.get(a, {}).get("status") != "selected", acfg.get(a, {}).get("arm", 99), a))
    palette = PALETTE + EXTRA_GRAYS
    return {a: (palette[i] if i < len(palette) else MUTED, KIND_MARKERS.get(acfg.get(a, {}).get("kind"), "D"))
            for i, a in enumerate(order)}


def _legend_handles(styles: dict, arms: list[str]) -> list:
    return [Line2D([], [], marker=styles[a][1], color=styles[a][0], ls="none", ms=7, label=a) for a in arms if a in styles]


def chart_accuracy_vs_cost(by_tier: pd.DataFrame, baseline_acc: dict, pairs: list, styles: dict, path: Path,
                           cost_col: str, cost_label: str, title: str) -> Path:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True)
    plotted: list[str] = []
    for ax, tier in zip(axes, TIERS):
        d = by_tier[(by_tier["tier"] == tier) & (by_tier["arm"] != BASELINE) & (by_tier["n"] > 0)]
        costs = d[cost_col][d[cost_col] > 0]
        lo, hi = (costs.min() / 5, costs.max() * 5) if not costs.empty else (1e-3, 1e2)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.05)
        pos = {}
        for r in d.itertuples(index=False):
            c, y = getattr(r, cost_col), r.accuracy
            if np.isfinite(y):
                pos[r.arm] = (c if np.isfinite(c) and c > 0 else lo, y, np.isfinite(c) and c > 0)
        for pa, ba in pairs:
            if pa in pos and ba in pos:
                ax.plot([pos[pa][0], pos[ba][0]], [pos[pa][1], pos[ba][1]], color=AXIS, lw=1, zorder=1)
        for i, (arm, (x, y, known)) in enumerate(pos.items()):
            color, marker = styles[arm]
            ax.plot(x, y, marker=marker, ms=8, color=color, mfc=color if known else "white", mec="white" if known else color,
                    mew=1.2, ls="none", zorder=3, clip_on=False)
            ax.annotate(arm + ("" if known else " (cost n/a)"), (x, y), xytext=(6, 5 if i % 2 == 0 else -10),
                        textcoords="offset points", fontsize=7.5, color=INK2)
            if arm not in plotted:
                plotted.append(arm)
        b = baseline_acc.get(tier)
        if b is not None and np.isfinite(b):
            ax.axhline(b, ls="--", lw=1, color=MUTED, zorder=2)
            ax.plot(lo, b, marker="D", ms=6, color=MUTED, ls="none", clip_on=False, zorder=3)
            ax.annotate("dumb baseline (cost 0)", (lo, b), xytext=(6, -11), textcoords="offset points", fontsize=7.5, color=MUTED)
        if d.empty:
            ax.text(0.5, 0.85, "no arm data", transform=ax.transAxes, ha="center", va="center", color=MUTED)
        ax.set_title(f"Tier {tier}")
        ax.set_xlabel(cost_label)
    axes[0].set_ylabel("accuracy (invalid = wrong)")
    handles = _legend_handles(styles, plotted) + [Line2D([], [], ls="--", color=MUTED, label="dumb baseline")]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_reliability(rel: pd.DataFrame, styles: dict, charts_dir: Path, arms: list[str]) -> list[Path]:
    """One grid (rows = arms, cols = tiers) when <= 10 arms have probabilities, else one 1x3 figure per arm."""
    _style()
    arms = [a for a in arms if (rel["arm"] == a).any()]
    if not arms:
        return []

    def panel(ax, arm, tier):
        d = rel[(rel["arm"] == arm) & (rel["tier"] == tier)]
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1)
        if d.empty:
            ax.text(0.5, 0.5, "not available", transform=ax.transAxes, ha="center", va="center", color=MUTED)
        else:
            n = d["count"].sum()
            sizes = 5 + 9 * np.sqrt(d["count"] / d["count"].max())
            ax.plot(d["conf_mean"], d["acc"], color=styles[arm][0], lw=2, zorder=2)
            ax.scatter(d["conf_mean"], d["acc"], s=sizes ** 2, color=styles[arm][0], edgecolor="white", lw=1, zorder=3)
            e = sum(c / n * abs(a - m) for c, a, m in zip(d["count"], d["acc"], d["conf_mean"]))
            ax.text(0.04, 0.93, f"ECE {e:.3f}  n={int(n)}", transform=ax.transAxes, fontsize=8, color=INK2, va="top")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{arm} - Tier {tier}", fontsize=8.5)

    paths = []
    if len(arms) <= 10:
        fig, axes = plt.subplots(len(arms), 3, figsize=(10, 2.7 * len(arms) + 0.6), squeeze=False)
        for i, arm in enumerate(arms):
            for j, tier in enumerate(TIERS):
                panel(axes[i][j], arm, tier)
                if i == len(arms) - 1:
                    axes[i][j].set_xlabel("confidence (max p), 10 bins")
            axes[i][0].set_ylabel("accuracy of argmax")
        fig.suptitle("Reliability diagrams (marker size = bin count)")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        p = charts_dir / "reliability.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    else:
        for arm in arms:
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
            for j, tier in enumerate(TIERS):
                panel(axes[j], arm, tier)
                axes[j].set_xlabel("confidence (max p), 10 bins")
            axes[0].set_ylabel("accuracy of argmax")
            fig.tight_layout()
            p = charts_dir / f"reliability_{re.sub(r'[^A-Za-z0-9_.-]', '_', arm)}.png"
            fig.savefig(p)
            plt.close(fig)
            paths.append(p)
    return paths


def chart_pair_deltas(pairs_df: pd.DataFrame, styles: dict, path: Path, alt_bases: set[str] | None = None) -> Path | None:
    """One bar group per (pair arm, base arm); pairs against an alternate base (the reasoning control) are hatched."""
    _style()
    alt_bases = alt_bases or set()
    if pairs_df.empty or not pairs_df["delta"].notna().any():
        return None
    valid = pairs_df[pairs_df["delta"].notna()]
    keys = list(dict.fromkeys(zip(valid["pair_arm"], valid["base_arm"])))
    tiers = [str(t) for t in TIERS] + ["all"]
    fig, axes = plt.subplots(2, 4, figsize=(15 + 1.2 * max(0, len(keys) - 3), 7))
    for ax, metric in zip(axes.flat, PAIR_METRICS):
        xs, ticks = [], []
        for g, (pa, ba) in enumerate(keys):
            hatch = "//" if ba in alt_bases else None
            for k, t in enumerate(tiers):
                x = g * (len(tiers) + 1) + k
                xs.append(x)
                ticks.append("T" + t if t != "all" else "all")
                r = pairs_df[(pairs_df["pair_arm"] == pa) & (pairs_df["base_arm"] == ba) & (pairs_df["tier"] == t) &
                             (pairs_df["metric"] == metric)]
                if r.empty or not np.isfinite(r["delta"].iloc[0]):
                    continue
                d, lo, hi = r["delta"].iloc[0], r["ci_lo"].iloc[0], r["ci_hi"].iloc[0]
                ax.bar(x, d, width=0.8, color=styles.get(pa, (MUTED,))[0], zorder=2, hatch=hatch,
                       edgecolor="white" if hatch else None, linewidth=0)
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.errorbar(x, d, yerr=[[max(0.0, d - lo)], [max(0.0, hi - d)]], color=INK2, lw=1, capsize=2, zorder=3)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(ticks, fontsize=7)
        ax.set_title(metric)
        ax.grid(axis="x", visible=False)
    handles = [Patch(facecolor=styles.get(pa, (MUTED,))[0], hatch="//" if ba in alt_bases else None,
                     edgecolor="white", linewidth=0, label=f"{pa} minus {ba}") for pa, ba in keys]
    fig.legend(handles=handles, loc="lower center", ncol=min(3, len(handles)), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Matched-pair deltas (pair arm minus base arm) with bootstrap 95% CI over questions; hatched = minus the "
                 "reasoning control")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_prompt_sensitivity(ps: pd.DataFrame, styles: dict, control: str, baseline_acc: float | None, tier: int,
                             path: Path) -> Path | None:
    """Three bars (one per system prompt) of the control's accuracy on one tier, bootstrap CI whiskers, baseline dashed."""
    _style()
    if ps is None or ps.empty:
        return None
    d = ps[(ps["scope"] == "all") & (ps["arm"] != "range")]
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    color = styles.get(control, (PALETTE[0], "s"))[0]
    xs = np.arange(len(d))
    acc = d["accuracy"].to_numpy(float)
    ax.bar(xs, acc, 0.6, color=color, zorder=2)
    lo, hi = d["ci_lo"].to_numpy(float), d["ci_hi"].to_numpy(float)
    ok = np.isfinite(lo) & np.isfinite(hi)
    if ok.any():
        ax.errorbar(xs[ok], acc[ok], yerr=[np.maximum(0, acc[ok] - lo[ok]), np.maximum(0, hi[ok] - acc[ok])], fmt="none",
                    ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
    for x, a, h in zip(xs, acc, hi):
        ax.text(x, (h if np.isfinite(h) else a) + 0.02, f"{a:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    if baseline_acc is not None and np.isfinite(baseline_acc):
        ax.axhline(baseline_acc, ls="--", lw=1, color=MUTED, zorder=1)
        ax.text(len(d) - 0.5, baseline_acc + 0.015, "dumb baseline", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"prompt {v}\n{a}" for a, v in zip(d["arm"], d["prompt_variant"])], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(f"Tier {tier} accuracy (invalid = wrong)")
    ax.grid(axis="x", visible=False)
    rng = ps[(ps["scope"] == "all") & (ps["arm"] == "range")]
    sub = ""
    if not rng.empty:
        r = rng.iloc[0]
        sub = f": range {f3(r['accuracy'])} [{f3(r['ci_lo'])}, {f3(r['ci_hi'])}]"
    ax.set_title(f"Prompt sensitivity of the one-token control, Tier {tier}{sub}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_routing(routing: pd.DataFrame, styles: dict, path: Path, ycol: str, ylabel: str, ref: dict | None,
                  log_y: bool, title: str) -> Path | None:
    _style()
    if routing.empty:
        return None
    arms = list(dict.fromkeys(routing["arm"]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    for ax, tier in zip(axes, TIERS):
        d = routing[routing["tier"] == tier]
        for arm in arms:
            e = d[d["arm"] == arm]
            if not e.empty:
                ax.plot(e["tau"], e[ycol], color=styles[arm][0], marker=styles[arm][1], ms=4, markevery=5, lw=2)
        if ref and tier in ref and np.isfinite(ref[tier]):
            ax.axhline(ref[tier], ls="--", lw=1, color=MUTED)
        if d.empty:
            ax.text(0.5, 0.5, "not available", transform=ax.transAxes, ha="center", va="center", color=MUTED)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlim(TAUS[0], TAUS[-1])
        ax.set_title(f"Tier {tier}")
        ax.set_xlabel("threshold tau (route when max p < tau)")
    axes[0].set_ylabel(ylabel)
    handles = _legend_handles(styles, arms)
    if ref:
        handles.append(Line2D([], [], ls="--", color=MUTED, label="reasoning arm alone"))
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_labelswap(per_set: pd.DataFrame, per_arm: pd.DataFrame, names: list[str], path: Path) -> Path | None:
    _style()
    arms = list(dict.fromkeys(per_set["arm"]))
    if not arms:
        return None
    fig, ax = plt.subplots(figsize=(max(6.5, 1.7 * len(arms) + 2.5), 4.4))
    width = 0.8 / len(names)
    for j, name in enumerate(names):
        vals = [per_set[(per_set["arm"] == a) & (per_set["label_set"] == name)]["accuracy"].max() for a in arms]
        xs = np.arange(len(arms)) + (j - (len(names) - 1) / 2) * width
        ax.bar(xs, [v if np.isfinite(v) else 0 for v in vals], width * 0.9, color=PALETTE[j % len(PALETTE)], label=name, zorder=2)
    for i, a in enumerate(arms):
        r = per_arm[per_arm["arm"] == a]
        if not r.empty:
            ax.text(i, 1.02, f"sensitivity {f3(r['sensitivity'].iloc[0])}\n(n={int(r['n_pairs'].iloc[0])})",
                    ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, fontsize=8)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("accuracy on label-swap items")
    ax.grid(axis="x", visible=False)
    ax.legend(title="label set", loc="upper right", ncol=len(names))
    ax.set_title("Label swap: accuracy per label set (sensitivity = share of items whose answer changes)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ======================================================================================
# Report
# ======================================================================================
def _pair_get(pairs_df: pd.DataFrame, pair_arm: str, tier: str, metric: str, base_arm: str | None = None) -> tuple[float, float, float]:
    r = pairs_df[(pairs_df["pair_arm"] == pair_arm) & (pairs_df["tier"] == tier) & (pairs_df["metric"] == metric)]
    if base_arm is not None:
        r = r[r["base_arm"] == base_arm]
    if r.empty:
        return NAN, NAN, NAN
    return float(r["delta"].iloc[0]), float(r["ci_lo"].iloc[0]), float(r["ci_hi"].iloc[0])


def _by_tier_get(by_tier: pd.DataFrame, arm: str, tier: int, col: str) -> float:
    r = by_tier[(by_tier["arm"] == arm) & (by_tier["tier"] == tier)]
    if r.empty:
        return NAN
    try:
        return float(r[col].iloc[0])
    except (TypeError, ValueError):
        return NAN


def followup_lines(R: dict) -> list[str]:
    """HANDOFF-2.md: the two follow-up answers, one sentence each, plus the prompt-sensitivity table."""
    cfg, by_tier, pairs_df, ps = R["cfg"], R["by_tier"], R["pairs_df"], R.get("prompt_sens")
    alts = [a for a in alternate_bases(cfg) if a in R["providers"]]
    has_ps = ps is not None and not ps.empty
    if not alts and not has_ps:
        return []
    L = ["\n## Follow-up (HANDOFF-2.md): does the head recover reasoning, or just the format?\n"]
    for ra in sorted(alts):
        base = cfg["arms"][ra]["pairs_with"]
        heads = [a for a, c in cfg["arms"].items() if c.get("kind") == "decision" and c.get("pairs_with") == base]
        acc = _by_tier_get(by_tier, ra, 2, "accuracy")
        if not np.isfinite(acc):
            L.append(f"1. **Reasoning control.** `{ra}`: not available (no completed Tier 2 calls).\n")
            continue
        cost, lat, inv = (_by_tier_get(by_tier, ra, 2, c) for c in ("cost_per_1k_actual", "latency_p50_ms", "invalid_rate"))
        parts = []
        for h in heads:
            d, lo, hi = _pair_get(pairs_df, h, "2", "accuracy", base_arm=ra)
            parts.append(f"{h} {f3(_by_tier_get(by_tier, h, 2, 'accuracy'))} (head minus reasoning base {fs3(d)} [{f3(lo)}, {f3(hi)}]; "
                         f"{fusd(_by_tier_get(by_tier, h, 2, 'cost_per_1k_actual'))} per 1,000 and "
                         f"{fint(_by_tier_get(by_tier, h, 2, 'latency_p50_ms'))} ms p50)")
        L.append(f"1. **Reasoning control.** On Tier 2, `{ra}` ({base} weights with thinking on, then one token) scores {f3(acc)} "
                 f"at {fusd(cost)} per 1,000 decisions and {fint(lat)} ms p50 (invalid rate {f3(inv)}), against the heads built on "
                 f"the same weights: " + "; ".join(parts) + ".\n")
    if has_ps:
        allrows = ps[(ps["scope"] == "all") & (ps["arm"] != "range")]
        rng = ps[(ps["scope"] == "all") & (ps["arm"] == "range")]
        if not allrows.empty and not rng.empty:
            best = allrows.loc[allrows["accuracy"].idxmax()]
            worst = allrows.loc[allrows["accuracy"].idxmin()]
            r = rng.iloc[0]
            L.append(f"2. **Prompt sensitivity.** On Tier 2 the one-token control moves from {f3(worst['accuracy'])} (prompt "
                     f"{worst['prompt_variant']}, `{worst['arm']}`) to {f3(best['accuracy'])} (prompt {best['prompt_variant']}, "
                     f"`{best['arm']}`) across the three system prompts: range {f3(r['accuracy'])} [{f3(r['ci_lo'])}, "
                     f"{f3(r['ci_hi'])}] (max minus min accuracy, bootstrap 95% CI over questions).\n")
        L.append("\nPrompt sensitivity, Tier 2 (`prompt_sensitivity.csv`; CI = bootstrap 95% over questions, on accuracy for an "
                 "arm and on max minus min for the `range` rows):\n\n")
        L.append(md_table(ps, [("scope", "scope", str), ("arm", "arm", str), ("prompt_variant", "prompt", str), ("n", "n", fint),
                               ("accuracy", "accuracy", f3), ("ci_lo", "CI lo", f3), ("ci_hi", "CI hi", f3), ("brier", "Brier", f3),
                               ("ece", "ECE", f3), ("invalid_rate", "invalid", f3), ("flip_rate", "flip", f3)]))
        if R.get("ps_chart"):
            L.append("\n![prompt sensitivity](charts/prompt_sensitivity.png)\n")
    return L


def headline(pairs_df: pd.DataFrame, pairs: list[tuple[str, str]]) -> str:
    """One sentence on the 9B decision heads vs raw Qwen3.5-9B, from the pooled (all-tier) matched-pair deltas."""
    nine = [(pa, ba) for pa, ba in pairs if ba == "qwen3.5-9b"]
    if not nine:
        return "No arm is configured to pair with qwen3.5-9b, so the 9B question is not yet measurable."
    clauses = []
    for pa, ba in nine:
        acc = _pair_get(pairs_df, pa, "all", "accuracy", ba)
        if not np.isfinite(acc[0]):
            clauses.append(f"{pa} vs {ba}: not yet measurable (no overlapping completed calls)")
            continue
        cal = _pair_get(pairs_df, pa, "all", "ece", ba)
        bri = _pair_get(pairs_df, pa, "all", "brier", ba)
        flp = _pair_get(pairs_df, pa, "all", "flip_rate", ba)
        cost = _pair_get(pairs_df, pa, "all", "cost_per_1k_actual", ba)
        better, worse = [], []
        for name, (d, lo, hi), good_sign in [("accuracy", acc, 1), ("calibration (Brier)", bri, -1), ("calibration (ECE)", cal, -1),
                                             ("consistency (flip rate)", flp, -1)]:
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue
            if (lo > 0 and good_sign > 0) or (hi < 0 and good_sign < 0):
                better.append(name)
            elif (hi < 0 and good_sign > 0) or (lo > 0 and good_sign < 0):
                worse.append(name)
        verdict = ("beats it on " + ", ".join(better)) if better else "does not beat it on accuracy, calibration or consistency (95% CIs include 0)"
        if worse:
            verdict += " and is worse on " + ", ".join(worse)
        cost_txt = (f"{'+' if cost[0] >= 0 else '-'}{fusd(abs(cost[0]))} per 1,000 decisions (actual)"
                    if np.isfinite(cost[0]) else "a cost difference that is not yet measurable")
        clauses.append(f"{pa} {verdict}: accuracy {fs3(acc[0])} [{f3(acc[1])}, {f3(acc[2])}], Brier {fs3(bri[0])}, ECE {fs3(cal[0])}, "
                       f"flip rate {fs3(flp[0])}, at {cost_txt}")
    return "Versus raw qwen3.5-9b over all tiers: " + "; ".join(clauses) + "."


BY_TABLE_COLS = [("arm", "arm", str), ("n", "n", fint), ("accuracy", "accuracy", f3), ("macro_f1", "macro-F1", f3),
                 ("brier", "Brier", f3), ("ece", "ECE", f3), ("flip_rate", "flip", f3), ("invalid_rate", "invalid", f3),
                 ("missing_rate", "missing", f3), ("latency_p50_ms", "p50 ms", fint), ("latency_p95_ms", "p95 ms", fint),
                 ("cost_per_1k_actual", "cost/1k actual", fusd), ("cost_per_1k_list", "cost/1k list", fusd),
                 ("prob_source", "probs", str)]


def write_report(path: Path, ctx: dict) -> None:
    cfg, R = ctx["cfg"], ctx
    L: list[str] = []
    L.append("# Decision models vs LLMs - analysis report\n")
    L.append(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. ")
    if R["run_id"] is None:
        L.append("**No run data**: the database has no calls, so every run-based metric below is not available. "
                 "Only the dumb baseline (computed from data/questions.jsonl) is reported.\n")
    else:
        L.append(f"Run `{R['run_id']}` ({'dry run; ' if R['is_dry'] else ''}{R['n_base']} base calls over "
                 f"{R['n_questions_seen']} questions, {R['n_runs']} run(s) per question expected; label-swap rows from run "
                 f"`{R['labelswap_run_id']}`: {R['n_ls']}).\n")
    for n in R["notes"]:
        L.append(f"> {n}\n")
    L += followup_lines(R)

    L.append("\n## 1. Does the 9B decision head beat raw Qwen3.5-9B?\n")
    L.append(R["headline"] + "\n")

    L.append("\n## 2. Accuracy vs cost\n")
    if R["main_chart"]:
        L.append("![accuracy vs cost per 1,000 decisions (actual)](charts/accuracy_vs_cost.png)\n")
        L.append("\nList-price version: [charts/accuracy_vs_cost_list.png](charts/accuracy_vs_cost_list.png)\n")
    else:
        L.append("not available\n")

    L.append("\n## 3. Matched-pair deltas (pair arm minus base arm; bootstrap 95% CI over question ids, "
             f"{BOOT_REPS} reps, seed {cfg.get('seed')})\n")
    pairs_df = R["pairs_df"]
    if pairs_df.empty or not pairs_df["delta"].notna().any():
        L.append("not available (no tier has completed calls for both arms of a matched pair)\n")
    else:
        d = pairs_df.copy()
        money = d["metric"].str.startswith("cost")
        d["ci"] = ["[" + (fusd(lo) if m else f3(lo)) + ", " + (fusd(hi) if m else f3(hi)) + "]"
                   for lo, hi, m in zip(d["ci_lo"], d["ci_hi"], money)]
        for c in ["pair_value", "base_value", "delta"]:
            d[c + "_s"] = [fusd(v) if m else f3(v) for v, m in zip(d[c], money)]
        has_data = d.groupby("pair_arm")["delta"].transform(lambda s: s.notna().any())
        for pa, g in d[~has_data].groupby("pair_arm", sort=False):
            L.append(f"- {pa} vs {g['base_arm'].iloc[0]}: not available (no tier has completed calls for both arms)\n")
        d = d[has_data & (d["delta"].notna() | (d["tier"] == "all"))]
        L.append("\n" + md_table(d, [("pair_arm", "pair arm", str), ("base_arm", "base arm", str), ("tier", "tier", str),
                                     ("metric", "metric", str), ("pair_value_s", "pair", str), ("base_value_s", "base", str),
                                     ("delta_s", "delta", str), ("ci", "95% CI", str), ("n_questions", "questions", fint)]))
        if R["pair_chart"]:
            L.append("\n![matched-pair deltas](charts/matched_pair_deltas.png)\n")

    L.append("\n## 4. Metrics per tier\n")
    by_tier = R["by_tier"]
    if by_tier.empty:
        L.append("not available\n")
    for tier in TIERS:
        d = by_tier[by_tier["tier"] == tier]
        if d.empty:
            continue
        L.append(f"\n### Tier {tier}\n")
        L.append(md_table(d, BY_TABLE_COLS))
    if R["rel_paths"]:
        L.append("\nReliability diagrams: " + ", ".join(f"[{p.name}](charts/{p.name})" for p in R["rel_paths"]) + "\n")
    L.append("\n### Dumb baseline (no API)\n")
    L.append(md_table(R["baseline_df"], [("tier", "tier", fint), ("format", "format", str), ("n", "n", fint),
                                         ("accuracy", "accuracy", f3), ("macro_f1", "macro-F1", f3), ("rule", "rule", str)]))
    L.append("\n" + R["baseline_line"] + "\n")

    L.append("\n## 5. Metrics per format\n")
    by_fmt = R["by_format"]
    if by_fmt.empty:
        L.append("not available\n")
    for fmt in FORMATS:
        d = by_fmt[by_fmt["format"] == fmt]
        if d.empty:
            continue
        L.append(f"\n### {fmt}\n")
        L.append(md_table(d, BY_TABLE_COLS))

    L.append("\n## 6. Label-swap sensitivity (yesno items rendered with yes/no, 1/0, A/B)\n")
    if R["ls_set"].empty:
        L.append("not available\n")
    else:
        L.append(md_table(R["ls_arm"], [("arm", "arm", str), ("sensitivity", "sensitivity", f3), ("n_pairs", "items (question x run) with all 3 sets", fint),
                                        ("mean_abs_pyes_diff", "mean |dP(yes)| between sets", f3)]))
        L.append("\n")
        L.append(md_table(R["ls_set"], [("arm", "arm", str), ("label_set", "label set", str), ("n", "n", fint),
                                        ("accuracy", "accuracy", f3), ("auc", "AUC of P(yes)", f3), ("n_with_probs", "n with probs", fint)]))
        if R["ls_chart"]:
            L.append("\n![label swap](charts/labelswap.png)\n")

    L.append("\n## 7. Routing simulation (decision arm answers when max p >= tau, else the reasoning arm)\n")
    if R["routing_note"]:
        L.append(R["routing_note"] + "\n")
    routing = R["routing"]
    if not routing.empty:
        L.append(f"Reasoning arm: `{R['reasoning_arm']}`. Full grid in routing.csv; selected thresholds:\n\n")
        sel = routing[routing["tau"].isin([0.6, 0.8, 0.9, 0.96, 1.0])]
        L.append(md_table(sel, [("arm", "arm", str), ("tier", "tier", fint), ("tau", "tau", lambda x: f"{float(x):.2f}"),
                                ("frac_routed", "routed", f3), ("accuracy", "accuracy", f3), ("cost_per_1k", "cost/1k", fusd)]))
        L.append("\n![routing accuracy](charts/routing_accuracy.png)\n\n![routing cost](charts/routing_cost.png)\n")

    L.append("\n## 8. Caveats\n")
    for c in R["caveats"]:
        L.append(f"- {c}\n")
    path.write_text("".join(L), encoding="utf-8")


def build_caveats(ctx: dict) -> list[str]:
    cfg, R = ctx["cfg"], ctx
    c = ["Tier 3 ground truth is the agreement of two frontier models (generator and an independent checker), "
         "which is weak evidence compared with the machine-verified Tier 1 and Tier 2 answers."]
    reas_arms = [a for a, v in cfg["arms"].items() if v.get("kind") == "reasoning" and (not R["run_id"] or a in R["providers"])]
    c.append("Reasoning-arm calibration uses a verbalized 0-100 confidence, not logprobs; its Brier/ECE are not directly "
             f"comparable with the logprob/decision probabilities of the other arms (reasoning arms: {', '.join(reas_arms) or 'none'}).")
    t3 = cfg.get("tier3", {})
    t3_vendors = {s.split("/")[0] for s in (t3.get("generator_slug"), t3.get("checker_slug")) if s}
    same_vendor = [a for a in R["providers"] if cfg["arms"].get(a, {}).get("slug", "").split("/")[0] in t3_vendors] if R["run_id"] else []
    if t3_vendors:
        c.append(f"Tier 3 questions were generated by {t3.get('generator_slug')} and cross-checked by {t3.get('checker_slug')}. "
                 + (f"Arms from the same vendors ({', '.join(same_vendor)}) share those models' beliefs, so their Tier 3 scores are partly "
                    "circular and should not be read as independent evidence." if same_vendor else
                    "No arm in this run comes from those vendors."))
    c.append("Brier is the multi-class form sum_k (p_k - y_k)^2 averaged over items; for yesno questions this equals "
             "twice the binary Brier score.")
    if R["providers"]:
        c.append("Provider/region per arm (gen_provider, falling back to the response provider): " +
                 "; ".join(f"{a}: {v}" for a, v in R["providers"].items()) + ".")
    else:
        c.append("Provider/region per arm: not available (no calls).")
    if R["missing_arms"]:
        c.append("Arms with failed or missing calls (triples without an ok/invalid row count as missing): " +
                 "; ".join(R["missing_arms"]) + ".")
    else:
        c.append("Arms with failed or missing calls: none" + ("" if R["run_id"] else " (no run data)") + ".")
    if R["none_arms"]:
        c.append("Arms whose prob_source is 'none' (no probability vector, so Brier/ECE/routing are not available): " +
                 ", ".join(R["none_arms"]) + ".")
    else:
        c.append("No arm has prob_source 'none'" + ("" if R["run_id"] else " (no run data)") + ".")
    if R["run_id"]:
        c.append("Expected call counts use the questions seen in this run (any arm) within each arm's config `scope` times the "
                 f"runs per question ({R['n_runs']}); accuracy pools all runs, so n = questions x runs.")
    for a in sorted(alternate_bases(cfg)):
        if a not in R["providers"]:
            continue
        cap = cfg["arms"][a].get("reasoning_max_tokens", cfg["rendering"]["reasoning_max_tokens"])
        n_cut, n_all = R.get("truncated", {}).get(a, (0, 0))
        c.append(f"{a} is the follow-up reasoning control (HANDOFF-2.md): the same weights and provider as "
                 f"{cfg['arms'][a]['pairs_with']} with thinking on and a {cap:,}-token completion cap; {n_cut} of {n_all} base calls "
                 "produced no ANSWER line within the cap and count as invalid (= wrong). Its calibration is verbalized like the "
                 "other reasoning arm's; the answer token's logprobs are recorded in parse_extra only.")
    variants = [a for a, v in prompt_variant_arms(cfg) if v != "a" and a in R["providers"]]
    if variants:
        c.append(f"Prompt-variant arms ({', '.join(variants)}) are the one-token control with a different system prompt, run on "
                 "Tier 2 only without the label-swap subset; they appear in the per-tier/per-format tables and prompt_sensitivity.csv "
                 "and are left out of the accuracy-vs-cost and reliability charts.")
    c.append("Charts: the categorical palette holds 8 fixed arm colors; further arms are drawn in grays and every point is "
             "labelled by name, with marker shape by arm kind (circle = decision, square = instant, triangle = reasoning).")
    return c


# ======================================================================================
# Orchestration
# ======================================================================================
def run_analysis(db_path=DB_PATH, out_dir=RESULTS_DIR, run_id: str | None = None, labelswap_run_id: str | None = None) -> dict:
    """Compute everything, write CSVs, charts and REPORT.md under out_dir. Returns the tables and file paths."""
    cfg = load_config()
    out = Path(out_dir)
    charts = out / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    questions = load_questions() if QUESTIONS_PATH.exists() else []
    notes: list[str] = []
    files: list[Path] = []

    conn = _connect(db_path)
    run_meta: dict = {"started_at": None, "args": {}}
    if conn is None:
        run_id = None
        notes.append("The database is missing or has no `calls` table.")
    else:
        run_id, run_meta, picked = pick_run(conn, run_id)
        notes += picked
        if run_id is None:
            notes.append("The database contains no calls.")
    labelswap_run_id = labelswap_run_id or run_id
    raw = load_calls(conn, run_id, cfg)
    raw_ls = raw if labelswap_run_id == run_id else load_calls(conn, labelswap_run_id, cfg)
    if conn is not None:
        conn.close()
    if run_id is not None and raw.empty:
        notes.append(f"Run `{run_id}` has no calls.")

    systemone = cfg["openrouter"].get("systemone_path")
    base = dedup_calls(raw[raw["label_set"] == ""], systemone)
    base_ok = base[base["status"].isin(["ok", "invalid"])]
    ls = dedup_calls(raw_ls[raw_ls["label_set"] != ""], systemone)
    ls_ok = ls[ls["status"].isin(["ok", "invalid"])]
    seen = raw[["question_id", "tier", "format"]].drop_duplicates()
    is_dry = bool(run_id) and (str(run_id).startswith("dry") or (not raw.empty and set(raw["phase"].dropna()) <= {"dry"}))
    if is_dry:
        n_runs = int(base["run_index"].max()) + 1 if not base.empty else 1
    else:
        n_runs = int(run_meta.get("args", {}).get("runs") or cfg["counts"]["runs_per_question"])
    acfg = cfg["arms"]
    arms = sorted(set(raw["arm"].dropna()) | set(raw_ls["arm"].dropna()),
                  key=lambda a: (acfg.get(a, {}).get("status") != "selected", acfg.get(a, {}).get("arm", 99), a))
    styles = arm_styles(arms, cfg)
    pairs = [(pa, ba) for pa, ba in matched_pairs(cfg)]
    alt_bases = alternate_bases(cfg)
    variants = prompt_variant_arms(cfg)
    variant_arms = [a for a, v in variants if v != "a"]   # kept out of the scatter / reliability charts

    # ---- baseline (questions only) ----
    preds = baseline_predictions(questions)
    baseline_df = baseline_table(preds)
    baseline_df.to_csv(out / "baseline.csv", index=False)
    files.append(out / "baseline.csv")
    baseline_acc = {int(r["tier"]): float(r["accuracy"]) for _, r in baseline_df[baseline_df["format"] == "all"].iterrows()}

    # ---- metrics tables ----
    by_tier = metrics_table(base, seen, "tier", list(TIERS), n_runs, arms, cfg)
    by_tier = pd.concat([by_tier, pd.DataFrame(baseline_rows(preds, "tier", list(TIERS)), columns=by_tier.columns)], ignore_index=True)
    by_format = metrics_table(base, seen, "format", list(FORMATS), n_runs, arms, cfg)
    by_format = pd.concat([by_format, pd.DataFrame(baseline_rows(preds, "format", list(FORMATS)), columns=by_format.columns)], ignore_index=True)
    by_tier.to_csv(out / "metrics_by_tier.csv", index=False)
    by_format.to_csv(out / "metrics_by_format.csv", index=False)
    rel = reliability_table(base_ok, arms)
    rel.to_csv(out / "reliability.csv", index=False)
    files += [out / "metrics_by_tier.csv", out / "metrics_by_format.csv", out / "reliability.csv"]

    # ---- label swap ----
    ls_set, ls_arm = labelswap_tables(ls_ok, cfg, arms)
    ls_set.to_csv(out / "labelswap.csv", index=False)
    ls_arm.to_csv(out / "labelswap_sensitivity.csv", index=False)
    files += [out / "labelswap.csv", out / "labelswap_sensitivity.csv"]

    # ---- matched pairs ----
    pairs_df = matched_pair_deltas(base_ok, pairs, int(cfg.get("seed", 0)))
    pairs_df.to_csv(out / "matched_pair_deltas.csv", index=False)
    files.append(out / "matched_pair_deltas.csv")

    # ---- prompt sensitivity (HANDOFF-2.md; the prompt-variant arms run tier 2 only) ----
    prompt_sens = prompt_sensitivity(base_ok, variants, 2, int(cfg.get("seed", 0))) if len(variants) > 1 else pd.DataFrame(columns=PS_COLS)
    prompt_sens.to_csv(out / "prompt_sensitivity.csv", index=False)
    files.append(out / "prompt_sensitivity.csv")

    # ---- routing ----
    dec_arms = [a for a in arms if acfg.get(a, {}).get("kind") == "decision"]
    reas_sel = [a for a, v in acfg.items() if v.get("kind") == "reasoning" and v.get("status") == "selected"]
    routing_note, reasoning_arm = "", None
    if not reas_sel:
        routing_note = "Routing skipped: no reasoning arm has status `selected` in config.yaml (candidates are decided by the dry run)."
    else:
        reasoning_arm = reas_sel[0]
        if not (base_ok["arm"] == reasoning_arm).any():
            routing_note = f"Routing skipped: the selected reasoning arm `{reasoning_arm}` has no completed calls in this run."
    routing = routing_simulation(base_ok, dec_arms, reasoning_arm) if reasoning_arm and not routing_note else pd.DataFrame(columns=ROUTING_COLS)
    if reasoning_arm and not routing_note and routing.empty:
        routing_note = "Routing skipped: no decision arm has completed calls to route."
    routing.to_csv(out / "routing.csv", index=False)
    files.append(out / "routing.csv")

    # ---- calls summary ----
    allrows = pd.concat([raw, raw_ls]).drop_duplicates("call_id") if labelswap_run_id != run_id else raw
    summ = []
    for arm in arms:
        d = allrows[allrows["arm"] == arm]
        summ.append(dict(arm=arm, ok=int((d["status"] == "ok").sum()), invalid=int((d["status"] == "invalid").sum()),
                         failed=int((d["status"] == "failed").sum()), total_cost_actual=float(np.nansum(d["cost_actual"])) if d["cost_actual"].notna().any() else NAN,
                         total_cost_list=float(np.nansum(d["cost_list"])) if d["cost_list"].notna().any() else NAN))
    calls_summary = pd.DataFrame(summ, columns=["arm", "ok", "invalid", "failed", "total_cost_actual", "total_cost_list"])
    calls_summary.to_csv(out / "calls_summary.csv", index=False)
    files.append(out / "calls_summary.csv")

    # ---- charts ----
    scatter = by_tier[~by_tier["arm"].isin(variant_arms)]
    main_chart = chart_accuracy_vs_cost(scatter, baseline_acc, pairs, styles, charts / "accuracy_vs_cost.png", "cost_per_1k_actual",
                                        "cost per 1,000 decisions, USD (actual, log scale)", "Accuracy vs actual cost per 1,000 decisions")
    chart_accuracy_vs_cost(scatter, baseline_acc, pairs, styles, charts / "accuracy_vs_cost_list.png", "cost_per_1k_list",
                           "cost per 1,000 decisions, USD (list price, log scale)", "Accuracy vs list-price cost per 1,000 decisions")
    files += [charts / "accuracy_vs_cost.png", charts / "accuracy_vs_cost_list.png"]
    rel_paths = chart_reliability(rel[~rel["arm"].isin(variant_arms)], styles, charts, [a for a in arms if a not in variant_arms])
    files += rel_paths
    pair_chart = chart_pair_deltas(pairs_df, styles, charts / "matched_pair_deltas.png", alt_bases)
    ps_chart = chart_prompt_sensitivity(prompt_sens, styles, CONTROL_ARM, baseline_acc.get(2), 2, charts / "prompt_sensitivity.png")
    if ps_chart:
        files.append(ps_chart)
    ref = {t: float(by_tier[(by_tier["arm"] == reasoning_arm) & (by_tier["tier"] == t)]["accuracy"].max()) for t in TIERS} if reasoning_arm else None
    r1 = chart_routing(routing, styles, charts / "routing_accuracy.png", "accuracy", "accuracy", ref, False, "Routing simulation: accuracy vs threshold")
    r2 = chart_routing(routing, styles, charts / "routing_cost.png", "cost_per_1k", "cost per 1,000 decisions, USD (actual, log)", None, True,
                       "Routing simulation: cost vs threshold")
    ls_chart = chart_labelswap(ls_set, ls_arm, [s["name"] for s in cfg["label_sets"]], charts / "labelswap.png")
    files += [p for p in [pair_chart, r1, r2, ls_chart] if p]

    # ---- report ----
    providers = {}
    for arm in arms:
        vals = Counter(v for v in allrows[allrows["arm"] == arm]["prov"] if isinstance(v, str) and v)
        providers[arm] = ", ".join(f"{k} ({n})" for k, n in vals.most_common()) or "not available"
    missing_arms = []
    for arm in arms:
        d = by_tier[(by_tier["arm"] == arm)]
        failed = int(calls_summary[calls_summary["arm"] == arm]["failed"].sum())
        miss = d["missing_rate"].max() if not d.empty else NAN
        if failed > 0 or (np.isfinite(miss) and miss > 0):
            missing_arms.append(f"{arm} (failed rows {failed}, worst missing rate {f3(miss)})")
    none_arms = sorted(set(base_ok[base_ok["prob_source"].fillna("none") == "none"]["arm"]))
    truncated = {}
    for a in alt_bases:
        d = base_ok[base_ok["arm"] == a]
        if not d.empty:
            truncated[a] = (int((d["invalid_reason"] == "no ANSWER line").sum()), int(len(d)))
    jev = by_tier[(by_tier["arm"] == "jev-1.14") & (by_tier["tier"] == 1)]["accuracy"]
    b1 = baseline_acc.get(1)
    if b1 is None or jev.empty or not np.isfinite(jev.iloc[0]):
        baseline_line = "Tier 1 baseline vs jev-1.14: not available (no completed jev-1.14 Tier 1 calls)."
    else:
        gap = jev.iloc[0] - b1
        baseline_line = (f"Tier 1: the dumb baseline scores {f3(b1)} vs jev-1.14 {f3(jev.iloc[0])} (gap {fs3(gap)}); the baseline is "
                         f"{'within' if abs(gap) <= 0.05 else 'not within'} 5 points of jev-1.14.")
    ctx = dict(cfg=cfg, run_id=run_id, labelswap_run_id=labelswap_run_id, is_dry=is_dry, n_base=int(len(base_ok)),
               n_questions_seen=int(seen["question_id"].nunique()), n_runs=n_runs, n_ls=int(len(ls_ok)), notes=notes,
               headline=headline(pairs_df, pairs), main_chart=main_chart, pairs_df=pairs_df, pair_chart=pair_chart,
               by_tier=by_tier, by_format=by_format, rel_paths=rel_paths, baseline_df=baseline_df, baseline_line=baseline_line,
               ls_set=ls_set, ls_arm=ls_arm, ls_chart=ls_chart, routing=routing, routing_note=routing_note,
               reasoning_arm=reasoning_arm, providers=providers, missing_arms=missing_arms, none_arms=none_arms,
               prompt_sens=prompt_sens, ps_chart=ps_chart, truncated=truncated, run_ids=run_meta.get("run_ids", []))
    ctx["caveats"] = build_caveats(ctx)
    write_report(out / "REPORT.md", ctx)
    files.append(out / "REPORT.md")
    ctx["files"] = files
    ctx["calls_summary"] = calls_summary
    return ctx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", help="run to analyse (default: most recent non-dry run)")
    ap.add_argument("--labelswap-run-id", help="run holding the label-swap rows (default: same run)")
    ap.add_argument("--out", default=str(RESULTS_DIR), help="output directory (default: results/)")
    ap.add_argument("--db", default=str(DB_PATH), help="SQLite database (default: data/runs.sqlite)")
    args = ap.parse_args(argv)
    ctx = run_analysis(Path(args.db), Path(args.out), args.run_id, args.labelswap_run_id)
    print(f"run_id={ctx['run_id']} base_calls={ctx['n_base']} labelswap_calls={ctx['n_ls']}")
    for n in ctx["notes"]:
        print("note:", n)
    for p in ctx["files"]:
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
