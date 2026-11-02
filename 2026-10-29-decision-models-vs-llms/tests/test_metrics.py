"""Pure metric functions in src.analyze against hand-computed values, plus an end-to-end smoke test on a tiny DB."""
import json
import math
import sqlite3

import numpy as np
import pandas as pd
import pytest

from src import analyze as A
from src.common import QUESTIONS_PATH, load_questions
from src.run import SCHEMA


# ---- brier -----------------------------------------------------------------------------------
def test_brier_perfect_and_uniform():
    assert A.brier([{"yes": 1.0, "no": 0.0}], ["yes"]) == pytest.approx(0.0)
    assert A.brier([{"yes": 0.5, "no": 0.5}], ["yes"]) == pytest.approx(0.5)          # 2 x binary Brier (0.25)
    assert A.brier([{"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}], ["2"]) == pytest.approx(0.75)
    # mean over items; None items skipped; a truth key missing from the dict counts as p = 0
    assert A.brier([{"yes": 1.0, "no": 0.0}, None, {"yes": 0.5, "no": 0.5}], ["yes", "yes", "no"]) == pytest.approx(0.25)
    assert A.brier([{"no": 1.0}], ["yes"]) == pytest.approx(2.0)
    assert math.isnan(A.brier([None], ["yes"]))


# ---- ece -------------------------------------------------------------------------------------
def test_ece_calibrated_and_overconfident():
    assert A.ece([0.75] * 4, [1, 1, 1, 0]) == pytest.approx(0.0)
    assert A.ece([0.95] * 4, [1, 0, 0, 0]) == pytest.approx(0.70)
    # two bins, equal weight: |0.5 - 0.65| * 0.5 + |1.0 - 0.95| * 0.5
    assert A.ece([0.65, 0.65, 0.95, 0.95], [1, 0, 1, 1]) == pytest.approx(0.10)
    assert math.isnan(A.ece([], []))


def test_reliability_bins_and_weights():
    rows = A.reliability_bins([0.65, 0.65, 0.95, 0.95], [1, 0, 1, 1])
    assert [r["bin"] for r in rows] == [6, 9]
    assert rows[0]["count"] == 2 and rows[0]["acc"] == pytest.approx(0.5) and rows[0]["conf_mean"] == pytest.approx(0.65)
    assert rows[1]["count"] == 2 and rows[1]["acc"] == pytest.approx(1.0)
    # weights act like repeated items
    assert A.ece([0.65, 0.95], [0, 1], weights=[3, 1]) == pytest.approx(A.ece([0.65] * 3 + [0.95], [0, 0, 0, 1]))


# ---- macro-F1 --------------------------------------------------------------------------------
def test_macro_f1_with_invalid_prediction():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "invalid", "b", "a"]
    # a: TP 1, FP 1, FN 1 -> 0.5 ; b: TP 1, FP 0, FN 1 -> 2/3 ; "invalid" is not a class
    assert A.macro_f1(y_true, y_pred) == pytest.approx((0.5 + 2 / 3) / 2)
    assert A.macro_f1(["yes", "no"], ["yes", "no"]) == pytest.approx(1.0)
    assert A.macro_f1(["yes", "yes"], ["invalid", "invalid"]) == pytest.approx(0.0)
    assert math.isnan(A.macro_f1([], []))


# ---- auc -------------------------------------------------------------------------------------
def test_auc_perfect_reversed_ties():
    assert A.auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert A.auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == pytest.approx(0.0)
    assert A.auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == pytest.approx(0.5)
    assert A.auc([0.9, 0.5, 0.5, 0.1], [1, 1, 0, 0]) == pytest.approx(0.875)  # 3 ordered pairs + one tie at 0.5, of 4
    assert math.isnan(A.auc([0.3, 0.4], [1, 1]))


# ---- flip rate -------------------------------------------------------------------------------
def test_flip_rate_groups():
    assert A.flip_rate([["yes", "yes"], ["no", "no", "no"]]) == pytest.approx(0.0)
    assert A.flip_rate([["yes", "yes"], ["yes", "no"], ["no"]]) == pytest.approx(0.5)   # singleton not eligible
    assert A.flip_rate([["yes", None]]) == pytest.approx(1.0)                          # None is its own value
    assert math.isnan(A.flip_rate([["yes"]]))


# ---- bootstrap -------------------------------------------------------------------------------
def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    values = rng.normal(size=40)
    point = values.mean()
    lo, hi = A.bootstrap_ci(lambda w: np.average(values, weights=w), len(values), reps=500, seed=3)
    assert lo <= point <= hi
    assert hi - lo < 4 * values.std() / math.sqrt(len(values))      # not absurdly wide
    # vector statistics get one CI per element
    lo_v, hi_v = A.bootstrap_ci(lambda w: np.array([np.average(values, weights=w), 1.0]), len(values), reps=50, seed=0)
    assert lo_v[0] <= point <= hi_v[0] and lo_v[1] == hi_v[1] == 1.0


# ---- end-to-end smoke test -------------------------------------------------------------------
def _row(call_id, run_id, q, arm, kind, run_index, answer, label_set="", status="ok", probs=None, phase="main",
         cost_usage=1e-5, gen_total_cost=None, latency=800.0):
    gt = str(q.ground_truth)
    return {
        "call_id": call_id, "run_id": run_id, "phase": phase, "question_id": q.id, "tier": q.tier, "format": q.format,
        "arm": arm, "model_slug": arm, "kind": kind, "run_index": run_index, "label_set": label_set, "status": status,
        "latency_ms": latency, "prompt_tokens": 300, "completion_tokens": 1, "cost_usage": cost_usage,
        "gen_total_cost": gen_total_cost, "gen_provider": "TestProvider", "provider": "TestProvider",
        "probs": json.dumps(probs) if probs is not None else None,
        "prob_source": "decision" if kind == "decision" else "logprobs",
        "parsed_answer": answer if status == "ok" else None, "ground_truth": gt,
        "is_correct": int(status == "ok" and answer == gt), "confidence": max(probs.values()) if probs else None,
    }


def _probs(q, answer, p=0.8):
    keys = ["yes", "no"] if q.format == "yesno" else [str(i) for i in range(q.n_options())]
    other = (1 - p) / (len(keys) - 1)
    return {k: (p if k == answer else other) for k in keys}


def _wrong(q):
    keys = ["yes", "no"] if q.format == "yesno" else [str(i) for i in range(q.n_options())]
    return next(k for k in keys if k != str(q.ground_truth))


@pytest.mark.skipif(not QUESTIONS_PATH.exists(), reason="data/questions.jsonl not generated yet")
def test_run_analysis_smoke(tmp_path):
    qs = load_questions()
    by_fmt = {}
    for q in qs:                       # three questions: one per format, tier 1 for yesno/score, tier 2 for choice
        want = 2 if q.format == "choice" else 1
        if q.tier == want and q.format not in by_fmt:
            by_fmt[q.format] = q
    q_yes, q_choice, q_score = by_fmt["yesno"], by_fmt["choice"], by_fmt["score"]
    db = tmp_path / "t.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO runs VALUES (?,?,?,?,?)", ("test-01", "2026-10-29T00:00:00+00:00", json.dumps({"runs": 2}), "h", ""))
    rows = []
    # elephant-1 (decision): tier 1 -> 3 of 4 correct, tier 2 -> 1 of 2 correct
    plan = {"elephant-1": [(q_yes, [True, True]), (q_score, [True, False]), (q_choice, [True, False])],
            "qwen3.5-9b": [(q_yes, [True, False]), (q_score, [True, False]), (q_choice, [False, False])]}
    kinds = {"elephant-1": "decision", "qwen3.5-9b": "instant"}
    for arm, per_q in plan.items():
        for q, outcomes in per_q:
            for k, right in enumerate(outcomes):
                ans = str(q.ground_truth) if right else _wrong(q)
                rows.append(_row(f"{arm}-{q.id}-{k}", "test-01", q, arm, kinds[arm], k, ans, probs=_probs(q, ans),
                                 gen_total_cost=2e-5 if arm == "elephant-1" else None))
    # a failed duplicate of a completed triple (resumed run) must be ignored
    rows.append(_row("elephant-1-dup-failed", "test-01", q_yes, "elephant-1", "decision", 1, None, status="failed", probs=None))
    # label-swap rows: elephant-1 flips on one label set, qwen3.5-9b does not
    for arm, answers in {"elephant-1": ["yes", "yes", "no"], "qwen3.5-9b": ["yes", "yes", "yes"]}.items():
        for name, ans in zip(["yes_no", "zero_one", "a_b"], answers):
            rows.append(_row(f"{arm}-ls-{name}", "test-01", q_yes, arm, kinds[arm], 0, ans, label_set=name,
                             probs=_probs(q_yes, ans), phase="labelswap"))
    for r in rows:
        conn.execute(f"INSERT INTO calls ({', '.join(r)}) VALUES ({', '.join('?' for _ in r)})", list(r.values()))
    conn.commit()
    conn.close()

    out = tmp_path / "results"
    ctx = A.run_analysis(db, out)
    assert ctx["run_id"] == "test-01"
    for name in ["metrics_by_tier.csv", "metrics_by_format.csv", "reliability.csv", "labelswap.csv", "labelswap_sensitivity.csv",
                 "matched_pair_deltas.csv", "routing.csv", "baseline.csv", "calls_summary.csv", "REPORT.md"]:
        assert (out / name).exists(), name
    assert (out / "charts" / "accuracy_vs_cost.png").exists()
    assert (out / "charts" / "accuracy_vs_cost_list.png").exists()
    assert (out / "charts" / "reliability.png").exists()
    assert (out / "charts" / "labelswap.png").exists()

    m = pd.read_csv(out / "metrics_by_tier.csv")
    e1 = m[(m.arm == "elephant-1") & (m.tier == 1)].iloc[0]
    assert e1.n == 4 and e1.accuracy == pytest.approx(0.75)
    assert e1.missing_rate == pytest.approx(0.0)                       # 2 questions x 2 runs expected, 4 completed
    assert e1.invalid_rate == pytest.approx(0.0)
    assert e1.cost_per_1k_actual == pytest.approx(2e-5 * 1000)         # gen_total_cost preferred over cost_usage
    assert e1.cost_per_1k_list == pytest.approx((300 * 0.035 / 1e6) * 1000)
    assert m[(m.arm == "elephant-1") & (m.tier == 2)].iloc[0].accuracy == pytest.approx(0.5)
    q9 = m[(m.arm == "qwen3.5-9b") & (m.tier == 1)].iloc[0]
    assert q9.accuracy == pytest.approx(0.5)
    assert q9.cost_per_1k_actual == pytest.approx(1e-5 * 1000)         # falls back to cost_usage
    assert (m.arm == "baseline").any()

    pairs = pd.read_csv(out / "matched_pair_deltas.csv")
    d = pairs[(pairs.pair_arm == "elephant-1") & (pairs.tier.astype(str) == "1") & (pairs.metric == "accuracy")].iloc[0]
    assert d.delta == pytest.approx(0.25) and d.ci_lo <= d.delta <= d.ci_hi

    sens = pd.read_csv(out / "labelswap_sensitivity.csv").set_index("arm")
    assert sens.loc["elephant-1", "sensitivity"] == pytest.approx(1.0) and sens.loc["elephant-1", "n_pairs"] == 1
    assert sens.loc["qwen3.5-9b", "sensitivity"] == pytest.approx(0.0)

    summary = pd.read_csv(out / "calls_summary.csv").set_index("arm")
    assert summary.loc["elephant-1", "failed"] == 1 and summary.loc["elephant-1", "ok"] == 9

    report = (out / "REPORT.md").read_text(encoding="utf-8")
    assert "test-01" in report and "charts/accuracy_vs_cost.png" in report and "Caveats" in report


def test_run_analysis_empty_db(tmp_path):
    db = tmp_path / "empty.sqlite"
    sqlite3.connect(db).executescript(SCHEMA)
    ctx = A.run_analysis(db, tmp_path / "results")
    assert ctx["run_id"] is None
    report = (tmp_path / "results" / "REPORT.md").read_text(encoding="utf-8")
    assert "No run data" in report
    assert (tmp_path / "results" / "metrics_by_tier.csv").exists()
