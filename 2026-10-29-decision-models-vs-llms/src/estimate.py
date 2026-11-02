"""Writes COST_ESTIMATE.md: measured tokens per call (from the dry run in data/runs.sqlite) x full call count x list price.

  python -m src.estimate [--run-id dry-a,dry-b]  # comma-separated dry runs to merge; default: most recent dry run;
                                                  # if none, estimates tokens from prompt length
  python -m src.estimate --append --section-title "Follow-up" --run-id dry-06 --arms a,b,c
                                                  # append a section for the given arms (call counts follow each arm's
                                                  # config `scope`) instead of rewriting the file

The reasoning arm is reported separately because it dominates. Wall-clock is estimated from measured p50 latency at the
concurrency cap (per model, all models in parallel).
"""
from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
from datetime import date

from .common import DB_PATH, ROOT, load_config, load_questions
from .render import render
from .run import arm_scope

CHARS_PER_TOKEN = 3.6   # rough pre-dry-run fallback only; replaced by measured tokens once a dry run exists


def call_counts(cfg: dict, n_questions: int, n_labelswap: int) -> tuple[int, int]:
    runs = cfg["counts"]["runs_per_question"]
    return n_questions * runs, n_labelswap * len(cfg["label_sets"]) * runs


def arm_calls(cfg: dict, questions, arm: str) -> tuple[int, int]:
    """(main calls, label-swap calls) for one arm under its config `scope`, from the actual questions file."""
    tiers, ls = arm_scope(cfg, arm)
    runs = cfg["counts"]["runs_per_question"]
    n_main = sum(1 for q in questions if q.tier in tiers) * runs
    n_ls = sum(1 for q in questions if q.tier in tiers and q.in_labelswap) * len(cfg["label_sets"]) * runs if ls else 0
    return n_main, n_ls


def arm_calls_by_tier(cfg: dict, questions, arm: str) -> dict[int, int]:
    """Calls per tier for one arm under its scope (main runs plus label-swap calls for the yes/no subset)."""
    tiers, ls = arm_scope(cfg, arm)
    runs = cfg["counts"]["runs_per_question"]
    out: dict[int, int] = {}
    for q in questions:
        if q.tier not in tiers:
            continue
        out[q.tier] = out.get(q.tier, 0) + runs + (len(cfg["label_sets"]) * runs if ls and q.in_labelswap else 0)
    return out


def section_lines(cfg: dict, questions, meas: dict[str, dict], arms: list[str], run_id: str, title: str) -> list[str]:
    """A self-contained estimate section for `arms` (appended to COST_ESTIMATE.md by --append).

    Where the dry run measured a tier, the full-run cost and wall-clock weight that tier's own token/latency means by
    its call count (reasoning length differs a lot between tiers); tiers the dry run did not touch use the overall mean.
    """
    per_model = cfg["concurrency"]["per_model"]
    lines = [f"## {title}", "",
             f"Generated {date.today().isoformat()}. Basis: **measured** from dry run(s) `{run_id}`. Call counts follow each arm's "
             f"`scope` in config.yaml ({cfg['counts']['runs_per_question']} runs per question; label-swap = "
             f"{len(cfg['label_sets'])} label sets x the yes/no subset within the arm's tiers). List prices are the pinned "
             "provider's endpoint prices (USD per million tokens), re-resolved on the date noted in config.yaml. Full-run $ "
             "and wall-clock are tier-weighted: each measured tier's own mean tokens / median latency x its call count; "
             "unmeasured tiers use the overall dry-run mean.", "",
             "| arm | slug | kind | tiers | label-swap | calls | in tok/call | out tok/call | reasoning tok/call | $/M in | $/M out "
             "| $/call (list, flat mean) | $/call (actual, flat mean) | p50 latency ms | full-run $ (list, tier-weighted) | provider | notes |",
             "|" + "---|" * 17]
    total, n_total = 0.0, 0
    wall: dict[str, float] = {}
    reasoning: dict[str, float] = {}
    detail: list[str] = []
    for arm in arms:
        a = cfg["arms"][arm]
        m = meas.get(arm)
        pm = a["price_per_million"]
        tiers, ls = arm_scope(cfg, arm)
        by_tier = arm_calls_by_tier(cfg, questions, arm)
        n_calls = sum(by_tier.values())
        if m and m["pt"]:
            pt, ct = statistics.mean(m["pt"]), statistics.mean(m["ct"]) if m["ct"] else 0.0
            rt = statistics.mean(m["rt"]) if m["rt"] else 0.0
            lat = statistics.median(m["lat"]) if m["lat"] else None
            actual = statistics.mean(m["gcost"] or m["cost"]) if (m["gcost"] or m["cost"]) else None
            notes = f"ok {m['ok']}, invalid {m['invalid']}, failed {m['failed']}; probs: {','.join(sorted(m['prob_sources'])) or '-'}"
            prov = ",".join(sorted(m["providers"])) or "-"
        else:
            pt, ct = estimated_tokens(cfg, questions, arm)
            rt, lat, actual, prov = 0.0, None, None, "-"
            notes = "estimated (no dry-run rows)" if not m else f"failed {m['failed']}: " + "; ".join(m["errors"][:2])
        per_call = pt / 1e6 * pm["prompt"] + ct / 1e6 * pm["completion"]

        def t_mean(key: str, t: int, fallback: float) -> float:
            vals = (m or {}).get(key, {}).get(t)
            return statistics.mean(vals) if vals else fallback

        full = 0.0
        wall_s = 0.0
        for t, n in sorted(by_tier.items()):
            pt_t, ct_t = t_mean("pt_by_tier", t, pt), t_mean("ct_by_tier", t, ct)
            full += n * (pt_t / 1e6 * pm["prompt"] + ct_t / 1e6 * pm["completion"])
            lat_vals = (m or {}).get("lat_by_tier", {}).get(t)
            lat_t = statistics.median(lat_vals) if lat_vals else lat
            if lat_t:
                wall_s += n * lat_t / 1000.0
            if m and (m.get("pt_by_tier", {}).get(t) or m.get("ct_by_tier", {}).get(t)):
                detail.append(f"- {arm}, tier {t}: {n:,} calls; measured on {len(m['pt_by_tier'].get(t, []))} dry call(s): "
                              f"{pt_t:,.0f} in / {ct_t:,.0f} out tokens per call"
                              + (f" ({t_mean('rt_by_tier', t, 0.0):,.0f} reasoning)" if a["kind"] == "reasoning" else "")
                              + (f", median latency {lat_t / 1000:.1f} s" if lat_t else "")
                              + f" -> ${n * (pt_t / 1e6 * pm['prompt'] + ct_t / 1e6 * pm['completion']):,.2f}")
            else:
                detail.append(f"- {arm}, tier {t}: {n:,} calls; not in the dry run, uses the overall mean -> "
                              f"${n * per_call:,.2f}")
        total += full
        n_total += n_calls
        if a["kind"] == "reasoning":
            reasoning[arm] = full
        if wall_s:
            wall[arm] = wall_s / per_model / 3600.0
        lines.append(f"| {arm} | {a['slug']} | {a['kind']} | {','.join(map(str, tiers))} | {'yes' if ls else 'no'} | {n_calls:,} | "
                     f"{pt:,.0f} | {ct:,.1f} | {rt:,.0f} | {pm['prompt']} | {pm['completion']} | {per_call:.6f} | "
                     f"{('%.6f' % actual) if actual is not None else '-'} | {('%.0f' % lat) if lat else '-'} | {full:,.2f} | {prov} | {notes} |")
    lines += ["", f"**Total at list price for these arms: ${total:,.2f}** over {n_total:,} calls (completion tokens include the "
              "reasoning tokens; Parasail bills them as output).", "", "Per tier:", ""] + detail + [""]
    if reasoning:
        lines.append("Reasoning arm separately: " + ", ".join(f"{k}: ${v:,.2f}" for k, v in reasoning.items()) + ".")
        for arm in reasoning:
            m = meas.get(arm)
            if m and m.get("rt_by_tier"):
                by = {t: statistics.mean(v) for t, v in sorted(m["rt_by_tier"].items())}
                pm = cfg["arms"][arm]["price_per_million"]
                n_calls = sum(arm_calls(cfg, questions, arm))
                extra_300 = 300 / 1e6 * pm["completion"] * n_calls
                cap = cfg["rendering"]["reasoning_max_tokens"]
                cap_cost = cap / 1e6 * pm["completion"]
                lines.append(f"{arm}: mean reasoning tokens per call by tier in the dry run: " +
                             ", ".join(f"tier {t}: {v:.0f}" for t, v in by.items()) +
                             f". Every extra 300 reasoning tokens per call across its {n_calls:,} calls would add ${extra_300:,.2f} at list price. "
                             f"A call that runs to the {cap:,}-token cap costs at most ${cap_cost:.4f} in output tokens.")
        lines.append("")
    if wall:
        lines += [f"Wall-clock at per-model concurrency {per_model}, arms in parallel (tier-weighted median latency from the dry run, "
                  "no retries), so the run takes about as long as the slowest arm:", ""]
        for arm, h in sorted(wall.items(), key=lambda x: -x[1]):
            lines.append(f"- {arm}: ~{h:.1f} h ({h * 60:.0f} min)")
        lines.append("")
    spent = sum(sum(m["gcost"] or m["cost"]) for a, m in meas.items() if a in arms)
    n_dry = sum(m["n"] for a, m in meas.items() if a in arms)
    lines.append(f"Dry run `{run_id}` for these arms: {n_dry} calls, ${spent:.4f} actual (usage/generation endpoint).")
    return lines


def measured(conn: sqlite3.Connection, run_ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    qs = ",".join("?" for _ in run_ids)
    rows = conn.execute(f"""SELECT arm, model_slug, prompt_tokens, completion_tokens, reasoning_tokens, cost_usage, gen_total_cost,
                                   latency_ms, status, endpoint_path, COALESCE(gen_provider, provider), prob_source, invalid_reason, error, tier
                            FROM calls WHERE run_id IN ({qs})""", run_ids).fetchall()
    for (arm, slug, pt, ct, rt, cost, gcost, lat, status, path, prov, src, inv, err, tier) in rows:
        d = out.setdefault(arm, {"slug": slug, "pt": [], "ct": [], "rt": [], "rt_by_tier": {}, "pt_by_tier": {}, "ct_by_tier": {},
                                 "lat_by_tier": {}, "cost": [], "gcost": [], "lat": [], "n": 0,
                                 "ok": 0, "invalid": 0, "failed": 0, "providers": set(), "prob_sources": set(), "errors": []})
        d["n"] += 1
        d[status] += 1
        if status == "failed":
            d["errors"].append((err or "")[:160])
            continue
        if pt is not None:
            d["pt"].append(pt)
            d["pt_by_tier"].setdefault(tier, []).append(pt)
        if ct is not None:
            d["ct"].append(ct)
            d["ct_by_tier"].setdefault(tier, []).append(ct)
        if rt is not None:
            d["rt"].append(rt)
            d["rt_by_tier"].setdefault(tier, []).append(rt)
        if cost is not None:
            d["cost"].append(cost)
        if gcost is not None:
            d["gcost"].append(gcost)
        if lat is not None:
            d["lat"].append(lat)
            d["lat_by_tier"].setdefault(tier, []).append(lat)
        if prov:
            d["providers"].add(prov)
        if src:
            d["prob_sources"].add(src)
    return out


def estimated_tokens(cfg: dict, questions, arm: str) -> tuple[float, float]:
    """Fallback: mean prompt tokens from rendered request length; completion tokens by arm kind."""
    a = cfg["arms"][arm]
    lens = []
    for q in questions:
        r = render(q, arm, a, cfg)
        if r.endpoint == "decisions":
            import json
            lens.append(len(json.dumps(r.body["state"])) + len(q.question) + 120)
        else:
            lens.append(sum(len(m["content"]) for m in r.body["messages"]))
    pt = statistics.mean(lens) / CHARS_PER_TOKEN
    ct = {"decision": 0.0, "instant": 1.0, "reasoning": 1500.0}[a["kind"]]
    return pt, ct


def actual_lines(conn: sqlite3.Connection, run_id: str) -> list[str]:
    """Actual spend, call counts and latency of a completed run, per arm (ok counts ok + invalid rows)."""
    lines = [f"## Actual spend of the full run `{run_id}`", "",
             "| arm | calls ok | calls invalid | calls failed | actual $ | p50 ms | p95 ms |", "|---|---|---|---|---|---|---|"]
    tot_actual = 0.0
    for arm, in conn.execute("SELECT DISTINCT arm FROM calls WHERE run_id=? ORDER BY arm", (run_id,)):
        ok = conn.execute("SELECT COUNT(*) FROM calls WHERE run_id=? AND arm=? AND status='ok'", (run_id, arm)).fetchone()[0]
        inv = conn.execute("SELECT COUNT(*) FROM calls WHERE run_id=? AND arm=? AND status='invalid'", (run_id, arm)).fetchone()[0]
        failed = conn.execute("SELECT COUNT(*) FROM calls WHERE run_id=? AND arm=? AND status='failed'", (run_id, arm)).fetchone()[0]
        cost = conn.execute("SELECT SUM(COALESCE(gen_total_cost, cost_usage)) FROM calls WHERE run_id=? AND arm=?", (run_id, arm)).fetchone()[0] or 0.0
        lats = sorted(r[0] for r in conn.execute("SELECT latency_ms FROM calls WHERE run_id=? AND arm=? AND status IN ('ok','invalid')", (run_id, arm)))
        p50 = lats[len(lats) // 2] if lats else 0
        p95 = lats[int(len(lats) * 0.95)] if lats else 0
        tot_actual += cost
        lines.append(f"| {arm} | {ok} | {inv} | {failed} | {cost:.4f} | {p50:.0f} | {p95:.0f} |")
    first, last = conn.execute("SELECT MIN(started_at), MAX(finished_at) FROM calls WHERE run_id=?", (run_id,)).fetchone()
    lines += ["", f"**Total actual: ${tot_actual:.4f}** (OpenRouter generation records; failed calls cost nothing). "
              f"First call started {first}, last call finished {last}."]
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    ap.add_argument("--actual-run-id", help="a completed full run to append actual spend for")
    ap.add_argument("--append", action="store_true", help="append a section for --arms instead of rewriting the file")
    ap.add_argument("--section-title", default="Follow-up")
    ap.add_argument("--arms", help="comma-separated arm names for the appended section (default: arms present in the dry run)")
    ap.add_argument("--out", default=str(ROOT / "COST_ESTIMATE.md"))
    args = ap.parse_args(argv)
    cfg = load_config()
    questions = load_questions()
    n_q = len(questions)
    n_ls = sum(q.in_labelswap for q in questions)
    n_q_target = cfg["counts"]["tier1"] + cfg["counts"]["tier2"] + cfg["counts"]["tier3"]
    main_calls, ls_calls = call_counts(cfg, n_q_target, cfg["counts"]["labelswap"])
    conn = sqlite3.connect(DB_PATH) if DB_PATH.exists() else None
    run_ids = [r.strip() for r in args.run_id.split(",")] if args.run_id else []
    meas: dict[str, dict] = {}
    if conn is not None:
        if not run_ids:
            row = conn.execute("SELECT run_id FROM runs WHERE run_id LIKE 'dry%' ORDER BY started_at DESC LIMIT 1").fetchone()
            run_ids = [row[0]] if row else []
        if run_ids:
            meas = measured(conn, run_ids)
    run_id = ", ".join(run_ids)
    if args.append:
        if args.actual_run_id:
            if conn is None:
                raise SystemExit("no database")
            out = "\n" + "\n".join(actual_lines(conn, args.actual_run_id)) + "\n"
        else:
            arms = [a.strip() for a in args.arms.split(",") if a.strip()] if args.arms else [a for a in cfg["arms"] if a in meas]
            unknown = [a for a in arms if a not in cfg["arms"]]
            if unknown:
                raise SystemExit(f"unknown arms {unknown}")
            out = "\n" + "\n".join(section_lines(cfg, questions, meas, arms, run_id, args.section_title)) + "\n"
        open(args.out, "a", encoding="utf-8").write(out)
        print(out)
        return 0
    mode = "measured" if meas else "pre-dry-run estimate (tokens approximated from prompt length; NOT measured)"

    lines = [f"# Cost estimate — {cfg['experiment']}", "",
             f"Generated {date.today().isoformat()}. Basis: **{mode}**" + (f" from dry run(s) `{run_id}`." if meas else "."), "",
             f"Questions in `data/questions.jsonl`: {n_q} (target {n_q_target}; tier 3 pending if lower). "
             f"Full run = {n_q_target} questions x {cfg['counts']['runs_per_question']} runs = {main_calls:,} calls per arm, "
             f"plus label-swap {cfg['counts']['labelswap']} x {len(cfg['label_sets'])} labels x {cfg['counts']['runs_per_question']} runs "
             f"= {ls_calls:,} calls per arm; {main_calls + ls_calls:,} calls per arm in total.", "",
             "List prices are USD per million tokens from `GET /api/v1/models` (see config.yaml). "
             "Actual cost from the dry run is shown next to the list-price projection where measured.", ""]
    hdr = "| arm | slug | status | kind | in tok/call | out tok/call | reasoning tok/call | $/M in | $/M out | $/call (list) | $/call (actual) | p50 latency ms | full-run $ (list) | provider | notes |"
    lines += [hdr, "|" + "---|" * 15]
    total_list, total_reasoning = 0.0, {}
    wall = {}
    dropped = []
    for arm, a in cfg["arms"].items():
        if a.get("status") == "dropped":
            dropped.append(f"- **{arm}** (`{a['slug']}`): {a.get('drop_reason', 'dropped')}")
            continue
        m = meas.get(arm)
        pm = a["price_per_million"]
        if m and m["pt"]:
            pt, ct = statistics.mean(m["pt"]), statistics.mean(m["ct"]) if m["ct"] else 0.0
            rt = statistics.mean(m["rt"]) if m["rt"] else 0.0
            lat = statistics.median(m["lat"]) if m["lat"] else None
            actual = statistics.mean(m["gcost"] or m["cost"]) if (m["gcost"] or m["cost"]) else None
            notes = f"ok {m['ok']}, invalid {m['invalid']}, failed {m['failed']}; probs: {','.join(sorted(m['prob_sources'])) or '-'}"
            prov = ",".join(sorted(m["providers"])) or "-"
        else:
            pt, ct = estimated_tokens(cfg, questions, arm)
            rt, lat, actual, prov = 0.0, None, None, "-"
            notes = "estimated" if not m else f"failed {m['failed']}: " + "; ".join(m["errors"][:2])
        per_call = pt / 1e6 * pm["prompt"] + ct / 1e6 * pm["completion"]
        n_calls = main_calls + ls_calls
        full = per_call * n_calls
        if a.get("status") == "selected":
            total_list += full
        if a["kind"] == "reasoning":
            total_reasoning[arm] = full
        if lat:
            wall[arm] = n_calls * (lat / 1000.0) / cfg["concurrency"]["per_model"] / 3600.0
        lines.append(f"| {arm} | {a['slug']} | {a.get('status')} | {a['kind']} | {pt:,.0f} | {ct:,.1f} | {rt:,.0f} | {pm['prompt']} | {pm['completion']} | "
                     f"{per_call:.6f} | {('%.6f' % actual) if actual is not None else '-'} | {('%.0f' % lat) if lat else '-'} | {full:,.2f} | {prov} | {notes} |")
    lines += ["", f"**Total at list price, selected arms: ${total_list:,.2f}** (per-call token counts are dry-run means; the list price "
              "is the pinned provider's endpoint price where a provider is pinned).", ""]
    if total_reasoning:
        lines.append("Reasoning arm separately (it dominates): " +
                     ", ".join(f"{k}: ${v:,.2f}" for k, v in total_reasoning.items()) + ".")
        for arm in total_reasoning:
            m = meas.get(arm)
            if m and m.get("rt_by_tier"):
                by = {t: statistics.mean(v) for t, v in sorted(m["rt_by_tier"].items())}
                pm = cfg["arms"][arm]["price_per_million"]
                extra_300 = 300 / 1e6 * pm["completion"] * (main_calls + ls_calls)
                lines.append(f"{arm}: mean reasoning tokens per call by tier in the dry run: " +
                             ", ".join(f"tier {t}: {v:.0f}" for t, v in by.items()) +
                             f". Every extra 300 reasoning tokens per call across the full run would add ${extra_300:,.2f} at list price.")
        lines.append("")
    if dropped:
        lines += ["## Dropped arms", ""] + dropped + [""]
    if wall:
        lines += ["## Wall-clock at the concurrency cap", "",
                  f"Per-model concurrency {cfg['concurrency']['per_model']}; all arms run in parallel, so wall-clock is the slowest arm "
                  "(latency p50 from the dry run, no retries, generation lookups run alongside):", ""]
        for arm, h in sorted(wall.items(), key=lambda x: -x[1]):
            lines.append(f"- {arm}: ~{h:.1f} h")
        lines.append("")
    else:
        lines += ["## Wall-clock", "", "Not estimated: no measured latency yet (run the dry run).", ""]
    lines += ["## Dry run cost", ""]
    if meas:
        spent = sum(sum(m["gcost"] or m["cost"]) for m in meas.values())
        lines.append(f"Dry run `{run_id}`: {sum(m['n'] for m in meas.values())} calls, ${spent:.4f} actual (usage/generation endpoint).")
    else:
        lines.append("No dry run recorded yet.")
    if conn is not None and args.actual_run_id:
        lines += [""] + actual_lines(conn, args.actual_run_id)
    out = "\n".join(lines) + "\n"
    open(args.out, "w", encoding="utf-8").write(out)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
