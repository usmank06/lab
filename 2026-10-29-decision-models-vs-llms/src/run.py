"""Runner: sends every (question, arm, run) call through OpenRouter, checkpointing each call to SQLite.

  python -m src.run --dry-run                       # 3 questions x 1 run x every non-dropped arm (+ endpoint checks)
  python -m src.run --run-id main-01                # full run (selected arms, all tiers, 5 runs) + label-swap set
  python -m src.run --run-id main-01 --resume       # skip completed (question, arm, run, label_set) rows
  python -m src.run --run-id main-01 --models qwen3.5-9b,elephant-1 --tiers 1,2 --runs 5 --no-labelswap
  python -m src.run --status [--run-id main-01]

Every call records: arm, model slug, provider, run index, request body hash, raw response JSON, probability vector,
parsed answer, client-measured latency (request start to last byte), tokens, usage cost, generation-endpoint cost,
timestamp, error. Raw responses go to data/raw/<run_id>/<call_id>.json (and <call_id>.generation.json).
Nothing is deleted or overwritten; a resumed run appends new rows (failed rows stay).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .common import (DB_PATH, RAW_DIR, ROOT, Question, canonical_json, get_api_key, load_config, load_questions,
                     scrub_secret, sha256_text)
from .render import Rendered, parse_response, render

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, started_at TEXT, args TEXT, config_hash TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS calls (
  call_id TEXT PRIMARY KEY,
  run_id TEXT, phase TEXT,
  question_id TEXT, tier INTEGER, format TEXT, arm TEXT, model_slug TEXT, kind TEXT,
  run_index INTEGER, label_set TEXT,
  endpoint TEXT, endpoint_path TEXT, request_hash TEXT, request_body TEXT,
  status TEXT, http_status INTEGER, error TEXT, attempts INTEGER,
  response_json TEXT, response_headers TEXT,
  provider TEXT, response_model TEXT, generation_id TEXT,
  latency_ms REAL, started_at TEXT, finished_at TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER, reasoning_tokens INTEGER, cost_usage REAL,
  gen_total_cost REAL, gen_provider TEXT, gen_latency_ms REAL, gen_generation_time_ms REAL,
  gen_native_prompt INTEGER, gen_native_completion INTEGER, gen_native_reasoning INTEGER, gen_json TEXT,
  probs TEXT, prob_source TEXT, parsed_label TEXT, parsed_answer TEXT, ground_truth TEXT, is_correct INTEGER,
  invalid_reason TEXT, confidence REAL, allowed_mass REAL, parse_extra TEXT, raw_path TEXT);
CREATE INDEX IF NOT EXISTS idx_calls_key ON calls(run_id, question_id, arm, run_index, label_set);
CREATE INDEX IF NOT EXISTS idx_calls_arm ON calls(run_id, arm, status);
"""

DRY_RUN_QUESTIONS = ["t1-ticket_topic-yesno-000", "t2-meeting_rooms-choice-000", "t1-fields_present-score-000"]


@dataclass
class WorkItem:
    phase: str
    question: Question
    arm: str
    run_index: int
    label_set: dict | None
    endpoint_path: str | None = None   # override (dry run: exercise /api/v1/systemone too)
    arm_override: dict | None = None   # dry run: e.g. reasoning effort override

    @property
    def label_set_name(self) -> str:
        return self.label_set["name"] if self.label_set else ""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_upstream_error_body(resp: httpx.Response) -> bool:
    try:
        d = resp.json()
    except ValueError:
        return False
    return isinstance(d, dict) and "error" in d and "choices" not in d and "answers" not in d


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.lock = asyncio.Lock()

    def completed(self, run_id: str) -> set[tuple]:
        rows = self.conn.execute(
            "SELECT question_id, arm, run_index, label_set, endpoint_path FROM calls WHERE run_id=? AND status IN ('ok','invalid')",
            (run_id,)).fetchall()
        return set(rows)

    async def insert(self, row: dict) -> None:
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        async with self.lock:
            self.conn.execute(f"INSERT INTO calls ({cols}) VALUES ({qs})", list(row.values()))
            self.conn.commit()

    async def update(self, call_id: str, row: dict) -> None:
        sets = ", ".join(f"{k}=?" for k in row)
        async with self.lock:
            self.conn.execute(f"UPDATE calls SET {sets} WHERE call_id=?", [*row.values(), call_id])
            self.conn.commit()

    def register_run(self, run_id: str, args: str, config_hash: str, note: str) -> None:
        self.conn.execute("INSERT OR IGNORE INTO runs VALUES (?,?,?,?,?)", (run_id, now_iso(), args, config_hash, note))
        self.conn.commit()


class Runner:
    def __init__(self, cfg: dict, run_id: str, api_key: str, store: Store, reasoning_effort: str | None = None):
        self.cfg, self.run_id, self.key, self.store = cfg, run_id, api_key, store
        orc = cfg["openrouter"]
        self.base = orc["base_url"]
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", **orc.get("app_headers", {})}
        self.sems = {arm: asyncio.Semaphore(cfg["concurrency"]["per_model"]) for arm in cfg["arms"]}
        self.gen_sem = asyncio.Semaphore(cfg["concurrency"]["generation_lookup"])
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(cfg["timeouts"]["request_s"]),
                                        limits=httpx.Limits(max_connections=64, max_keepalive_connections=32))
        self.raw_dir = RAW_DIR / run_id
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.counts: dict[str, dict[str, int]] = {}
        self.reasoning_effort = reasoning_effort

    def arm_cfg(self, item: WorkItem) -> dict:
        a = dict(self.cfg["arms"][item.arm])
        if item.arm_override:
            a.update(item.arm_override)
        if self.reasoning_effort and a["kind"] == "reasoning":
            a["reasoning"] = {"effort": self.reasoning_effort}
        return a

    def _path_for(self, r: Rendered, item: WorkItem) -> str:
        if item.endpoint_path:
            return item.endpoint_path
        orc = self.cfg["openrouter"]
        return orc["decisions_path"] if r.endpoint == "decisions" else orc["chat_path"]

    async def _post(self, path: str, body: dict, timeout: float) -> tuple[httpx.Response | None, Exception | None, float, int]:
        """POST with retry/backoff. Returns (response, exception, latency_ms of final attempt, attempts)."""
        rc = self.cfg["retry"]
        attempts = 0
        last_exc: Exception | None = None
        resp: httpx.Response | None = None
        latency = 0.0
        while attempts < rc["max_attempts"]:
            attempts += 1
            t0 = time.perf_counter()
            try:
                resp = await self.client.post(self.base + path, headers=self.headers, json=body, timeout=timeout)
                await resp.aread()
                latency = (time.perf_counter() - t0) * 1000.0
                last_exc = None
                retry = resp.status_code in rc["retry_statuses"]
                if not retry and resp.status_code == 200 and _is_upstream_error_body(resp):
                    retry = True   # OpenRouter can relay an upstream error (e.g. provider rate limit) as HTTP 200 + error body
                if retry and attempts < rc["max_attempts"]:
                    await asyncio.sleep(self._backoff(attempts, resp))
                    continue
                return resp, None, latency, attempts
            except (httpx.HTTPError, OSError) as e:
                latency = (time.perf_counter() - t0) * 1000.0
                last_exc = e
                resp = None
                if attempts < rc["max_attempts"]:
                    await asyncio.sleep(self._backoff(attempts, None))
        return resp, last_exc, latency, attempts

    def _backoff(self, attempt: int, resp: httpx.Response | None) -> float:
        rc = self.cfg["retry"]
        if resp is not None:
            ra = resp.headers.get("retry-after")
            if ra and ra.isdigit():
                return min(float(ra), rc["backoff_max_s"])
        return min(rc["backoff_base_s"] * (2 ** (attempt - 1)) + random.uniform(0, 0.5), rc["backoff_max_s"])

    async def execute(self, item: WorkItem) -> None:
        q = item.question
        arm_cfg = self.arm_cfg(item)
        r = render(q, item.arm, arm_cfg, self.cfg, item.label_set)
        path = self._path_for(r, item)
        call_id = f"{item.arm}-{q.id}-r{item.run_index}-{item.label_set_name or 'base'}-{uuid.uuid4().hex[:8]}"
        timeout = self.cfg["timeouts"]["reasoning_request_s"] if r.kind == "reasoning" else self.cfg["timeouts"]["request_s"]
        started = now_iso()
        async with self.sems[item.arm]:
            resp, exc, latency_ms, attempts = await self._post(path, r.body, timeout)
        finished = now_iso()
        raw_file = self.raw_dir / f"{call_id}.json"
        try:
            raw_rel = str(raw_file.relative_to(ROOT))
        except ValueError:
            raw_rel = str(raw_file)
        row = {
            "call_id": call_id, "run_id": self.run_id, "phase": item.phase, "question_id": q.id, "tier": q.tier,
            "format": q.format, "arm": item.arm, "model_slug": arm_cfg["slug"], "kind": r.kind, "run_index": item.run_index,
            "label_set": item.label_set_name, "endpoint": r.endpoint, "endpoint_path": path, "request_hash": r.request_hash,
            "request_body": canonical_json(r.body), "attempts": attempts, "latency_ms": latency_ms,
            "started_at": started, "finished_at": finished, "ground_truth": str(q.ground_truth),
            "raw_path": raw_rel,
        }
        headers = dict(resp.headers) if resp is not None else {}
        row["response_headers"] = json.dumps(headers)
        data = None
        if resp is not None:
            row["http_status"] = resp.status_code
            try:
                data = resp.json()
            except ValueError:
                data = {"_non_json_body": resp.text[:4000]}
            row["response_json"] = json.dumps(scrub_secret(data, self.key), ensure_ascii=False)
        if exc is not None or resp is None or resp.status_code != 200 or (isinstance(data, dict) and "error" in data and "choices" not in data and "answers" not in data):
            row["status"] = "failed"
            row["error"] = repr(exc) if exc is not None else json.dumps((data or {}).get("error", data))[:2000]
            self._count(item.arm, "failed")
        else:
            parsed = parse_response(r, data)
            usage = data.get("usage") or {}
            row.update({
                "provider": data.get("provider") or next((v for k, v in headers.items() if "provider" in k.lower()), None),
                "response_model": data.get("model"), "generation_id": data.get("id"),
                "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
                "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "cost_usage": usage.get("cost"),
                "probs": json.dumps(parsed.probs) if parsed.probs is not None else None,
                "prob_source": parsed.prob_source, "parsed_label": parsed.label, "parsed_answer": parsed.answer,
                "is_correct": int(parsed.valid and parsed.answer == str(q.ground_truth)),
                "invalid_reason": parsed.invalid_reason, "confidence": parsed.confidence, "allowed_mass": parsed.allowed_mass,
                "parse_extra": json.dumps({**parsed.extra, "missing_labels": parsed.missing_labels, "raw_text": parsed.raw_text}),
                "status": "ok" if parsed.valid else "invalid",
            })
            self._count(item.arm, row["status"])
        await self.store.insert(row)
        raw = {"call_id": call_id, "run_id": self.run_id, "arm": item.arm, "question_id": q.id, "run_index": item.run_index,
               "label_set": item.label_set_name, "endpoint_path": path, "request": r.body, "request_hash": r.request_hash,
               "http_status": row.get("http_status"), "latency_ms": latency_ms, "attempts": attempts,
               "started_at": started, "finished_at": finished, "response_headers": headers, "response": data,
               "error": row.get("error")}
        raw_file.write_text(json.dumps(scrub_secret(raw, self.key), indent=1, ensure_ascii=False), encoding="utf-8")
        if row.get("generation_id") and self.cfg["concurrency"].get("generation_lookup", 0) > 0:
            await self.lookup_generation(call_id, row["generation_id"])

    async def lookup_generation(self, call_id: str, gen_id: str) -> None:
        cc = self.cfg["concurrency"]
        path = self.cfg["openrouter"]["generation_path"]
        last = None
        data = None
        await asyncio.sleep(cc.get("generation_lookup_delay_s", 6.0))   # records appear ~7 s after the call
        for attempt in range(cc.get("generation_lookup_attempts", 8)):
            async with self.gen_sem:   # only the GET holds the slot; waiting happens outside it
                try:
                    resp = await self.client.get(self.base + path, headers=self.headers, params={"id": gen_id}, timeout=30)
                    last = {"status": resp.status_code, "body": resp.text[:300]}
                    if resp.status_code == 200:
                        data = resp.json()
                        break
                    if resp.status_code not in (404, 429, 500, 502, 503):
                        data = {"_error_status": resp.status_code, "_body": resp.text[:1000]}
                        break
                except (httpx.HTTPError, OSError) as e:
                    last = {"exception": repr(e)}
            await asyncio.sleep(cc.get("generation_lookup_retry_s", 3.0))
        if data is None:
            data = {"_error": "generation lookup failed", "_last": last}
        d = data.get("data") if isinstance(data, dict) else None
        upd = {"gen_json": json.dumps(scrub_secret(data, self.key))}
        if isinstance(d, dict):
            upd.update({"gen_total_cost": d.get("total_cost"), "gen_provider": d.get("provider_name"),
                        "gen_latency_ms": d.get("latency"), "gen_generation_time_ms": d.get("generation_time"),
                        "gen_native_prompt": d.get("native_tokens_prompt"), "gen_native_completion": d.get("native_tokens_completion"),
                        "gen_native_reasoning": d.get("native_tokens_reasoning")})
        await self.store.update(call_id, upd)
        (self.raw_dir / f"{call_id}.generation.json").write_text(json.dumps(scrub_secret(data, self.key), indent=1), encoding="utf-8")

    def _count(self, arm: str, status: str) -> None:
        c = self.counts.setdefault(arm, {"ok": 0, "invalid": 0, "failed": 0})
        c[status] += 1

    async def run_all(self, items: list[WorkItem], progress_every: float = 15.0) -> None:
        totals: dict[str, int] = {}
        for it in items:
            totals[it.arm] = totals.get(it.arm, 0) + 1
        tasks = [asyncio.create_task(self.execute(it)) for it in items]
        t0 = time.time()

        async def reporter():
            while True:
                await asyncio.sleep(progress_every)
                self.print_progress(totals, time.time() - t0)

        rep = asyncio.create_task(reporter())
        try:
            await asyncio.gather(*tasks)
        finally:
            rep.cancel()
            self.print_progress(totals, time.time() - t0)
            await self.client.aclose()

    def print_progress(self, totals: dict[str, int], elapsed: float) -> None:
        parts = []
        for arm, total in totals.items():
            c = self.counts.get(arm, {"ok": 0, "invalid": 0, "failed": 0})
            done = sum(c.values())
            parts.append(f"{arm}: {done}/{total} (ok {c['ok']}, invalid {c['invalid']}, failed {c['failed']})")
        print(f"[{elapsed:7.0f}s] " + " | ".join(parts), flush=True)


# --------------------------------------------------------------------------------------
# Work list construction
# --------------------------------------------------------------------------------------
def select_arms(cfg: dict, models: str | None, dry_run: bool) -> list[str]:
    arms = cfg["arms"]
    if models:
        wanted = [m.strip() for m in models.split(",") if m.strip()]
        out = []
        for w in wanted:
            if w in arms:
                out.append(w)
                continue
            match = [a for a in arms if arms[a]["slug"] == w]
            if not match:
                raise SystemExit(f"unknown arm/model {w!r}; known: {list(arms)}")
            if len(match) > 1:   # several arms share a slug (the follow-up control arms); name the arm instead
                raise SystemExit(f"slug {w!r} matches several arms {match}; pass the arm name")
            out += match
        return out
    if dry_run:
        return [a for a in arms if arms[a].get("status") != "dropped"]
    return [a for a in arms if arms[a].get("status") == "selected"]


def arm_scope(cfg: dict, arm: str) -> tuple[list[int], bool]:
    """(tiers, labelswap) an arm is limited to by its config `scope`; default every tier and the label-swap subset."""
    sc = cfg["arms"][arm].get("scope") or {}
    tiers = [int(t) for t in sc.get("tiers", [1, 2, 3])]
    return tiers, bool(sc.get("labelswap", True))


def build_items(cfg: dict, questions: list[Question], arms: list[str], runs: int, tiers: list[int], labelswap: bool,
                phase: str, limit: int | None = None) -> list[WorkItem]:
    qs = [q for q in questions if q.tier in tiers]
    if limit:
        qs = qs[:limit]
    items: list[WorkItem] = []
    for arm in arms:
        arm_tiers, _ = arm_scope(cfg, arm)
        for q in qs:
            if q.tier not in arm_tiers:
                continue
            for k in range(runs):
                items.append(WorkItem(phase, q, arm, k, None))
    if labelswap:
        for arm in arms:
            arm_tiers, arm_ls = arm_scope(cfg, arm)
            if not arm_ls:
                continue
            for q in qs:
                if not q.in_labelswap or q.tier not in arm_tiers:
                    continue
                for ls in cfg["label_sets"]:
                    for k in range(runs):
                        items.append(WorkItem("labelswap", q, arm, k, ls))
    return items


def build_dry_items(cfg: dict, questions: list[Question], arms: list[str], qids: list[str]) -> list[WorkItem]:
    by_id = {q.id: q for q in questions}
    qs = [by_id[i] for i in qids]
    items: list[WorkItem] = []
    for arm in arms:
        for q in qs:   # the dry run exercises every format regardless of the arm's tier scope
            items.append(WorkItem("dry", q, arm, 0, None))
        # one label-swap call per arm (exercises the choice-with-label-keys path for decision models)
        yn = next((q for q in qs if q.format == "yesno"), None)
        if yn is not None and arm_scope(cfg, arm)[1]:
            items.append(WorkItem("dry", yn, arm, 0, cfg["label_sets"][1]))
        # the System One endpoint for TypeSafe (the handoff named /v1/systemone; docs say /api/alpha/decisions is general)
        if cfg["arms"][arm]["slug"].startswith("typesafe/"):
            for q in qs:
                items.append(WorkItem("dry", q, arm, 0, None, endpoint_path=cfg["openrouter"]["systemone_path"]))
    return items


def status_report(store: Store, run_id: str | None) -> None:
    where, params = ("WHERE run_id=?", (run_id,)) if run_id else ("", ())
    rows = store.conn.execute(f"""SELECT run_id, phase, arm, status, COUNT(*), AVG(latency_ms), SUM(cost_usage), SUM(gen_total_cost)
                                  FROM calls {where} GROUP BY run_id, phase, arm, status ORDER BY run_id, phase, arm, status""", params).fetchall()
    print(f"{'run_id':<22} {'phase':<10} {'arm':<22} {'status':<8} {'n':>6} {'lat_ms':>9} {'cost_usage':>11} {'cost_gen':>11}")
    for r in rows:
        print(f"{r[0]:<22} {r[1]:<10} {r[2]:<22} {r[3]:<8} {r[4]:>6} {r[5] or 0:>9.0f} {r[6] or 0:>11.5f} {r[7] or 0:>11.5f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-id")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--models", help="comma-separated arm names or slugs")
    ap.add_argument("--tiers", default="1,2,3")
    ap.add_argument("--runs", type=int)
    ap.add_argument("--no-labelswap", action="store_true")
    ap.add_argument("--limit", type=int, help="first N questions per tier filter (debug)")
    ap.add_argument("--questions", help="comma-separated question ids (dry run)")
    ap.add_argument("--reasoning-effort", help="override reasoning effort for reasoning arms (dry runs)")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    cfg = load_config()
    store = Store(DB_PATH)
    if args.status:
        status_report(store, args.run_id)
        return 0
    key = get_api_key(cfg)
    if not key:
        print(f"{cfg['openrouter']['api_key_env']} is not set; nothing sent.", file=sys.stderr)
        return 2
    questions = load_questions()
    tiers = [int(t) for t in args.tiers.split(",")]
    arms = select_arms(cfg, args.models, args.dry_run)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.dry_run:
        run_id = args.run_id or f"dry-{stamp}"
        qids = args.questions.split(",") if args.questions else DRY_RUN_QUESTIONS
        items = build_dry_items(cfg, questions, arms, qids)
    else:
        run_id = args.run_id or f"main-{stamp}"
        runs = args.runs or cfg["counts"]["runs_per_question"]
        items = build_items(cfg, questions, arms, runs, tiers, not args.no_labelswap, "main", args.limit)
    if args.resume:
        if not args.run_id:
            raise SystemExit("--resume needs --run-id")
        done = store.completed(run_id)
        before = len(items)
        items = [it for it in items if (it.question.id, it.arm, it.run_index, it.label_set_name,
                                        it.endpoint_path or _default_path(cfg, it)) not in done]
        print(f"resume {run_id}: {before - len(items)} completed, {len(items)} remaining")
    config_hash = sha256_text(canonical_json(cfg))
    store.register_run(run_id, json.dumps(vars(args)), config_hash, args.note)
    print(f"run_id={run_id} arms={arms} calls={len(items)} concurrency={cfg['concurrency']['per_model']}/model")
    runner = Runner(cfg, run_id, key, store, reasoning_effort=args.reasoning_effort)
    try:
        asyncio.run(runner.run_all(items))
    except KeyboardInterrupt:
        print("interrupted; completed calls are checkpointed. Re-run with --resume --run-id", run_id)
        return 130
    status_report(store, run_id)
    return 0


def _default_path(cfg: dict, it: WorkItem) -> str:
    kind = cfg["arms"][it.arm]["kind"]
    return cfg["openrouter"]["decisions_path"] if kind == "decision" else cfg["openrouter"]["chat_path"]


if __name__ == "__main__":
    sys.exit(main())
