"""Runner: checkpointing, resume, retry/backoff, failed recording, raw files. Uses an httpx mock transport (no network)."""
import asyncio
import json
import math

import httpx
import pytest

from src import run as RUN
from src.common import load_config, load_questions


@pytest.fixture
def cfg():
    c = load_config()
    c = json.loads(json.dumps(c))
    c["retry"]["backoff_base_s"] = 0.001
    c["retry"]["backoff_max_s"] = 0.002
    c["concurrency"]["generation_lookup_delay_s"] = 0.0
    return c


@pytest.fixture
def questions():
    qs = load_questions()
    by_id = {q.id: q for q in qs}
    return [by_id[i] for i in RUN.DRY_RUN_QUESTIONS]


class Fake:
    """Scripted OpenRouter: decisions, chat, generation endpoints; optional 429s before success."""

    def __init__(self, fail_first=0, hard_fail=False):
        self.calls = 0
        self.fail_first = fail_first
        self.hard_fail = hard_fail
        self.seen_bodies = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith("/api/v1/generation"):
            gid = request.url.params["id"]
            return httpx.Response(200, json={"data": {"id": gid, "total_cost": 0.000123, "provider_name": "FakeProv",
                                                      "latency": 321, "generation_time": 100, "native_tokens_prompt": 50,
                                                      "native_tokens_completion": 1, "native_tokens_reasoning": 0}})
        self.calls += 1
        body = json.loads(request.content)
        self.seen_bodies.append(body)
        if self.hard_fail:
            return httpx.Response(400, json={"error": {"code": 400, "message": "Invalid request parameters"}})
        if self.calls <= self.fail_first:
            return httpx.Response(429, json={"error": {"code": 429, "message": "Rate limit exceeded"}})
        if "questions" in body:
            qo = body["questions"]["q"]
            if qo["type"] == "noul":
                ans = {"type": "noul", "noul": 0.9}
            elif qo["type"] == "choice":
                keys = list(qo["criteria"])
                ans = {"type": "choice", "choice": keys[0], "confidence": 0.8,
                       "probabilities": {k: (0.8 if i == 0 else 0.2 / (len(keys) - 1)) for i, k in enumerate(keys)}}
            else:
                ans = {"type": "score", "score": 2.0, "confidence": 0.9, "probabilities": {str(i): (0.9 if i == 2 else 0.025) for i in range(5)}}
            return httpx.Response(200, json={"id": f"gen-dec-{self.calls}", "model": body["model"] + "-20261014", "provider": "FakeProv",
                                             "answers": {"q": ans}, "usage": {"input_tokens": 400, "output_tokens": 10, "cost": 0.0000168}})
        # chat
        allowed = body["messages"][0]["content"].split("{")[1].split("}")[0].split(", ")
        tok = allowed[0]
        lp = {"content": [{"token": tok, "logprob": math.log(0.7),
                           "top_logprobs": [{"token": a, "logprob": math.log(0.7 if a == tok else 0.3 / (len(allowed) - 1))} for a in allowed]}]}
        return httpx.Response(200, json={"id": f"gen-{self.calls}", "model": body["model"], "provider": "FakeProv",
                                         "choices": [{"message": {"role": "assistant", "content": tok}, "logprobs": lp, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 300, "completion_tokens": 1, "cost": 0.0000513}})


def _runner(cfg, store, run_id, fake, tmp_path):
    r = RUN.Runner(cfg, run_id, "sk-test-secret", store)
    r.client = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    r.raw_dir = tmp_path / "raw" / run_id
    r.raw_dir.mkdir(parents=True, exist_ok=True)
    return r


def test_dry_items_and_checkpoint(cfg, questions, tmp_path):
    store = RUN.Store(tmp_path / "runs.sqlite")
    arms = ["jev-1.14", "qwen3.5-9b"]
    items = RUN.build_dry_items(cfg, questions, arms, [q.id for q in questions])
    # jev: 3 base + 1 labelswap + 3 systemone-endpoint; qwen: 3 + 1
    assert len(items) == 7 + 4
    fake = Fake()
    runner = _runner(cfg, store, "t-dry", fake, tmp_path)
    asyncio.run(runner.run_all(items, progress_every=100))
    rows = store.conn.execute("SELECT arm, status, endpoint_path, label_set, is_correct, probs, cost_usage, gen_total_cost, gen_provider, raw_path FROM calls").fetchall()
    assert len(rows) == 11 and all(r[1] == "ok" for r in rows)
    assert {r[2] for r in rows if r[0] == "jev-1.14"} == {cfg["openrouter"]["decisions_path"], cfg["openrouter"]["systemone_path"]}
    assert any(r[3] == "zero_one" for r in rows)
    assert all(r[6] is not None and r[7] == 0.000123 and r[8] == "FakeProv" for r in rows)
    for r in rows:
        probs = json.loads(r[5])
        assert abs(sum(probs.values()) - 1) < 1e-9
    # raw files exist, contain the request but never the key
    files = list((tmp_path / "raw" / "t-dry").glob("*.json"))
    assert len(files) == 22  # call + generation file each
    for f in files:
        assert "sk-test-secret" not in f.read_text(encoding="utf-8")
    # request hashes recorded and bodies carry max_tokens=1 for the instant arm
    assert all("max_tokens" in b and b["max_tokens"] == 1 for b in fake.seen_bodies if "messages" in b)


def test_resume_skips_completed_and_retries_failed(cfg, questions, tmp_path):
    store = RUN.Store(tmp_path / "runs.sqlite")
    arms = ["elephant-1"]
    items = RUN.build_items(cfg, questions, arms, 2, [1, 2], True, "main")
    assert len(items) == 3 * 2 + 1 * 3 * 2  # base + labelswap (one yesno question in the dry set)
    # first pass: everything hard-fails -> recorded as failed, 5 attempts each? (400 is not retried)
    fake = Fake(hard_fail=True)
    runner = _runner(cfg, store, "t-main", fake, tmp_path)
    asyncio.run(runner.run_all(items, progress_every=100))
    rows = store.conn.execute("SELECT status, attempts, error, http_status FROM calls WHERE run_id='t-main'").fetchall()
    assert len(rows) == len(items) and all(r[0] == "failed" and r[1] == 1 and r[3] == 400 for r in rows)
    assert store.completed("t-main") == set()
    # second pass with --resume semantics: nothing completed, all retried, now succeeding after 2 rate limits
    fake = Fake(fail_first=2)
    runner = _runner(cfg, store, "t-main", fake, tmp_path)
    remaining = [it for it in items if (it.question.id, it.arm, it.run_index, it.label_set_name, RUN._default_path(cfg, it)) not in store.completed("t-main")]
    assert len(remaining) == len(items)
    asyncio.run(runner.run_all(remaining, progress_every=100))
    rows = store.conn.execute("SELECT status, attempts FROM calls WHERE run_id='t-main'").fetchall()
    assert len(rows) == 2 * len(items)   # failed rows are kept, new rows appended
    ok = [r for r in rows if r[0] == "ok"]
    assert len(ok) == len(items)
    assert sum(r[1] for r in ok) == len(items) + 2   # two 429s were retried with backoff
    assert len(store.completed("t-main")) == len(items)
    # third pass: resume skips everything
    remaining = [it for it in items if (it.question.id, it.arm, it.run_index, it.label_set_name, RUN._default_path(cfg, it)) not in store.completed("t-main")]
    assert remaining == []


def test_retry_exhaustion_records_failed(cfg, questions, tmp_path):
    store = RUN.Store(tmp_path / "runs.sqlite")
    items = RUN.build_items(cfg, questions[:1], ["qwen3.5-27b"], 1, [1], False, "main")
    fake = Fake(fail_first=99)
    runner = _runner(cfg, store, "t-x", fake, tmp_path)
    asyncio.run(runner.run_all(items, progress_every=100))
    (status, attempts, http) = store.conn.execute("SELECT status, attempts, http_status FROM calls").fetchone()
    assert status == "failed" and attempts == cfg["retry"]["max_attempts"] and http == 429


def test_upstream_error_in_200_body_is_retried(cfg, questions, tmp_path):
    store = RUN.Store(tmp_path / "runs.sqlite")
    items = RUN.build_items(cfg, questions[:1], ["qwen3.5-27b"], 1, [1], False, "main")
    fake = Fake()
    real = fake.handler

    def handler(request):
        if request.url.path.startswith("/api/v1/generation"):
            return real(request)
        fake.calls += 1
        if fake.calls <= 2:
            return httpx.Response(200, json={"id": "gen-x", "error": {"message": "Upstream error from Alibaba: Request rate increased too quickly", "code": 429}})
        fake.calls -= 1
        return real(request)

    runner = _runner(cfg, store, "t-u", fake, tmp_path)
    runner.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    asyncio.run(runner.run_all(items, progress_every=100))
    (status, attempts) = store.conn.execute("SELECT status, attempts FROM calls").fetchone()
    assert status == "ok" and attempts == 3


def test_arm_scope_limits_tiers_and_labelswap(cfg, questions):
    """HANDOFF-2.md: prompt arms run tier 2 only with no label-swap; the reasoning arm runs everything. The CLI tier
    filter still applies on top, and the dry run exercises every format but skips the label-swap call for an arm
    whose scope excludes it."""
    assert RUN.arm_scope(cfg, "qwen3.5-9b") == ([1, 2, 3], True)
    assert RUN.arm_scope(cfg, "qwen3.5-9b-reasoning") == ([1, 2, 3], True)
    assert RUN.arm_scope(cfg, "qwen3.5-9b-prompt-b") == ([2], False) == RUN.arm_scope(cfg, "qwen3.5-9b-prompt-c")
    items = RUN.build_items(cfg, questions, ["qwen3.5-9b-prompt-b", "qwen3.5-9b-reasoning"], 2, [1, 2, 3], True, "main")
    pb = [it for it in items if it.arm == "qwen3.5-9b-prompt-b"]
    rs = [it for it in items if it.arm == "qwen3.5-9b-reasoning"]
    assert len(pb) == 1 * 2 and {it.question.tier for it in pb} == {2} and all(it.label_set is None for it in pb)
    assert len(rs) == 3 * 2 + 1 * 3 * 2
    assert RUN.build_items(cfg, questions, ["qwen3.5-9b-prompt-b"], 1, [1, 3], True, "main") == []
    dry = RUN.build_dry_items(cfg, questions, ["qwen3.5-9b-prompt-b", "qwen3.5-9b-reasoning"], [q.id for q in questions])
    assert len([it for it in dry if it.arm == "qwen3.5-9b-prompt-b"]) == 3
    assert len([it for it in dry if it.arm == "qwen3.5-9b-reasoning"]) == 4
    # the real follow-up work list: 3,000 + 1,000 + 1,000 calls
    allq = load_questions()
    full = RUN.build_items(cfg, allq, ["qwen3.5-9b-reasoning", "qwen3.5-9b-prompt-b", "qwen3.5-9b-prompt-c"], 5, [1, 2, 3], True, "main")
    by_arm = {}
    for it in full:
        by_arm[it.arm] = by_arm.get(it.arm, 0) + 1
    assert by_arm == {"qwen3.5-9b-reasoning": 3000, "qwen3.5-9b-prompt-b": 1000, "qwen3.5-9b-prompt-c": 1000}


def test_select_arms(cfg):
    assert RUN.select_arms(cfg, None, False) == [a for a in cfg["arms"] if cfg["arms"][a]["status"] == "selected"]
    dry = RUN.select_arms(cfg, None, True)
    assert all(cfg["arms"][a]["status"] != "dropped" for a in dry)
    assert [a for a in dry if cfg["arms"][a]["kind"] == "reasoning"] == [a for a in cfg["arms"] if cfg["arms"][a]["kind"] == "reasoning" and cfg["arms"][a]["status"] == "selected"]
    assert RUN.select_arms(cfg, "qwen3.5-9b,typesafe/jev-1.14", False) == ["qwen3.5-9b", "jev-1.14"]
    assert RUN.select_arms(cfg, "qwen3.5-9b-reasoning,qwen3.5-9b-prompt-b,qwen3.5-9b-prompt-c", False) == \
        ["qwen3.5-9b-reasoning", "qwen3.5-9b-prompt-b", "qwen3.5-9b-prompt-c"]
    with pytest.raises(SystemExit):
        RUN.select_arms(cfg, "nope", False)
    with pytest.raises(SystemExit):   # four arms share this slug since the follow-up; the arm name is required
        RUN.select_arms(cfg, "qwen/qwen3.5-9b", False)
