"""Every question must validate against its own checker; structural invariants of the question set."""
import re
from collections import Counter

import pytest

from src import generate as G
from src.common import FORMATS, QUESTIONS_PATH, SCORE_LEVELS, Question, load_config, load_questions


@pytest.fixture(scope="module")
def questions() -> list[Question]:
    if not QUESTIONS_PATH.exists():
        pytest.skip("data/questions.jsonl not generated yet")
    return load_questions()


def test_every_question_passes_its_checker(questions):
    failures = [q.id for q in questions if not G.check_question(q)]
    assert failures == []


def test_ids_unique(questions):
    ids = [q.id for q in questions]
    assert len(ids) == len(set(ids))


def test_schema(questions):
    for q in questions:
        assert q.format in FORMATS
        assert q.tier in (1, 2, 3)
        assert q.question.strip()
        assert isinstance(q.generator_seed, int)
        assert q.domain
        if q.format == "yesno":
            assert q.options is None and q.ground_truth in ("yes", "no")
        elif q.format == "choice":
            assert 4 <= len(q.options) <= 8, q.id
            assert len(set(q.options)) == len(q.options), q.id
            assert isinstance(q.ground_truth, int) and 0 <= q.ground_truth < len(q.options)
            assert q.ground_truth_text == q.options[q.ground_truth]
        else:
            assert len(q.options) == SCORE_LEVELS
            assert isinstance(q.ground_truth, int) and 0 <= q.ground_truth < SCORE_LEVELS


def test_counts_and_format_mix(questions):
    cfg = load_config()
    by_tier = Counter(q.tier for q in questions)
    assert by_tier[1] == cfg["counts"]["tier1"]
    assert by_tier[2] == cfg["counts"]["tier2"]
    for t in (1, 2):
        c = Counter(q.format for q in questions if q.tier == t)
        n = sum(c.values())
        assert abs(c["yesno"] / n - 0.4) < 0.05
        assert abs(c["choice"] / n - 0.4) < 0.05
        assert abs(c["score"] / n - 0.2) < 0.05
    if by_tier[3]:
        assert by_tier[3] == cfg["counts"]["tier3"]


def test_yesno_balance(questions):
    for t in (1, 2):
        ys = [q.ground_truth for q in questions if q.tier == t and q.format == "yesno"]
        frac = ys.count("yes") / len(ys)
        assert 0.35 <= frac <= 0.65, (t, frac)


def test_labelswap_subset(questions):
    cfg = load_config()
    ls = [q for q in questions if q.in_labelswap]
    assert len(ls) == cfg["counts"]["labelswap"]
    assert all(q.format == "yesno" and q.tier in (1, 2) for q in ls)
    assert Counter(q.tier for q in ls) == {1: cfg["counts"]["labelswap"] // 2, 2: cfg["counts"]["labelswap"] // 2}


def test_domains_vary_in_tier1(questions):
    doms = {q.domain for q in questions if q.tier == 1}
    assert {"business", "legal", "medical", "technical", "casual"} <= doms


def test_tier3_flagged_as_weak_ground_truth(questions):
    for q in questions:
        if q.tier == 3:
            assert q.meta.get("cross_checked") is True
            assert "ground_truth_note" in q.meta


def test_generation_is_deterministic():
    cfg = load_config()
    a = G.generate_tiers_1_2(cfg["seed"])
    b = G.generate_tiers_1_2(cfg["seed"])
    assert [q.to_dict() for q in a] == [q.to_dict() for q in b]


def test_filler_contains_no_topic_keywords():
    """Tier-1 ticket filler must never contain a topic keyword, or the planted 'no' answers would be wrong."""
    for dom, sents in G.FILLER.items():
        for s in sents + G.GENERIC_NOTES:
            for t, d in G.TOPICS.items():
                assert re.search(d["kw"], s, re.I) is None, (dom, s, t)


def test_topic_sentences_only_mention_their_own_topic():
    for t, d in G.TOPICS.items():
        for s in d["sents"]:
            for t2, d2 in G.TOPICS.items():
                if t2 != t:
                    assert re.search(d2["kw"], s, re.I) is None, (t, s, t2)


def test_checkers_reject_wrong_answers():
    """Flip the ground truth of generated questions and make sure the checker notices (checker is not vacuous)."""
    cfg = load_config()
    qs = G.generate_tiers_1_2(cfg["seed"])
    by_gen: dict[str, Question] = {}
    for q in qs:
        by_gen.setdefault((q.generator, q.format), q)
    for (gen, fmt), q in by_gen.items():
        d = q.to_dict()
        if fmt == "yesno":
            d["ground_truth"] = "no" if d["ground_truth"] == "yes" else "yes"
        else:
            d["ground_truth"] = (d["ground_truth"] + 1) % len(d["options"])
        assert not G.check_question(Question.from_dict(d)), (gen, fmt)


def test_tier3_loader_keeps_only_agreed_and_well_formed(monkeypatch, tmp_path):
    import json
    cache = {"generator": "gen/model", "checker": "chk/model", "generated_at": "2026-10-29", "candidates": [
        {"format": "yesno", "domain": "geo", "question": "Is X a capital?", "options": None, "answer": "yes", "agree": True},
        {"format": "yesno", "domain": "geo", "question": "Is Y a capital?", "options": None, "answer": "no", "agree": False},
        {"format": "choice", "domain": "sci", "question": "Which?", "options": ["a", "b", "c", "d"], "answer": 2, "agree": True},
        {"format": "choice", "domain": "sci", "question": "Too few?", "options": ["a", "b"], "answer": 0, "agree": True},
        {"format": "score", "domain": "hist", "question": "When?", "options": ["<1800", "1800s", "1900-49", "1950-99", "2000+"], "answer": 3, "agree": True},
        {"format": "score", "domain": "hist", "question": "Bad bins", "options": ["a", "b"], "answer": 0, "agree": True},
    ]}
    p = tmp_path / "tier3.json"
    p.write_text(json.dumps(cache), encoding="utf-8")
    monkeypatch.setattr(G, "TIER3_CACHE", p)
    qs = G.load_tier3(1, 50)
    assert [q.format for q in qs] == ["yesno", "choice", "score"]
    assert qs[0].ground_truth == "yes" and qs[1].ground_truth == 2 and qs[2].ground_truth == 3
    assert all(q.tier == 3 and q.meta["cross_checked"] and q.state == G.TIER3_STATE for q in qs)
    assert all(G.check_question(q) for q in qs)


def test_tier2_solvers_independent_examples():
    # meeting rooms: overlap at the boundary is not an overlap
    st = {"rooms": [{"name": "Room A", "capacity": 5}],
          "meetings": [{"name": "m1", "start": "09:00", "end": "10:00", "attendees": 3},
                       {"name": "m2", "start": "10:00", "end": "11:00", "attendees": 5}],
          "assignment": {"m1": "Room A", "m2": "Room A"}}
    assert G.meeting_violations(st) == (set(), set())
    st["meetings"][1]["start"] = "09:30"
    assert G.meeting_violations(st)[0] == {"m1", "m2"}
    st["meetings"][1]["start"] = "10:00"
    st["meetings"][1]["attendees"] = 6
    assert G.meeting_violations(st) == ({"m2"}, {"Room A"})
    # expense: meal per person
    p = {"meal_max_per_person": 40, "receipt_required_over": 25, "trip": {"start": "2026-03-10", "end": "2026-03-12"}}
    it = {"category": "meal", "amount": 120.0, "people": 3, "receipt": True, "preapproved": False, "date": "2026-03-11"}
    assert G.item_violations(p, it) == []
    it["amount"] = 121.0
    assert G.item_violations(p, it) == [1]
    it.update({"category": "hotel", "amount": 300.0, "receipt": False, "date": "2026-03-13"})
    assert G.item_violations(p, it) == [2, 4, 5]
    # event order
    ev = ["a", "b", "c"]
    cons = [{"t": "before", "a": "a", "b": "b"}, {"t": "imm_before", "a": "b", "b": "c"}]
    assert G.consistent_orders(ev, cons) == [("a", "b", "c")]
