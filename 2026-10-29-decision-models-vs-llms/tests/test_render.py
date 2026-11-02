"""Rendering: one function, all arms; identical state/question per arm; label sets; request hashing."""
import json

import pytest

from src import render as R
from src.common import LETTERS, Question, load_config

CFG = load_config()
ARMS = CFG["arms"]
DEC = next(a for a in ARMS if ARMS[a]["kind"] == "decision")
INST = "qwen3.5-9b"
REAS = next(a for a in ARMS if ARMS[a]["kind"] == "reasoning")


def q_yesno(state="Ticket: please refund me."):
    return Question(id="x-yn", tier=1, format="yesno", state=state, question="Does this mention a refund?", options=None,
                    ground_truth="yes", domain="business", generator="t", generator_seed=1)


def q_choice():
    return Question(id="x-ch", tier=1, format="choice", state={"to": "Legal"}, question="Which department?",
                    options=["Finance", "Legal", "Facilities", "Marketing", "Procurement"], ground_truth=1, domain="business",
                    generator="t", generator_seed=1)


def q_score():
    return Question(id="x-sc", tier=1, format="score", state="Priority: High", question="What priority?",
                    options=["Low", "Medium", "High", "Urgent", "Critical"], ground_truth=2, domain="business",
                    generator="t", generator_seed=1)


def test_decision_yesno_is_noul():
    r = R.render(q_yesno(), DEC, ARMS[DEC], CFG)
    assert r.endpoint == "decisions"
    assert r.body["model"] == ARMS[DEC]["slug"]
    assert r.body["state"] == "Ticket: please refund me."
    qo = r.body["questions"]["q"]
    assert qo["type"] == "noul" and set(qo["criteria"]) == {"true", "false"}
    assert qo["instructions"] == "Does this mention a refund?"
    assert r.labels == ["yes", "no"] and r.answer_keys == ["yes", "no"]


def test_decision_choice_and_score():
    r = R.render(q_choice(), DEC, ARMS[DEC], CFG)
    qo = r.body["questions"]["q"]
    assert qo["type"] == "choice"
    assert list(qo["criteria"]) == q_choice().options       # text keys, native style
    assert r.body["state"] == {"to": "Legal"}                 # JSON state passed as an object
    assert r.label_to_answer["Legal"] == 1
    r = R.render(q_score(), DEC, ARMS[DEC], CFG)
    qo = r.body["questions"]["q"]
    assert qo["type"] == "score" and qo["criteria"] == q_score().options
    assert r.labels == ["0", "1", "2", "3", "4"]


def test_decision_labelswap_uses_choice_with_label_keys():
    ls = CFG["label_sets"][2]
    r = R.render(q_yesno(), DEC, ARMS[DEC], CFG, label_set=ls)
    qo = r.body["questions"]["q"]
    assert qo["type"] == "choice" and list(qo["criteria"]) == [ls["yes"], ls["no"]]
    assert r.label_to_answer == {ls["yes"]: "yes", ls["no"]: "no"}
    assert r.label_set == ls["name"]


def test_instant_body_shape():
    r = R.render(q_choice(), INST, ARMS[INST], CFG)
    b = r.body
    assert r.endpoint == "chat" and b["max_tokens"] == 1 and b["logprobs"] is True
    assert b["temperature"] == 0 and "seed" in b and b["top_logprobs"] == CFG["rendering"]["top_logprobs"]
    assert r.labels == list(LETTERS[:5]) and r.label_to_answer["B"] == 1
    assert "A) Finance" in b["messages"][1]["content"] and "B) Legal" in b["messages"][1]["content"]
    assert "{A, B, C, D, E}" in b["messages"][0]["content"]
    # logit bias covers bare and space-prefixed letter tokens with a uniform value
    lb = b["logit_bias"]
    assert len(lb) == 10 and set(lb.values()) == {CFG["rendering"]["logit_bias_value"]}
    # JSON state is pretty-printed into the prompt
    assert json.dumps({"to": "Legal"}, indent=2) in b["messages"][1]["content"]


def test_instant_score_digits_and_yesno_labels():
    r = R.render(q_score(), INST, ARMS[INST], CFG)
    assert r.labels == ["1", "2", "3", "4", "5"] and r.label_to_answer["3"] == 2
    assert "1 = Low" in r.prompt_text and "5 = Critical" in r.prompt_text
    for ls in CFG["label_sets"]:
        r = R.render(q_yesno(), INST, ARMS[INST], CFG, label_set=ls)
        assert r.labels == [ls["yes"], ls["no"]]
        assert f"Answer {ls['yes']} if the answer is yes" in r.prompt_text
        assert all(k in r.body["logit_bias"] for k in [])  # bias present
        assert r.body["logit_bias"]


def test_reasoning_body_shape():
    r = R.render(q_yesno(), REAS, ARMS[REAS], CFG)
    b = r.body
    assert "logprobs" not in b and b["max_tokens"] == CFG["rendering"]["reasoning_max_tokens"]
    assert b["reasoning"] == ARMS[REAS]["reasoning"]
    assert "ANSWER:" in b["messages"][0]["content"] and "CONFIDENCE:" in b["messages"][0]["content"]


def test_same_state_and_question_across_arms():
    q = q_choice()
    dec = R.render(q, DEC, ARMS[DEC], CFG)
    ins = R.render(q, INST, ARMS[INST], CFG)
    rea = R.render(q, REAS, ARMS[REAS], CFG)
    assert dec.body["state"] == q.state
    assert ins.prompt_text == rea.prompt_text
    assert q.state_text() in ins.prompt_text and q.question in ins.prompt_text
    assert dec.body["questions"]["q"]["instructions"] == q.question


def test_request_hash_is_deterministic_and_changes_with_label_set():
    a = R.render(q_yesno(), INST, ARMS[INST], CFG)
    b = R.render(q_yesno(), INST, ARMS[INST], CFG)
    c = R.render(q_yesno(), INST, ARMS[INST], CFG, label_set=CFG["label_sets"][1])
    assert a.request_hash == b.request_hash != c.request_hash


def test_label_set_rejected_for_non_yesno():
    with pytest.raises(ValueError):
        R.render(q_choice(), INST, ARMS[INST], CFG, label_set=CFG["label_sets"][1])


def test_prompt_variants_only_change_system():
    """HANDOFF-2.md: under variants a, b, c the user message and the allowed-token handling are identical; only the
    system content differs (and it carries the same allowed set)."""
    base = ARMS[INST]
    for q in (q_yesno(), q_choice(), q_score()):
        rend = {v: R.render(q, f"variant-{v}", {**base, "prompt_variant": v}, CFG) for v in ("a", "b", "c")}
        a = rend["a"]
        allowed = R._allowed_str(a.labels)
        assert a.body["messages"][0]["content"] == R.INSTANT_SYSTEM.format(allowed=allowed)   # a == the original prompt
        assert R.render(q, INST, base, CFG).body["messages"][0]["content"] == a.body["messages"][0]["content"]
        for v in ("b", "c"):
            r = rend[v]
            assert r.body["messages"][1] == a.body["messages"][1]                       # user message identical
            assert r.prompt_text == a.prompt_text
            assert (r.labels, r.label_to_answer, r.answer_keys) == (a.labels, a.label_to_answer, a.answer_keys)
            assert r.body["logit_bias"] == a.body["logit_bias"]                         # same allowed-token bias
            rest = {k: val for k, val in r.body.items() if k != "messages"}
            assert rest == {k: val for k, val in a.body.items() if k != "messages"}     # max_tokens, logprobs, seed, ...
            sys_v = r.body["messages"][0]["content"]
            assert sys_v != a.body["messages"][0]["content"]
            assert sys_v == R.INSTANT_SYSTEM_VARIANTS[v].format(allowed=allowed) and allowed in sys_v
            assert r.request_hash != a.request_hash
        assert rend["b"].body["messages"][0]["content"] != rend["c"].body["messages"][0]["content"]
    # the wording is exactly the handoff's
    assert R.INSTANT_SYSTEM_VARIANTS["b"] == "Reply with one token: {allowed}. Nothing else."
    assert R.INSTANT_SYSTEM_VARIANTS["c"].startswith("You are grading a document against a question.")
    assert R.INSTANT_SYSTEM_VARIANTS["c"].endswith("Output the single token only.")
    assert ARMS["qwen3.5-9b-prompt-b"]["prompt_variant"] == "b" and ARMS["qwen3.5-9b-prompt-c"]["prompt_variant"] == "c"
    with pytest.raises(ValueError):
        R.render(q_yesno(), "bad", {**base, "prompt_variant": "z"}, CFG)


def test_followup_prompt_arms_match_control_except_system():
    """The configured prompt-b/-c arms differ from qwen3.5-9b only in the system message."""
    for name in ("qwen3.5-9b-prompt-b", "qwen3.5-9b-prompt-c"):
        for q in (q_yesno(), q_choice(), q_score()):
            a = R.render(q, INST, ARMS[INST], CFG)
            r = R.render(q, name, ARMS[name], CFG)
            assert r.body["messages"][1] == a.body["messages"][1]
            assert {k: v for k, v in r.body.items() if k != "messages"} == {k: v for k, v in a.body.items() if k != "messages"}
            assert r.body["messages"][0]["content"] != a.body["messages"][0]["content"]


def test_followup_reasoning_arm_shape():
    """qwen3.5-9b-reasoning: same slug, provider pin, seed and temperature as the control; thinking on; one-token
    ANSWER line; logprobs requested (recorded only); user prompt identical to the one-token arm's."""
    name = "qwen3.5-9b-reasoning"
    a = ARMS[name]
    assert a["slug"] == ARMS[INST]["slug"] and a["provider"] == ARMS[INST]["provider"]
    assert a["seed"] == ARMS[INST]["seed"] and a["temperature"] == 0 and a["pairs_with"] == INST
    for q in (q_yesno(), q_choice(), q_score()):
        r = R.render(q, name, a, CFG)
        b = r.body
        assert b["reasoning"] == {"enabled": True} and b["max_tokens"] == a["reasoning_max_tokens"] == 12000
        assert R.render(q, REAS, ARMS[REAS], CFG).body["max_tokens"] == CFG["rendering"]["reasoning_max_tokens"] == 6000
        assert b["logprobs"] is True and b["top_logprobs"] == a["top_logprobs"] and "logit_bias" not in b
        assert b["temperature"] == 0 and b["seed"] == ARMS[INST]["seed"] and b["provider"] == ARMS[INST]["provider"]
        assert "ANSWER:" in b["messages"][0]["content"] and "CONFIDENCE:" in b["messages"][0]["content"]
        assert b["messages"][1] == R.render(q, INST, ARMS[INST], CFG).body["messages"][1]
        assert r.labels == R.render(q, INST, ARMS[INST], CFG).labels


def test_all_configured_arms_render_every_format():
    for name, arm in ARMS.items():
        for q in (q_yesno(), q_choice(), q_score()):
            r = R.render(q, name, arm, CFG)
            assert r.body["model"] == arm["slug"]
            assert len(r.labels) == len(r.answer_keys) == (2 if q.format == "yesno" else len(q.options))
