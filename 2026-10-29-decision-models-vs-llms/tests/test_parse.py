"""Response parsing for each arm kind, including invalid outputs."""
import math

import pytest

from src import render as R
from src.common import Question, load_config

CFG = load_config()
ARMS = CFG["arms"]
DEC = next(a for a in ARMS if ARMS[a]["kind"] == "decision")
INST = "qwen3.5-9b"
REAS = next(a for a in ARMS if ARMS[a]["kind"] == "reasoning")


def q_yesno():
    return Question(id="x", tier=1, format="yesno", state="s", question="q?", options=None, ground_truth="yes",
                    domain="d", generator="t", generator_seed=1)


def q_choice():
    return Question(id="x", tier=1, format="choice", state="s", question="q?", options=["Alpha", "Beta", "Gamma", "Delta"],
                    ground_truth=2, domain="d", generator="t", generator_seed=1)


def q_score():
    return Question(id="x", tier=1, format="score", state="s", question="q?", options=["L1", "L2", "L3", "L4", "L5"],
                    ground_truth=3, domain="d", generator="t", generator_seed=1)


# ---- decision ------------------------------------------------------------------------------
def test_decision_noul():
    r = R.render(q_yesno(), DEC, ARMS[DEC], CFG)
    p = R.parse_response(r, {"answers": {"q": {"type": "noul", "noul": 0.83}}, "usage": {}})
    assert p.valid and p.answer == "yes" and p.probs == {"yes": 0.83, "no": pytest.approx(0.17)}
    p = R.parse_response(r, {"answers": {"q": {"type": "noul", "noul": 0.2}}})
    assert p.answer == "no" and p.prob_source == "decision"


def test_decision_choice_text_keys():
    r = R.render(q_choice(), DEC, ARMS[DEC], CFG)
    resp = {"answers": {"q": {"type": "choice", "choice": "Gamma", "confidence": 0.7,
                              "probabilities": {"Alpha": 0.1, "Beta": 0.1, "Gamma": 0.7, "Delta": 0.1}}}}
    p = R.parse_response(r, resp)
    assert p.valid and p.answer == "2" and p.label == "Gamma" and p.probs["2"] == pytest.approx(0.7)
    # unknown choice key -> invalid, probabilities still recorded
    resp["answers"]["q"]["choice"] = "Omega"
    p = R.parse_response(r, resp)
    assert not p.valid and p.invalid_reason.startswith("choice")


def test_decision_score():
    r = R.render(q_score(), DEC, ARMS[DEC], CFG)
    resp = {"answers": {"q": {"type": "score", "score": 3.1, "confidence": 0.9,
                              "probabilities": {"0": 0.0, "1": 0.0, "2": 0.1, "3": 0.8, "4": 0.1}}}}
    p = R.parse_response(r, resp)
    assert p.valid and p.answer == "3" and p.extra["score"] == 3.1
    assert sum(p.probs.values()) == pytest.approx(1.0)


def test_decision_missing_answers():
    r = R.render(q_yesno(), DEC, ARMS[DEC], CFG)
    p = R.parse_response(r, {"error": {"message": "bad"}})
    assert not p.valid and p.probs is None


# ---- instant ---------------------------------------------------------------------------------
def _chat(content, top=None, first_lp=None, finish="stop"):
    lp = None
    if top is not None:
        lp = {"content": [{"token": content, "logprob": first_lp if first_lp is not None else math.log(0.5),
                           "top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in top]}]}
    return {"choices": [{"message": {"role": "assistant", "content": content}, "logprobs": lp, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1}}


def test_instant_logprobs_renormalised():
    r = R.render(q_choice(), INST, ARMS[INST], CFG)
    p = R.parse_response(r, _chat("C", top=[("C", 0.5), ("A", 0.2), (" C", 0.1), ("The", 0.1), ("B", 0.05)]))
    assert p.valid and p.answer == "2" and p.label == "C" and p.prob_source == "logprobs"
    assert p.allowed_mass == pytest.approx(0.85)
    assert p.probs["2"] == pytest.approx(0.6 / 0.85)        # "C" and " C" variants summed
    assert p.probs["3"] == 0.0 and "D" in p.missing_labels
    assert sum(p.probs.values()) == pytest.approx(1.0)


def test_instant_yesno_case_variants_and_labelsets():
    r = R.render(q_yesno(), INST, ARMS[INST], CFG)
    p = R.parse_response(r, _chat("Yes", top=[("Yes", 0.4), (" yes", 0.2), ("No", 0.3), ("maybe", 0.1)]))
    assert p.valid and p.answer == "yes" and p.probs["yes"] == pytest.approx(0.6 / 0.9)
    ls = CFG["label_sets"][1]  # 1 = yes, 0 = no
    r = R.render(q_yesno(), INST, ARMS[INST], CFG, label_set=ls)
    p = R.parse_response(r, _chat("0", top=[("0", 0.7), ("1", 0.3)]))
    assert p.valid and p.answer == "no" and p.probs == {"yes": pytest.approx(0.3), "no": pytest.approx(0.7)}


def test_instant_invalid_token_is_recorded_not_hidden():
    r = R.render(q_score(), INST, ARMS[INST], CFG)
    p = R.parse_response(r, _chat("The", top=[("The", 0.9), ("3", 0.1)]))
    assert not p.valid and p.invalid_reason == "token not in allowed set" and p.raw_text == "The"
    assert p.probs is not None and p.probs["2"] == pytest.approx(1.0)   # distribution over allowed tokens still kept
    p = R.parse_response(r, _chat("", top=[]))
    assert not p.valid


def test_instant_without_logprobs_falls_back_to_onehot():
    r = R.render(q_yesno(), INST, ARMS[INST], CFG)
    p = R.parse_response(r, _chat("no"))
    assert p.valid and p.answer == "no" and p.prob_source == "none" and p.probs == {"yes": 0.0, "no": 1.0}


def test_instant_chosen_token_outside_top_list_is_added():
    r = R.render(q_yesno(), INST, ARMS[INST], CFG)
    p = R.parse_response(r, _chat("no", top=[("yes", 0.5), ("Yes", 0.1)], first_lp=math.log(0.4)))
    assert p.probs["no"] == pytest.approx(0.4)


# ---- reasoning -------------------------------------------------------------------------------
def _reason(content, reasoning="thinking..."):
    return {"choices": [{"message": {"role": "assistant", "content": content, "reasoning": reasoning}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 300, "completion_tokens_details": {"reasoning_tokens": 280}}}


def test_reasoning_verbalized():
    r = R.render(q_choice(), REAS, ARMS[REAS], CFG)
    p = R.parse_response(r, _reason("Some thoughts.\n\nANSWER: C\nCONFIDENCE: 85"))
    assert p.valid and p.answer == "2" and p.confidence == 0.85 and p.prob_source == "verbalized"
    assert p.probs["2"] == pytest.approx(0.85) and p.probs["0"] == pytest.approx(0.05)
    p = R.parse_response(r, _reason("**ANSWER:** B\n**CONFIDENCE:** 60"))
    assert p.valid and p.answer == "1" and p.confidence == 0.6
    p = R.parse_response(r, _reason("I think the answer is C."))
    assert not p.valid and p.invalid_reason == "no ANSWER line"
    p = R.parse_response(r, _reason("ANSWER: Z\nCONFIDENCE: 50"))
    assert not p.valid and p.invalid_reason == "answer not in allowed set"
    p = R.parse_response(r, _reason("ANSWER: A"))
    assert p.valid and p.confidence is None and p.extra.get("confidence_missing")


def test_reasoning_yesno_labelset():
    ls = CFG["label_sets"][2]
    r = R.render(q_yesno(), REAS, ARMS[REAS], CFG, label_set=ls)
    p = R.parse_response(r, _reason("ANSWER: B\nCONFIDENCE: 70"))
    assert p.valid and p.answer == "no" and p.probs == {"yes": pytest.approx(0.3), "no": pytest.approx(0.7)}


def _lp(tok, p, top=None):
    e = {"token": tok, "logprob": math.log(p)}
    if top:
        e["top_logprobs"] = [{"token": t, "logprob": math.log(q)} for t, q in top]
    return e


def test_reasoning_answer_token_logprobs_recorded_but_calibration_stays_verbalized():
    """Per-token logprobs (as a vLLM provider returns them) are reduced to the answer token's label distribution and
    stored in parse_extra; probs/prob_source still come from the verbalized confidence. The forward search from the
    last ANSWER: match must not pick the CONFIDENCE digits, which collide with score labels."""
    arm = ARMS["qwen3.5-9b-reasoning"]
    r = R.render(q_score(), "qwen3.5-9b-reasoning", arm, CFG)
    content = "Let me think.\n\nANSWER: 4\nCONFIDENCE: 85"
    toks = [_lp("Let", 0.9), _lp(" me", 0.9), _lp(" think", 0.9), _lp(".\n\n", 0.9), _lp("ANSWER", 0.9), _lp(":", 0.99),
            _lp(" 4", 0.6, [(" 4", 0.6), (" 3", 0.3), (" 5", 0.05), (" 2", 0.03)]), _lp("\n", 0.99),
            _lp("CONF", 0.9), _lp("IDENCE", 0.99), _lp(":", 0.99), _lp(" 8", 0.5, [(" 8", 0.5), (" 9", 0.4)]), _lp("5", 0.7, [("5", 0.7), ("0", 0.3)])]
    resp = _reason(content)
    resp["choices"][0]["logprobs"] = {"content": toks}
    p = R.parse_response(r, resp)
    assert p.valid and p.answer == "3" and p.confidence == 0.85 and p.prob_source == "verbalized"
    assert p.probs["3"] == pytest.approx(0.85)
    t = p.extra["answer_token_logprobs"]
    assert t["index"] == 6 and t["token"] == " 4" and t["n_tokens"] == 13
    assert t["allowed_mass"] == pytest.approx(0.98) and t["label_probs"]["3"] == pytest.approx(0.6 / 0.98)
    assert t["label_probs"]["2"] == pytest.approx(0.3 / 0.98) and t["label_mass"]["1"] == 0.0
    # no logprobs in the response -> nothing recorded, parse unchanged
    p2 = R.parse_response(r, _reason(content))
    assert p2.valid and "answer_token_logprobs" not in p2.extra
    # logprobs present but the ANSWER line is missing -> invalid, nothing recorded
    resp3 = _reason("I give up.")
    resp3["choices"][0]["logprobs"] = {"content": toks[:4]}
    p3 = R.parse_response(r, resp3)
    assert not p3.valid and "answer_token_logprobs" not in p3.extra
