"""One rendering function for every arm, and the matching response parsers.

render(question, arm_name, arm_cfg, cfg, label_set=None) -> Rendered
parse_response(rendered, response_json) -> Parsed

Arm kinds:
  decision  — OpenRouter Decisions request: state + one typed question (noul | choice | score).
  instant   — chat completion, max_tokens=1, logprobs on, answer constrained to one token from the allowed set.
  reasoning — chat completion with reasoning enabled; final answer as one token + verbalized 0-100 confidence.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .common import LETTERS, ROOT, SCORE_LEVELS, Question, canonical_json, sha256_text

NOUL_CRITERIA = {"true": "The answer to the question is yes.", "false": "The answer to the question is no."}

INSTANT_SYSTEM = ("You are a precise classifier. Read the STATE and answer the QUESTION. Reply with exactly one token "
                  "from the allowed set {allowed} and nothing else: no explanation, no punctuation, no extra words.")
# Prompt-sensitivity variants (HANDOFF-2.md). Only the system wording differs; the user message and the allowed-token
# handling are identical. An arm selects one with `prompt_variant: a|b|c` (default a).
INSTANT_SYSTEM_VARIANTS = {
    "a": INSTANT_SYSTEM,
    "b": "Reply with one token: {allowed}. Nothing else.",
    "c": ("You are grading a document against a question. Read the document carefully, then answer the question with "
          "exactly one token from this set: {allowed}. Do not explain. Do not add punctuation. Output the single token only."),
}
REASONING_SYSTEM = ("You are a careful analyst. Read the STATE and answer the QUESTION. Think it through, then finish your "
                    "reply with exactly these two lines and nothing after them:\n"
                    "ANSWER: <exactly one of {allowed}>\n"
                    "CONFIDENCE: <an integer from 0 to 100: your probability that your answer is correct>")


@dataclass
class Rendered:
    arm: str
    kind: str
    endpoint: str                      # "decisions" | "chat"
    body: dict
    labels: list[str]                  # output labels, in canonical order
    label_to_answer: dict[str, Any]    # output label -> canonical answer ("yes"/"no" | option index | level index)
    answer_keys: list[str]             # canonical answer keys as strings, in canonical order (probability vector keys)
    label_set: str | None = None
    request_hash: str = ""
    prompt_text: str = ""              # chat arms: the user prompt (for inspection)

    def canonical_of(self, label: str) -> str:
        return str(self.label_to_answer[label])


@dataclass
class Parsed:
    valid: bool
    answer: str | None                 # canonical answer key as string, or None
    label: str | None                  # output label as produced by the model
    probs: dict[str, float] | None     # canonical answer key -> probability (renormalised over the allowed set)
    prob_source: str                   # "decision" | "logprobs" | "verbalized" | "none"
    raw_text: str | None = None
    confidence: float | None = None    # reasoning arm: verbalized 0-1; decision arms: reported confidence if any
    invalid_reason: str | None = None
    allowed_mass: float | None = None  # instant: probability mass on allowed tokens before renormalisation
    missing_labels: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------
def output_labels(q: Question, kind: str, cfg: dict, label_set: dict | None) -> tuple[list[str], dict[str, Any], list[str]]:
    """Returns (labels, label_to_answer, answer_keys) for the question under this arm kind."""
    if q.format == "yesno":
        ls = label_set or cfg["label_sets"][0]
        labels = [ls["yes"], ls["no"]]
        return labels, {ls["yes"]: "yes", ls["no"]: "no"}, ["yes", "no"]
    if q.format == "choice":
        n = len(q.options)
        keys = [str(i) for i in range(n)]
        if kind == "decision":
            if cfg["rendering"].get("decision_choice_keys", "text") == "text":
                labels = list(q.options)
            else:
                labels = list(LETTERS[:n])
        else:
            labels = list(LETTERS[:n])
        return labels, {lab: i for i, lab in enumerate(labels)}, keys
    # score
    keys = [str(i) for i in range(SCORE_LEVELS)]
    if kind == "decision":
        labels = keys[:]                                   # probabilities are keyed "0".."4"
        return labels, {k: int(k) for k in keys}, keys
    labels = [str(i + 1) for i in range(SCORE_LEVELS)]     # digits 1-5
    return labels, {lab: i for i, lab in enumerate(labels)}, keys


def _label_variants(label: str) -> list[str]:
    """Token spellings that count as this label (bare and space-prefixed; case variants for word labels)."""
    base = [label]
    if label.isalpha() and len(label) > 1:
        base = list(dict.fromkeys([label, label.lower(), label.capitalize(), label.upper()]))
    out = []
    for b in base:
        out += [b, " " + b]
    return out


_TOKEN_IDS: dict | None = None


def load_token_ids(cfg: dict) -> dict:
    global _TOKEN_IDS
    if _TOKEN_IDS is None:
        path = ROOT / cfg["rendering"]["token_ids_file"]
        _TOKEN_IDS = json.loads(path.read_text(encoding="utf-8"))
    return _TOKEN_IDS


def logit_bias_for(labels: list[str], cfg: dict) -> dict[str, float]:
    ids = load_token_ids(cfg)["single_token_ids"]
    value = cfg["rendering"]["logit_bias_value"]
    bias: dict[str, float] = {}
    for lab in labels:
        for v in _label_variants(lab):
            key = v.strip()
            if key in ids and v in ids[key]:
                bias[str(ids[key][v])] = value
    if not bias:
        raise ValueError(f"no token ids for labels {labels}")
    return bias


# --------------------------------------------------------------------------------------
# Prompt text (chat arms)
# --------------------------------------------------------------------------------------
def _options_block(q: Question, labels: list[str], label_set: dict | None, cfg: dict) -> str:
    if q.format == "yesno":
        ls = label_set or cfg["label_sets"][0]
        return f"Answer {ls['yes']} if the answer is yes, or {ls['no']} if the answer is no."
    if q.format == "choice":
        lines = [f"{lab}) {opt}" for lab, opt in zip(labels, q.options)]
        return "OPTIONS:\n" + "\n".join(lines) + "\n\nAnswer with the letter of the correct option."
    lines = [f"{lab} = {opt}" for lab, opt in zip(labels, q.options)]
    return "SCALE (ordered from lowest to highest):\n" + "\n".join(lines) + "\n\nAnswer with the digit of the correct level."


def user_prompt(q: Question, labels: list[str], label_set: dict | None, cfg: dict) -> str:
    return f"STATE:\n{q.state_text()}\n\nQUESTION: {q.question}\n\n{_options_block(q, labels, label_set, cfg)}"


def _allowed_str(labels: list[str]) -> str:
    return "{" + ", ".join(labels) + "}"


def instant_system(arm_cfg: dict, labels: list[str]) -> str:
    variant = str(arm_cfg.get("prompt_variant", "a"))
    if variant not in INSTANT_SYSTEM_VARIANTS:
        raise ValueError(f"unknown prompt_variant {variant!r}; known: {sorted(INSTANT_SYSTEM_VARIANTS)}")
    return INSTANT_SYSTEM_VARIANTS[variant].format(allowed=_allowed_str(labels))


# --------------------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------------------
def render(q: Question, arm_name: str, arm_cfg: dict, cfg: dict, label_set: dict | None = None) -> Rendered:
    kind = arm_cfg["kind"]
    if label_set is not None and q.format != "yesno":
        raise ValueError("label sets apply to yesno questions only")
    labels, l2a, keys = output_labels(q, kind, cfg, label_set)
    slug = arm_cfg["slug"]

    if kind == "decision":
        if q.format == "yesno" and label_set is None:
            qobj = {"type": "noul", "instructions": q.question, "criteria": dict(NOUL_CRITERIA)}
        elif q.format == "yesno":
            qobj = {"type": "choice", "instructions": q.question,
                    "criteria": {label_set["yes"]: NOUL_CRITERIA["true"], label_set["no"]: NOUL_CRITERIA["false"]}}
        elif q.format == "choice":
            if cfg["rendering"].get("decision_choice_keys", "text") == "text":
                criteria = {opt: None for opt in q.options}
            else:
                criteria = {lab: opt for lab, opt in zip(labels, q.options)}
            qobj = {"type": "choice", "instructions": q.question, "criteria": criteria}
        else:
            qobj = {"type": "score", "instructions": q.question, "criteria": list(q.options)}
        body = {"model": slug, "state": q.state, "questions": {"q": qobj}}
        r = Rendered(arm=arm_name, kind=kind, endpoint="decisions", body=body, labels=labels, label_to_answer=l2a,
                     answer_keys=keys, label_set=label_set["name"] if label_set else None)
    elif kind == "instant":
        prompt = user_prompt(q, labels, label_set, cfg)
        body: dict[str, Any] = {
            "model": slug,
            "messages": [{"role": "system", "content": instant_system(arm_cfg, labels)},
                         {"role": "user", "content": prompt}],
            "max_tokens": 1,
            "logprobs": True,
        }
        sup = arm_cfg.get("supports", {})
        if sup.get("top_logprobs", True):
            body["top_logprobs"] = arm_cfg.get("top_logprobs", cfg["rendering"]["top_logprobs"])
        if sup.get("temperature", True) and "temperature" in arm_cfg:
            body["temperature"] = arm_cfg["temperature"]
        if sup.get("seed", True) and "seed" in arm_cfg:
            body["seed"] = arm_cfg["seed"]
        if arm_cfg.get("logit_bias"):
            body["logit_bias"] = logit_bias_for(labels, cfg)
        if arm_cfg.get("reasoning") is not None:
            body["reasoning"] = dict(arm_cfg["reasoning"])
        if arm_cfg.get("provider"):
            body["provider"] = dict(arm_cfg["provider"])
        r = Rendered(arm=arm_name, kind=kind, endpoint="chat", body=body, labels=labels, label_to_answer=l2a,
                     answer_keys=keys, label_set=label_set["name"] if label_set else None, prompt_text=prompt)
    elif kind == "reasoning":
        prompt = user_prompt(q, labels, label_set, cfg)
        body = {
            "model": slug,
            "messages": [{"role": "system", "content": REASONING_SYSTEM.format(allowed=_allowed_str(labels))},
                         {"role": "user", "content": prompt}],
            "max_tokens": arm_cfg.get("reasoning_max_tokens", cfg["rendering"]["reasoning_max_tokens"]),   # per-arm override
            "reasoning": dict(arm_cfg.get("reasoning", {"enabled": True})),
        }
        sup = arm_cfg.get("supports", {})
        if sup.get("seed") and "seed" in arm_cfg:
            body["seed"] = arm_cfg["seed"]
        if sup.get("temperature") and "temperature" in arm_cfg:
            body["temperature"] = arm_cfg["temperature"]
        if arm_cfg.get("logprobs") and sup.get("logprobs"):
            # Record the answer token's logprobs where the provider returns them (HANDOFF-2.md); calibration for a
            # reasoning arm stays verbalized, the logprob distribution only goes into parse_extra.
            body["logprobs"] = True
            if sup.get("top_logprobs"):
                body["top_logprobs"] = arm_cfg.get("top_logprobs", cfg["rendering"]["top_logprobs"])
        if arm_cfg.get("provider"):
            body["provider"] = dict(arm_cfg["provider"])
        r = Rendered(arm=arm_name, kind=kind, endpoint="chat", body=body, labels=labels, label_to_answer=l2a,
                     answer_keys=keys, label_set=label_set["name"] if label_set else None, prompt_text=prompt)
    else:
        raise ValueError(f"unknown arm kind {kind}")
    r.request_hash = sha256_text(canonical_json(r.body))
    return r


# --------------------------------------------------------------------------------------
# parse
# --------------------------------------------------------------------------------------
def _match_label(text: str | None, labels: list[str]) -> str | None:
    if text is None:
        return None
    t = text.strip()
    if t in labels:
        return t
    for lab in labels:
        if lab.isalpha() and len(lab) > 1 and t.lower() == lab.lower():
            return lab
    return None


def _normalise(p: dict[str, float]) -> dict[str, float]:
    s = sum(p.values())
    if s <= 0:
        return p
    return {k: v / s for k, v in p.items()}


def parse_decision(r: Rendered, resp: dict) -> Parsed:
    ans = (resp.get("answers") or {}).get("q")
    if not isinstance(ans, dict):
        return Parsed(False, None, None, None, "decision", invalid_reason="no answers.q in response")
    t = ans.get("type")
    try:
        if t == "noul":
            p = float(ans["noul"])
            probs = {"yes": p, "no": 1.0 - p}
            answer = "yes" if p >= 0.5 else "no"
            return Parsed(True, answer, answer, probs, "decision", confidence=max(p, 1 - p), extra={"noul": p})
        if t == "choice":
            raw = ans.get("probabilities") or {}
            probs = {}
            missing = []
            for lab in r.labels:
                if lab in raw:
                    probs[r.canonical_of(lab)] = float(raw[lab])
                else:
                    probs[r.canonical_of(lab)] = 0.0
                    missing.append(lab)
            unknown = [k for k in raw if k not in r.labels]
            choice = ans.get("choice")
            if choice not in r.labels:
                return Parsed(False, None, str(choice), _normalise(probs), "decision",
                              invalid_reason=f"choice {choice!r} not in allowed set", missing_labels=missing,
                              extra={"unknown_keys": unknown, "confidence": ans.get("confidence")})
            return Parsed(True, r.canonical_of(choice), choice, _normalise(probs), "decision",
                          confidence=ans.get("confidence"), missing_labels=missing, extra={"unknown_keys": unknown})
        if t == "score":
            raw = ans.get("probabilities") or {}
            probs = {k: float(raw.get(k, 0.0)) for k in r.answer_keys}
            missing = [k for k in r.answer_keys if k not in raw]
            if sum(probs.values()) <= 0:
                return Parsed(False, None, None, probs, "decision", invalid_reason="empty probabilities", missing_labels=missing)
            answer = max(probs, key=probs.get)
            return Parsed(True, answer, answer, _normalise(probs), "decision", confidence=ans.get("confidence"),
                          missing_labels=missing, extra={"score": ans.get("score"), "legend": ans.get("legend")})
    except (KeyError, TypeError, ValueError) as e:
        return Parsed(False, None, None, None, "decision", invalid_reason=f"parse error: {e!r}")
    return Parsed(False, None, None, None, "decision", invalid_reason=f"unexpected answer type {t!r}")


def _first_token_logprobs(resp: dict) -> tuple[list[dict], dict | None]:
    """(top_logprobs list for the first generated token, the first token's own entry) or ([], None)."""
    try:
        lp = resp["choices"][0]["logprobs"]
        content = lp.get("content") or []
        if not content:
            return [], None
        first = content[0]
        return list(first.get("top_logprobs") or []), {"token": first.get("token"), "logprob": first.get("logprob")}
    except (KeyError, IndexError, TypeError, AttributeError):
        return [], None


def parse_instant(r: Rendered, resp: dict) -> Parsed:
    try:
        msg = resp["choices"][0]["message"]
        text = msg.get("content")
    except (KeyError, IndexError, TypeError):
        return Parsed(False, None, None, None, "none", invalid_reason="no choices[0].message")
    label = _match_label(text, r.labels)
    top, first = _first_token_logprobs(resp)
    entries = list(top)
    if first and first.get("token") is not None and first.get("logprob") is not None:
        if not any(e.get("token") == first["token"] for e in entries):
            entries.append(first)
    probs_raw: dict[str, float] = {}
    missing: list[str] = []
    if entries:
        for lab in r.labels:
            variants = set(_label_variants(lab))
            mass = sum(math.exp(e["logprob"]) for e in entries if e.get("token") in variants and e.get("logprob") is not None)
            probs_raw[r.canonical_of(lab)] = mass
            if mass == 0:
                missing.append(lab)
    allowed_mass = sum(probs_raw.values()) if probs_raw else None
    if probs_raw and allowed_mass and allowed_mass > 0:
        probs, src = _normalise(probs_raw), "logprobs"
    elif label is not None:
        probs, src = {k: (1.0 if k == r.canonical_of(label) else 0.0) for k in r.answer_keys}, "none"
    else:
        probs, src = None, "none"
    finish = None
    try:
        finish = resp["choices"][0].get("finish_reason") or resp["choices"][0].get("native_finish_reason")
    except (KeyError, IndexError, TypeError, AttributeError):
        pass
    extra = {"n_top_logprobs": len(top), "finish_reason": finish,
             "reasoning_present": bool(msg.get("reasoning")) if isinstance(msg, dict) else False}
    if label is None:
        return Parsed(False, None, None, probs, src, raw_text=text, invalid_reason="token not in allowed set",
                      allowed_mass=allowed_mass, missing_labels=missing, extra=extra)
    return Parsed(True, r.canonical_of(label), label, probs, src, raw_text=text, allowed_mass=allowed_mass,
                  missing_labels=missing, extra=extra)


ANSWER_RE = re.compile(r"ANSWER:\s*\**\s*([^\s*]+)", re.I)
CONF_RE = re.compile(r"CONFIDENCE:\s*\**\s*(\d{1,3})", re.I)


def answer_token_logprobs(r: Rendered, resp: dict) -> dict | None:
    """The logprob entry at the position of the final ANSWER token, when the provider returned per-token logprobs.

    Tokens are joined into text, the last `ANSWER:` match is located by character offset, and the first token at or
    after the answer's start whose spelling is one of the allowed labels is taken as the answer token (a backwards
    search would hit the CONFIDENCE digits for score questions). Returns None when no logprobs came back or the
    answer token cannot be located. `label_mass` is the probability on each allowed label among that position's
    top_logprobs (plus the chosen token itself); `label_probs` is that mass renormalised over the canonical answer keys.
    """
    try:
        content = resp["choices"][0]["logprobs"]["content"] or []
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    if not content:
        return None
    tokens = [str(e.get("token") or "") for e in content]
    text = "".join(tokens)
    matches = list(ANSWER_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].start(1)
    offsets = []
    pos = 0
    for t in tokens:
        offsets.append(pos)
        pos += len(t)
    variants_by_label = {lab: set(_label_variants(lab)) for lab in r.labels}
    idx = None
    for i, (off, t) in enumerate(zip(offsets, tokens)):
        if off + len(t) <= start:
            continue
        if any(t in v for v in variants_by_label.values()):
            idx = i
            break
        if off > start + 8:   # the answer should begin within a few characters of the match
            break
    if idx is None:
        return None
    entry = content[idx]
    top = list(entry.get("top_logprobs") or [])
    if entry.get("logprob") is not None and not any(e.get("token") == entry.get("token") for e in top):
        top.append({"token": entry.get("token"), "logprob": entry.get("logprob")})
    mass = {}
    for lab, variants in variants_by_label.items():
        mass[lab] = sum(math.exp(e["logprob"]) for e in top if e.get("token") in variants and e.get("logprob") is not None)
    total = sum(mass.values())
    probs = {r.canonical_of(lab): (m / total if total > 0 else 0.0) for lab, m in mass.items()}
    return {"index": idx, "n_tokens": len(content), "token": entry.get("token"), "logprob": entry.get("logprob"),
            "n_top_logprobs": len(entry.get("top_logprobs") or []), "allowed_mass": total, "label_mass": mass,
            "label_probs": probs}


def parse_reasoning(r: Rendered, resp: dict) -> Parsed:
    try:
        msg = resp["choices"][0]["message"]
        text = msg.get("content") or ""
    except (KeyError, IndexError, TypeError):
        return Parsed(False, None, None, None, "verbalized", invalid_reason="no choices[0].message")
    finish = None
    try:
        finish = resp["choices"][0].get("finish_reason")
    except (AttributeError, KeyError, IndexError, TypeError):
        pass
    a = ANSWER_RE.findall(text)
    c = CONF_RE.findall(text)
    extra = {"finish_reason": finish, "reasoning_chars": len(msg.get("reasoning") or "") if isinstance(msg, dict) else 0}
    tok = answer_token_logprobs(r, resp)
    if tok is not None:
        extra["answer_token_logprobs"] = tok
    if not a:
        return Parsed(False, None, None, None, "verbalized", raw_text=text, invalid_reason="no ANSWER line", extra=extra)
    label = _match_label(a[-1].strip().strip(".)"), r.labels)
    conf = min(100, int(c[-1])) / 100.0 if c else None
    if label is None:
        return Parsed(False, None, a[-1], None, "verbalized", raw_text=text, confidence=conf,
                      invalid_reason="answer not in allowed set", extra=extra)
    n = len(r.answer_keys)
    key = r.canonical_of(label)
    if conf is None:
        probs = {k: (1.0 if k == key else 0.0) for k in r.answer_keys}
        extra["confidence_missing"] = True
    else:
        probs = {k: (conf if k == key else (1.0 - conf) / (n - 1)) for k in r.answer_keys}
    return Parsed(True, key, label, probs, "verbalized", raw_text=text, confidence=conf, extra=extra)


def parse_response(r: Rendered, resp: dict) -> Parsed:
    if r.kind == "decision":
        return parse_decision(r, resp)
    if r.kind == "instant":
        return parse_instant(r, resp)
    return parse_reasoning(r, resp)
