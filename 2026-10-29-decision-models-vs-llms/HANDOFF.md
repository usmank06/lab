# Lab experiment: Decision models vs LLMs — handoff

You are building and running this experiment end to end. Work in `lab/decision-models-vs-llms/`. Everything goes through the OpenRouter API (key in `OPENROUTER_API_KEY`, never committed). Python 3.11+, `uv` or plain venv, your call.

There are three phases with a hard stop after Phase 1. Do not spend money beyond the dry run until I say "go".

## The question

Is a "decision model" (Jev-style: state + typed questions → calibrated probabilities, no text) actually better than the LLM it was built from, or than a normal LLM forced to answer in one token? Nobody has compared a decision model to its own base model. We can, because two of them are Qwen3.5-9B finetunes.

One sentence the final report must answer: does the 9B decision head beat raw Qwen3.5-9B on accuracy, calibration, or consistency — and at what cost difference?

## Arms and models (9 total — don't add more)

Resolve the exact OpenRouter slugs from `GET /api/v1/models` at build time and write them to `config.yaml`. Don't guess slugs. Record list prices from the same call.

Decision models (OpenRouter Decisions / `/v1/systemone` endpoint — read OpenRouter's docs for the exact request shape, don't assume it):

1. TypeSafe Jev 1.14 — the reference
2. Perplexity Decider V1.2 (27B)
3. Baseten Elephant 1 (Qwen3.5-9B base) — matched pair A
4. Microsoft-Decision-1 (Qwen3.5-9B base) — matched pair B
5. Liquid d1-ultra — only non-Qwen architecture

Instant LLM (chat completions, `max_tokens=1`, logprobs on, answer constrained to one token):

6. Qwen3.5-9B raw — the control that matters, same base as 3 and 4
7. Qwen3.5-27B raw — pairs with Decider
8. One cheap closed model: GPT-5 Nano or Claude Haiku 4.5. Pick whichever returns token logprobs through OpenRouter. If neither does, drop this arm and say so in the report.

Reasoning LLM (ceiling reference, reasoning enabled, then a final one-token answer):

9. One frontier model: Claude Fable 5.1 or GPT-6 Astra, whichever is cheaper per our estimate.

Dumb baseline (no API): regex/keyword matcher for Tier 1, majority-class for Tier 2. If it's within 5 points of Jev on Tier 1, that's a headline.

## Questions: 450 total, all private, all machine-verifiable

The tiers are about where the answer lives, not how much knowledge is needed. Do not use any public benchmark — these models have seen them.

Every question has: `id`, `tier`, `format` (`yesno` | `choice` | `score`), `state` (text or JSON), `question`, `options` (for choice/score), `ground_truth`, `domain`, `generator_seed`.

Format mix per tier: ~40% yesno, ~40% choice (4–8 options), ~20% score (5-level ordinal). Every arm gets the identical state and question.

Tier 1 — read-off-the-page (200). The answer is literally in the state. Generate synthetic documents (support tickets, invoices, memos, emails, JSON event logs, short contracts) with code-planted facts, so ground truth is known by construction. Questions like "does this ticket mention a refund," "which department is this addressed to," "how many of these five required fields are present (0–5)." Vary domain across business, legal-ish, medical-ish, technical, casual. Add distractor text so it's not trivial.

Tier 2 — reason-over-the-state (200). Everything needed is in the context but must be combined. Programmatically generated constraint problems with a solver-verified answer: scheduling conflicts ("given these 6 meetings and 3 rooms, is this assignment valid"), rule compliance ("does this expense report violate any of these 5 policy rules, which one"), small zebra/logic-grid puzzles, order-of-events consistency. Difficulty knob: number of constraints. Ground truth from the generator, double-checked by an independent checker function.

Tier 3 — outside knowledge (50). Labeled in the report as a confound demo, not a real tier: shows that size wins when the answer isn't in the context. Generate factual questions with a frontier model, cross-check answers with a second frontier model, keep only the ones both agree on, and flag that ground truth here is weaker.

Label-swap subset (50 yesno items from Tier 1+2). Run three times with option labels `yes/no`, `0/1`, `A/B`. Measures whether a model's answer moves when only the label name changes (replicates the Sun et al. finding).

Write all questions to `data/questions.jsonl`. Write a unit test that validates every question against its own generator/checker before anything hits the API.

## Prompt/request rendering

* Decision models: state + question + options in the native typed format (`noul`/yes-no, `choice`, `score`). One question per request so latency is comparable.
* Instant LLMs: system prompt that says answer with exactly one token from the allowed set; choice options mapped to letters A–H, score to digits 1–5, yesno to the current label pair. Use `logit_bias` where the provider supports it. Read the logprobs of the allowed tokens, renormalize, that's the probability vector. If a model returns a token outside the allowed set, record it as invalid (that's data, not a bug to hide).
* Reasoning LLM: reasoning on, then ask for the final answer as one token. If final-token logprobs aren't exposed, record the verbalized answer and a verbalized 0–100 confidence, and mark in the report that this arm's calibration is verbalized, not logprob.
* Same rendering code for every arm. One function, parametrized by arm. Test it.

## Runs and fairness

* 5 runs per question per model (temperature 0 where it exists; decision models as-is). Total ≈ 450 × 5 × 9 = 20,250 calls plus label-swap (50 × 3 labels × 5 runs × 9).
* Record per call: model slug, provider header, run index, request body hash, raw response JSON, probability vector, parsed answer, latency ms (client-measured, request start to last byte), prompt/completion tokens, cost from OpenRouter's usage/generation endpoint, timestamp, error if any.
* Concurrency cap of 4 per model, all from the same machine, so latency numbers are comparable. Note the region/provider in the report.
* Checkpoint to SQLite (`data/runs.sqlite`) after every call. The run must be resumable; `--resume` skips completed (question, model, run) triples. Retry with backoff on 429/5xx, max 5, then record as failed.

## Metrics (per model × tier, and per format)

* Accuracy and macro-F1
* Brier score, ECE (10 bins), reliability diagram
* Flip rate: fraction of questions where the 5 runs don't all agree
* Invalid-output rate
* Latency p50 / p95
* Cost per 1,000 decisions: actual (from OpenRouter usage) and at list price
* Label-swap sensitivity: % of answers that change across label sets, and AUC per label set
* Matched-pair delta: Elephant 1 and Microsoft-Decision-1 minus raw Qwen3.5-9B on every metric, with bootstrap 95% CI
* Routing simulation: for each decision model, send items below confidence threshold τ to the reasoning model; plot accuracy and cost vs τ

## Deliverables

```
lab/decision-models-vs-llms/
  README.md            what this is, how to run
  HANDOFF.md           this file
  config.yaml          slugs, prices, counts, concurrency, seeds
  src/generate.py      question generators + checkers
  src/render.py        one rendering function, all arms
  src/run.py           runner with --dry-run, --resume, --models, --tiers
  src/analyze.py       metrics, charts, REPORT.md
  tests/               generator validity, rendering, parsing, metric math
  data/questions.jsonl
  data/runs.sqlite
  data/raw/            every raw response as JSON, one file per call
  results/*.csv        every metric table
  results/charts/*.png accuracy vs cost per tier, reliability diagrams, matched-pair bars, routing curves
  results/REPORT.md
  COST_ESTIMATE.md
```

Main chart: accuracy vs cost per 1k decisions, log x-axis, one panel per tier, points colored by arm, matched pair connected by a line.

## Phases

Phase 1 — build and estimate. Then STOP.

1. Set up the project, resolve slugs and prices, write `config.yaml`.
2. Write generators and checkers. Generate all 450 questions + label-swap set. Run tests.
3. Write the runner. Dry run: 3 questions × 1 run × every model, just to confirm each endpoint works, logprobs are present where expected, and the parser handles each model's response shape. Save those responses.
4. Write `COST_ESTIMATE.md`: measured tokens per call from the dry run × full call count × list price per model, total, and the reasoning model separately since it dominates. Also estimated wall-clock at the concurrency cap.
5. Tell me: what works, what didn't (which models lack logprobs, any endpoint surprises), the cost, and anything you had to decide that I should know. Then stop.

Phase 2 — run (after I say go). Full run with checkpoints. Report progress per model. Don't change config mid-run; if something must change, stop and tell me.

Phase 3 — analyze. `analyze.py` produces every CSV, every chart, and `results/REPORT.md`. The report leads with the one-sentence answer, then the main chart, then the matched-pair result, then per-tier tables, then label-swap, then routing, then caveats (ground-truth weakness on Tier 3, verbalized vs logprob calibration, provider/region, anything that failed). Plain language, no hype. Numbers only where they're measured.

## Rules

* Never commit the API key or any raw response that contains it.
* Don't invent model names, slugs, or prices — resolve them live and write them down.
* If an endpoint's schema differs from what's described here, follow the real schema and note the difference.
* Keep all raw data. Nothing is deleted or overwritten; new runs get a new run id.
* Don't add models, tiers, or metrics beyond this spec without asking.
