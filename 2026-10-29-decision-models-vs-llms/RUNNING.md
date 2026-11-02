# Running the experiment

The technical notes for `2026-10-29-decision-models-vs-llms`: layout, setup, commands, how each arm is called, and what the dry runs changed. The question, method and results are in [README.md](README.md); the original brief is [HANDOFF.md](HANDOFF.md).

Everything goes through OpenRouter. The API key is read from `OPENROUTER_API_KEY`, or from a `.env` file in this directory (`OPENROUTER_API_KEY=...`) when the variable is unset. `.env` is git-ignored and never copied anywhere.

## Layout

```
config.yaml          arms (slugs + list prices resolved live from /api/v1/models on 2026-10-29), counts, label sets, concurrency, seed
src/common.py        Question schema, paths, config loading
src/generate.py      tier 1 + 2 generators with independent checkers; tier 3 via two frontier models (cached)
src/render.py        render(question, arm) -> request body for every arm kind; parse_response(...) -> answer + probability vector
src/run.py           async runner: 4 in flight per model, SQLite checkpoint per call, retries, --resume, raw JSON per call
src/estimate.py      COST_ESTIMATE.md from dry-run measurements
src/analyze.py       metrics, charts, results/REPORT.md
src/charts.py        the SVG figures for the website page, from results/*.csv
tests/               generator validity (every question vs its checker), rendering, parsing, runner (mock transport), metric math
data/questions.jsonl 450 questions (200 tier 1, 200 tier 2, 50 tier 3) with a 50-item label-swap subset flagged
data/runs.sqlite     every call (table `calls`), every run (table `runs`)   — not in git (180 MB)
data/raw/<run>/      one JSON per call (request, response, headers, timing)  — not in git (~48k files)
data/token_ids.json  Qwen3 token ids for the allowed answer tokens (for logit_bias)
data/openrouter_models_snapshot.json   the model records as fetched
results/             CSVs, charts/, REPORT.md
COST_ESTIMATE.md
```

## Setup

Python 3.11+ (built and tested on 3.14.7). From this directory:

```bash
py -m venv .venv
.venv\Scripts\python.exe -m pip install httpx pyyaml numpy pandas matplotlib pytest
```

Set the key in the shell that runs the commands below (PowerShell): `$env:OPENROUTER_API_KEY = "..."`, or put `OPENROUTER_API_KEY=...` in a `.env` file next to `config.yaml`.

## Run

```bash
.venv\Scripts\python.exe -m pytest tests -q                 # all tests; questions are validated against their checkers
.venv\Scripts\python.exe -m src.generate                    # (re)build data/questions.jsonl deterministically (tiers 1+2, tier 3 from cache)
.venv\Scripts\python.exe -m src.generate --tier3            # API: generate + cross-check tier 3, then rebuild (needs the key)
.venv\Scripts\python.exe -m src.run --dry-run               # 3 questions x 1 run x every non-dropped arm (+ endpoint/label-swap checks)
.venv\Scripts\python.exe -m src.estimate                    # COST_ESTIMATE.md from the latest dry run
.venv\Scripts\python.exe -m src.run --run-id main-01        # Phase 2: full run (selected arms, all tiers, 5 runs, label-swap set)
.venv\Scripts\python.exe -m src.run --run-id main-01 --resume
.venv\Scripts\python.exe -m src.run --status --run-id main-01
.venv\Scripts\python.exe -m src.analyze --run-id main-01    # Phase 3: results/*.csv, results/charts/*.png, results/REPORT.md
.venv\Scripts\python.exe -m src.charts                     # SVG figures for website/decision-models-vs-llms/
```

`--models a,b` (arm names, or a slug where only one arm uses it), `--tiers 1,2`, `--runs N`, `--no-labelswap`, `--limit N` narrow a run.
Runs never delete or overwrite: a resumed run appends rows; failed rows stay; a new run id starts a new set.

### Follow-up run (HANDOFF-2.md, 2026-10-30)

Three control arms were added after the main run: `qwen3.5-9b-reasoning` (the control's weights and provider with
thinking on, then `ANSWER:` / `CONFIDENCE:` lines; completion cap 12,000 tokens) and `qwen3.5-9b-prompt-b` / `-c` (the
one-token control under two other system prompts). An arm's config `scope` limits its tiers and whether it runs the
label-swap subset (the prompt arms: tier 2 only, no label-swap), so one invocation covers all three:

```bash
.venv\Scripts\python.exe -m src.run --dry-run --run-id dry-06-followup --models qwen3.5-9b-reasoning,qwen3.5-9b-prompt-b,qwen3.5-9b-prompt-c
.venv\Scripts\python.exe -m src.estimate --append --section-title "Follow-up (HANDOFF-2.md)" --run-id dry-06-followup,dry-07-followup-t2 --arms qwen3.5-9b-reasoning,qwen3.5-9b-prompt-b,qwen3.5-9b-prompt-c
.venv\Scripts\python.exe -m src.run --run-id followup-01 --models qwen3.5-9b-reasoning,qwen3.5-9b-prompt-b,qwen3.5-9b-prompt-c
.venv\Scripts\python.exe -m src.analyze --run-id main-01,followup-01    # the union is one dataset; adds prompt_sensitivity.csv and the Follow-up report section
.venv\Scripts\python.exe -m src.charts                                   # also writes fig-prompts.svg
```

`main-01` rows are never touched; the analysis reads both runs and pairs each 9B head with the one-token base and,
as a second line, with the reasoning base.

## How each arm is called

Resolved from the live OpenRouter docs on 2026-10-29 (the handoff named `/v1/systemone`; the general decisions
endpoint is `POST /api/alpha/decisions` with the same schema, and `/api/v1/systemone` maps bare TypeSafe ids into the
`typesafe/` namespace; the dry run hits both for Jev).

- **Decision arms**: `{"model", "state", "questions": {"q": {...}}}`, one question per request. `yesno` → `noul` with
  `criteria.true/false`; `choice` → `choice` with the option text as criteria keys; `score` → `score` with the five
  ordered level labels. Answer probabilities come from `answers.q.noul` / `.probabilities`.
- **Instant arms**: chat completion, `max_tokens=1`, `logprobs=true`, `top_logprobs=20`, `temperature=0`, `seed`,
  uniform `logit_bias` on the allowed tokens where the provider supports it (uniform bias preserves relative
  probabilities among the allowed set). Options are lettered A–H, score levels are digits 1–5, yes/no uses the label
  pair. The probability vector is the renormalised mass on the allowed tokens (bare and space-prefixed spellings
  summed). A generated token outside the allowed set is recorded as invalid.
- **Reasoning arm**: reasoning enabled, then `ANSWER: <token>` and `CONFIDENCE: <0-100>` lines parsed from the
  final text. Neither candidate exposes logprobs on OpenRouter, so its calibration is verbalized.
- **Label-swap**: the 50 flagged yes/no questions are re-run with labels yes/no, 1/0 and A/B. For decision models
  these use the `choice` type with the labels as criteria keys, so only the label changes.

## Final arm set (after the Phase 1 dry runs, 2026-10-29)

| # | arm | slug | provider | notes |
|---|---|---|---|---|
| 1 | jev-1.14 | typesafe/jev-1.14 | TypeSafe | reference; also verified on /api/v1/systemone |
| 2 | decider-v1.2 | perplexity/pplx-decider-v1.2 | Perplexity | 27B |
| 3 | elephant-1 | baseten/elephant-1 | Baseten | built on Qwen3.5-9B (matched pair A) |
| 4 | microsoft-decision-1 | microsoft/microsoft-decision-1 | Azure | post-trained from Qwen3.5-9B (matched pair B) |
| 5 | d1-ultra | liquid/d1-ultra | Liquid | architecture not stated on its OpenRouter page |
| 6 | qwen3.5-9b | qwen/qwen3.5-9b | **pinned: Parasail (bf16)** | only bf16 endpoint returning logprobs; logit_bias on |
| 7 | qwen3.5-27b | qwen/qwen3.5-27b | **pinned: Alibaba** | only endpoint returning logprobs; top_logprobs capped at 5; no logit_bias |
| 8 | (dropped) | gpt-5-nano / claude-haiku-4.5 | | neither returns token logprobs through OpenRouter (see config.yaml `drop_reason`) |
| 9 | gpt-6.1-sol | openai/gpt-6.1-sol | OpenAI | medium effort; standard high-end tier, chosen over Astra (5x the price) on request; Terra and Luna measured as alternatives |
| 10 | qwen3.5-9b-reasoning | qwen/qwen3.5-9b | **pinned: Parasail (bf16)** | follow-up: `reasoning: {enabled: true}`, temperature 0, cap 12,000 tokens; verbalized confidence, answer-token logprobs recorded only |
| 11 | qwen3.5-9b-prompt-b | qwen/qwen3.5-9b | **pinned: Parasail (bf16)** | follow-up: the control with the terse system prompt; tier 2 only |
| 12 | qwen3.5-9b-prompt-c | qwen/qwen3.5-9b | **pinned: Parasail (bf16)** | follow-up: the control with the "careful" system prompt; tier 2 only |

Provider pinning uses `provider: {order: [...], allow_fallbacks: false, require_parameters: true}` in the request body.
Without it the router sent Qwen3.5-9B to three different providers in four calls, two of which returned no logprobs
and one of which serves an fp4 quantisation.

## Known limitations recorded in the dry runs

- The 27B arm sees only the top 5 token logprobs (Alibaba's cap), so on 6-8 option questions any option outside the
  top 5 gets probability 0 and is listed in `missing_labels`.
- Whether a provider applies `logit_bias` before computing the returned logprobs is not observable; the renormalised
  vector over allowed tokens is unaffected either way, and allowed-token mass was above 0.99 on every dry-run call.
- Jev tokenises the same state into roughly twice as many input tokens as the other decision models (its own tokenizer).
- The generation record (`GET /api/v1/generation`) becomes available about 7 s after a call; the runner waits 6 s and
  retries up to 8 times, outside the lookup concurrency slot.

## Counts

Score questions use a 1–5 scale throughout (the handoff's "0–5 fields present" example became "1–5 of five
required fields" so that every score question has exactly five ordinal levels mapped to digits 1–5).
