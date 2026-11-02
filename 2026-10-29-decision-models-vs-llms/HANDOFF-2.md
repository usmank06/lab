# Follow-up: two control checks before publishing

Work in the original experiment working directory (the one with `data/runs.sqlite` and `data/raw/`). Same rules as HANDOFF.md: OpenRouter only, key from `OPENROUTER_API_KEY`, never delete or overwrite raw data, don't change anything about run `main-01`.

Phase 1 (build + dry run + cost estimate) then **stop and tell me**. Phase 2 runs on "go". Phase 3 analyzes and updates the public folder.

## Why

The experiment's control is raw Qwen3.5-9B forced to one token with no reasoning. That shows the 9B decision heads beat a *non-reasoning* 9B, but not whether the head recovers reasoning or just teaches the format. Two checks close that:

1. **Reasoning control**: the same Qwen3.5-9B with thinking on, then a one-token answer. If it scores near Elephant 1 on tier 2, the head gives reasoning-level accuracy at one-token cost. If it scores well above, the head teaches the format, not the reasoning.
2. **Prompt sensitivity**: the existing one-token 9B arm with two different system prompts. If the control moves by several points, part of the matched-pair gap is prompt engineering.

## New arms (3 total — don't add more)

Add to `config.yaml` under `arms:` with `status: selected`. All three use slug `qwen/qwen3.5-9b`, pinned to the same provider as the existing arm (`parasail/bf16`, `allow_fallbacks: false`, `require_parameters: true`), same pricing block, same `seed`, `temperature: 0`.

| arm name | kind | what changes |
|---|---|---|
| `qwen3.5-9b-reasoning` | reasoning | `reasoning: {enabled: true}` (Qwen3.5 thinking mode; if OpenRouter needs `effort` instead, use `medium` and note it). Render like the existing reasoning arm: think, then `ANSWER: <token>` and `CONFIDENCE: <0-100>`. Set `reasoning_max_tokens` to 6000 like GPT. If Parasail returns logprobs on the final token, record them too, but calibration for this arm is verbalized like GPT's. `pairs_with: qwen3.5-9b` so it lands in the matched-pair table as a second comparison line for Elephant 1 and Microsoft-Decision-1 (head minus reasoning-base). |
| `qwen3.5-9b-prompt-b` | instant | identical to `qwen3.5-9b` except the system prompt: `prompt_variant: b` |
| `qwen3.5-9b-prompt-c` | instant | identical to `qwen3.5-9b` except the system prompt: `prompt_variant: c` |

### Prompt variants

Add a `prompt_variant` field to arms (default `a` = the existing `INSTANT_SYSTEM`). Implement `b` and `c` in `render.py` beside it. Same allowed-token list substitution; only the wording differs. Use these two, verbatim apart from the `{allowed}` substitution:

**b — terse:**
```
Reply with one token: {allowed}. Nothing else.
```

**c — careful:**
```
You are grading a document against a question. Read the document carefully, then answer the question with exactly one token from this set: {allowed}. Do not explain. Do not add punctuation. Output the single token only.
```

Add a test in `tests/test_render.py` that renders one question of each format under variants a, b, c and asserts the user message and allowed-token handling are identical and only the system content differs.

## Scope

- `qwen3.5-9b-reasoning`: all tiers, 5 runs, plus the label-swap subset. 3,000 calls.
- `qwen3.5-9b-prompt-b` and `-c`: **tier 2 only**, 5 runs, no label-swap. 1,000 calls each.
- Run id: `followup-01`. `--models qwen3.5-9b-reasoning,qwen3.5-9b-prompt-b,qwen3.5-9b-prompt-c`. Everything else from `main-01` is reused by the analysis, not rerun.

## Phase 1 — build, dry run, estimate. Then stop.

1. Config + render changes + test. Run the full test suite.
2. Dry run: 3 questions × 1 run × the 3 new arms (`--dry-run --models ...`). For the reasoning arm, confirm thinking actually happens (reasoning tokens > 0 on tier-2 items; Qwen3.5 may need the explicit enable flag or it silently answers without thinking — check the raw response), confirm the `ANSWER:` line parses, and record whether final-token logprobs come back. For the prompt arms, confirm allowed-token mass is still > 0.99 and invalid rate is 0.
3. `COST_ESTIMATE.md`: append a "Follow-up" section with measured tokens per call × call counts × list price, per arm, with the reasoning arm separate. Also wall-clock at concurrency 4.
4. Tell me: whether Qwen 9B actually reasons through Parasail, the parse rate, the cost, anything you had to decide. Stop.

## Phase 2 — run (after "go")

`--run-id followup-01`, checkpoints, `--resume` on failure, same retry policy. Report per-arm progress.

## Phase 3 — analyze and update the public folder

`analyze.py` must accept multiple run ids (`--run-id main-01,followup-01`) and treat the union as one dataset. Then:

**New outputs:**
- `results/metrics_by_tier.csv` and `_by_format.csv` gain the three arms.
- `results/matched_pair_deltas.csv` gains rows for `elephant-1 minus qwen3.5-9b-reasoning` and `microsoft-decision-1 minus qwen3.5-9b-reasoning`, all tiers, same bootstrap.
- `results/prompt_sensitivity.csv`: tier 2, for arms `qwen3.5-9b` (a), `-prompt-b`, `-prompt-c`: accuracy, Brier, ECE, invalid rate, flip rate, plus the range (max − min) of accuracy across the three prompts with a bootstrap CI on that range. Also per format.
- `results/REPORT.md`: add a "Follow-up" section at the top of results with the two answers in one sentence each: (1) reasoning-Qwen tier-2 accuracy vs Elephant 1 and Microsoft-Decision-1, with the delta and CI, and its cost and p50 latency vs theirs; (2) the prompt range on the one-token control.
- Charts: add `qwen3.5-9b-reasoning` to `accuracy_vs_cost.png` and `matched_pair_deltas.png`; new `prompt_sensitivity.png` (three bars, tier 2 accuracy, with CI whiskers, baseline dashed).

**Then update the public lab repo** folder `2026-10-29-decision-models-vs-llms/`:
- Copy over the updated `config.yaml`, `src/`, `tests/`, `results/*.csv`, `results/REPORT.md`, `results/charts/*.png`, `COST_ESTIMATE.md`. Copy this file in as `HANDOFF-2.md`. Do not copy `data/raw/` or `runs.sqlite`.
- Regenerate the website SVGs: `python -m src.charts` from the public folder. Extend `src/charts.py` first so that `fig-tier2.svg` includes the reasoning-Qwen bar (kind `reasoning`, so it draws green with the triangle marker) and `fig-matched-pairs.svg` shows, for the two 9B pairs, a second grey marker for the reasoning base on the same row (legend: "base model, reasoning on"). Add `fig-prompts.svg`: three horizontal bars (prompt a / b / c), tier 2 accuracy, value labels, same style as the other figures. Keep the 680px viewBox and the existing palette.
- Append to the folder `README.md` under Results: a short "Follow-up" subsection with the two findings in plain numbers, and a Log line dated with the run date.
- In `website/decision-models-vs-llms/index.html`: the SVGs are inlined. Re-inline the regenerated `fig-tier2` and `fig-matched-pairs` in place, and add a new subsection after "The matched pairs" titled "Does the head recover reasoning, or just the format?" with: one paragraph stating the reasoning-Qwen result and what it means for the matched-pair claim, the `fig-prompts.svg` inlined with a caption, and one paragraph on the prompt range. Also update the "Caveats" bullet about the control if the prompt range is more than 3 points. Keep the writing plain; match the tone of the surrounding text; no new adjectives.

Report back with the two numbers and the diff of the page section so I can read it before pushing.

## Rules

- Don't touch `main-01` rows; the new arms get `followup-01`.
- Don't invent slugs, prices or provider names — resolve live, write them in config with the date.
- If Qwen3.5-9B can't be made to reason through Parasail, try the other bf16 providers that return the parsed `ANSWER:` line; logprobs aren't required for this arm. If none reasons, stop and tell me rather than substituting a different model.
