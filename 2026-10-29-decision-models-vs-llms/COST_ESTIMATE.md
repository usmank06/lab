# Cost estimate — decision-models-vs-llms

Generated 2026-10-29. Basis: **measured** from dry run(s) `dry-03-final, dry-04-27b, dry-05-arm9`.

Questions in `data/questions.jsonl`: 450 (target 450; tier 3 pending if lower). Full run = 450 questions x 5 runs = 2,250 calls per arm, plus label-swap 50 x 3 labels x 5 runs = 750 calls per arm; 3,000 calls per arm in total.

List prices are USD per million tokens from `GET /api/v1/models` (see config.yaml). Actual cost from the dry run is shown next to the list-price projection where measured.

| arm | slug | status | kind | in tok/call | out tok/call | reasoning tok/call | $/M in | $/M out | $/call (list) | $/call (actual) | p50 latency ms | full-run $ (list) | provider | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev-1.14 | typesafe/jev-1.14 | selected | decision | 561 | 34.3 | 0 | 0.042 | 0.0 | 0.000024 | 0.000024 | 243 | 0.07 | TypeSafe | ok 11, invalid 0, failed 0; probs: decision |
| decider-v1.2 | perplexity/pplx-decider-v1.2 | selected | decision | 319 | 1.0 | 0 | 0.02 | 0.0 | 0.000006 | 0.000006 | 496 | 0.02 | Perplexity | ok 6, invalid 0, failed 0; probs: decision |
| elephant-1 | baseten/elephant-1 | selected | decision | 337 | 1.0 | 0 | 0.035 | 0.0 | 0.000012 | 0.000012 | 342 | 0.04 | BaseTen | ok 6, invalid 0, failed 0; probs: decision |
| microsoft-decision-1 | microsoft/microsoft-decision-1 | selected | decision | 252 | 1.0 | 0 | 0.042 | 0.0 | 0.000011 | 0.000011 | 536 | 0.03 | Azure | ok 6, invalid 0, failed 0; probs: decision |
| d1-ultra | liquid/d1-ultra | selected | decision | 284 | 0.0 | 0 | 0.042 | 0.0 | 0.000012 | 0.000012 | 1323 | 0.04 | Liquid | ok 6, invalid 0, failed 0; probs: decision |
| qwen3.5-9b | qwen/qwen3.5-9b | selected | instant | 368 | 1.0 | 0 | 0.1 | 0.25 | 0.000037 | 0.000037 | 377 | 0.11 | Parasail | ok 6, invalid 0, failed 0; probs: logprobs |
| qwen3.5-27b | qwen/qwen3.5-27b | selected | instant | 368 | 1.0 | 0 | 0.195 | 1.56 | 0.000073 | 0.000073 | 695 | 0.22 | Alibaba | ok 6, invalid 0, failed 6; probs: logprobs |
| gpt-6.1-sol | openai/gpt-6.1-sol | selected | reasoning | 350 | 53.3 | 38 | 2.0 | 10.0 | 0.001233 | 0.001356 | 2619 | 3.70 | OpenAI | ok 6, invalid 0, failed 0; probs: verbalized |

**Total at list price, selected arms: $4.22** (per-call token counts are dry-run means; the list price is the pinned provider's endpoint price where a provider is pinned).

Reasoning arm separately (it dominates): gpt-6.1-sol: $3.70.
gpt-6.1-sol: mean reasoning tokens per call by tier in the dry run: tier 1: 5, tier 2: 70. Every extra 300 reasoning tokens per call across the full run would add $9.00 at list price.

## Dropped arms

- **gpt-5-nano** (`openai/gpt-5-nano`): dry run 2026-10-29: HTTP 400 from both OpenAI and Azure: 'logprobs are not supported with reasoning models'
- **claude-haiku-4.5** (`anthropic/claude-haiku-4.5`): dry run 2026-10-29: no logprobs in any response (Amazon Bedrock); 3 of 4 one-token answers were outside the allowed set. Arm 8 is dropped: neither candidate returns token logprobs through OpenRouter.
- **claude-fable-5.1** (`anthropic/claude-fable-5.1`): dry run 2026-10-29: $0.0106 per call vs GPT-6 Astra $0.0052 (same questions, medium effort; low effort the same). Also returned 0 reasoning tokens at both efforts (reasoned in visible text instead).
- **gpt-6-astra** (`openai/gpt-6-astra`): was the cheaper frontier candidate ($0.0058/call); replaced on the user's request by a standard-tier OpenAI model (see gpt-6.1-sol / gpt-6.1-terra / gpt-6-luna)
- **gpt-6.1-terra** (`openai/gpt-6.1-terra`): alternative for arm 9: dry run 6/6 correct, $0.00063/call; not chosen because the arm is a ceiling reference and GPT-6.1 Sol is the standard high-end model. Flip status to use it instead.
- **gpt-6-luna** (`openai/gpt-6-luna`): alternative for arm 9: dry run 6/6 correct, $0.00011/call, but it is the fast/small tier (124 reasoning tokens/call mean), not a ceiling reference. Flip status to use it instead.

## Wall-clock at the concurrency cap

Per-model concurrency 4; all arms run in parallel, so wall-clock is the slowest arm (latency p50 from the dry run, no retries, generation lookups run alongside):

- gpt-6.1-sol: ~0.5 h
- d1-ultra: ~0.3 h
- qwen3.5-27b: ~0.1 h
- microsoft-decision-1: ~0.1 h
- decider-v1.2: ~0.1 h
- qwen3.5-9b: ~0.1 h
- elephant-1: ~0.1 h
- jev-1.14: ~0.1 h

## Dry run cost

Dry run `dry-03-final, dry-04-27b, dry-05-arm9`: 77 calls, $0.0460 actual (usage/generation endpoint).

## Actual spend of the full run `main-01`

| arm | calls ok | calls failed | actual $ | p50 ms | p95 ms |
|---|---|---|---|---|---|
| d1-ultra | 3000 | 0 | 0.0406 | 333 | 809 |
| decider-v1.2 | 3000 | 0 | 0.0210 | 341 | 832 |
| elephant-1 | 3000 | 0 | 0.0380 | 278 | 1601 |
| gpt-6.1-sol | 3000 | 0 | 3.2457 | 1861 | 3591 |
| jev-1.14 | 3000 | 0 | 0.0756 | 281 | 648 |
| microsoft-decision-1 | 3000 | 0 | 0.0340 | 330 | 703 |
| qwen3.5-27b | 3000 | 0 | 0.2464 | 715 | 6563 |
| qwen3.5-9b | 3000 | 30 | 0.1247 | 342 | 2373 |

**Total actual: $3.8258** (OpenRouter generation records; failed calls cost nothing). First call started 2026-10-30T00:17:45.872001+00:00, last call finished 2026-10-30T00:44:54.965660+00:00.

## Follow-up (HANDOFF-2.md)

Generated 2026-10-30. Basis: **measured** from dry run(s) `dry-06-followup, dry-07-followup-t2`. Call counts follow each arm's `scope` in config.yaml (5 runs per question; label-swap = 3 label sets x the yes/no subset within the arm's tiers). List prices are the pinned provider's endpoint prices (USD per million tokens), re-resolved on the date noted in config.yaml. Full-run $ and wall-clock are tier-weighted: each measured tier's own mean tokens / median latency x its call count; unmeasured tiers use the overall dry-run mean.

| arm | slug | kind | tiers | label-swap | calls | in tok/call | out tok/call | reasoning tok/call | $/M in | $/M out | $/call (list, flat mean) | $/call (actual, flat mean) | p50 latency ms | full-run $ (list, tier-weighted) | provider | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-9b-reasoning | qwen/qwen3.5-9b | reasoning | 1,2,3 | yes | 3,000 | 360 | 2,425.2 | 2,292 | 0.1 | 0.25 | 0.000642 | 0.000642 | 13982 | 1.72 | Parasail | ok 7, invalid 1, failed 0; probs: verbalized |
| qwen3.5-9b-prompt-b | qwen/qwen3.5-9b | instant | 2 | no | 1,000 | 403 | 1.0 | 0 | 0.1 | 0.25 | 0.000041 | 0.000041 | 443 | 0.07 | Parasail | ok 3, invalid 0, failed 0; probs: logprobs |
| qwen3.5-9b-prompt-c | qwen/qwen3.5-9b | instant | 2 | no | 1,000 | 436 | 1.0 | 0 | 0.1 | 0.25 | 0.000044 | 0.000044 | 456 | 0.08 | Parasail | ok 3, invalid 0, failed 0; probs: logprobs |

**Total at list price for these arms: $1.87** over 5,000 calls (completion tokens include the reasoning tokens; Parasail bills them as output).

Per tier:

- qwen3.5-9b-reasoning, tier 1: 1,375 calls; measured on 3 dry call(s): 285 in / 943 out tokens per call (926 reasoning), median latency 5.2 s -> $0.36
- qwen3.5-9b-reasoning, tier 2: 1,375 calls; measured on 5 dry call(s): 404 in / 3,315 out tokens per call (3,111 reasoning), median latency 18.5 s -> $1.19
- qwen3.5-9b-reasoning, tier 3: 250 calls; not in the dry run, uses the overall mean -> $0.16
- qwen3.5-9b-prompt-b, tier 2: 1,000 calls; measured on 1 dry call(s): 732 in / 1 out tokens per call, median latency 0.5 s -> $0.07
- qwen3.5-9b-prompt-c, tier 2: 1,000 calls; measured on 1 dry call(s): 765 in / 1 out tokens per call, median latency 0.5 s -> $0.08

Reasoning arm separately: qwen3.5-9b-reasoning: $1.72.
qwen3.5-9b-reasoning: mean reasoning tokens per call by tier in the dry run: tier 1: 926, tier 2: 3111. Every extra 300 reasoning tokens per call across its 3,000 calls would add $0.22 at list price. A call that runs to the 6,000-token cap costs at most $0.0015 in output tokens.

Wall-clock at per-model concurrency 4, arms in parallel (tier-weighted median latency from the dry run, no retries), so the run takes about as long as the slowest arm:

- qwen3.5-9b-reasoning: ~2.5 h (150 min)
- qwen3.5-9b-prompt-c: ~0.0 h (2 min)
- qwen3.5-9b-prompt-b: ~0.0 h (2 min)

Dry run `dry-06-followup, dry-07-followup-t2` for these arms: 14 calls, $0.0054 actual (usage/generation endpoint).

## Actual spend of the full run `followup-01`

| arm | calls ok | calls invalid | calls failed | actual $ | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| qwen3.5-9b-prompt-b | 1000 | 0 | 0 | 0.0524 | 278 | 440 |
| qwen3.5-9b-prompt-c | 1000 | 0 | 0 | 0.0557 | 284 | 451 |
| qwen3.5-9b-reasoning | 2875 | 125 | 8 | 1.8895 | 15839 | 87929 |

**Total actual: $1.9977** (OpenRouter generation records; failed calls cost nothing). First call started 2026-10-30T05:12:21.887853+00:00, last call finished 2026-10-30T18:36:12.413563+00:00.

The 8 failed calls are label-swap calls on the reasoning arm that hit Parasail's upstream rate limit (HTTP 429) five times each during the main pass (05:12–10:24 UTC). They were retried with `--resume` at 18:32–18:36 UTC: 7 answered, 1 looped to the 12,000-token cap. The retries are the extra rows under ok/invalid; the failed rows are kept, as all raw data is.
