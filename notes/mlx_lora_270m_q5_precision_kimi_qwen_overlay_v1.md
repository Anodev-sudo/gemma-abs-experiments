# q5 precision Kimi/Qwen overlay v1

Date: 2026-05-07

## Goal

Create a strict final-overlay candidate set for the q5 lineage, focused on the
remaining production-shape failures:

- number-role binding
- polarity and direction
- scoped claims
- malformed repetition or notation
- unsupported entity/mechanism substitutions

## Source pool

Candidate builder:

- `scripts/build_q5_precision_kimi_qwen_candidates.py`

Candidate artifact:

- `hf_bucket/experiments/mlx_lora/q5_precision_kimi_qwen_overlay_v1/source_candidates/q5_precision_candidates.jsonl`

Pool size: 253 rows.

Source mix:

- 213 mined q5-severe/q8-clean rows from the 1000-row q5 hardening review
- 14 current replay_75 risk-review rows
- 17 current replay_75 fresh blind-review rows
- 9 current replay_75 mechanical-risk rows

## Writer

Writer model:

- `moonshotai/kimi-k2.6`

Prompt:

- `prompts/synthetic_summary/kimi_k2_6_q5_precision_rewriter_v1.json`

Important writer setting:

- reasoning effort: `none`

The first writer prompt still leaked too many 16-18 word summaries, so the
prompt was tightened to aim for 10-12 words while the local release gate stayed
at 10-15 words.

Writer result:

- 253 generated corrections
- 222 passed local eligibility
- 31 excluded locally
- local exclusions: 29 outside 10-15 words, 24 too long, 1 title duplicate, 1 formula shorthand

## Reviewer

Primary intended reviewer was `qwen/qwen3-235b-a22b-2507`, but current
OpenRouter/provider routing was unstable:

- WandB provider returned 404
- Cerebras and Novita hit repeated 429s
- default routing was very slow and 503-prone

The completed strict review used:

- `qwen/qwen3.5-122b-a10b`

Prompt:

- `prompts/synthetic_summary/qwen3_235b_a22b_2507_q5_precision_verifier_v1.json`

The reviewer prompt was adjusted after a too-strict pass rejected faithful
no-number repairs for omitting risky metrics. The final verifier remains strict
on unsupported details, number-role binding, polarity, and copied q5 failures,
but does not require fragile numbers to be included.

Reviewer result:

- 222 eligible rows requested
- 220 completed verifications
- 82 accepted by Qwen
- 138 rejected by Qwen
- 2 malformed-response failures

## Materialized release

Release:

- `hf_dataset/releases/recent_3y_balanced_10746_q5_precision_kimi_qwen_overlay_v1/training_rows.jsonl`

Final materialized rows: 81.

One Qwen-accepted row was excluded at release materialization because it missed
the final period.

Release QA:

- average word count: 12.49
- min word count: 10
- max word count: 15
- over 15 words: 0
- missing final period: 0

Accepted issue-tag coverage:

- numeric: 32 rows
- polarity: 14 rows
- entity: 12 rows
- long notation: 5 rows
- fresh replay_75 rows: 11 rows across risk, blind, and mechanical buckets

## Read

This is not a large overlay, but it is a clean one. It is comparable in size to
the earlier q5 Kimi/Qwen accepted set while being more focused on the exact
late-stage q5 failure modes. It is suitable as a strict precision overlay
candidate for the next small training run.
