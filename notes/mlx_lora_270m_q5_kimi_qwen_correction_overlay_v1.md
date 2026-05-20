# q5 Kimi/Qwen Correction Overlay v1

## Purpose

Create a focused corrective overlay from the q5 severe-case review set. The writer rewrites only rows where q5 behavior was marked severe, and the verifier accepts only corrections that are grounded in the abstract and suitable for the mobile feed summary style.

## Source

- Input reviewed rows: `hf_bucket/experiments/mlx_lora/q5_hardening_distill_v1/identified_rows/q5_severe.jsonl`
- Candidate pool: 274 q5 severe rows
- Writer prompt: `prompts/synthetic_summary/kimi_k2_6_q5_correction_rewriter_v1.json`
- Verifier prompt: `prompts/synthetic_summary/qwen3_235b_a22b_2507_q5_correction_verifier_v1.json`

## Run Shape

- Writer model: `moonshotai/kimi-k2.6`
- Reviewer model: `qwen/qwen3-235b-a22b-2507`
- Kimi reasoning was disabled with the OpenRouter reasoning control.
- Qwen review used strict structured output. DeepInfra handled most rows; a few transport-level failures were completed by rerouting the same Qwen model through SiliconFlow and WandB.

## Counts

- Kimi rewrites attempted: 274
- Kimi rewrites completed: 274
- Locally eligible for Qwen review: 168
- Locally excluded before Qwen: 106
- Qwen accepted: 71
- Qwen rejected: 97

Primary local exclusion causes:

- `outside_10_15_words`: 105
- `too_long`: 103
- `banned_opening`: 1
- `formula_shorthand`: 1
- `title_duplicate`: 1
- `not_single_sentence`: 1

Primary verifier rejection tags:

- `copies_q5_failure`: 59
- `unsupported_entity`: 11
- `style_drift`: 6
- `wrong_number_binding`: 6
- `unsupported_mechanism`: 4
- `non_answer`: 3

## Outputs

- Experiment root: `hf_bucket/experiments/mlx_lora/q5_kimi_qwen_correction_overlay_v1/`
- Accepted review rows: `hf_bucket/experiments/mlx_lora/q5_kimi_qwen_correction_overlay_v1/materialized_release/selected_review_rows.jsonl`
- Excluded review rows: `hf_bucket/experiments/mlx_lora/q5_kimi_qwen_correction_overlay_v1/materialized_release/excluded_review_rows.jsonl`
- Release rows: `hf_dataset/releases/recent_3y_balanced_10746_q5_kimi_qwen_correction_overlay_v1/training_rows.jsonl`
- MLX chat dataset: `hf_dataset/releases/recent_3y_balanced_10746_q5_kimi_qwen_correction_overlay_v1/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_glanceable_summary_v1/`

MLX split counts:

- train: 63
- valid: 4
- test: 4

## Notes

This is intentionally a narrow overlay, not a replacement corpus. The reviewer rejected many readable-looking rewrites because they preserved the q5 failure, overclaimed beyond the abstract, or bound a number/entity incorrectly. That is useful pressure for a corrective phase: these 71 rows are the high-confidence subset.

## Rejection Repair Pass

A second pass attempted to improve yield by feeding Qwen's explicit rejection tags and `reason_span` back to Kimi. This used `prompts/synthetic_summary/kimi_k2_6_q5_rejection_repair_v1.json` and `scripts/run_q5_rejection_repair_pipeline.py`.

Qwen's non-thinking endpoint did not expose hidden reasoning in this run. The usable feedback was the verifier schema output: `verifier_issue_tags`, `verifier_evidence_span`, and `verifier_reason_span`. The thinking Qwen endpoint advertises reasoning controls through OpenRouter, but this repair pass kept the same non-thinking reviewer and used only the explicit rejection rationale we asked it to return.

Repair counts:

- Rejected rows targeted: 97
- Kimi repair rewrites completed: 97
- Locally eligible for second Qwen review: 72
- Locally excluded before second Qwen review: 25
- Second Qwen reviewed: 71
- Second Qwen accepted: 11
- Second Qwen rejected: 60
- Remaining unverified due provider errors: 1

Primary second-pass local exclusion causes:

- `outside_10_15_words`: 21
- `too_long`: 20
- `copies_rejected_correction`: 4

Primary second-pass verifier rejection tags:

- `copies_q5_failure`: 42
- `style_drift`: 8
- `unsupported_entity`: 4
- `invented_number`: 1

Combined repair release:

- Release rows: `hf_dataset/releases/recent_3y_balanced_10746_q5_kimi_qwen_correction_overlay_repair_v1/training_rows.jsonl`
- Experiment root: `hf_bucket/experiments/mlx_lora/q5_kimi_qwen_correction_overlay_v1/rejection_repair_v1/`
- Original accepted rows: 71
- Repair accepted rows: 11
- Combined training rows: 82
- MLX split counts: train 74, valid 4, test 4

The yield improvement is real but modest. Most remaining rejected repairs are not formatting failures; Qwen still sees them as preserving the original q5 failure or drifting away from a grounded feed-summary target.
