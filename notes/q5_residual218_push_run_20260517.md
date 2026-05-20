# q5 residual-218 push run - 2026-05-17

Goal: improve the promoted q5 llama.cpp lineage without increasing model size.

Constraints:

- Do not delete existing repo artifacts.
- Keep generated artifacts and logs.
- Do not hand-write labels.
- Use Kimi for writing repair labels and Qwen for verification.
- Use blind review as the source of truth, with GPT-5.4-Mini subagents for review.
- Stop if OpenRouter credit is exhausted or clearly insufficient.

Starting point:

- Current q5 promotion: checkpoint 218 from `q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7`.
- Current q5 runtime: Q5_K_M base plus f16 LoRA sidecar.
- q8 remains the quality reference and beat checkpoint 218 in the latest 500-row review.

Budget:

- OpenRouter credits checked through `/api/v1/credits`.
- Starting remaining balance was about `$6.22`.

Initial plan:

1. Use the reviewed 6000-row q5 hardening mine as source evidence.
2. Take q5-severe / q8-clean rows, excluding rows already in checkpoint-218 training.
3. Ask Kimi to rewrite them into cautious 10-12 word feed-lead targets.
4. Ask Qwen to verify them.
5. Continue training from checkpoint 218 using accepted repairs plus preservation anchors.
6. Compare candidate checkpoints against q8 with blind review before any promotion decision.

Source candidate prep:

- Input: `hf_bucket/experiments/mlx_lora/q5_hardening_distill_6000_v1/q5_severe_q8_clean_review_rows.jsonl`
- Source rows: 285
- Excluded checkpoint-218 training overlap: 18
- Prepared candidate rows: 267

Prepared candidate mix:

- Buckets: numeric 186, entity 44, polarity 27, control 8, long_notation 2
- Issue tags: invented_number 88, unsupported_entity 70, misses_point 32,
  overstrong_claim 29, malformed_repetition 15, polarity_flip 15,
  wrong_number_binding 13, unsupported_mechanism 10, non_answer 5


Kimi/Qwen label pass:

- Kimi writer processed 267 candidates.
- Writer decisions: 217 rewrite, 50 drop.
- Local gates kept 183 for Qwen verification.
- Qwen reviewer decisions: 51 accept, 132 reject.
- Accepted rows by bucket: numeric 38, polarity 7, entity 6.
- Remaining OpenRouter balance after materialization: about `$5.82`.

Training plan after label pass:

- Continue from checkpoint 218.
- Use 51 accepted repairs at weight 2.0.
- Reuse checkpoint-218 self-anchor and hard77 preservation rows.
- Keep learning rate at `2e-7` and run a short checkpoint sweep.

Training result:

- Command: `hf_bucket/experiments/dpo/runtime/py312/bin/python scripts/run_weighted_sft_gemma.py --config configs/sft/q5_residual218_push_285_v1_lr2e7.yaml`
- Output root: `hf_bucket/experiments/sft/q5_residual218_push_285_v1/lr2e7`
- Steps: 182
- Checkpoints: 25, 50, 75, 100, 125, 150, 175, 182
- Runtime: 537.6s
- Final train loss: 2.644
- Notes: training completed cleanly; no promotion decision is implied by this.

Residual-push sweep result:

- Runtime sweep: `hf_bucket/experiments/sft/q5_residual218_push_285_v1/lr2e7/eval/HEURISTIC_PUSH_TRIAGE_SUMMARY.md`
- Checkpoints 50, 100, 150, and 182 all generated 454/454 rows with no malformed rows.
- The sweep showed only small movement versus checkpoint 218, so this branch was not taken to q8 blind review.

Q8-shape continuation baseline:

- Existing branch inspected: `hf_bucket/experiments/sft/q5_step2400_sft700_residual218_q8_shape_distill_v1_lr2e7`
- Prior blind review showed checkpoint 150 beat promoted checkpoint 218, but not q8.
- A small continuation was trained from that checkpoint using the 51 residual-push rows plus q8-shape preservation rows.
- Config: `configs/sft/q5_q8shape150_residual_push_v1_lr1e7.yaml`
- Output root: `hf_bucket/experiments/sft/q5_q8shape150_residual_push_v1/lr1e7`
- Resulting checkpoint 150 was blind-reviewed against q8 on 500 reused rows.

Q8-shape plus push blind review:

- Review root: `hf_bucket/experiments/sft/q5_q8shape150_residual_push_v1/lr1e7/eval/blind_ab_q8_mlx_python_vs_q8shape_push150_500`
- GPT-5.4-Mini reviewer shards: 10 x 50 rows.
- Preferred: q8 330, q5 164, neither 5, tie 1.
- More accurate: q8 330, q5 164, neither 5, tie 1.
- Decision: not promotable.

First q8-win value-aware label pass:

- Source: q8 wins from the q8-shape-plus-push blind review.
- Prepared candidates: 330.
- Kimi writer decisions: 311 rewrite, 19 drop.
- Local gates kept 169 for Qwen verification.
- Qwen reviewer decisions: 87 accept, 82 reject.
- Release: `hf_dataset/releases/recent_3y_balanced_10746_q5_q8shape150_q8win_value_aware_v1/training_rows.jsonl`
- Remaining OpenRouter balance after this pass: about `$5.46`.

First value-aware continuation:

- Dataset: `hf_bucket/experiments/sft/q5_q8shape150_value_aware_v1/dataset_87q8win_x3_518q8shape_193residual_77hard_v1/train.jsonl`
- Conservative config: `configs/sft/q5_q8shape150_value_aware_v1_lr2e7.yaml`
- Conservative run output: `hf_bucket/experiments/sft/q5_q8shape150_value_aware_v1/lr2e7`
- The `2e-7` run did not move the q5 runtime outputs on the 454-row triage set.
- Stronger config: `configs/sft/q5_q8shape150_value_aware_v1_lr1e6.yaml`
- Stronger run output: `hf_bucket/experiments/sft/q5_q8shape150_value_aware_v1/lr1e6`
- Runtime sweep selected checkpoint 125 as the least-bad moved checkpoint.

Value-aware checkpoint 125 blind review:

- Review root: `hf_bucket/experiments/sft/q5_q8shape150_value_aware_v1/lr1e6/eval/blind_ab_q8_mlx_python_vs_value_aware125_500`
- GPT-5.4-Mini reviewer shards: 10 x 50 rows.
- Preferred: q8 319, q5 169, neither 12.
- More accurate: q8 317, q5 171, neither 12.
- Decision: not promotable.

Value-aware checkpoint 50 fallback review:

- Review root: `hf_bucket/experiments/sft/q5_q8shape150_value_aware_v1/lr1e6/eval/blind_ab_q8_mlx_python_vs_value_aware50_500`
- GPT-5.4-Mini reviewer shards: 10 x 50 rows.
- Preferred: q8 337, q5 154, neither 8, tie 1.
- More accurate: q8 339, q5 152, neither 8, tie 1.
- Decision: not promotable.

Second q8-win value-aware label pass:

- Source: q8 wins from the value-aware checkpoint 125 blind review.
- Prepared candidates: 315.
- Kimi writer completed: 315.
- Local gates kept 138 for Qwen verification.
- Qwen reviewer decisions: 54 accept, 84 reject.
- Release: `hf_dataset/releases/recent_3y_balanced_10746_q5_value125_q8win_value_aware_v1/training_rows.jsonl`
- Remaining OpenRouter balance after this pass: about `$5.12`.

Second-pass continuation:

- Dataset: `hf_bucket/experiments/sft/q5_q8shape150_value_aware_v1/lr1e6/secondpass_after_ckpt125/dataset_54second_x5_87first_518q8shape_193residual_77hard_v1/train.jsonl`
- Config: `configs/sft/q5_value125_secondpass_v1_lr8e7.yaml`
- Output root: `hf_bucket/experiments/sft/q5_q8shape150_value_aware_v1/lr1e6/secondpass_after_ckpt125/lr8e7`
- Steps: 287
- Runtime: 756.4s
- Final train loss: 7.408
- Heuristic sweep showed no malformed rows, but mixed qualitative movement; checkpoint 50 was selected for one final q8 blind review rather than promotion.

Second-pass checkpoint 50 blind review:

- Review root: `hf_bucket/experiments/sft/q5_q8shape150_value_aware_v1/lr1e6/secondpass_after_ckpt125/lr8e7/eval/blind_ab_q8_mlx_python_vs_secondpass50_500`
- GPT-5.4-Mini reviewer shards: 10 x 50 rows.
- Preferred: q8 298, q5 188, neither 11, tie 3.
- More accurate: q8 297, q5 189, neither 11, tie 3.
- By split:
  - control: q8 37, q5 12, neither 1.
  - entity: q8 54, q5 42, neither 3.
  - long notation: q8 35, q5 13, tie 1.
  - numeric: q8 97, q5 73, neither 3, tie 1.
  - polarity: q8 74, q5 45, neither 4, tie 1.
  - remembered stress: q5 3, q8 1 by preference.
- Decision: improved over earlier q5 candidates, but still not promotable and not close enough to claim q5 beats q8.

Final state:

- No q5 promotion pointer was changed.
- Current promoted q5 remains `q5_step2400_sft700_hard77_burnin150_residual_repair_v1_checkpoint_218`.
- Final OpenRouter remaining balance was about `$5.12`.
- Best q8 comparison from this run was second-pass checkpoint 50 at q5 188 vs q8 298 by preference.
- The main lesson is that small q8-win repair overlays can improve q5 on targeted failures, but they also introduce new brittle bindings and do not close the q8 gap.

## Broad basin-change branch

User noted that the current q5 lineage may have entered the wrong basin from
the start, and explicitly allowed retraining new adapters or changing ratios
and weighting for the existing 10k labels.

Specificity label top-up:

- Source: q8-winning and more-accurate rows from
  `blind_ab_q8_mlx_python_vs_secondpass50_500`
- Excluded already labeled rows from the prior q8-win repair releases
- Candidate rows: 209
- Prompt config:
  `configs/synthetic_summary_generation/q5_secondpass50_q8win_specificity_kimi_qwen_v1.json`
- Kimi writer: 209 completed
- Local gates: 83 eligible, 126 excluded
- Qwen reviewer: 31 accepted, 52 rejected
- Release:
  `hf_dataset/releases/recent_3y_balanced_10746_q5_secondpass50_q8win_specificity_v1/training_rows.jsonl`
- Remaining OpenRouter balance after this pass: about `$4.90`

Broad reweighted dataset:

- Builder: `scripts/build_q5_broad_reweighted_basin_v1_dataset.py`
- Output:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/dataset_broad_base_low_repair_high_v1/train.jsonl`
- Rows: 11,339 unique papers
- Selection policy: one target per paper; higher-priority repair/style labels
  override lower-priority broad-base labels
- Starting adapter:
  `hf_bucket/experiments/mlx_lora/runtime_shift/llama_cpp_q5_clean_slate_broad_v1_20260508/gguf_adapter/peft_2400`

Dataset mix:

| component | rows | weight |
|---|---:|---:|
| base_10746_low_weight | 8590 | 0.35 |
| faithfulness_overlay_1400 | 894 | 0.9 |
| recenter_broad_overlay | 576 | 0.8 |
| specificity_preserving_400 | 374 | 1.0 |
| step2400_mined_kimi_qwen_10000 | 352 | 1.45 |
| q5_precision_plus_q4_source_kimi_qwen | 122 | 1.6 |
| q5_correction_overlay_repair_kimi_qwen | 80 | 1.75 |
| q5_precision_kimi_qwen_combined | 76 | 1.65 |
| hard77_preserve | 77 | 2.0 |
| q5_q8shape150_q8win_value_aware | 62 | 2.1 |
| q5_value125_q8win_value_aware | 54 | 2.2 |
| q5_residual218_push_kimi_qwen | 51 | 1.9 |
| q5_secondpass50_q8win_specificity | 31 | 2.3 |

Training config:

- `configs/sft/q5_step2400_broad_reweighted_basin_v1_lr6e7.yaml`
- Runner:
  `hf_bucket/experiments/dpo/runtime/py312/bin/python scripts/run_weighted_sft_gemma.py`
- Smoke 2-step run completed successfully.
- Full run started after the smoke.

Live broad-run evaluation plan:

- Full output root:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7`
- Save cadence: every 250 training steps, up to the final step.
- First pass after training: llama.cpp Q5 runtime sweep on the combined
  454-row heuristic set.
- Source of truth after a candidate is selected: 500-row blind q8-vs-q5
  review with GPT-5.4-Mini reviewer shards.
- Promotion rule: do not move the q5 promotion pointer unless q5 beats q8
  convincingly in blind review.

Broad-run result and diagnostic reframing:

- Full broad run completed at 2,835 steps in about 44.8 minutes.
- Output root:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7`
- Runtime sweep summary:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/eval/heuristic_broad_basin_lr6e7_combined_454_summary.json`
- The sweep shows many exact-number and formula-heavy outputs, especially on
  the old numeric risk set. That is useful evidence, but it is not enough by
  itself to reject the branch.
- User clarification: the first broad run is expected to be dirty. The real
  question is whether it is a better base to clean than the existing q5 base.
- Follow-up correction: "old base" means the original upstream 10k q5 LoRA
  base, not checkpoint 218 or second-pass checkpoint 50.
- Correct old-base comparator:
  `hf_bucket/experiments/mlx_lora/runtime_shift/llama_cpp_q5_clean_slate_broad_v1_20260508/gguf_adapter/step_2400_lora.f16.gguf`
- Next diagnostic should be blind A/B review against that step-2400 old base,
  judged on faithfulness, simplicity, readability, and mobile-feed usefulness
  rather than digit presence alone.

Old-base vs broad-base diagnostic setup:

- Old-base outputs:
  `hf_bucket/experiments/mlx_lora/runtime_shift/llama_cpp_q5_clean_slate_broad_v1_20260508/eval/final500_old_base_step2400_diag_outputs/old_base_step2400_outputs_500.jsonl`
- Broad candidate outputs:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/eval/final500_broad_base_diag_outputs/checkpoint_2500_outputs_500.jsonl`
- Broad checkpoint selected for blind diagnostic: checkpoint 2500.
- Reason for selecting checkpoint 2500 over 2750: nearly identical surface
  profile, with checkpoint 2500 slightly lower on overlong and digit-heavy
  rows in the final500 quick screen.
- Neutral blind review root:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/eval/blind_ab_basin_diag_20260518_500`
- Review model: GPT-5.4-Mini subagents.

Broad checkpoint 2500 cleanup label pass:

- Source candidates: 176 rows where broad checkpoint 2500 lost to old
  step-2400 or both sides were judged bad in the blind basin diagnostic.
- Candidate prep:
  `scripts/prepare_q5_broad_basin_cleanup_candidates_v1.py`
- Kimi/Qwen config:
  `configs/synthetic_summary_generation/q5_broad2500_cleanup_kimi_qwen_v1.json`
- Writer: 176 completed.
- Local filter: 86 eligible, 90 excluded.
- Qwen verifier: 51 accepted, 35 rejected.
- Release:
  `hf_dataset/releases/recent_3y_balanced_10746_q5_broad2500_cleanup_v1/training_rows.jsonl`
- OpenRouter balance after the pass: about `$4.69`.

Broad checkpoint 2500 cleanup SFT:

- Dataset builder:
  `scripts/build_q5_broad2500_cleanup_v1_dataset.py`
- Dataset:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/dataset_51cleanup_x4_v1/train.jsonl`
- Rows: 204, from 51 accepted cleanup labels repeated 4x with weight 2.2.
- Training config:
  `configs/sft/q5_broad2500_cleanup_v1_lr4e7.yaml`
- Starting adapter:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/checkpoint-2500`
- Output root:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7`
- Full run completed 51 steps in 117.1 seconds, train loss 15.06.
- Runtime sweep:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/eval/final500_cleanup_lr4e7_diag_summary.json`
- Surface read: checkpoint 30 is the most plausible cleanup checkpoint for
  blind review. It changes 94 of 500 outputs, slightly reduces digit and
  unsupported-number counts, and does less repetition drift than later
  checkpoints.

Cleanup checkpoint 30 blind review vs broad checkpoint 2500:

- Review root:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/eval/blind_ab_cleanup_diag_20260518_500`
- GPT-5.4-Mini reviewer shards: 10 x 50 rows.
- Missing/duplicate/schema errors: 0/0/0.
- Preferred:
  - cleanup checkpoint 30: 45
  - broad checkpoint 2500 baseline: 41
  - tie: 386
  - neither: 28
- More accurate:
  - cleanup checkpoint 30: 46
  - broad checkpoint 2500 baseline: 40
  - tie: 386
  - neither: 28
- Decision: cleanup checkpoint 30 is a tiny positive move and is not damaging,
  but the effect is small. Use checkpoint 30 as the cleaned broad candidate for
  the q8 comparison; do not treat it as a major improvement over broad 2500.

Cleaned broad checkpoint 30 blind review vs q8:

- Review root:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/eval/blind_ab_final_reference_20260518_500`
- GPT-5.4-Mini reviewer shards: 10 x 50 rows.
- Missing/duplicate/schema errors: 0/0/0.
- Preferred:
  - q8 MLX-Python: 263
  - q5 broad cleanup checkpoint 30: 208
  - tie: 14
  - neither: 15
- More accurate:
  - q8 MLX-Python: 263
  - q5 broad cleanup checkpoint 30: 208
  - tie: 14
  - neither: 15
- Decision: not promotable yet, but this is the strongest q5-vs-q8 result so
  far in the current run. Continue with a targeted q8-win cleanup pass from
  this review.

Old-base vs broad-base blind diagnostic result:

- Rows reviewed: 500
- Missing/duplicate/schema errors: 0/0/0
- Preferred:
  - broad checkpoint 2500: 209
  - old step-2400 base: 152
  - tie: 115
  - neither: 24
- More accurate:
  - broad checkpoint 2500: 209
  - old step-2400 base: 152
  - tie: 115
  - neither: 24
- By split, broad checkpoint 2500 beats old step-2400 on control, entity,
  long-notation, numeric, and polarity rows. Old step-2400 wins the tiny
  remembered-stress split.
- Decision: broad checkpoint 2500 is a better base than the old upstream q5
  step-2400 base, but not a final candidate. Continue from broad checkpoint
  2500 with a cleanup overlay.

Targeted q8-win cleanup after broad cleanup checkpoint 30:

- Source review:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/eval/blind_ab_final_reference_20260518_500`
- Candidate prep:
  `scripts/prepare_q5_q8win_candidates_from_blind_review_v1.py`
- Kimi/Qwen config:
  `configs/synthetic_summary_generation/q5_broad_cleanup30_q8win_kimi_qwen_v1.json`
- Source candidates: 261 q8-win or neither rows.
- Qwen-reviewed rows: 148.
- Accepted labels: 78.
- Release:
  `hf_dataset/releases/recent_3y_balanced_10746_q5_broad_cleanup30_q8win_v1/training_rows.jsonl`
- OpenRouter balance after the pass: about `$4.39`.

Targeted q8-win cleanup SFT:

- Dataset builder:
  `scripts/build_q5_broad_cleanup30_q8win_v1_dataset.py`
- Dataset:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/dataset_78cleanup_x3_v1/train.jsonl`
- Rows: 234, from 78 accepted cleanup labels repeated 3x with weight 2.0.
- Training config:
  `configs/sft/q5_broad_cleanup30_q8win_v1_lr3e7.yaml`
- Starting adapter:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/checkpoint-30`
- Output root:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3`
- Runtime sweep:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/eval/final500_q8win_lr3e7_x3_diag_summary.json`
- Selected checkpoint 50 for blind review. It had a nearly unchanged surface
  profile while still moving enough outputs to matter.

q8-win checkpoint 50 blind review vs q8:

- Review root:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/eval/blind_ab_q8_vs_q8win50_20260518_500`
- GPT-5.4-Mini reviewer shards: 10 x 50 rows.
- Missing/duplicate/schema errors: 0/0/0.
- Preferred:
  - q8 MLX-Python: 253
  - q5 q8-win cleanup checkpoint 50: 217
  - tie: 14
  - neither: 16
- More accurate:
  - q8 MLX-Python: 252
  - q5 q8-win cleanup checkpoint 50: 218
  - tie: 14
  - neither: 16
- Decision: checkpoint 50 improves the q5-vs-q8 gap from 55 to 36 wins, but
  q5 is still behind. It is the best q5 candidate so far, not a promotion.

Stronger replay of the same 78 q8-win labels:

- Training config:
  `configs/sft/q5_broad_cleanup30_q8win_v1_lr6e7_ep2.yaml`
- Output root:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr6e7_x3_ep2`
- Runtime sweep:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr6e7_x3_ep2/eval/final500_q8win_lr6e7_x3_ep2_diag_summary.json`
- Decision: not worth blind review. The surface got noisier, with more digit
  rows and worse unsupported-number examples.

Round-two q8-win cleanup after checkpoint 50:

- Source review:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/eval/blind_ab_q8_vs_q8win50_20260518_500`
- Kimi/Qwen config:
  `configs/synthetic_summary_generation/q5_broad_q8win50_round2_kimi_qwen_v1.json`
- Source candidates: 251 current q8-win or neither rows.
- Qwen-reviewed rows: 78.
- Accepted labels: 51.
- Release:
  `hf_dataset/releases/recent_3y_balanced_10746_q5_broad_q8win50_round2_v1/training_rows.jsonl`
- OpenRouter balance after the pass: about `$4.14`.

Round-two overlay SFT:

- Dataset builder:
  `scripts/build_q5_broad_q8win50_round2_v1_dataset.py`
- Dataset:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/round2_after_ckpt50/dataset_51round2_x4_v1/train.jsonl`
- Rows: 204, from 51 accepted cleanup labels repeated 4x with weight 2.0.
- Training config:
  `configs/sft/q5_broad_q8win50_round2_v1_lr3e7.yaml`
- Output root:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/round2_after_ckpt50/lr3e7_x4`
- Runtime sweep:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/round2_after_ckpt50/lr3e7_x4/eval/final500_round2_lr3e7_x4_diag_summary.json`
- Decision: not worth blind review. Checkpoint 10 barely moved; later
  checkpoints added more noisy digit and overlong rows without a convincing
  quality signal.

Broad reweighted basin v2:

- Rationale: targeted overlays were helping but with diminishing returns. Try
  a new broad-basin pass from the correct upstream old q5 step-2400 base,
  mixing the latest Kimi/Qwen cleanup labels into the broad 10k-style corpus.
- Dataset builder:
  `scripts/build_q5_broad_reweighted_basin_v2_dataset.py`
- Dataset:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v2/dataset_broad_base_low_repair_high_v2/train.jsonl`
- Rows: 11,419.
- Training config:
  `configs/sft/q5_step2400_broad_reweighted_basin_v2_lr6e7.yaml`
- Output root:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v2/lr6e7`
- Early runtime sweep:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v2/lr6e7/eval/final500_v2_early_diag_summary.json`
- Early read: v2 checkpoint 250 has 135 digit rows, compared with 121 for the
  current q5 q8-win checkpoint-50 reference. This looks dirty, but the retro
  v1 early sweep shows v1 checkpoint 250 had 134 digit rows and recovered to
  120 by checkpoint 500:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/eval/final500_v1_early_retro_diag_summary.json`
- Decision: do not stop v2 at checkpoint 250. Let it reach checkpoint 500 and
  compare again before deciding whether this basin is worth blind review.
- Checkpoint-500 runtime sweep:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v2/lr6e7/eval/final500_v2_ckpt500_diag_summary.json`
- Checkpoint-500 read: v2 improves from 135 to 130 digit rows, but it is still
  behind the current q5 q8-win checkpoint-50 reference at 121 and behind the
  retro v1 checkpoint-500 surface at 120. It is also more overlong than v1 at
  the same step.
- Decision: continue to checkpoint 750/1000 only because broad-basin runs can
  recover late. Do not blind-review checkpoint 500.
- Checkpoint-750/1000 runtime sweep:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v2/lr6e7/eval/final500_v2_ckpt750_1000_diag_summary.json`
- Checkpoint-1000 read: v2 continues recovering, reaching 125 digit rows and
  76 over-15-word rows, but has one incomplete-looking output. It changes 193
  of 500 outputs against the current q5 q8-win checkpoint-50 reference, with
  mixed examples. Do not blind-review checkpoint 1000.
- Checkpoint-1250 runtime sweep:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v2/lr6e7/eval/final500_v2_ckpt1250_diag_summary.json`
- Checkpoint-1250 read: v2 reaches 124 digit rows, 78 over-15-word rows, 28
  under-10-word rows, and no bad endings. This is cleaner than checkpoint 1000
  but still behind the current q5 reference on digit rows.
- Checkpoint-1500 runtime sweep:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v2/lr6e7/eval/final500_v2_ckpt1500_diag_summary.json`
- Checkpoint-1500 read: v2 regresses to 126 digit rows while keeping no bad
  endings. It is shorter than the current q5 reference but still less clean
  on value attachment.
- Bridge blind review, current q5 q8-win checkpoint 50 vs v2 checkpoint 1250:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v2/lr6e7/eval/blind_ab_q8win50_vs_v2ckpt1250_20260518_500`
- GPT-5.4-Mini reviewer shards: 10 x 50 rows.
- Missing/duplicate/schema errors: 0/0/0.
- Preferred:
  - current q5 q8-win checkpoint 50: 91
  - v2 checkpoint 1250: 68
  - tie: 319
  - neither: 22
- More accurate:
  - current q5 q8-win checkpoint 50: 89
  - v2 checkpoint 1250: 67
  - tie: 322
  - neither: 22
- Decision: stop v2. It produces many ties but loses to the current best q5
  checkpoint, so it will not close the q8 gap. Pivot to a lighter-weight or
  structurally different mix rather than more training in this branch.

Adapter interpolation probe:

- Utility added:
  `scripts/interpolate_peft_lora_adapters.py`
- Mixed current q5 q8-win checkpoint 50 with v2 checkpoint 1250 at alpha
  0.10, 0.20, and 0.30:
  `hf_bucket/experiments/sft/q5_adapter_interpolation/current_q8win50_to_v2ckpt1250_v1`
- Runtime sweep:
  `hf_bucket/experiments/sft/q5_adapter_interpolation/current_q8win50_to_v2ckpt1250_v1/eval/final500_interp_v2mix_diag_summary.json`
- Surface:
  - current q5 reference: 121 digit rows, 86 over-15-word rows
  - alpha 0.10: 123 digit rows, 87 over-15-word rows, 28 changed outputs
  - alpha 0.20: 123 digit rows, 85 over-15-word rows, 50 changed outputs
  - alpha 0.30: 123 digit rows, 83 over-15-word rows, 71 changed outputs
- Read: low-alpha interpolation imports some v2 concision but also imports
  unsupported specifics and numeric/value attachment drift, including the
  1618-bit tangle failure at alpha 0.20 and 0.30.
- Decision: do not blind-review interpolated adapters.

Current-q5 anchored union overlay:

- Dataset builder:
  `scripts/build_q5_current_union_repair_anchored_v1_dataset.py`
- Dataset:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/anchored_union_after_ckpt50/dataset_union_repairs_anchored_v1/train.jsonl`
- Rows: 193, from 142 unique papers:
  - `broad2500_cleanup_anchor`: 33 rows, weight 0.8
  - `q8win_cleanup_anchor`: 58 rows, weight 1.1
  - `q8win_round2_repair`: 102 rows, weight 1.5, repeated 2x
- Training config:
  `configs/sft/q5_current_union_repair_anchored_v1_lr2e7.yaml`
- Starting adapter: current q5 q8-win checkpoint 50.
- Output root:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/anchored_union_after_ckpt50/lr2e7`
- Full run completed 49 steps in 93.7 seconds.
- Runtime sweep:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/anchored_union_after_ckpt50/lr2e7/eval/final500_current_union_repair_anchored_lr2e7_diag_summary.json`
- Surface:
  - current q5 reference: 121 digit rows, 86 over-15-word rows
  - checkpoint 10: 121 digit rows, 88 over-15-word rows, 15 changed outputs
  - checkpoint 20: 122 digit rows, 92 over-15-word rows, 28 changed outputs
  - checkpoint 30: 122 digit rows, 96 over-15-word rows, 44 changed outputs
  - checkpoint 40: 122 digit rows, 100 over-15-word rows, 52 changed outputs
  - final 49: 122 digit rows, 101 over-15-word rows, 59 changed outputs
- Read: too much of the movement is toward longer, more specific, and sometimes
  unsupported claims. Checkpoint 10 is too small a move; later checkpoints drift.
- Decision: do not blind-review this overlay.

Simple-intro Kimi/Qwen label attempts:

- Rationale: the no-new-label overlays were drifting toward dense specificity.
  Try labels that encode the clarified product goal: faithful, readable,
  intro-like feed hooks rather than mini-abstracts.
- Candidate source:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/eval/q8win_simple_intro_after_ckpt50/source_candidates/q8win_repair_candidates.jsonl`
- Candidate rows: 141 q8-preferred/current-q5-losing rows, excluding prior
  broad/q8win cleanup label releases.
- Credit before these live calls: OpenRouter usage `65.871226533` of `70`.
- v1 config:
  `configs/synthetic_summary_generation/q5_q8win50_simple_intro_kimi_qwen_v1.json`
- v1 release:
  `hf_dataset/releases/recent_3y_balanced_10746_q5_q8win50_simple_intro_v1/training_rows.jsonl`
- v1 yield: 141 writer rows, 42 local-eligible rows, 4 Qwen-accepted rows.
- v1 read: the 10-13 word hard gate discarded many reasonable 14-15 word
  hooks, and Qwen rejected most short summaries as missing central values.
- v2 balanced config:
  `configs/synthetic_summary_generation/q5_q8win50_simple_intro_kimi_qwen_v2_balanced.json`
- v2 balanced release:
  `hf_dataset/releases/recent_3y_balanced_10746_q5_q8win50_simple_intro_v2_balanced/training_rows.jsonl`
- v2 yield: 141 writer rows, 46 local-eligible rows, 5 Qwen-accepted rows.
- v2 read: allowing 10-15 words did not fix the core problem. Kimi still
  produced many overlong summaries, while Qwen rejected compact versions that
  dropped the q8-winning central hook.
- Retry candidate prep:
  `scripts/prepare_q5_q8win_simple_intro_retry_candidates_v1.py`
- Retry source:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/eval/q8win_simple_intro_after_ckpt50/retry_after_v2/source_candidates/retry_candidates.jsonl`
- Retry config:
  `configs/synthetic_summary_generation/q5_q8win50_simple_intro_retry_kimi_qwen_v1.json`
- Retry release:
  `hf_dataset/releases/recent_3y_balanced_10746_q5_q8win50_simple_intro_retry_v1/training_rows.jsonl`
- Retry yield: 134 retry candidates, 45 local-eligible rows, 4 Qwen-accepted
  rows.
- Credit after these Kimi/Qwen attempts: OpenRouter usage `66.312540684` of
  `70`, about `$3.69` remaining.
- Decision: stop this label family. It is high precision but far too low yield
  to train a meaningful adapter, and repeated retries spend budget without
  changing the accepted-row scale.

Blind-review preference DPO from current q5:

- Rationale: use the blind review itself as the preference signal without
  creating new written summaries. q8 outputs are chosen only on q8-preferred
  and q8-more-accurate rows; current q5 outputs are chosen on current-q5
  preferred and more-accurate rows. Ties, neither, and preference/accuracy
  disagreements are skipped.
- Dataset builder:
  `scripts/build_q5_q8win50_blind_dpo_pairs_v1.py`
- Dataset:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/dpo_blind_q8_q5_balanced_v1/dataset`
- Pairs: 467 total; 251 q8-chosen, 216 current-q5-chosen.
- Split counts: 373 train, 48 valid, 46 test.
- Config:
  `configs/dpo/gemma_3_270m_it_title_abstract_q5_q8win50_blind_balanced_v1.yaml`
- Starting adapter: current q5 q8-win checkpoint 50.
- Output root:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/dpo_blind_q8_q5_balanced_v1/lr2e7_beta01`
- Training completed 94 steps. The final DPO valid/test evaluation was stopped
  after training because macOS reported volume-space errors while MPS graph
  cache files were being created. Checkpoints were already written.
- Runtime sweep:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/dpo_blind_q8_q5_balanced_v1/lr2e7_beta01/eval/final500_dpo_blind_pref_v1_diag_summary.json`
- Surface:
  - current q5 reference: 121 digit rows, 70 over-15-word rows, 25 under-10 rows
  - checkpoint 25: 121 digit rows, 74 over-15-word rows, 24 under-10 rows, 22 changed outputs
  - checkpoint 50: 121 digit rows, 75 over-15-word rows, 23 under-10 rows, 18 changed outputs
  - checkpoint 75: 122 digit rows, 75 over-15-word rows, 23 under-10 rows, 24 changed outputs
  - checkpoint 94: 122 digit rows, 76 over-15-word rows, 23 under-10 rows, 26 changed outputs
- Read: not enough movement to close the q8 gap, and the movement slightly
  worsens overlong/digit surfaces. Do not blind-review this DPO branch.

Runtime prompt variants:

- `scripts/generate_llama_stream.py` now accepts optional `--prompt-spec` and
  `--input-mode`; defaults are unchanged.
- Concise/value-safe prompt:
  `prompts/llama_runtime/q5_feed_hook_concise_value_safe_v1.json`
- Output:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/eval/final500_prompt_variant_concise_value_safe_v1/current_q5_prompt_variant_outputs_500.jsonl`
- Surface vs default current q5: 132 digit rows vs 121, 70 over-15 rows vs 70,
  33 under-10 rows vs 25, 271 changed outputs.
- Read: worse; it immediately resurrects the old `1618` tangle failure.
- Digit-free prompt:
  `prompts/llama_runtime/q5_feed_hook_digit_free_v1.json`
- Output:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/eval/final500_prompt_variant_digit_free_v1/current_q5_prompt_digit_free_outputs_500.jsonl`
- Surface vs default current q5: 129 digit rows vs 121, 75 over-15 rows vs 70,
  36 under-10 rows vs 25, 286 changed outputs.
- Read: worse; despite the prompt, it also resurrects the `1618` tangle failure.
- Decision: prompt pressure is not a safe lever for this adapter. It changes a
  lot of outputs but tends to revive older numeric/format attractors.
