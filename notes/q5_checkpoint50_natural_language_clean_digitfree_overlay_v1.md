# q5 checkpoint-50 natural-language clean overlay v1

## Purpose

Test a basin shift instead of another small update on the promoted q5/q8
sidecar. This run starts from the previous q5-trained checkpoint-50 lineage and
uses only existing clean natural-language rows.

## Guardrails

- No new generated data.
- No Codex-authored training labels.
- No subagent review labels used as training rows.
- Do not promote without explicit approval.

## Dataset

- Dataset:
  `hf_bucket/experiments/sft/q5_checkpoint50_natural_language_clean_digitfree_overlay_v1/dataset_clean_digitfree_10to14_no_supdigit_v1/train.jsonl`
- Rows:
  `1648`
- Shape:
  clean digit-free, 10-14 word natural-language rows, with existing
  `loss_weight_hints`.
- Source:
  deterministic filtered copy of
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v4/dataset_clean_digitfree_10to14_v4/train.jsonl`.
- Filter:
  dropped one existing row whose summary contained a superscript digit in
  `M³IT`.

## Training

- Start adapter:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/checkpoint-50`
- Config:
  `configs/sft/q5_checkpoint50_natural_language_clean_digitfree_overlay_v1_lr1e6_steps160.yaml`
- Output:
  `hf_bucket/experiments/sft/q5_checkpoint50_natural_language_clean_digitfree_overlay_v1/lr1e6_steps160`
- LR:
  `1e-6`
- Steps:
  `160`
- Save:
  every `32` steps.

## Evaluation Plan

- Run llama.cpp fresh20 and held100 with the v5 feed-blurb prompt.
- Compare against the promoted q5/q8 sidecar baseline outputs.
- Only run blind review if a checkpoint looks like a real contender on changed
  rows and mechanics.

## Result

Training was stopped after checkpoint `64` because the early checkpoints had
already shown the basin-shift pattern clearly.

Fresh20 metrics:

- Baseline: avg `12.85`, max `21`, `>14` `4`, `>15` `3`, `<10` `1`,
  digits `0`, title-overlap `>=3` `16`.
- Checkpoint `32`: avg `12.65`, max `19`, `>14` `3`, `>15` `2`, `<10` `3`,
  digits `1`, title-overlap `>=3` `11`, changed `20`, regressions `3`.
- Checkpoint `64`: same mechanics as checkpoint `32`.

Read: this moved the old checkpoint-50 basin aggressively, but it reintroduced
numeric and faithfulness issues while shaving off useful specificity. Not a
blind-review candidate.
