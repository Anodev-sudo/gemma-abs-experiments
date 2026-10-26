# q5 Promoted q8 Sidecar: Clean Digit-Free Fresh v1

## Intent

Fresh run from the promoted q5/q8 sidecar using the existing clean-only
digit-free 10-14-word dataset.

The burn-in32 stabilizer branches improved some mechanics but repeatedly kept
or introduced sticky numeric failures. This branch tests whether the clean
dataset behaves better when applied directly to the promoted sidecar rather
than through the burn-in32 basin.

## Constraints

- No new generated data.
- No labels written by Codex.
- Existing labels only.
- No mined silver/bronze rows.
- No token spans.

## Dataset

- Dataset:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v4/dataset_clean_digitfree_10to14_v4/train.jsonl`
- Rows:
  `1649`
- Components:
  - `893` strict gold clean rows.
  - `756` faithfulness anchor clean rows.
- Label constraints:
  - digit-free
  - 10-14 words
  - existing labels only

## Training

- Start adapter:
  `hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1`
- Config:
  `configs/sft/q5_promoted_q8_sidecar_natural_language_clean_digitfree_10to14_fresh_v1_lr2e6_steps160.yaml`
- Output:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_clean_digitfree_10to14_fresh_v1/lr2e6_steps160`
- LR:
  `2.0e-6`
- Steps:
  `160`
- Save:
  every `32` steps.

## Eval Plan

- fresh20 llama.cpp sweep against promoted baseline.
- held100 llama.cpp sweep against promoted baseline.
- Compare against the best burn-in32 branch, currently v2 checkpoint 96.

## Verdict

Pending.
