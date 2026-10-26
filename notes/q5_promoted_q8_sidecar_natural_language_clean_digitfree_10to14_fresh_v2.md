# q5 Promoted q8 Sidecar: Clean Digit-Free Fresh v2

## Intent

Lower-LR repeat of the clean digit-free fresh run.

The v1 fresh run produced a plausible checkpoint at `64`, but manual read found
quiet semantic damage in some changed rows. This run keeps the same existing
clean dataset and promoted start adapter, but lowers the LR to look for a
smoother checkpoint.

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

## Training

- Start adapter:
  `hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1`
- Config:
  `configs/sft/q5_promoted_q8_sidecar_natural_language_clean_digitfree_10to14_fresh_v2_lr1e6_steps192.yaml`
- Output:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_clean_digitfree_10to14_fresh_v2/lr1e6_steps192`
- LR:
  `1.0e-6`
- Steps:
  `192`
- Save:
  every `32` steps.

## Eval Plan

- fresh20 llama.cpp sweep against promoted baseline with v5 prompt.
- held100 llama.cpp sweep against promoted baseline with v5 prompt.
- Compare against fresh v1 checkpoint 64 and burn-in32 stabilizer v2 checkpoint 96.

## Verdict

Pending.
