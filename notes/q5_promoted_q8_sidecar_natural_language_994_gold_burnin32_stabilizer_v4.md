# q5 Promoted q8 Sidecar: Burn-In 32 Stabilizer v4

## Intent

Fourth stabilizer branch after the 994-gold burn-in checkpoint 32.

v3 damped digit-bearing rows but did not materially reduce digit behavior. This
run removes those labels from the stabilizer view entirely and trains only on
existing digit-free 10-14-word labels.

## Constraints

- No new generated data.
- No labels written by Codex.
- Existing labels only.
- No mined silver/bronze rows.
- No token spans.

## Dataset

- Root:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v4/dataset_clean_digitfree_10to14_v4`
- Builder:
  `scripts/build_q5_natural_language_burnin32_stabilizer_v4_dataset.py`
- Prompt:
  `gemma_270m_it_abstract_to_feed_blurb_v5`
- Input mode:
  `title_abstract`

## Row Weights

Gold:

- Clean gold default: `0.70`
- Clean gold title-overlap risk: `0.50`

Anchors:

- Clean anchor default: `0.95`
- Clean anchor title-overlap risk: `0.75`

## Training

- Start adapter:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin_v1/lr6e6_steps192/checkpoint-32`
- Config:
  `configs/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v4_lr1e6_steps160.yaml`
- Output:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v4/lr1e6_steps160`
- LR:
  `1.0e-6`
- Steps:
  `160`
- Save:
  every `32` steps.

## Eval Plan

- fresh20 llama.cpp sweep against promoted baseline.
- held100 llama.cpp sweep against promoted baseline.
- Compare against v2 and v3 before deciding whether this is worth blind review.

## Verdict

Pending.
