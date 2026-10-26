# q5 Promoted q8 Sidecar: Burn-In 32 Stabilizer v2

## Intent

Second stabilizer branch after the 994-gold burn-in checkpoint 32.

The v1 stabilizer moved in the right direction but was too weak: it changed only
a small slice relative to burn-in checkpoint 32 and kept the sticky faithfulness
failures. This v2 run keeps the same data provenance but applies much stronger
existing-anchor pressure.

## Constraints

- No new generated data.
- No labels written by Codex.
- Existing labels only:
  - 994 strict gold rows.
  - 1000 existing faithfulness-anchor rows.
- No mined silver/bronze rows.
- No token spans.

## Dataset

- Root:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v2/dataset_994gold_1000faithanchors_anchorheavy_v2`
- Builder:
  `scripts/build_q5_natural_language_burnin32_stabilizer_v1_dataset.py`
- Prompt:
  `gemma_270m_it_abstract_to_feed_blurb_v5`
- Input mode:
  `title_abstract`
- Eval exclusions:
  - fresh20 packet.
  - held100 packet used for q5 token-weighted review.

## Row Weights

- Strict gold default: `0.45`
- Strict gold title-overlap risk: `0.30`
- Strict gold under-10 cap: `0.20`
- Faithfulness anchors: `0.70`

## Training

- Start adapter:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin_v1/lr6e6_steps192/checkpoint-32`
- Config:
  `configs/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v2_lr1e6_steps160.yaml`
- Output:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v2/lr1e6_steps160`
- LR:
  `1.0e-6`
- Steps:
  `160`
- Save:
  every `32` steps.

## Eval Plan

- fresh20 llama.cpp sweep against promoted baseline.
- held100 llama.cpp sweep against promoted baseline.
- Compare against:
  - promoted baseline.
  - burn-in checkpoint 32.
  - stabilizer v1 checkpoint 32.

## Verdict

Pending.
