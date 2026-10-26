# q5 Promoted q8 Sidecar: Burn-In 32 Stabilizer v3

## Intent

Third stabilizer branch after the 994-gold burn-in checkpoint 32.

v2 proved that stronger anchor pressure can reduce long-output regressions, but
it also pulled numeric habits back into the candidate. This run keeps anchor
pressure, but only lets clean digit-free anchors carry most of that pressure.

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
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v3/dataset_994gold_1000faithanchors_digitdamped_v3`
- Builder:
  `scripts/build_q5_natural_language_burnin32_stabilizer_v3_dataset.py`
- Prompt:
  `gemma_270m_it_abstract_to_feed_blurb_v5`
- Input mode:
  `title_abstract`

## Row Weights

Gold:

- Strict gold default: `0.55`
- Strict gold title-overlap risk: `0.35`
- Strict gold digit cap: `0.30`
- Strict gold under-10 cap: `0.15`

Anchors:

- Clean digit-free 10-14 word anchor: `0.90`
- Title-overlap-risk digit-free 10-14 word anchor: `0.70`
- Digit-bearing anchor: `0.10`
- Under-10 anchor: `0.25`
- Over-14 anchor: `0.30`

## Training

- Start adapter:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin_v1/lr6e6_steps192/checkpoint-32`
- Config:
  `configs/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v3_lr1e6_steps160.yaml`
- Output:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v3/lr1e6_steps160`
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
  - stabilizer v1.
  - stabilizer v2.

## Verdict

Pending.
