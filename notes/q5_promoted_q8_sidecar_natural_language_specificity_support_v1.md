# q5 promoted q8-sidecar natural-language specificity support v1

## Purpose

Test whether low-weight existing broad data can preserve specificity while the
clean natural-language rows push readability.

## Guardrails

- No new generated data.
- No Codex-authored training labels.
- No review labels used as training rows.
- Do not promote without explicit approval.

## Dataset

- Dataset:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_specificity_support_v1/dataset_clean1648_broadsupport_v1/train.jsonl`
- Rows:
  `3248`
- Clean rows:
  `1648`
- Low-weight broad support rows:
  `1600`
- Broad source:
  `hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v2/dataset_broad_base_low_repair_high_v2/train.jsonl`
- Support filter:
  10-14 words, no ASCII/superscript/subscript digits, no formula-ish text, no
  boilerplate opening, no title-overlap risk, no eval IDs, no clean-dataset
  duplicates.

## Training

- Start adapter:
  `hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1`
- Config:
  `configs/sft/q5_promoted_q8_sidecar_natural_language_specificity_support_v1_lr8e7_steps96.yaml`
- Output:
  `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_specificity_support_v1/lr8e7_steps96`
- LR:
  `8e-7`
- Steps:
  `96`
- Save:
  every `32` steps.

## Result

Fresh20 metrics:

- Baseline: avg `12.85`, max `21`, `>14` `4`, `>15` `3`, `<10` `1`,
  digits `0`, title-overlap `>=3` `16`.
- Checkpoint `32`: identical to baseline.
- Checkpoint `64`: avg `12.55`, max `21`, `>14` `5`, `>15` `3`, `<10` `2`,
  digits `0`, title-overlap `>=3` `16`, changed `3`, regressions `1`.
- Checkpoint `96`: same mechanics as checkpoint `64`.

Read: the low-weight broad support did preserve numeric mechanics, but movement
was tiny and mostly looked like truncation/detail loss. Not a blind-review
candidate.
