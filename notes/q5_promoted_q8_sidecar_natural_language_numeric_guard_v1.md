# q5 promoted q8 sidecar natural-language numeric guard v1

Existing-data-only diagnostic run from the promoted q5/q8 sidecar.

## Dataset

- Builder: `scripts/build_q5_natural_language_numeric_guard_v1_dataset.py`
- Output: `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_numeric_guard_v1/dataset_clean1648_numericguard200_v1/train.jsonl`
- Rows: 1813
- Clean stabilizer rows: 1648
- Numeric guard rows: 165
- Source for guard rows: prior `q5_step2400_sft700_targeted_booster_clean_v1` patch dataset
- No text generated in this run; the builder only converts existing rows to the v5 prompt and scales existing span weights.

Guard filter:

- keep only numeric targeted-weight or booster patch rows
- 10-15 words
- no ASCII/superscript/subscript digit glyphs
- no formula-ish text
- exclude clean dataset IDs and eval IDs
- dedupe patch IDs

Guard weighting:

- row base weight: 0.15
- existing patch span weights scaled by 0.35
- max scaled span weights observed: 1.4, 1.75, 3.15, 3.5

## Training

- Config: `configs/sft/q5_promoted_q8_sidecar_natural_language_numeric_guard_v1_lr1e6_steps128.yaml`
- Start adapter: promoted q5/q8 sidecar
- LR: 1e-6
- Steps: 128
- Save every 32 steps

## Status

Training and fresh20 eval completed.

Fresh20 metrics:

- Baseline: avg `12.85`, max `21`, `>14` `4`, `>15` `3`, `<10` `1`,
  digits `0`, title-overlap `>=3` `16`.
- Checkpoint `32`: avg `12.75`, max `21`, `>14` `5`, `>15` `3`, `<10` `1`,
  digits `0`, title-overlap `>=3` `16`, changed `2`, regressions `1`.
- Checkpoint `64`: same mechanics as checkpoint `32`.
- Checkpoint `96`: avg `12.55`, max `21`, `>14` `5`, `>15` `3`, `<10` `2`,
  digits `0`, title-overlap `>=3` `16`, changed `3`, regressions `1`.
- Checkpoint `128`: avg `12.40`, max `21`, `>14` `4`, `>15` `2`, `<10` `2`,
  digits `0`, title-overlap `>=3` `16`, changed `5`, regressions `1`.

Read: the numeric guard helped prevent digit relapse, but by itself it was too
timid. The useful changes were mostly compression, not a clear feed-quality
gain. Not a blind-review candidate.
