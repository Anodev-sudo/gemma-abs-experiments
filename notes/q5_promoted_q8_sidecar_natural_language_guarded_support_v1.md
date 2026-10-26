# q5 promoted q8 sidecar natural-language guarded support v1

Existing-data-only diagnostic run from the promoted q5/q8 sidecar.

## Dataset

- Builder: `scripts/build_q5_natural_language_guarded_support_v1_dataset.py`
- Output: `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_guarded_support_v1/dataset_clean1648_under14support482_numericguard165_v1/train.jsonl`
- Rows: 2277
- Clean stabilizer rows: 1648
- Under-14 support rows: 464
- Numeric guard rows: 165
- No generated text in this run; all rows come from previously recorded artifacts.

Priority order for duplicate IDs:

1. clean stabilizer
2. numeric guard
3. under-14 support

The builder dropped 18 under-14 rows that overlapped numeric guard rows.

## Training

- Config: `configs/sft/q5_promoted_q8_sidecar_natural_language_guarded_support_v1_lr1p5e6_steps128.yaml`
- Start adapter: promoted q5/q8 sidecar
- LR: 1.5e-6
- Steps: 128
- Save every 32 steps

## Status

Training and fresh20 eval completed.

Fresh20 metrics:

- Baseline: avg `12.85`, max `21`, `>14` `4`, `>15` `3`, `<10` `1`,
  digits `0`, title-overlap `>=3` `16`.
- Checkpoint `32`: avg `12.50`, max `21`, `>14` `4`, `>15` `3`, `<10` `2`,
  digits `0`, title-overlap `>=3` `16`, changed `3`, regressions `0`.
- Checkpoint `64`: avg `12.55`, max `21`, `>14` `5`, `>15` `3`, `<10` `2`,
  digits `0`, title-overlap `>=3` `16`, changed `4`, regressions `1`.
- Checkpoint `96`: same mechanics as checkpoint `64`.
- Checkpoint `128`: avg `12.55`, max `21`, `>14` `5`, `>15` `3`, `<10` `2`,
  digits `0`, title-overlap `>=3` `16`, changed `5`, regressions `1`.

Read: combining under-14 support with numeric guard prevented the digit relapse,
but the run still barely moved and the changed rows did not clear the manual-read
bar. Not a blind-review candidate.
