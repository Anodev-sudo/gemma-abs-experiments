# q5 promoted q8 sidecar natural-language under14 high-support v1

Existing-data-only diagnostic run from the promoted q5/q8 sidecar.

## Dataset

- Builder: `scripts/build_q5_natural_language_under14_highsupport_v1_dataset.py`
- Output: `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_under14_highsupport_v1/dataset_clean1648_under14_highsupport_v1/train.jsonl`
- Rows: 2130
- Clean stabilizer rows: 1648
- Filtered under-14 support rows: 482
- No generated text from this run; support text comes from previously recorded q8 under-14 Kimi/Qwen repair data.

Support filter:

- exclude `base_10746_low_weight`
- 10-13 words
- no ASCII/superscript/subscript digit glyphs
- no formula-ish text
- no boilerplate opening
- no title-overlap risk
- exclude clean dataset IDs and eval IDs

Support rows are low-weight:

- faithfulness: 0.10
- recenter: 0.12
- specificity: 0.20
- mined/q5 repair rows: 0.22-0.24
- hard preserve rows: 0.25

## Training

- Config: `configs/sft/q5_promoted_q8_sidecar_natural_language_under14_highsupport_v1_lr1p2e6_steps128.yaml`
- Start adapter: promoted q5/q8 sidecar
- LR: 1.2e-6
- Steps: 128
- Save every 32 steps

## Status

Training and fresh20 eval completed.

Fresh20 metrics:

- Baseline: avg `12.85`, max `21`, `>14` `4`, `>15` `3`, `<10` `1`,
  digits `0`, title-overlap `>=3` `16`.
- Checkpoint `32`: avg `12.25`, max `21`, `>14` `4`, `>15` `3`, `<10` `3`,
  digits `1`, title-overlap `>=3` `15`, changed `4`, regressions `1`.
- Checkpoint `64`: avg `12.35`, max `21`, `>14` `5`, `>15` `3`, `<10` `3`,
  digits `1`, title-overlap `>=3` `15`, changed `6`, regressions `2`.
- Checkpoint `96`: avg `12.40`, max `21`, `>14` `5`, `>15` `3`, `<10` `3`,
  digits `1`, title-overlap `>=3` `16`, changed `5`, regressions `2`.
- Checkpoint `128`: avg `12.80`, max `21`, `>14` `6`, `>15` `4`, `<10` `2`,
  digits `1`, title-overlap `>=3` `16`, changed `6`, regressions `3`.

Read: support rows pushed some shortness, but the branch reintroduced a numeric
attractor and added regressions before producing enough useful movement. Not a
blind-review candidate.
