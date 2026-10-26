# q5 promoted q8 sidecar curated200 DPO v1

Diagnostic preference-only run from the current promoted q5/q8 sidecar.

## Intent

Test whether the existing curated 200-pair DPO set can apply targeted pressure
on failure modes that SFT has not moved reliably:

- invented or wrongly bound numbers
- polarity flips
- title restatement
- wrong entity/object
- too-generic feed blurbs
- side-detail fixation

This run uses existing preference labels only. No new target summaries are
written for training.

## Inputs

- Adapter: `hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1`
- Dataset: `hf_dataset/releases/recent_3y_balanced_5000_pilot_v2/dpo/title_abstract_preference_balanced200_curated_v2`
- Config: `configs/dpo/q5_promoted_q8_sidecar_curated200_dpo_v1.yaml`

## Training Plan

- LR: `2e-6`
- Beta: `0.1`
- Epochs: `2`
- Effective batch: `4`
- Save every `20` steps
- Expected checkpoints: `20`, `40`, `60`, `80`

## Evaluation Plan

Run q5 llama.cpp eval on the standard fresh20 packet first. Only consider a
blind review if the run produces meaningful movement without the recurring
numeric or faithfulness regressions.

## Result

Training completed cleanly in `80` steps, with checkpoints `20`, `40`, `60`,
and `80`.

Fresh20 metrics:

- Baseline: avg `12.85`, max `21`, `>14` `4`, `>15` `3`, `<10` `1`,
  digits `0`, title-overlap `>=3` `16`.
- Checkpoint `20`: avg `12.95`, max `21`, `>14` `5`, `>15` `4`, `<10` `1`,
  digits `0`, title-overlap `>=3` `16`, changed `5`, regressions `1`.
- Checkpoint `40`: avg `12.85`, max `21`, `>14` `4`, `>15` `3`, `<10` `1`,
  digits `0`, title-overlap `>=3` `16`, changed `10`, regressions `1`.
- Checkpoint `60`: avg `12.95`, max `21`, `>14` `4`, `>15` `3`, `<10` `1`,
  digits `0`, title-overlap `>=3` `16`, changed `10`, regressions `1`.
- Checkpoint `80`: avg `12.95`, max `21`, `>14` `4`, `>15` `3`, `<10` `1`,
  digits `0`, title-overlap `>=3` `16`, changed `11`, regressions `1`.

Manual read: not a contender. The run preserved digit mechanics, but changed
rows included bad specificity drift and repeated-token failures such as
`detect and detect`, unsupported exact-looking math (`m/n`), and `factor
factorization`.
