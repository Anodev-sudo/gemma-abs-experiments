# q5 promoted sidecar natural-language 500 low-LR no-anchor diagnostic

Purpose: test whether the promoted q5 sidecar has a narrow early steering
window toward natural-language summaries when trained gently on the 500-row
natural-language target without replay anchors.

This is not a promotion candidate by default.

Starting adapter:

```text
hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1
```

Dataset:

```text
hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_aggressive_v1/dataset_title_abstract_500/train.jsonl
```

Config:

```text
configs/sft/q5_promoted_q8_sidecar_natural_language_500_lowlr_noanchors_v1_lr1e6_steps50.yaml
```

Training command:

```bash
hf_bucket/experiments/dpo/runtime/py312/bin/python \
  scripts/run_weighted_sft_gemma.py \
  --config configs/sft/q5_promoted_q8_sidecar_natural_language_500_lowlr_noanchors_v1_lr1e6_steps50.yaml
```

Planned checkpoints: `10`, `20`, `30`, `40`, `50`.

Planned eval: the same fresh20 PEFT/Transformers and llama.cpp natural-language
prompt checks used for the aggressive run.

## Result

Training completed for `50` optimizer steps, saving checkpoints at `10`, `20`,
`30`, `40`, and `50`.

```text
output: hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_lowlr_noanchors_v1/lr1e6_steps50
train rows: 500
assistant-label tokens: 7,736
effective batch size: 4
learning rate: 1.0e-6
final train loss: 11.86
runtime: 99.3 seconds
epoch: 0.4
```

### Shape Metrics

Fresh20 PEFT/Transformers runtime:

| label | avg words | max | over 14 | digit rows | forbidden openings | title overlap >= 3 | changed vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| promoted baseline | 12.05 | 16 | 3 | 0 | 0 | 13 | - |
| checkpoint 10 | 11.95 | 16 | 2 | 0 | 0 | 15 | 2 |
| checkpoint 20 | 12.45 | 16 | 2 | 0 | 0 | 14 | 4 |
| checkpoint 30 | 12.40 | 16 | 2 | 0 | 0 | 14 | 5 |
| checkpoint 40 | 12.25 | 16 | 2 | 0 | 0 | 14 | 6 |
| checkpoint 50 | 12.25 | 16 | 2 | 0 | 0 | 14 | 6 |

Fresh20 llama.cpp runtime:

| label | avg words | max | over 14 | digit rows | forbidden openings | title overlap >= 3 | changed vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| promoted baseline | 12.85 | 17 | 6 | 2 | 0 | 14 | - |
| checkpoint 10 | 12.80 | 17 | 6 | 2 | 0 | 14 | 1 |
| checkpoint 20 | 12.80 | 17 | 6 | 2 | 0 | 14 | 1 |
| checkpoint 30 | 12.80 | 17 | 6 | 2 | 0 | 14 | 3 |
| checkpoint 40 | 12.80 | 17 | 6 | 2 | 0 | 14 | 3 |
| checkpoint 50 | 12.70 | 17 | 6 | 2 | 0 | 13 | 4 |

### Read

This run is too cold to matter in the product runtime. PEFT changes a few rows,
but llama.cpp is nearly unchanged even at checkpoint `50`, and the shape metrics
do not materially improve.

The small PEFT movement is not clearly useful:

- `2402.03953` drifts from decentralized exchange behavior into Bitcoin trading
  phrasing by checkpoint `10`, then grows into stronger overreaction claims.
- `2212.01114` becomes shorter and more natural in PEFT, but the llama output
  keeps the baseline `model model` repetition unchanged.
- `2601.13411` gets shorter in PEFT at checkpoints `40` and `50`, but the
  shorter wording is still wrong and llama stays unchanged.

Decision: no promotion. `1e-6` with no anchors and early stopping preserves the
baseline, but it does not create enough natural-language movement in llama.cpp.
The interesting next no-anchor bracket would be a midpoint such as `3e-6` or
`5e-6`, still with early checkpoints, if the goal is to find the movement
threshold before faithfulness breaks.
