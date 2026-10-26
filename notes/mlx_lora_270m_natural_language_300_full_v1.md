# MLX full finetune 270M natural-language 300-row v1

## Goal

Compare an all-layer MLX full-mode run against the 300-row natural-language LoRA
run using the same abstract-only prompt, data split, batch size, and iteration
budget.

This is a control experiment. It checks whether full mode gives better behavior
than a fresh LoRA adapter on the same 300 strict-audited teacher rows.

## Data

Source release:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_300_milestone/training_rows.jsonl`

MLX dataset:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_300_milestone/mlx_lora/abstract_only_chat_gemma_270m_it_abstract_to_natural_language_summary_v1`

Split:

- train: `270`
- valid: `15`
- test: `15`

LoRA comparison checkpoint:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_300_lora_v1/adapters/0000120_adapters.safetensors`

## Training

Config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_300_full_all_layers_batch4_v1.yaml`

Command:

```bash
HF_HOME=hf_bucket/experiments/mlx_lora/.hf-home \
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_natural_language_300_full_all_layers_batch4_v1.yaml \
  --mask-prompt
```

Output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_300_full_all_layers_batch4_v1/full`

Training details:

- base model: `google/gemma-3-270m-it`
- fine-tune type: `full`
- trainable params: `100.325M / 268.098M`
- trainable share: `37.421%`
- batch size: `4`
- iterations: `300`
- learning rate: `1e-6`
- peak memory: about `5.83 GB`

MLX full mode caveat:

- With `num_layers: -1`, MLX full mode updates all selected transformer layers,
  not every parameter in the model. Embeddings and some non-block parameters are
  still outside the trained parameter set.

Training curve:

| step | train loss | valid loss |
| ---: | ---: | ---: |
| 1 | n/a | 6.105 |
| 20 | 3.398 | 3.859 |
| 40 | 3.041 | 3.535 |
| 60 | 2.770 | 3.417 |
| 80 | 2.661 | 3.312 |
| 100 | 2.472 | 3.263 |
| 120 | 2.725 | 3.242 |
| 140 | 2.634 | 3.221 |
| 160 | 2.522 | 3.213 |
| 180 | 2.709 | 3.212 |
| 200 | 2.531 | 3.214 |
| 220 | 2.389 | 3.204 |
| 240 | 2.352 | 3.198 |
| 260 | 2.328 | 3.206 |
| 280 | 2.390 | 3.203 |
| 300 | 2.512 | 3.189 |

Final checkpoint:

- best validation checkpoint: `0000300`
- test loss: `2.586`
- test perplexity: `13.272`

## Generation Checks

Artifacts:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_300_full_all_layers_batch4_v1/review_samples/lora120_vs_full300_fresh40`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_300_full_all_layers_batch4_v1/review_samples/lora120_vs_full300_test15`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_300_full_all_layers_batch4_v1/review_samples/lora120_vs_full300_train20`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_300_full_all_layers_batch4_v1/review_samples/full120_vs_full300_fresh40`

Fresh 40-row corpus sample:

| variant | avg words | median words | max words | over 14 | weird flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA 120 | 13.30 | 13 | 21 | 15 / 40 | 2 |
| full 120 | 14.80 | 13 | 37 | 15 / 40 | 7 |
| full 300 | 15.25 | 14 | 37 | 18 / 40 | 5 |

Held-out teacher test split:

| variant | avg words | median words | max words | over 14 | weird flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA 120 | 12.47 | 14 | 16 | 3 / 15 | 0 |
| full 300 | 13.07 | 13 | 19 | 6 / 15 | 0 |

Seen training diagnostic:

| variant | avg words | median words | max words | over 14 | weird flags |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA 120 | 12.05 | 12 | 19 | 4 / 20 | 0 |
| full 300 | 12.95 | 13 | 23 | 6 / 20 | 0 |

## Read

Full mode did not beat the 300-row LoRA run.

What changed:

- Full mode kept improving on validation through step 300, but the validation
  loss stayed worse than the LoRA run.
- Generation behavior is also worse than LoRA 120 on the fresh corpus sample.
- Full 300 is longer, more abstract-like, and more likely to spill into dense
  metrics, acronyms, or notation.
- Full 120 is not a better product checkpoint. It is slightly shorter than full
  300 but has more malformed or suspicious outputs on the fresh sample.

The full run does not support the idea that simply touching more weights gives
better control at 300 rows. The bottleneck still looks like target coverage,
hard-row supervision, and possibly model capacity, not just adapter capacity.

Current product read:

- Best LoRA checkpoint so far: `0000120`
- Best full checkpoint by validation: `0000300`
- Best full checkpoint by product behavior: no clear winner; neither beats LoRA
  120

The next useful move is probably not another same-size full run. Better options
are more strict-good rows, stronger hard-negative/length-pressure data, or a
cheap generation-time repair/filter layer.
