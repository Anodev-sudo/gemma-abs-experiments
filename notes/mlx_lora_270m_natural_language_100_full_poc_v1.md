# MLX full finetune 270M natural-language 100-row POC v1

## Goal

Compare full finetuning against the 100-row natural-language LoRA POC using
the same abstract-only prompt and strict-audited teacher rows.

Source dataset:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v7_100_good/mlx_lora/abstract_only_chat_gemma_270m_it_abstract_to_natural_language_summary_v1`

LoRA baseline:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_100_poc_lora_v1/adapters/0000060_adapters.safetensors`

## Full-mode caveat

`mlx_lm.lora --fine-tune-type full` does not update all `268M` parameters for
this model. With the default `num_layers: 16`, it updated:

- `89.178M / 268.098M`
- `33.263%`

With `num_layers: -1`, it updated:

- `100.325M / 268.098M`
- `37.421%`

So this is full training over the selected transformer layers, not a whole-model
all-parameter update including embeddings and every non-block parameter.

## Runs

### Default 16-layer full probe

Config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_100_poc_full_v1.yaml`

Output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_100_poc_full_v1/full`

Training curve:

| step | train loss | valid loss |
| ---: | ---: | ---: |
| 1 | n/a | 5.786 |
| 10 | 5.619 | 5.374 |
| 20 | 4.890 | 4.900 |
| 30 | 4.152 | 4.394 |
| 40 | 3.592 | 4.155 |
| 50 | 3.871 | 3.883 |
| 60 | 3.024 | 3.787 |
| 70 | 3.266 | 3.667 |
| 80 | 2.963 | 3.598 |

Final test loss:

- `3.372`
- test perplexity: `29.151`

This run used batch size `1`, so it undertrained relative to the LoRA POC.

### All-layer batch-1 probe

Config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_100_poc_full_all_layers_v1.yaml`

Output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_100_poc_full_all_layers_v1/full`

Training curve:

| step | train loss | valid loss |
| ---: | ---: | ---: |
| 1 | n/a | 5.786 |
| 10 | 5.697 | 5.691 |
| 20 | 5.369 | 5.489 |
| 30 | 4.927 | 5.200 |
| 40 | 4.410 | 5.044 |
| 50 | 4.904 | 4.841 |
| 60 | 4.047 | 4.738 |
| 70 | 4.185 | 4.600 |
| 80 | 3.855 | 4.525 |

Final test loss:

- `4.389`
- test perplexity: `80.522`

This also undertrained. It proves the all-layer setting runs locally, but it is
not the comparison result.

### All-layer batch-4 comparison run

Config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_100_poc_full_all_layers_batch4_v1.yaml`

Output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_100_poc_full_all_layers_batch4_v1/full`

Training curve:

| step | train loss | valid loss |
| ---: | ---: | ---: |
| 1 | n/a | 5.444 |
| 20 | 3.231 | 3.366 |
| 40 | 2.732 | 3.053 |
| 60 | 2.634 | 2.935 |
| 80 | 2.408 | 2.808 |
| 100 | 2.347 | 2.763 |
| 120 | 2.250 | 2.743 |

Final test loss:

- `2.850`
- test perplexity: `17.280`

This is the closest comparison to the LoRA POC. It used the same batch size and
step count as the LoRA run.

## Quick generation comparison

Review artifacts:

- fresh sample: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_100_poc_full_all_layers_batch4_v1/review_samples/lora60_vs_full120_fresh12`
- held-out teacher test: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_100_poc_full_all_layers_batch4_v1/review_samples/lora60_vs_full120_test10`
- seen training rows: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_100_poc_full_all_layers_batch4_v1/review_samples/lora60_vs_full120_train10`

### Fresh 12-row sample

| model | avg words | max words | over 14 | with number | math notation |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA ckpt60 | 14.33 | 24 | 5 | 4 | 3 |
| full ckpt120 | 20.50 | 32 | 10 | 3 | 2 |

### Held-out 10-row teacher test

| model | avg words | max words | over 14 | with number | math notation |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA ckpt60 | 15.00 | 22 | 5 | 3 | 0 |
| full ckpt120 | 14.60 | 23 | 4 | 1 | 0 |

### Seen 10-row training diagnostic

| model | avg words | max words | over 14 | with number | math notation |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA ckpt60 | 11.20 | 16 | 2 | 0 | 0 |
| full ckpt120 | 12.20 | 15 | 1 | 2 | 0 |

## Read

Full finetuning did not produce a clear quality win on the 100-row proof set.

What it did differently:

- It matched the LoRA final test loss when trained with the same batch and step
  shape.
- It was slightly better than LoRA on held-out word-count and numeric-density
  heuristics.
- It was worse on fresh out-of-set abstracts, where it became longer and more
  abstract-like.
- It still invented or distorted details, including seen training rows.

The main bottleneck still looks like data volume and target coverage, not just
adapter capacity. Full finetuning may give more control later, but this 100-row
test does not show that it fixes faithfulness or natural-language behavior by
itself.

## Next Step

The next meaningful comparison should use `300-500` strict-audited rows and keep
the same side-by-side design:

- LoRA checkpoint sweep
- full batch-4 checkpoint sweep
- held-out teacher test
- fresh corpus test
- same local natural-language heuristics

If full finetuning only starts to beat LoRA after the larger data run, then it is
worth considering as the control-first path. If it still mostly changes length
without improving faithfulness, the 270M model is likely data- or capacity-limited
for this target.
