# MLX LoRA 270M natural-language 994-row title+abstract prompt-v3 v1

## Goal

Test a candidate-inspired title+abstract prompt contract on the same clean
994-row natural-language gold set. Compared with prompt v2, this prompt is
shorter and borrows the promoted q5 candidate's language:

- title is already visible
- one readable sentence
- 10 to 13 words
- broad reader
- main takeaway over implementation detail
- exact numbers only when central and easy to bind

## Data

Source release:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/training_rows.jsonl`

MLX dataset:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_natural_language_summary_v3`

Prompt spec:

- `prompts/mlx_lora/gemma_270m_it_abstract_to_natural_language_summary_v3.json`

Split:

- train: `894`
- valid: `50`
- test: `50`

## Training

Config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_promptv3_lora_lr1e5_long_v1.yaml`

Command:

```bash
HF_HOME=hf_bucket/experiments/mlx_lora/.hf-home \
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_promptv3_lora_lr1e5_long_v1.yaml \
  --mask-prompt
```

Adapter output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_promptv3_lora_lr1e5_long_v1/adapters`

Settings:

- base model: `google/gemma-3-270m-it`
- fine-tune type: `lora`
- LoRA target modules: attention q/k/v/o projections
- rank: `16`
- scale: `10.0`
- batch size: `4`
- iterations: `500`
- learning rate: `1e-5`
- save/eval interval: `25`

## Result

Training completed.

Training details:

- trainable params: `1.475M / 268.098M`
- peak memory: about `9.25 GB`
- final test loss: `2.313`
- final test perplexity: `10.102`

Validation curve:

| step | valid loss |
| ---: | ---: |
| 1 | 5.383 |
| 25 | 2.622 |
| 50 | 2.445 |
| 75 | 2.412 |
| 100 | 2.351 |
| 125 | 2.322 |
| 150 | 2.310 |
| 175 | 2.296 |
| 200 | 2.301 |
| 225 | 2.292 |
| 250 | 2.311 |
| 275 | 2.284 |
| 300 | 2.289 |
| 325 | 2.292 |
| 350 | 2.270 |
| 375 | 2.268 |
| 400 | 2.272 |
| 425 | 2.243 |
| 450 | 2.237 |
| 475 | 2.261 |
| 500 | 2.237 |

Best validation checkpoints:

- `0000450_adapters.safetensors`
- `0000500_adapters.safetensors`

## Fresh Holdout Checks

Fresh 40-paper holdout reused from the abstract-only 994-row review sample:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_lora_lr1e5_long_v1/review_samples/ckpt360_800_vs_ckpt500_994_fresh40/generations.jsonl`

Generated review samples:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_promptv3_lora_lr1e5_long_v1/review_samples/ckpt425_vs_ckpt450_fresh40_title_abstract_v3`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_promptv3_lora_lr1e5_long_v1/review_samples/ckpt450_vs_ckpt500_fresh40_title_abstract_v3`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_promptv3_lora_lr1e5_long_v1/review_samples/title_abstract_v3_fresh40_mechanical`

Length and shape, generated with `max_tokens=36`:

| variant | avg words | median | max | over 13 | over 14 | over 15 | under 10 | digit rows | formula-like | title overlap >=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ta994_v2_ckpt425` | 12.20 | 12 | 16 | 13 / 40 | 8 / 40 | 4 / 40 | 5 / 40 | 4 / 40 | 1 / 40 | 21 / 40 |
| `ta994_v3_ckpt425` | 12.68 | 12 | 23 | 11 / 40 | 11 / 40 | 6 / 40 | 3 / 40 | 5 / 40 | 1 / 40 | 21 / 40 |
| `ta994_v3_ckpt450` | 13.45 | 13 | 23 | 17 / 40 | 13 / 40 | 10 / 40 | 4 / 40 | 3 / 40 | 1 / 40 | 26 / 40 |
| `ta994_v3_ckpt500` | 12.85 | 13 | 20 | 17 / 40 | 10 / 40 | 7 / 40 | 3 / 40 | 3 / 40 | 1 / 40 | 26 / 40 |

## Read

Prompt v3 does not clearly beat prompt v2 on the quick mechanical/product read.
It slightly improves test loss and sometimes gives nicer broad-reader wording,
but its best-looking checkpoint, `425`, has more over-14 and over-15 rows than
v2 `425` on the same fresh 40-paper sample.

The candidate-inspired wording seems directionally useful, especially
`broad reader` and `main takeaway over implementation detail`, but the stricter
`10 to 13 words` contract does not transfer cleanly when the target labels are
still 8-14 words. Later checkpoints also become more title-attracted.

Current read: keep prompt v3 as an experiment, but do not replace v2 from this
run alone. If we keep iterating on prompt-only changes, the next prompt should
probably preserve the candidate's simple title framing while dropping the
10-13 mismatch or pairing it with regenerated labels.
