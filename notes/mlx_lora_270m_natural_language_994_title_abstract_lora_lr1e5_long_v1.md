# MLX LoRA 270M natural-language 994-row title+abstract v1

## Goal

Run the clean 994-row natural-language gold set again with the student seeing
both the title and abstract. This is a sibling of the abstract-only 994-row
run, not a continuation.

## Data

Source release:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/training_rows.jsonl`

MLX dataset:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_natural_language_summary_v2`

Prompt spec:

- `prompts/mlx_lora/gemma_270m_it_abstract_to_natural_language_summary_v2.json`

Split:

- train: `894`
- valid: `50`
- test: `50`

## Training

Config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_lora_lr1e5_long_v1.yaml`

Command:

```bash
HF_HOME=hf_bucket/experiments/mlx_lora/.hf-home \
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_lora_lr1e5_long_v1.yaml \
  --mask-prompt
```

Adapter output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_lora_lr1e5_long_v1/adapters`

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
- final test loss: `2.323`
- final test perplexity: `10.202`

Validation curve:

| step | valid loss |
| ---: | ---: |
| 1 | 5.480 |
| 25 | 2.608 |
| 50 | 2.451 |
| 75 | 2.398 |
| 100 | 2.348 |
| 125 | 2.321 |
| 150 | 2.304 |
| 175 | 2.298 |
| 200 | 2.292 |
| 225 | 2.287 |
| 250 | 2.296 |
| 275 | 2.279 |
| 300 | 2.277 |
| 325 | 2.289 |
| 350 | 2.270 |
| 375 | 2.264 |
| 400 | 2.270 |
| 425 | 2.241 |
| 450 | 2.232 |
| 475 | 2.245 |
| 500 | 2.239 |

Best validation checkpoint:

- `0000450_adapters.safetensors`

## Fresh Holdout Checks

Fresh 40-paper holdout reused from the abstract-only 994-row review sample:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_lora_lr1e5_long_v1/review_samples/ckpt360_800_vs_ckpt500_994_fresh40/generations.jsonl`

Generated review samples:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_lora_lr1e5_long_v1/review_samples/ckpt275_vs_ckpt425_fresh40_title_abstract_v2`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_lora_lr1e5_long_v1/review_samples/ckpt300_vs_ckpt450_fresh40_title_abstract_v2`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_lora_lr1e5_long_v1/review_samples/ckpt450_vs_ckpt500_fresh40_title_abstract_v2`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_lora_lr1e5_long_v1/review_samples/title_abstract_v2_fresh40_mechanical`

Length and shape, generated with `max_tokens=36`:

| variant | avg words | median | max | over 14 | over 15 | under 8 | digit rows | formula-like | title overlap >=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ta994_ckpt275` | 13.45 | 12.5 | 24 | 11 / 40 | 8 / 40 | 0 / 40 | 4 / 40 | 1 / 40 | 23 / 40 |
| `ta994_ckpt300` | 12.72 | 12 | 21 | 14 / 40 | 9 / 40 | 1 / 40 | 4 / 40 | 1 / 40 | 24 / 40 |
| `ta994_ckpt425` | 12.20 | 12 | 16 | 8 / 40 | 4 / 40 | 0 / 40 | 4 / 40 | 1 / 40 | 21 / 40 |
| `ta994_ckpt450` | 13.68 | 13 | 23 | 14 / 40 | 9 / 40 | 0 / 40 | 4 / 40 | 1 / 40 | 26 / 40 |
| `ta994_ckpt500` | 13.25 | 13 | 24 | 12 / 40 | 6 / 40 | 1 / 40 | 3 / 40 | 1 / 40 | 23 / 40 |

## Read

The title+abstract input clearly improves validation/test loss versus the
abstract-only run, but the fresh-holdout generation read is not automatically
better.

Checkpoint `425` is the best initial product candidate by mechanical shape. It
has the lowest average length, lowest over-limit count, and lowest strong title
overlap count among the sampled checkpoints. It also avoids the long runaway
rows that appear at `450` and `500`.

The main remaining problem is title attraction. The model often uses the title
as a lexical anchor instead of only using it to avoid repetition, especially on
papers where the title names the core object. This likely needs either a prompt
change, title-dropout/mixed-input training, or data that explicitly rewards
non-rehash title-aware summaries.
