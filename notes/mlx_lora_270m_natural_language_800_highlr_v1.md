# MLX LoRA 270M natural-language 800-row high-LR LoRA v1

## Goal

Run a fresh 270M LoRA adapter on the 800-row strict-audited natural-language
teacher milestone with a higher learning rate than the 500-row baseline.

This is an exploration run, not a promotion claim. The main question is whether
the larger gold set supports faster style movement without damaging summary
faithfulness.

## Data

Source release:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_800_milestone/training_rows.jsonl`

MLX dataset:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_800_milestone/mlx_lora/abstract_only_chat_gemma_270m_it_abstract_to_natural_language_summary_v1`

Split:

- train: `720`
- valid: `40`
- test: `40`

Prompt spec:

- `prompts/mlx_lora/gemma_270m_it_abstract_to_natural_language_summary_v1.json`

## Training

Config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_800_lora_highlr_v1.yaml`

Command:

```bash
HF_HOME=hf_bucket/experiments/mlx_lora/.hf-home \
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_natural_language_800_lora_highlr_v1.yaml \
  --mask-prompt
```

Adapter output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_800_lora_highlr_v1/adapters`

Planned settings:

- base model: `google/gemma-3-270m-it`
- fine-tune type: `lora`
- LoRA target modules: attention q/k/v/o projections
- rank: `16`
- scale: `10.0`
- batch size: `4`
- iterations: `360`
- learning rate: `1e-5`

## Progress

Training completed.

Training details:

- trainable params: `1.475M / 268.098M`
- peak memory: about `8.91 GB`

Training curve:

| step | train loss | valid loss |
| ---: | ---: | ---: |
| 1 | n/a | 5.621 |
| 20 | 2.762 | 2.800 |
| 40 | 2.545 | 2.629 |
| 60 | 2.540 | 2.562 |
| 80 | 2.494 | 2.501 |
| 100 | 2.375 | 2.479 |
| 120 | 2.310 | 2.469 |
| 140 | 2.469 | 2.448 |
| 160 | 2.405 | 2.459 |
| 180 | 2.273 | 2.468 |
| 200 | 2.004 | 2.470 |
| 220 | 2.157 | 2.446 |
| 240 | 2.199 | 2.456 |
| 260 | 2.083 | 2.449 |
| 280 | 2.265 | 2.442 |
| 300 | 2.197 | 2.439 |
| 320 | 2.125 | 2.450 |
| 340 | 2.213 | 2.425 |
| 360 | 2.111 | 2.424 |

Final-checkpoint test loss:

- test loss: `2.382`
- test perplexity: `10.831`

Best validation checkpoint:

- `0000360_adapters.safetensors`

## Quick Generation Checks

Fresh 20-paper holdout sampled from the recent-balanced corpus, excluding the
800 teacher-training rows:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_800_lora_highlr_v1/review_samples/base_vs_ckpt360_fresh20`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_800_lora_highlr_v1/review_samples/ckpt300_vs_ckpt360_fresh20`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_800_lora_highlr_v1/review_samples/ckpt340_vs_ckpt360_fresh20`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_800_lora_highlr_v1/review_samples/ckpt220_500_vs_ckpt360_800_fresh20`

Length and shape:

| variant | avg words | median words | max words | over 14 | forbidden prefix | formula-like |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 25.45 | 26 | 36 | 19 / 20 | 0 | 2 |
| 500-row ckpt220 | 13.05 | 13 | 21 | 7 / 20 | 0 | 1 |
| 800-row ckpt300 | 13.00 | 14 | 21 | 7 / 20 | 0 | 0 |
| 800-row ckpt340 | 12.80 | 13 | 24 | 7 / 20 | 0 | 1 |
| 800-row ckpt360 | 12.70 | 11 | 23 | 6 / 20 | 0 | 1 |

## Read

The high-LR 800-row LoRA is a clear loss-curve improvement over the 500-row
run: best validation moved from `2.632` to `2.424`, and final test loss moved
from `2.432` to `2.382`.

The quick generation sample is more mixed. Checkpoint `0000360` is shorter than
the 500-row checkpoint on the same 20-paper holdout and has one fewer over-limit
output, but it still produces technical/math failures, including one formula-like
nucl-th sentence and one nonsensical Hopf-coquasigroup sentence. It is progress,
but not a promotion-quality read without broader manual review.
