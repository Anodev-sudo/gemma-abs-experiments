# MLX LoRA 1B natural-language 500-row LoRA v1

## Goal

Run an orthogonal capacity check: same 500-row natural-language teacher dataset,
same abstract-only student prompt, but with `google/gemma-3-1b-it` instead of
the 270M model.

The goal is not promotion. It is to see whether a larger student learns the
natural-language summary behavior with fewer semantic compression failures.

## Data

Source release:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_500_milestone/training_rows.jsonl`

MLX dataset:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_500_milestone/mlx_lora/abstract_only_chat_gemma_270m_it_abstract_to_natural_language_summary_v1`

Split:

- train: `450`
- valid: `25`
- test: `25`

Prompt spec:

- `prompts/mlx_lora/gemma_270m_it_abstract_to_natural_language_summary_v1.json`

## Training

Config:

- `configs/mlx_lora/gemma_3_1b_it_natural_language_500_lora_v1.yaml`

Command:

```bash
HF_HOME=hf_bucket/experiments/mlx_lora/.hf-home \
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_1b_it_natural_language_500_lora_v1.yaml \
  --mask-prompt
```

Adapter output:

- `hf_bucket/experiments/mlx_lora/gemma_3_1b_it_natural_language_500_lora_v1/adapters`

Planned settings:

- base model: `google/gemma-3-1b-it`
- fine-tune type: `lora`
- LoRA target modules: attention q/k/v/o projections
- rank: `16`
- scale: `10.0`
- batch size: `4`
- iterations: `320`
- learning rate: `5e-6`
- gradient checkpointing: `true`

## Progress

Training completed.

Training details:

- base model downloaded into repo-local HF cache
- trainable params: `2.982M / 999.886M`
- peak memory: about `7.09 GB`
- 1B model cache size after pull: about `1.9G`
- adapter checkpoint directory size: about `194M`

Training curve:

| step | train loss | valid loss |
| ---: | ---: | ---: |
| 1 | n/a | 8.964 |
| 20 | 2.988 | 2.744 |
| 40 | 2.272 | 2.333 |
| 60 | 1.986 | 2.169 |
| 80 | 2.024 | 2.141 |
| 100 | 1.782 | 2.133 |
| 120 | 1.871 | 2.095 |
| 140 | 1.747 | 2.104 |
| 160 | 1.746 | 2.086 |
| 180 | 1.698 | 2.101 |
| 200 | 1.748 | 2.085 |
| 220 | 1.764 | 2.064 |
| 240 | 1.567 | 2.075 |
| 260 | 1.620 | 2.068 |
| 280 | 1.680 | 2.098 |
| 300 | 1.633 | 2.066 |
| 320 | 1.513 | 2.065 |

Final-checkpoint test loss:

- test loss: `1.885`
- test perplexity: `6.583`

Best validation checkpoint:

- `0000220_adapters.safetensors`

## Quick Generation Checks

Fresh 20-paper holdout reused from the 270M natural-language 500-row review:

- `hf_bucket/experiments/mlx_lora/gemma_3_1b_it_natural_language_500_lora_v1/review_samples/ckpt160_vs_ckpt220_fresh20`
- `hf_bucket/experiments/mlx_lora/gemma_3_1b_it_natural_language_500_lora_v1/review_samples/ckpt220_vs_ckpt320_fresh20`

Length and shape:

| variant | avg words | median words | max words | over 14 | forbidden prefix |
| --- | ---: | ---: | ---: | ---: | ---: |
| ckpt160 | 12.70 | 12 | 17 | 3 / 20 | 0 |
| ckpt220 | 13.95 | 13 | 24 | 6 / 20 | 0 |
| ckpt320 | 15.25 | 14 | 35 | 7 / 20 | 0 |

## Manual Review: Fresh 20

Manual review artifact:

- `hf_bucket/experiments/mlx_lora/gemma_3_1b_it_natural_language_500_lora_v1/review_samples/ckpt160_vs_ckpt220_fresh20/manual_review_codex.md`

Manual review counts:

| bucket | rows |
| --- | ---: |
| clearly usable | 9 |
| borderline usable | 5 |
| reject | 6 |

Winner read:

| winner | rows |
| --- | ---: |
| ckpt220 | 8 |
| ckpt160 | 6 |
| tie | 3 |
| neither | 3 |

## Read

The 1B LoRA is clearly stronger than the 270M LoRA on loss:

| model | best valid loss | final test loss |
| --- | ---: | ---: |
| 270M LoRA 500 | 2.632 | 2.432 |
| 1B LoRA 500 | 2.064 | 1.885 |

It also looks stronger in manual review. On the same fresh 20-paper sample, the
270M review had `8` clearly usable, `4` borderline, and `8` rejects. The 1B
review has `9` clearly usable, `5` borderline, and `6` rejects.

This is real progress, but still not promotion-ready. The remaining failures are
semantic compression errors, not basic formatting errors:

- plausible but unsupported comparison
- wrong direction or timing of an effect
- over-specific named entity
- theorem/math compression into the wrong object

Checkpoint `0000220` is the best first candidate to audit more broadly.
Checkpoint `0000160` should stay in the comparison set because it is shorter and
occasionally safer.
