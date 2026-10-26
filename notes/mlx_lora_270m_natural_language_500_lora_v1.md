# MLX LoRA 270M natural-language 500-row LoRA v1

## Goal

Run a fresh LoRA adapter on the 500-row strict-audited natural-language teacher
milestone after the math readability prune.

This is still a proof-of-concept run. The goal is to see whether the larger,
cleaner teacher set improves style control over the 300-row LoRA while keeping
the student input as abstract-only.

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

- `configs/mlx_lora/gemma_3_270m_it_natural_language_500_lora_v1.yaml`

Command:

```bash
HF_HOME=hf_bucket/experiments/mlx_lora/.hf-home \
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_natural_language_500_lora_v1.yaml \
  --mask-prompt
```

Adapter output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_500_lora_v1/adapters`

Planned settings:

- base model: `google/gemma-3-270m-it`
- fine-tune type: `lora`
- LoRA target modules: attention q/k/v/o projections
- rank: `16`
- scale: `10.0`
- batch size: `4`
- iterations: `320`
- learning rate: `5e-6`

## Progress

Training completed.

Training details:

- trainable params: `1.475M / 268.098M`
- peak memory: about `7.24 GB`

Training curve:

| step | train loss | valid loss |
| ---: | ---: | ---: |
| 1 | n/a | 5.742 |
| 20 | 3.339 | 3.167 |
| 40 | 2.634 | 2.845 |
| 60 | 2.532 | 2.754 |
| 80 | 2.484 | 2.737 |
| 100 | 2.267 | 2.713 |
| 120 | 2.301 | 2.678 |
| 140 | 2.316 | 2.694 |
| 160 | 2.261 | 2.657 |
| 180 | 2.211 | 2.661 |
| 200 | 2.285 | 2.652 |
| 220 | 2.346 | 2.632 |
| 240 | 2.115 | 2.656 |
| 260 | 2.131 | 2.658 |
| 280 | 2.149 | 2.658 |
| 300 | 2.205 | 2.652 |
| 320 | 2.038 | 2.654 |

Final-checkpoint test loss:

- test loss: `2.432`
- test perplexity: `11.383`

Best validation checkpoint:

- `0000220_adapters.safetensors`

## Quick Generation Checks

Fresh 20-paper holdout sampled from the recent-balanced corpus, excluding the
500 teacher-training rows:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_500_lora_v1/review_samples/base_vs_ckpt220_fresh20`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_500_lora_v1/review_samples/ckpt120_vs_ckpt220_fresh20`

Length and shape:

| variant | avg words | median words | max words | over 14 | forbidden prefix |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 21.80 | 21 | 36 | 18 / 20 | 18 |
| ckpt120 | 11.05 | 10 | 17 | 3 / 20 | 1 |
| ckpt220 | 12.55 | 12 | 22 | 4 / 20 | 0 |

## Read

The 500-row LoRA is a real step forward on the training curve. Validation loss
reaches `2.632`, materially better than the 300-row LoRA's best recorded
validation loss of `2.991`.

The fresh generation check also shows strong style transfer versus the base:
shorter outputs, fewer empty app-context echoes, and no forbidden openings for
checkpoint `0000220`.

Checkpoint choice is still a product question. `0000220` has the best validation
loss and reads more specific in several examples, but `0000120` is shorter on
the fresh sample and has fewer over-limit outputs. Both should be manually
audited before treating either as a promotion candidate.

## Manual Review: Fresh 20

Manual review artifact:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_500_lora_v1/review_samples/ckpt120_vs_ckpt220_fresh20/manual_review_codex.md`

Manual review counts:

| bucket | rows |
| --- | ---: |
| clearly usable | 8 |
| borderline usable | 4 |
| reject | 8 |

Winner read:

| winner | rows |
| --- | ---: |
| ckpt220 | 8 |
| ckpt120 | 3 |
| tie | 2 |
| neither | 7 |

Read:

- `0000220` is usually more specific and avoids forbidden openings.
- `0000120` is shorter and sometimes safer, but more likely to be vague or use
  a banned opening.
- The remaining failure mode is mostly semantic compression, not formatting:
  unsupported comparisons, wrong math/physics object, vague title-level
  paraphrase, and over-broad result claims.

This sample does not look promotion-ready, but it does show real progress.
