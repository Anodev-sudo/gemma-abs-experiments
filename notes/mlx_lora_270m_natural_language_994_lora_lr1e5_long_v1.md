# MLX LoRA 270M natural-language 994-row long LoRA v1

## Goal

Run a fresh 270M LoRA adapter on the clean strict-only 1000-target natural
language pool after removing manual-tail contamination.

This is a clean-base run, not a patch on the promoted q5/q8-sidecar lineage.
The question is whether the larger gold set improves product behavior before
mixing in lower-confidence data.

## Data

Source release:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/training_rows.jsonl`

Source rows:

- rows: `994`
- unique paper ids: `994`
- summary word range: `8-14`
- target gap: missing `physics=2`, `math=4` from the exact 1000 target
- no manual-tail rows included

MLX dataset:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/mlx_lora/abstract_only_chat_gemma_270m_it_abstract_to_natural_language_summary_v1`

Split:

- train: `894`
- valid: `50`
- test: `50`

Prompt spec:

- `prompts/mlx_lora/gemma_270m_it_abstract_to_natural_language_summary_v1.json`

## Training

Config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_994_lora_lr1e5_long_v1.yaml`

Command:

```bash
HF_HOME=hf_bucket/experiments/mlx_lora/.hf-home \
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_natural_language_994_lora_lr1e5_long_v1.yaml \
  --mask-prompt
```

Adapter output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_lora_lr1e5_long_v1/adapters`

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
- peak memory: about `8.91 GB`
- final test loss: `2.401`
- final test perplexity: `11.031`

Validation curve:

| step | valid loss |
| ---: | ---: |
| 1 | 5.543 |
| 25 | 2.652 |
| 50 | 2.533 |
| 75 | 2.487 |
| 100 | 2.432 |
| 125 | 2.443 |
| 150 | 2.399 |
| 175 | 2.387 |
| 200 | 2.359 |
| 225 | 2.337 |
| 250 | 2.337 |
| 275 | 2.328 |
| 300 | 2.327 |
| 325 | 2.325 |
| 350 | 2.337 |
| 375 | 2.343 |
| 400 | 2.328 |
| 425 | 2.303 |
| 450 | 2.305 |
| 475 | 2.309 |
| 500 | 2.301 |

Best validation checkpoint:

- `0000500_adapters.safetensors`

Product checkpoints worth reading:

- `0000275_adapters.safetensors`
- `0000325_adapters.safetensors`
- `0000500_adapters.safetensors`

## Fresh Holdout Checks

Fresh 40-paper holdout sampled from the recent-balanced corpus, excluding the
994 teacher-training rows:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_lora_lr1e5_long_v1/review_samples/ckpt360_800_vs_ckpt500_994_fresh40`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_lora_lr1e5_long_v1/review_samples/ckpt275_vs_ckpt325_fresh40`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_lora_lr1e5_long_v1/review_samples/ckpt325_vs_ckpt425_fresh40`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_lora_lr1e5_long_v1/review_samples/ckpt425_vs_ckpt500_fresh40`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_lora_lr1e5_long_v1/review_samples/four_way_800_325_425_500_fresh40_review.md`

Length and shape:

| variant | avg words | median | max | over 14 | over 15 | under 8 | digit rows | formula-like | title overlap >=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 800-row ckpt360 | 12.80 | 12 | 18 | 10 / 40 | 6 / 40 | 0 / 40 | 3 / 40 | 1 / 40 | 19 / 40 |
| 994-row ckpt275 | 12.60 | 12 | 19 | 9 / 40 | 8 / 40 | 0 / 40 | 3 / 40 | 2 / 40 | 17 / 40 |
| 994-row ckpt325 | 12.20 | 12 | 21 | 6 / 40 | 6 / 40 | 1 / 40 | 4 / 40 | 3 / 40 | 16 / 40 |
| 994-row ckpt425 | 13.38 | 13 | 21 | 12 / 40 | 9 / 40 | 0 / 40 | 6 / 40 | 3 / 40 | 17 / 40 |
| 994-row ckpt500 | 12.50 | 13 | 16 | 9 / 40 | 8 / 40 | 1 / 40 | 4 / 40 | 3 / 40 | 17 / 40 |

## Read

The larger clean set clearly improves the validation curve versus the 800-row
run, but the generation read is mixed rather than promotion-ready.

Checkpoint `325` is the best mechanical checkpoint on the fresh holdout: it has
the fewest over-14 rows and lowest title-overlap count. It also produces more
aggressive simplifications, including one under-8-word row and some math/physics
compression that needs manual review.

Checkpoint `275` is a useful product contender despite weaker mechanical stats.
It often preserves more specific meaning than `325`, but it is wordier and has
more over-15 rows.

Checkpoint `500` has the best validation loss but is not clearly the best
product checkpoint. It remains fairly short, but still leaks formula-like math
and produces several over-15 summaries.

Current read: this run is progress, but not a clean promotion candidate from
the mechanical and spot-read checks alone. The best next audit target is a manual
review of `275` vs `325`, with `500` kept as the loss-curve reference.

## Llama Real-Read A/B Against Promoted Q5

Date: `2026-05-26`

Eval root:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_lora_lr1e5_long_v1/eval/llama_ab_325_vs_promoted_q5_fresh40_realread`

Setup:

- `ckpt325_llama`: 994-row fresh LoRA checkpoint `325`, llama runtime, abstract-only prompt.
- `promoted_q5_llama`: promoted q5 contender with old q8 sidecar, llama runtime, title+abstract prompt.
- same 40-paper fresh holdout as `ckpt360_800_vs_ckpt500_994_fresh40`
- blind real-read review split across four `gpt-5.4-mini` subagents, 10 rows each

Blind review result:

| candidate | preferred | more accurate |
| --- | ---: | ---: |
| `promoted_q5_llama` | 23 / 40 | 23 / 40 |
| `ckpt325_llama` | 14 / 40 | 14 / 40 |
| neither | 3 / 40 | 3 / 40 |

Mechanical llama output checks:

| candidate | avg words | max | over 14 | over 15 | digit rows | title overlap >=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `994_ckpt325_llama` | 13.60 | 19 | 14 / 40 | 14 / 40 | 4 / 40 | 18 / 40 |
| `promoted_q5_llama` | 12.72 | 18 | 9 / 40 | 5 / 40 | 10 / 40 | 30 / 40 |

Read: checkpoint `325` is not beating the promoted q5 contender in llama. It
does reduce digit use and title overlap, but it over-runs the word target more
often and loses the blind feed-preference and faithfulness read by the same
margin.
