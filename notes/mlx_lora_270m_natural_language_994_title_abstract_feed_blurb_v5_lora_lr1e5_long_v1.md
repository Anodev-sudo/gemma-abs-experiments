# MLX LoRA 270M natural-language 994-row title+abstract feed-blurb v5 v1

## Goal

Test whether changing the product target language from "summary" to "feed
blurb" changes the student behavior. This run keeps the prompt-v4 contract and
training setup, but removes summary-language from the prompt spec and prompt
version.

Prompt-v5 target wording:

- `paper-feed blurbs`
- `feed blurb`

The rest of the prompt stays conservative:

- title is already visible
- one readable sentence
- 14 words or less
- never exceed 14 words
- broad reader
- main takeaway over implementation detail
- exact numbers only when central and easy to bind

## Data

Source release:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/training_rows.jsonl`

MLX dataset:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5`

Prompt spec:

- `prompts/mlx_lora/gemma_270m_it_abstract_to_feed_blurb_v5.json`

Split:

- train: `894`
- valid: `50`
- test: `50`

## Training

Config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_feed_blurb_v5_lora_lr1e5_long_v1.yaml`

Command:

```bash
HF_HOME=hf_bucket/experiments/mlx_lora/.hf-home \
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_feed_blurb_v5_lora_lr1e5_long_v1.yaml \
  --mask-prompt
```

Adapter output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_feed_blurb_v5_lora_lr1e5_long_v1/adapters`

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
- final test loss: `2.324`
- final test perplexity: `10.217`

Validation curve:

| step | valid loss |
| ---: | ---: |
| 1 | 5.315 |
| 25 | 2.605 |
| 50 | 2.440 |
| 75 | 2.398 |
| 100 | 2.337 |
| 125 | 2.316 |
| 150 | 2.296 |
| 175 | 2.293 |
| 200 | 2.294 |
| 225 | 2.284 |
| 250 | 2.302 |
| 275 | 2.281 |
| 300 | 2.278 |
| 325 | 2.291 |
| 350 | 2.268 |
| 375 | 2.259 |
| 400 | 2.263 |
| 425 | 2.239 |
| 450 | 2.231 |
| 475 | 2.254 |
| 500 | 2.238 |

Best validation checkpoint:

- `0000450_adapters.safetensors`

## Fresh Holdout Checks

Fresh 40-paper holdout reused from the abstract-only 994-row review sample:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_lora_lr1e5_long_v1/review_samples/ckpt360_800_vs_ckpt500_994_fresh40/generations.jsonl`

Generated review samples:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_feed_blurb_v5_lora_lr1e5_long_v1/review_samples/ckpt425_vs_ckpt450_fresh40_title_abstract_feed_blurb_v5`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_feed_blurb_v5_lora_lr1e5_long_v1/review_samples/ckpt450_vs_ckpt500_fresh40_title_abstract_feed_blurb_v5`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_feed_blurb_v5_lora_lr1e5_long_v1/review_samples/title_abstract_feed_blurb_v5_fresh40_mechanical`

Length and shape, generated with `max_tokens=36`:

| variant | avg words | median | max | <=14 | over 14 | over 15 | under 10 | digit rows | formula-like | title overlap >=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ta994_v2_ckpt425` | 12.20 | 12 | 16 | 32 / 40 | 8 / 40 | 4 / 40 | 5 / 40 | 4 / 40 | 1 / 40 | 21 / 40 |
| `ta994_v4_ckpt425` | 12.25 | 12 | 18 | 33 / 40 | 7 / 40 | 6 / 40 | 6 / 40 | 4 / 40 | 1 / 40 | 23 / 40 |
| `ta994_v5_ckpt425` | 12.18 | 12 | 18 | 33 / 40 | 7 / 40 | 5 / 40 | 5 / 40 | 3 / 40 | 1 / 40 | 15 / 40 |
| `ta994_v5_ckpt450` | 13.00 | 13 | 23 | 27 / 40 | 13 / 40 | 8 / 40 | 6 / 40 | 4 / 40 | 1 / 40 | 20 / 40 |
| `ta994_v5_ckpt500` | 12.55 | 13 | 18 | 31 / 40 | 9 / 40 | 4 / 40 | 5 / 40 | 1 / 40 | 1 / 40 | 20 / 40 |

## Read

Prompt v5 is a small but real-looking prompt experiment, not a breakthrough.
The best validation checkpoint, `450`, has worse fresh-sample length behavior
than `425`, so validation loss is not enough to pick the product checkpoint.

The interesting checkpoint is `425`. Compared with prompt-v4 `425`, v5 `425`
keeps the same `33/40` rows at or below 14 words, slightly reduces the severe
long tail, and appears less title-attracted on this sample. Some outputs read
more like feed blurbs:

- `Long-read sequencing helps build genome assemblies that are more contiguous than traditional methods.`
- `Data portals and web tools help researchers track SARS-CoV-2 variants in Canada.`
- `Optimizing Alibaba's microservice cluster resource allocation helps latency-critical services run faster.`
- `Atomic-resolution scanning reveals a new way to align perovskite films.`

The same persistent failures remain:

- malformed math or technical abstractions
- title-adjacent statements of existence
- occasional over-14 rows
- a few outputs that sound fluent but weakly grounded

Current read: v5 `425` is worth comparing against v2 `425` and v4 `425` in a
real-read or llama-runtime check. Do not prefer v5 `450` just because it has the
lowest validation loss.
