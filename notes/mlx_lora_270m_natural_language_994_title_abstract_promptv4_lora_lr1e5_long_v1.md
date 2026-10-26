# MLX LoRA 270M natural-language 994-row title+abstract prompt-v4 v1

## Goal

Test the prompt-v3 product wording with the stricter prompt-v2 length contract.
This keeps the candidate-inspired framing:

- title is already visible
- broad reader
- main takeaway over implementation detail
- exact numbers only when central and easy to bind

But changes the length instruction back to:

- one readable sentence
- 14 words or less
- never exceed 14 words

## Data

Source release:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/training_rows.jsonl`

MLX dataset:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_natural_language_summary_v4`

Prompt spec:

- `prompts/mlx_lora/gemma_270m_it_abstract_to_natural_language_summary_v4.json`

Split:

- train: `894`
- valid: `50`
- test: `50`

## Training

Config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_promptv4_lora_lr1e5_long_v1.yaml`

Command:

```bash
HF_HOME=hf_bucket/experiments/mlx_lora/.hf-home \
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_promptv4_lora_lr1e5_long_v1.yaml \
  --mask-prompt
```

Adapter output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_promptv4_lora_lr1e5_long_v1/adapters`

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
- final test loss: `2.322`
- final test perplexity: `10.192`

Validation curve:

| step | valid loss |
| ---: | ---: |
| 1 | 5.407 |
| 25 | 2.613 |
| 50 | 2.441 |
| 75 | 2.404 |
| 100 | 2.350 |
| 125 | 2.323 |
| 150 | 2.303 |
| 175 | 2.293 |
| 200 | 2.299 |
| 225 | 2.291 |
| 250 | 2.307 |
| 275 | 2.285 |
| 300 | 2.285 |
| 325 | 2.297 |
| 350 | 2.279 |
| 375 | 2.263 |
| 400 | 2.269 |
| 425 | 2.246 |
| 450 | 2.232 |
| 475 | 2.258 |
| 500 | 2.238 |

Best validation checkpoint:

- `0000450_adapters.safetensors`

## Fresh Holdout Checks

Fresh 40-paper holdout reused from the abstract-only 994-row review sample:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_lora_lr1e5_long_v1/review_samples/ckpt360_800_vs_ckpt500_994_fresh40/generations.jsonl`

Generated review samples:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_promptv4_lora_lr1e5_long_v1/review_samples/ckpt425_vs_ckpt450_fresh40_title_abstract_v4`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_promptv4_lora_lr1e5_long_v1/review_samples/ckpt450_vs_ckpt500_fresh40_title_abstract_v4`
- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_title_abstract_promptv4_lora_lr1e5_long_v1/review_samples/title_abstract_v4_fresh40_mechanical`

Length and shape, generated with `max_tokens=36`:

| variant | avg words | median | max | <=14 | over 14 | over 15 | under 10 | digit rows | formula-like | title overlap >=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ta994_v2_ckpt425` | 12.20 | 12 | 16 | 32 / 40 | 8 / 40 | 4 / 40 | 5 / 40 | 4 / 40 | 1 / 40 | 21 / 40 |
| `ta994_v3_ckpt425` | 12.68 | 12 | 23 | 29 / 40 | 11 / 40 | 6 / 40 | 3 / 40 | 5 / 40 | 1 / 40 | 21 / 40 |
| `ta994_v3_ckpt450` | 13.45 | 13 | 23 | 27 / 40 | 13 / 40 | 10 / 40 | 4 / 40 | 3 / 40 | 1 / 40 | 26 / 40 |
| `ta994_v3_ckpt500` | 12.85 | 13 | 20 | 30 / 40 | 10 / 40 | 7 / 40 | 3 / 40 | 3 / 40 | 1 / 40 | 26 / 40 |
| `ta994_v4_ckpt425` | 12.25 | 12 | 18 | 33 / 40 | 7 / 40 | 6 / 40 | 6 / 40 | 4 / 40 | 1 / 40 | 23 / 40 |
| `ta994_v4_ckpt450` | 13.25 | 13 | 20 | 29 / 40 | 11 / 40 | 9 / 40 | 5 / 40 | 3 / 40 | 1 / 40 | 27 / 40 |
| `ta994_v4_ckpt500` | 12.75 | 13 | 19 | 29 / 40 | 11 / 40 | 4 / 40 | 4 / 40 | 3 / 40 | 1 / 40 | 21 / 40 |

## Read

Prompt v4 partially fixes the prompt-v3 length failure mode. The `425`
checkpoint has the best under-14 compliance in this small fresh sample:
`33/40` rows at or below 14 words, compared with `32/40` for prompt-v2 `425`.

It is not a clean replacement for prompt v2. The same v4 `425` checkpoint has
more severe long rows than v2 `425`: `6/40` over 15 words versus `4/40`, with a
max length of `18` instead of `16`. It also has slightly higher title overlap on
this mechanical pass.

The useful signal is that restoring `14 words or less` plus `Never exceed 14
words` was better than the prompt-v3 `10 to 13 words` instruction. The
candidate-inspired wording may still help phrasing, but it needs a real-read
comparison before promoting. For now, the most relevant next comparison is v2
`425` against v4 `425`, ideally in the same llama runtime used for candidate
checks.
