# MLX LoRA 270M natural-language 100-row POC v1

## Goal

Test whether a fresh small LoRA adapter can shift `google/gemma-3-270m-it`
toward natural-language, 14-word-or-less paper summaries using the new
strict-audited 100-row teacher set.

This is a style-learnability proof of concept, not a promotion candidate.

## Data

Source release:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v7_100_good/training_rows.jsonl`

MLX dataset:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v7_100_good/mlx_lora/abstract_only_chat_gemma_270m_it_abstract_to_natural_language_summary_v1`

Split:

- train: `80`
- valid: `10`
- test: `10`

Prompt spec:

- `prompts/mlx_lora/gemma_270m_it_abstract_to_natural_language_summary_v1.json`

## Training

Config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_100_poc_lora_v1.yaml`

Command:

```bash
HF_HOME=hf_bucket/experiments/mlx_lora/.hf-home \
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_natural_language_100_poc_lora_v1.yaml \
  --mask-prompt
```

Adapter output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_100_poc_lora_v1/adapters`

Training curve:

| step | train loss | valid loss |
| ---: | ---: | ---: |
| 1 | n/a | 5.444 |
| 20 | 2.893 | 2.978 |
| 40 | 2.241 | 2.605 |
| 60 | 1.926 | 2.562 |
| 80 | 1.734 | 2.580 |
| 100 | 1.576 | 2.650 |
| 120 | 1.442 | 2.741 |

Final test loss:

- `2.859`
- test perplexity: `17.445`

The validation curve suggests checkpoint `0000060` is the best early checkpoint;
later checkpoints show overfit pressure.

## Quick Testing

Fresh 12-row sample, base vs checkpoint `0000060`:

- artifact: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_100_poc_lora_v1/review_samples/base_vs_ckpt60_fresh12`
- base average words: `28.67`
- checkpoint 60 average words: `14.33`
- base over 14 words: `12/12`
- checkpoint 60 over 14 words: `5/12`

Held-out 10-row teacher test, base vs checkpoint `0000060`:

- artifact: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_100_poc_lora_v1/review_samples/base_vs_ckpt60_test10`
- base average words: `27.0`
- checkpoint 60 average words: `15.0`
- base over 14 words: `9/10`
- checkpoint 60 over 14 words: `5/10`

Training-row 10-row diagnostic, base vs checkpoint `0000060`:

- artifact: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_100_poc_lora_v1/review_samples/base_vs_ckpt60_train10`
- base average words: `27.7`
- checkpoint 60 average words: `11.2`
- base over 14 words: `9/10`
- checkpoint 60 over 14 words: `2/10`

## Read

The adapter clearly learns the short-output shape. It cuts length sharply on
fresh, held-out, and even training-row diagnostics.

It does not yet learn faithful natural-language summarization. Even checkpoint
`0000060` introduces or overstates details, keeps technical notation in some
cases, and sometimes misses the teacher target even on seen training rows.

The current result is a useful proof of style pressure, but not enough evidence
that `100` rows can produce a reliable deployable adapter.

## Next Step

For the next POC, use more rows before changing model size or training method:

- keep the same prompt and abstract-only input
- scale to `300-500` strict-audited rows
- keep early checkpoint testing
- compare against checkpoint `0000060` as the 100-row baseline

If faithfulness remains poor after `300-500` rows, then the next question should
be full fine-tune vs larger model, not more adapter tweaking.
