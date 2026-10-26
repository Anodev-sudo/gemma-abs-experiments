# MLX LoRA 270M natural-language 300-row LoRA v1

## Goal

Run a fresh LoRA adapter on the 300-row strict-audited natural-language teacher
milestone.

This is still a proof-of-concept run. The goal is to test whether more
strict-good rows improve the small adapter's ability to produce short,
plain-language paper summaries.

## Data

Source release:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_300_milestone/training_rows.jsonl`

MLX dataset:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_300_milestone/mlx_lora/abstract_only_chat_gemma_270m_it_abstract_to_natural_language_summary_v1`

Split:

- train: `270`
- valid: `15`
- test: `15`

Prompt spec:

- `prompts/mlx_lora/gemma_270m_it_abstract_to_natural_language_summary_v1.json`

## Training

Config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_300_lora_v1.yaml`

Command:

```bash
HF_HOME=hf_bucket/experiments/mlx_lora/.hf-home \
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_natural_language_300_lora_v1.yaml \
  --mask-prompt
```

Adapter output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_300_lora_v1/adapters`

Training details:

- base model: `google/gemma-3-270m-it`
- fine-tune type: `lora`
- trainable params: `1.475M / 268.098M`
- LoRA target modules: attention q/k/v/o projections
- rank: `16`
- scale: `10.0`
- batch size: `4`
- iterations: `300`
- learning rate: `5e-6`
- peak memory: about `7.24 GB`

Training curve:

| step | train loss | valid loss |
| ---: | ---: | ---: |
| 1 | n/a | 6.105 |
| 20 | 2.988 | 3.426 |
| 40 | 2.602 | 3.142 |
| 60 | 2.394 | 3.056 |
| 80 | 2.212 | 3.053 |
| 100 | 2.105 | 3.037 |
| 120 | 2.281 | 3.014 |
| 140 | 2.215 | 3.014 |
| 160 | 2.031 | 3.015 |
| 180 | 2.259 | 3.022 |
| 200 | 2.101 | 3.024 |
| 220 | 1.865 | 3.004 |
| 240 | 1.828 | 3.013 |
| 260 | 1.782 | 2.991 |
| 280 | 1.739 | 3.024 |
| 300 | 1.848 | 3.034 |

Final-checkpoint test loss:

- test loss: `2.318`
- test perplexity: `10.158`

Best validation checkpoint:

- `0000260_adapters.safetensors`

## Quick Generation Checks

Held-out test split, base vs checkpoint `0000260`:

- artifact: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_300_lora_v1/review_samples/base_vs_ckpt260_test15`
- base average words: `28.27`
- checkpoint 260 average words: `14.27`
- base over 14 words: `15/15`
- checkpoint 260 over 14 words: `7/15`

Held-out test split, checkpoint `0000120` vs checkpoint `0000260`:

- artifact: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_300_lora_v1/review_samples/ckpt120_vs_ckpt260_test15`
- checkpoint 120 average words: `12.47`
- checkpoint 260 average words: `14.27`
- checkpoint 120 over 14 words: `3/15`
- checkpoint 260 over 14 words: `7/15`

Held-out test split, checkpoint `0000060` vs checkpoint `0000120`:

- artifact: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_300_lora_v1/review_samples/ckpt60_vs_ckpt120_test15`
- checkpoint 60 average words: `10.47`
- checkpoint 120 average words: `12.47`
- checkpoint 60 over 14 words: `0/15`
- checkpoint 120 over 14 words: `3/15`

## Read

The adapter learned strong length and style pressure. Compared with the base
model, every checked checkpoint is much shorter and much closer to the feed
summary target.

The best-loss checkpoint is not obviously the best product checkpoint. Checkpoint
`0000260` has the lowest validation loss, but the held-out generations get
wordier and show more drift. Checkpoint `0000060` obeys the word budget best but
can be vague or awkward. Checkpoint `0000120` may be the most interesting middle
ground: shorter than 260, more specific than 60, but still not reliably under 14
words.

The quality is improved over the 100-row run, but this still does not look
promotion-ready without a broader generation audit.

## Fresh Holdout Checkpoint Audit

Fresh sample:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_300_lora_v1/review_samples/fresh_holdout40_source_20260522.jsonl`
- 40 papers sampled from the recent-balanced 20k corpus
- excludes the 300 teacher-training rows
- umbrella mix: `cs=12`, `physics=11`, `math=10`, `q-fin=2`, `eess=2`,
  `econ=1`, `q-bio=1`, `stat=1`

Artifacts:

- `base_vs_ckpt120_fresh40`
- `ckpt60_vs_ckpt120_fresh40`
- `ckpt120_vs_ckpt260_fresh40`

Length and shape:

| variant | avg words | median words | max words | over 14 | forbidden prefix |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 26.00 | 26 | 43 | 36 / 40 | 30 |
| ckpt60 | 12.43 | 12 | 21 | 14 / 40 | 0 |
| ckpt120 | 13.30 | 13 | 21 | 15 / 40 | 0 |
| ckpt260 | 14.55 | 15 | 28 | 20 / 40 | 0 |

Manual read:

- `ckpt60` has the strongest length pressure, but it is more likely to sound
  clipped, vague, or mechanically copied from the abstract.
- `ckpt120` looks like the best middle checkpoint. It is usually more readable
  and specific than `ckpt60`, while avoiding the worst verbosity and drift of
  `ckpt260`.
- `ckpt260` has the best validation loss, but the fresh generations are
  wordier, denser, and more prone to speculative or awkward claims.

Recurring failures:

- The 14-word limit is still not reliably learned on fresh hard papers.
- Math and theoretical physics rows often remain jargon-heavy.
- Some outputs keep equations, notation, or raw metrics despite the prompt.
- A few rows are clearly malformed or semantically suspicious, for example:
  - `by a factor of 1`
  - `classically-quantum`
  - escaped TeX fragments in `ckpt60`

Checkpoint read after fresh audit:

- Best product candidate to audit next: `0000120`
- Best length-control checkpoint: `0000060`
- Best validation-loss checkpoint: `0000260`

The fresh audit strengthens the main conclusion: the 300-row LoRA is learning the
new style, but it is not yet reliable enough. It needs either more rows, stronger
length-pressure examples, or a post-generation enforcement layer before it can be
treated as a serious promotion candidate.
