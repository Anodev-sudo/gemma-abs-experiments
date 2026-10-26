# MLX LoRA 270M 994-gold plus heuristic-3000 replay v1

## Goal

Run one fresh adapter with strict gold as the primary signal and the
deterministic old-data subset as low-frequency replay. This tests whether broad
coverage helps when it is mixed throughout training instead of learned first
and then overwritten by a gold finisher.

This is not a promotion run.

## Dataset

Dataset builder:

- `scripts/build_mlx_fullmix_gold_support_v1_dataset.py`

Build command:

```bash
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  scripts/build_mlx_fullmix_gold_support_v1_dataset.py \
  --gold-repeat 6 \
  --support-path hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_10746_heuristic3000_goldfinish_v1/dataset_heuristic3000_title_abstract_feed_blurb_v5/train.jsonl \
  --output-dir hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994gold_heuristic3000_replay_v1/dataset_gold6_heuristic3000_feed_blurb_v5
```

Output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994gold_heuristic3000_replay_v1/dataset_gold6_heuristic3000_feed_blurb_v5`

Counts:

- train examples: `8364`
- unique train paper IDs: `3894`
- strict gold train rows: `894`, repeated `6x` for `5364` exposures
- heuristic replay rows: `3000`, once each
- valid examples: `50` strict-gold holdout rows
- test examples: `50` strict-gold holdout rows

Because MLX LoRA does not use row-level `loss_weight_hints`, the low replay
weight is approximated by frequency. Each replay row has about one-sixth the
per-row exposure of a gold row.

## Training Plan

Config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_994gold_heuristic3000_replay_v1_lr8e6.yaml`

Output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994gold_heuristic3000_replay_v1/lora_lr8e6`

Training details:

- base model: `google/gemma-3-270m-it`
- fresh LoRA, attention projections only
- batch size: `4`
- iterations: `1400`
- learning rate: `8e-6`
- save every `100`

## Evaluation Plan

Run fresh-40 against the current title+abstract v5 LoRA checkpoint `425`.
Inspect checkpoints around `600`, `800`, `1000`, `1200`, and `1400`, with
extra attention to title overlap, unsupported numbers, and whether the replay
rows bring back old summary wording.

## Result

Training completed through `1400` iterations.

Strict-gold validation loss improved steadily instead of peaking early:

- iter `100`: `2.408`
- iter `500`: `2.269`
- iter `1000`: `2.222`
- iter `1200`: `2.217`
- iter `1400`: `2.191`

Fresh-40 mechanics versus the current title+abstract v5 LoRA checkpoint `425`:

| checkpoint | avg words | >14 | >15 | <10 | digits | formula | title >=3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoRA425 baseline | 12.15 | 7 | 5 | 5 | 3 | 1 | 15 |
| replay800 | 11.90 | 5 | 3 | 6 | 4 | 1 | 25 |
| replay1000 | 12.00 | 4 | 1 | 4 | 3 | 1 | 23 |
| replay1200 | 11.78 | 4 | 2 | 7 | 2 | 1 | 24 |
| replay1300 | 11.78 | 7 | 3 | 7 | 2 | 1 | 22 |
| replay1400 | 11.53 | 4 | 1 | 5 | 3 | 1 | 22 |

Corrected exact-change counts:

| checkpoint | changed rows | exact repeats |
|---|---:|---:|
| replay800 | 36 | 4 |
| replay1000 | 37 | 3 |
| replay1200 | 39 | 1 |
| replay1300 | 39 | 1 |
| replay1400 | 38 | 2 |

Manual read:

- The run moved the model strongly and improved length mechanics, especially
  at `1000` and `1400`.
- It did not produce a clean contender against LoRA425.
- Main regression pattern: selected broad replay rows brought back title-shaped
  phrasing and brittle technical specificity.
- Examples included invented or distorted details around graph cliques, exact
  vaccine-recovery windows, covariational-reasoning repetition, math/physics
  object drift, and a misspelled or hallucinated `Seicher networks` phrase.
- The best rows are useful evidence that broad replay can help tone and coverage,
  but the current replay pool is too permissive for a promotion-style run.

Decision: no blind review was run. The mechanical gains are real, but the manual
read shows enough systematic regression that a 100-row blind review would likely
spend budget confirming a non-contender.

Suggested next move if continuing this branch: make the replay pool stricter
before retraining. A good first cut is to keep only heuristic replay rows with
low title overlap, then use a higher gold-to-replay exposure ratio such as
`gold x8 + 1500-2200 title-clean replay rows`.
