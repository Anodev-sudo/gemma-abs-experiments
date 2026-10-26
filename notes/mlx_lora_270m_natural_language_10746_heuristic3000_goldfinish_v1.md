# MLX LoRA 270M heuristic-3000 burn-in plus gold finisher v1

## Goal

Try a two-stage fresh-adapter run:

1. Burn in on the best deterministic subset of the old `10,746`-row release.
2. Follow with an aggressive strict-gold pass on the `994` feed-blurb v5 rows.

This is not a promotion run.

## Data Policy

No new labels are generated and no labels are manually written. The broad-stage
labels come from existing old-release summaries. The selector only decides which
rows to keep.

The selected broad rows are recast into the current
`gemma_270m_it_abstract_to_feed_blurb_v5` title+abstract prompt so the training
surface matches the current natural-language inference path.

## Dataset

Builder:

- `scripts/build_mlx_heuristic_10746_support_dataset.py`

Output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_10746_heuristic3000_goldfinish_v1/dataset_heuristic3000_title_abstract_feed_blurb_v5`

Selection:

- source rows: old `recent_3y_balanced_10746_v1/training_rows.jsonl`
- exclude all `994` strict-gold IDs
- exclude the standard fresh-40 eval IDs
- select `3000` rows
- preserve broad target buckets: `cs 37%`, `physics 28%`, `math 16%`, `other 19%`
- hard reject missing fields, summaries outside `8-14` words, boilerplate
  openings, multi-sentence rows, formula-like summaries, numeric summaries,
  acronym stacks, and title-overlap count `>=5`

Counts from the initial build:

- train rows: `3000`
- valid rows: `50` strict-gold holdout rows
- test rows: `50` strict-gold holdout rows
- selected buckets: `1110 cs`, `840 physics`, `480 math`, `570 other`
- selected score range: `110.0` to `131.0`, average `118.924`

## Training Plan

Burn-in config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_10746_heuristic3000_burnin_v1_lr8e6.yaml`

Burn-in output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_10746_heuristic3000_goldfinish_v1/burnin_lr8e6`

Burn-in details:

- base model: `google/gemma-3-270m-it`
- fresh LoRA, attention projections only
- batch size: `4`
- iterations: `800`
- learning rate: `8e-6`
- save every `100`

Gold finisher config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_10746_heuristic3000_goldfinish_v1_lr1p5e5.yaml`

Gold finisher output:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_10746_heuristic3000_goldfinish_v1/goldfinish_lr1p5e5`

Gold finisher details:

- resume from burn-in checkpoint `0000800`
- strict `994` title+abstract feed-blurb v5 dataset
- batch size: `4`
- iterations: `300`
- learning rate: `1.5e-5`
- save every `25`

## Evaluation Plan

Run the standard fresh-40 comparison against the current title+abstract v5 LoRA
checkpoint `425`. Check burn-in final only as a diagnostic, then gold-finisher
checkpoints `100`, `150`, `200`, `250`, and `300`.

## Result

### Burn-In

Training completed cleanly.

Validation loss on the strict-gold holdout:

- step `1`: `5.315`
- step `100`: `2.581`
- step `200`: `2.499`
- step `300`: `2.470`
- step `400`: `2.455`
- step `500`: `2.443`
- step `600`: `2.435`
- step `700`: `2.419`
- step `800`: `2.433`

Test loss at step `800` was `2.441` (`11.490` perplexity). Step `700` had the
best validation loss, but this first pass used the planned step `800` for the
gold finisher.

Fresh-40 mechanics for burn-in `800` against LoRA `425`:

| checkpoint | avg words | over 14 | over 15 | under 10 | digit rows | formula rows | title overlap >=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA 425 | `12.15` | `7/40` | `5/40` | `5/40` | `3/40` | `1/40` | `15/40` |
| burn-in 800 | `11.57` | `6/40` | `4/40` | `6/40` | `4/40` | `0/40` | `24/40` |

Read: the heuristic broad rows helped length/formula mechanics, but title
overlap stayed high and several outputs read like old-data summaries rather
than feed blurbs.

### Aggressive Gold Finisher

Training completed cleanly from burn-in `800`.

Validation loss:

- step `1`: `2.439`
- step `25`: `2.376`
- step `50`: `2.318`
- step `75`: `2.278`
- step `100`: `2.228`
- step `125`: `2.220`
- step `150`: `2.200`
- step `175`: `2.192`
- step `200`: `2.214`
- step `225`: `2.192`
- step `250`: `2.228`
- step `275`: `2.204`
- step `300`: `2.213`

Test loss at step `300` was `2.251` (`9.495` perplexity).

Fresh-40 mechanics:

| checkpoint | avg words | over 14 | over 15 | under 10 | digit rows | formula rows | title overlap >=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gold 125 | `12.28` | `5/40` | `3/40` | `1/40` | `4/40` | `1/40` | `24/40` |
| gold 150 | `12.88` | `10/40` | `6/40` | `1/40` | `4/40` | `1/40` | `21/40` |
| gold 175 | `11.72` | `7/40` | `4/40` | `8/40` | `2/40` | `0/40` | `24/40` |
| gold 225 | `13.82` | `14/40` | `9/40` | `3/40` | `4/40` | `1/40` | `23/40` |
| gold 300 | `12.45` | `8/40` | `5/40` | `2/40` | `4/40` | `0/40` | `23/40` |

Read: `125` was the best mechanical checkpoint, but not a blind-review
candidate. It introduced or preserved exact-value claims and had several
semantic regressions. Later checkpoints did not fix the pattern and often got
longer or more title-bound.

### Numeric-Clean Gold Branch

Because the first gold finisher showed exact-number relapse, a branch was run
with digit/formula target rows removed from the strict-gold train split:

- builder: `scripts/build_mlx_filtered_gold_dataset.py`
- output dataset:
  `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_10746_heuristic3000_goldfinish_v1/dataset_gold_digitformula_free_feed_blurb_v5`
- kept train rows: `824`
- dropped train rows: `70`
- drop reasons: `69` digit targets, `2` formula-like targets

Config:

- `configs/mlx_lora/gemma_3_270m_it_natural_language_10746_heuristic3000_goldclean_finish_v1_lr1e5.yaml`

Validation loss:

- step `1`: `2.439`
- step `25`: `2.315`
- step `50`: `2.314`
- step `75`: `2.275`
- step `100`: `2.223`
- step `125`: `2.218`
- step `150`: `2.211`
- step `175`: `2.214`
- step `200`: `2.188`
- step `225`: `2.199`

Test loss at step `225` was `2.256` (`9.543` perplexity).

Fresh-40 mechanics:

| checkpoint | avg words | over 14 | over 15 | under 10 | digit rows | formula rows | title overlap >=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean 150 | `12.15` | `7/40` | `4/40` | `4/40` | `1/40` | `1/40` | `25/40` |
| clean 200 | `12.30` | `9/40` | `4/40` | `3/40` | `4/40` | `1/40` | `25/40` |
| clean 225 | `12.65` | `10/40` | `10/40` | `4/40` | `2/40` | `1/40` | `20/40` |

Read: `clean 150` is the best mechanics/faithfulness tradeoff in this branch,
but still not a blind-review candidate. It reduces digit rows, but title
overlap is high and manual read showed semantic oddities, title copying, and
occasional unsupported specificity.

## Takeaway

The deterministic 3k selection is not useless: it produces a burn-in with much
better length/formula mechanics than the full-finetune path. The problem is the
follow-up finisher. A normal aggressive gold pass pulls in exact-number and
semantic drift; a numeric-clean pass reduces that but leaves title copying and
odd specificity.

Do not spend blind-review budget on these checkpoints yet. The next likely move
is to make the broad selection itself more anti-title-copy and maybe train a
smaller `1500-2000` support set, or use the selected broad data as low-weight
replay during a primarily gold LoRA run instead of as a separate burn-in.
