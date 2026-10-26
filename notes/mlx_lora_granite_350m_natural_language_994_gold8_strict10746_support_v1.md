# MLX LoRA Granite 350M 994-gold x8 plus strict-10746 support v1

## Goal

Invest in Granite after the 994-row Granite LoRA beat fresh Gemma and narrowly
beat the current q5 sidecar in blind model-judge review.

This run tests whether a stricter old-data support pool can add coverage and
paper-specific detail without pulling Granite away from the natural feed-blurb
style it already learned.

This is not a promotion run.

## Data Policy

No new labels are generated and no labels are manually written. The support rows
are existing old-release summaries selected by deterministic gates only.

Support rows are lower-trust coverage/detail replay, not equivalent to the 994
strict gold rows.

## Strict Support Selection

Selector:

- `scripts/build_mlx_granite_strict_support_v1_dataset.py`

Source:

- `hf_dataset/releases/recent_3y_balanced_10746_v1/training_rows.jsonl`

Output:

- `hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_gold8_strict10746_support_v1/dataset_strict10746_support1600_title_abstract_feed_blurb_v5`

Policy:

- target `1600` rows
- preserve collapsed bucket ratio: `cs 37%`, `physics 28%`, `math 16%`, `other 19%`
- exclude all 994 strict-gold IDs and existing eval IDs
- require `9-14` words
- require title overlap `<=2`
- reject copied title trigrams
- reject formulas, digits, unsupported numbers, weak openings, hype/comparison terms, and acronym stacks
- require moderate source support and at least two non-title abstract-backed content hits

Selected support counts:

- `1600` rows exactly
- `592 cs`
- `448 physics`
- `256 math`
- `304 other`

Selected support shape:

- title overlap: `0` = `194`, `1` = `514`, `2` = `892`
- word counts: `9` = `106`, `10` = `265`, `11` = `445`, `12` = `430`, `13` = `234`, `14` = `120`
- acronym counts: `0` = `1249`, `1` = `351`

## Mixed Training Dataset

Builder:

- `scripts/build_mlx_fullmix_gold_support_v1_dataset.py`

Build command:

```bash
python3 scripts/build_mlx_fullmix_gold_support_v1_dataset.py \
  --gold-repeat 8 \
  --support-path hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_gold8_strict10746_support_v1/dataset_strict10746_support1600_title_abstract_feed_blurb_v5/train.jsonl \
  --output-dir hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_gold8_strict10746_support_v1/dataset_gold8_strictsupport1600_feed_blurb_v5
```

Output:

- `hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_gold8_strict10746_support_v1/dataset_gold8_strictsupport1600_feed_blurb_v5`

Counts:

- train examples: `8752`
- train unique paper IDs: `2494`
- gold train exposures: `7152`
- support rows: `1600`
- valid examples: `50` strict-gold holdout rows
- test examples: `50` strict-gold holdout rows

## Training Plan

Config:

- `configs/mlx_lora/granite_4_0_350m_natural_language_994_gold8_strict10746_support_v1_lr5e6.yaml`

Output:

- `hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_gold8_strict10746_support_v1/adapters_lr5e6`

Settings:

- base model: `ibm-granite/granite-4.0-350m`
- fresh LoRA, attention q/k/v/o projections
- rank `16`, scale `10`
- batch size `4`
- learning rate `5e-6`
- iterations `1600`
- save/eval every `100`

## Evaluation Plan

Compare candidate checkpoints to the prior Granite `425` checkpoint:

- baseline adapter:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/adapters/0000425_adapters.safetensors`
- reuse the same held-out 100-row packet used for q5-current vs Granite `425`
- report avg words, max, `>14`, `>15`, `<10`, digit rows, formula rows, title overlap, and changed rows
- manually read changed rows before deciding whether a blind review is warranted

## Result

### Training

The first leg reached checkpoint `1000` before the process/session was
interrupted.

Observed validation curve:

- step `1`: `3.848`
- step `100`: `1.702`
- step `200`: `1.644`
- step `300`: `1.630`
- step `400`: `1.630`
- step `500`: `1.624`
- step `600`: `1.626`
- step `700`: `1.607`
- resumed check from `1000`: `1.605`

The step `1000` checkpoint was close to the prior Granite `425` best validation
loss (`1.604`) despite the support replay.

A continuation from checkpoint `1000` was attempted:

- config:
  `configs/mlx_lora/granite_4_0_350m_natural_language_994_gold8_strict10746_support_v1_lr5e6_resume1000_to1600.yaml`
- resume checkpoint:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_gold8_strict10746_support_v1/adapters_lr5e6/0001000_adapters.safetensors`

Continuation step `1` reported validation loss `1.605`, but continuation step
`100` jumped to `1.696`. The continuation was stopped. Treat the first-leg
checkpoints as the only serious candidates.

### Mechanics

Comparison packet:

- `hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_gold8_strict10746_support_v1/review_samples/granite425_vs_strictsupport700_blind100_title_abstract_feed_blurb_v5`
- `hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_gold8_strict10746_support_v1/review_samples/granite425_vs_strictsupport900_blind100_title_abstract_feed_blurb_v5`
- `hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_gold8_strict10746_support_v1/review_samples/granite425_vs_strictsupport1000_blind100_title_abstract_feed_blurb_v5`

All comparisons reused the same 100 held-out papers as the q5-current vs
Granite `425` blind packet.

| checkpoint | avg words | max | <=14 | >14 | >15 | <10 | digits | formula | title >=3 | changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Granite `425` baseline | `13.09` | `24` | `71` | `29` | `18` | `5` | `5` | `4` | `56` | n/a |
| strict-support `700` | `12.68` | `23` | `79` | `21` | `15` | `15` | `8` | `2` | `51` | `93` |
| strict-support `900` | `12.11` | `20` | `85` | `15` | `11` | `17` | `6` | `2` | `53` | `93` |
| strict-support `1000` | `12.20` | `21` | `84` | `16` | `11` | `13` | `8` | `3` | `52` | `95` |

Surface read:

- The support replay clearly increased movement and improved length compliance.
- It also increased under-10 rows and digit rows.
- Title-overlap improved only modestly.

### Manual Read

This is not a blind-review candidate.

Regression pattern:

- unsupported-looking exact values reappeared
- old abstract-compression phrasing reappeared
- support-like technical specificity often displaced the cleaner feed hook
- some rows became malformed or semantically distorted

Examples:

- `2511.12380`: baseline correctly kept `500 newtons`; support checkpoints
  produced malformed or wrong actuator phrasing, including `500 hertz`.
- `2302.06647`: baseline was bad but support output became an unfinished
  over-specific galaxy/black-hole sentence.
- `2402.08795`: support checkpoints shifted a Uranus/xenon hook into a comet
  comparison that reads unsupported.
- `2506.18024`: support `700` introduced an exact `86%` latency claim.
- `2310.07333`: support `1000` collapsed into the awkward fragment
  `Polynomial-ity extends to multi-dimensional monotone functions.`
- `2507.07130`: support checkpoints produced confusing split-federated-learning
  claims around communication and accuracy.

Decision: no blind review. The strict support selection was much better than
the old 3k pool mechanically, but old labels still seem too behaviorally
contaminating for Granite when mixed into the 994 gold set this way.

## Takeaway

The selective-support idea is not dead, but this version should not be extended.
For Granite, old supervised rows need either much lower exposure than `gold x8 +
support x1`, or a different source family. The current evidence favors staying
close to strict gold and using broader old data only after a stronger review or
rewrite step, not as raw support replay.
