# Granite 350M v5 q6 Fusion-Aware v1

This is the q6-native replication of the q8 fusion-aware run. It is due
diligence only, not a candidate promotion.

No new labels were generated. Training used the same existing 994 strict gold
title+abstract v5 rows as the q8 run.

## Artifacts

- Run root:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q6_fusion_aware_v1`
- Clean q6 base:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q6_fusion_aware_v1/base_q6_g64`
- Stage A adapters:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q6_fusion_aware_v1/stage_a_q6_base_lr5e6/adapters`
- Stage A selected fused artifact:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q6_fusion_aware_v1/fused_round1_q6_g64`
- Stage B adapters:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q6_fusion_aware_v1/stage_b_fused_round1_lr1e6/adapters`
- Stage B selected fused diagnostic:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q6_fusion_aware_v1/fused_round2_q6_g64`

The preserved q6 fused artifacts are about `280M`.

## Validation

Dataset validation passed:

| split | rows |
| --- | ---: |
| train | `894` |
| valid | `50` |
| test | `50` |
| strict gold training rows | `994` |

Prompt version: `gemma_270m_it_abstract_to_feed_blurb_v5`.

The q6 base was freshly converted from `ibm-granite/granite-4.0-350m` with
`mlx_lm convert --quantize --q-bits 6 --q-group-size 64`.

## Configs

- `configs/mlx_lora/granite_4_0_350m_v5_q6_fusion_aware_stage_a_lr5e6.yaml`
- `configs/mlx_lora/granite_4_0_350m_v5_q6_fusion_aware_stage_b_lr1e6.yaml`

Shared LoRA shape matches q8:

| setting | value |
| --- | --- |
| rank | `16` |
| scale | `10` |
| dropout | `0` |
| targets | `q_proj,k_proj,v_proj,o_proj` |
| layers | all |
| batch | `4` |
| prompt masking | enabled |

Stage A trained from the clean q6 base at `5e-6` for `500` iterations. Stage B
trained from `fused_round1_q6_g64` at `1e-6` for `200` iterations.

## Stage A q6 Fused Swift Sweep

All rows use the same 100-paper Granite blind packet used by the q8 run.

| step | avg | max | <=14 | >14 | >15 | <10 | digits | formula | title >=3 | changed vs original MLX |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `250` | `15.15` | `26` | `49` | `51` | `41` | `3` | `8` | `6` | `70` | `98` |
| `300` | `15.29` | `26` | `50` | `50` | `41` | `3` | `10` | `6` | `70` | `98` |
| `350` | `14.37` | `25` | `58` | `42` | `32` | `5` | `6` | `3` | `67` | `98` |
| `400` | `13.54` | `26` | `73` | `27` | `17` | `9` | `5` | `5` | `67` | `98` |
| `425` | `13.99` | `28` | `65` | `35` | `25` | `7` | `7` | `4` | `65` | `98` |
| `450` | `14.60` | `31` | `58` | `42` | `32` | `2` | `7` | `6` | `71` | `98` |
| `500` | `13.93` | `48` | `65` | `35` | `20` | `4` | `8` | `6` | `70` | `96` |

Stage A `400` had the best mechanics, but it lost the Superposition
`incomplete` caveat. Stage A `500` was selected only as a q6 Stage B input
because it preserved that polarity, despite a 48-word runaway row.

## Stage B q6 Fused Swift Sweep

| step | avg | max | <=14 | >14 | >15 | <10 | digits | formula | title >=3 | changed vs original MLX |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `25` | `13.43` | `21` | `69` | `31` | `19` | `4` | `7` | `7` | `67` | `96` |
| `50` | `13.48` | `21` | `71` | `29` | `18` | `4` | `9` | `6` | `68` | `97` |
| `75` | `13.90` | `26` | `62` | `38` | `21` | `4` | `8` | `6` | `69` | `98` |
| `100` | `13.83` | `21` | `65` | `35` | `24` | `5` | `7` | `5` | `68` | `97` |
| `125` | `13.89` | `21` | `61` | `39` | `25` | `5` | `8` | `6` | `67` | `98` |

Stage B `25` is the least-bad q6-native checkpoint. It fixed the Stage A
runaway tail and preserved the Superposition polarity, so it was fused as a
diagnostic artifact:

`hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q6_fusion_aware_v1/fused_round2_q6_g64`

## Sticky Rows

Stage B `25` q6 fused Swift preserves the critical sticky row:

- `2405.03367`: stays `refutationally incomplete`

Other sticky read:

- `2311.00575`: readable but less specific than q8; it collapses four time
  scales into a general time-scale-separation line.
- `2308.04773`: good enough; keeps the Dalitz decay-width target.
- `2301.02200`: good enough; keeps autonomous-driving dataset influence.
- `2007.03757`: plausible but more theory-shaped than the q8 candidate.

## Comparison To q8

| label | avg | max | <=14 | >14 | >15 | <10 | digits | formula | title >=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage B `25` q8 fused Swift | `12.60` | `24` | `81` | `19` | `13` | `6` | `6` | `4` | `56` |
| Stage B `25` q8-to-q6 probe | `13.44` | `21` | `69` | `31` | `25` | `10` | `7` | `2` | `56` |
| Stage B `25` q6-native fused Swift | `13.43` | `21` | `69` | `31` | `19` | `4` | `7` | `7` | `67` |

Native q6 training slightly improved the q8-to-q6 probe on `>15` and `<10`
counts, but worsened title overlap and formula-ish rows. It remains clearly
worse than q8.

## Decision

Do not blind-review this q6 artifact unless the product constraint changes.
The saved size is not buying enough quality: q6 has heavier drift, weaker
length mechanics, and worse title overlap than the q8 Granite candidate.

Current Granite candidate remains:

`hf_bucket/experiments/mlx_lora/promoted/granite_4_0_350m_title_abstract_candidate_q8.json`
