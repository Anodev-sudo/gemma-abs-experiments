# Granite 350M v5 q8 Fusion-Aware v1

This run tests Granite in the runtime shape we care about for the next pass:
MLX q8 group-size 64, fused, then read through the Swift probe.

No new labels were generated. Training used only the existing 994 strict gold
title+abstract v5 rows.

## Artifacts

- Run root:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q8_fusion_aware_v1`
- Clean q8 base:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q8_fusion_aware_v1/base_q8_g64`
- Stage A adapters:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q8_fusion_aware_v1/stage_a_q8_base_lr5e6/adapters`
- Stage A selected fused artifact:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q8_fusion_aware_v1/fused_round1_q8_g64`
- Stage B adapters:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q8_fusion_aware_v1/stage_b_fused_round1_lr1e6/adapters`
- Stage B selected fused artifact:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q8_fusion_aware_v1/fused_round2_q8_g64`
- Granite candidate pointer:
  `hf_bucket/experiments/mlx_lora/promoted/granite_4_0_350m_title_abstract_candidate_q8.json`

Both preserved fused q8 artifacts are about `364M`.

## Validation

Dataset validation passed:

| split | rows |
| --- | ---: |
| train | `894` |
| valid | `50` |
| test | `50` |
| strict gold training rows | `994` |

Prompt version: `gemma_270m_it_abstract_to_feed_blurb_v5`.

The q8 base was freshly converted from `ibm-granite/granite-4.0-350m` with
`mlx_lm convert --quantize --q-bits 8 --q-group-size 64`. It was not based on
the earlier ckpt425 fused artifact.

## Configs

- `configs/mlx_lora/granite_4_0_350m_v5_q8_fusion_aware_stage_a_lr5e6.yaml`
- `configs/mlx_lora/granite_4_0_350m_v5_q8_fusion_aware_stage_b_lr1e6.yaml`

Shared LoRA shape:

| setting | value |
| --- | --- |
| rank | `16` |
| scale | `10` |
| dropout | `0` |
| targets | `q_proj,k_proj,v_proj,o_proj` |
| layers | all |
| batch | `4` |
| prompt masking | enabled |

Stage A trained from the clean q8 base at `5e-6` for `500` iterations. Stage B
trained from `fused_round1_q8_g64` at `1e-6` for `200` iterations.

## Stage A q8 Fused Swift Sweep

All rows use the same 100-paper Granite blind packet used in the earlier
checkpoint-425 work.

| step | avg | <=14 | >14 | >15 | <10 | digits | title >=3 | changed vs original MLX |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `250` | `12.46` | `76` | `24` | `17` | `15` | `6` | `50` | `91` |
| `300` | `13.16` | `67` | `33` | `21` | `8` | `11` | `52` | `87` |
| `350` | `13.23` | `72` | `28` | `19` | `2` | `7` | `55` | `74` |
| `400` | `12.21` | `81` | `19` | `10` | `16` | `5` | `46` | `85` |
| `425` | `12.70` | `77` | `23` | `15` | `6` | `4` | `57` | `48` |
| `450` | `13.19` | `71` | `29` | `19` | `7` | `9` | `59` | `75` |
| `500` | `12.47` | `74` | `26` | `16` | `14` | `6` | `56` | `79` |

Checkpoint `425` is the Stage A pick. It is not the shortest checkpoint, but it
has the best balance: it fixes the severe Superposition polarity flip in q8
fused Swift while staying close to the original Granite checkpoint-425 behavior.

The tempting mechanical alternative was `400`, but it reintroduced the
Superposition "complete" vs "incomplete" failure in Swift, so it is out.

## Stage B q8 Fused Swift Sweep

Stage B is a correction LoRA trained on top of the Stage A q8 fused artifact.

| step | avg | <=14 | >14 | >15 | <10 | digits | title >=3 | changed vs original MLX | changed vs Stage A 425 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `25` | `12.60` | `81` | `19` | `13` | `6` | `6` | `56` | `50` | `47` |
| `50` | `13.06` | `72` | `28` | `18` | `6` | `6` | `56` | `60` | `53` |
| `75` | `12.57` | `78` | `22` | `17` | `11` | `6` | `54` | `60` | `59` |
| `100` | `12.82` | `73` | `27` | `17` | `10` | `5` | `56` | `58` | `59` |
| `125` | `12.88` | `73` | `27` | `16` | `9` | `4` | `56` | `58` | `59` |

Checkpoint `25` is the only Stage B checkpoint that looks like a real
correction candidate. It improves the length profile over Stage A `425` and
keeps the sticky polarity fix. Later checkpoints drift harder, lengthen the
tail, and do not offer a clear qualitative benefit.

Checkpoint `150` started evaluation but only sidecar Python output completed
before local disk pressure interrupted the q8 fuse. Given the worse trend from
`50` through `125`, it was not worth spending more disk or review budget on
`150` or `200` before diagnosing.

## Sticky Rows

Stage A `425` and Stage B `25` both preserve the critical Superposition row in
q8 fused Swift:

- `2405.03367`: stays `refutationally incomplete`

Stage B `25` improves the Brusselator sticky row versus Stage A:

- Stage A `425`: `Time scales separate the Brusselator model near the line at infinity.`
- Stage B `25`: `Desingularizing reveals four distinct time scales for the Brusselator model near the line at infinity.`

Other sticky read:

- `2308.04773`: Stage B keeps the meson-cloud transition claim stable.
- `2301.02200`: Stage B exactly matches the Stage A/q8 line.
- `2007.03757`: Stage B changes to `Micromechanical modeling reveals a more realistic crack path for brittle fracture.` This is readable but slightly less concrete than the Stage A line, so it is a watch item.

## Blind Review

Because the prior q8 fused Swift export had a known severe polarity regression,
the most decision-useful single blind review is Stage A `425` vs Stage B `25`,
both in q8 fused Swift.

Blind packet:

`hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q8_fusion_aware_v1/blind_reviews/stage_a425_vs_stage_b25_q8_fused_swift_100_20260528`

It has `100` rows split into five `20`-row shards. Reviewers are asked only:
which summary is better for deciding whether to open the paper in a broad
mobile research feed?

Unblinded result:

| choice | count | share |
| --- | ---: | ---: |
| Stage B `25` q8 fused Swift | `26` | `26%` |
| Stage A `425` q8 fused Swift | `18` | `18%` |
| tie | `53` | `53%` |
| neither | `3` | `3%` |

Review quality checks:

- reviewed rows: `100 / 100`
- missing IDs: `0`
- duplicate IDs: `0`
- schema errors: `0`

Read: this is a soft Stage B win, not a decisive preference shift. The high tie
rate means the correction did not radically improve feed usefulness, but it also
did not create enough obvious regressions for reviewers to prefer Stage A.

## Blind Review Against Original Granite 425

After the Stage A/B comparison, Stage B `25` q8 fused Swift was reviewed
directly against the original Granite `425` MLX output on the same 100-paper
packet.

Blind packet:

`hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q8_fusion_aware_v1/blind_reviews/old425_mlx_vs_stage_b25_q8_fused_swift_100_20260528`

The packet had `50 / 100` exact matching summaries before review.

Unblinded result:

| choice | count | share |
| --- | ---: | ---: |
| Stage B `25` q8 fused Swift | `33` | `33%` |
| original Granite `425` MLX | `15` | `15%` |
| tie | `49` | `49%` |
| neither | `3` | `3%` |

Review quality checks:

- reviewed rows: `100 / 100`
- missing IDs: `0`
- duplicate IDs: `0`
- schema errors: `0`

Read: Stage B `25` does not merely recover old `425`; in this blind feed
preference pass it beats it. Because half the packet was exact matches, the
non-tie preference among changed rows is meaningfully in Stage B's direction.

## Blind Review Against q5 Current Sidecar

Stage B `25` q8 fused Swift was also reviewed directly against the current q5
candidate sidecar output from the earlier q5-vs-Granite packet.

Blind packet:

`hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q8_fusion_aware_v1/blind_reviews/q5_current_vs_stage_b25_q8_fused_swift_100_20260528`

The packet had `0 / 100` exact matching summaries before review.

Unblinded result:

| choice | count | share |
| --- | ---: | ---: |
| Stage B `25` q8 fused Swift | `57` | `57%` |
| q5 current q8 sidecar | `41` | `41%` |
| tie | `2` | `2%` |

Review quality checks:

- reviewed rows: `100 / 100`
- missing IDs: `0`
- duplicate IDs: `0`
- schema errors: `0`

Read: Stage B `25` now beats the old q5 candidate on the same app-style
preference question. The margin is similar to the earlier Granite `425` vs q5
review, but this comparison is stronger for deployment planning because the
candidate side is the actual q8 fused Swift artifact rather than original MLX.

## q6 Probe

Stage B `25` q8 fused Swift was dequantized and requantized to MLX q6 group-size
64, then evaluated through the Swift probe on the same 100-paper packet.

Artifacts:

- q6 model:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q8_fusion_aware_v1/q6_probe/fused_round2_q6_g64`
- q6 generations:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q8_fusion_aware_v1/q6_probe/generations/stage_b25_q6_fused_swift_blind100.jsonl`
- q6 mechanics:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q8_fusion_aware_v1/q6_probe/reports/blind100_q6_vs_q8_q5_old/MECHANICS.md`

Size:

| artifact | size |
| --- | ---: |
| Stage B `25` q8 fused | `364M` |
| Stage B `25` q6 fused | `280M` |

Mechanics:

| label | avg | max | <=14 | >14 | >15 | <10 | digits | formula | title >=3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old Granite `425` MLX | `13.09` | `24` | `71` | `29` | `18` | `5` | `5` | `4` | `56` |
| q5 current q8 sidecar | `12.88` | `19` | `73` | `27` | `17` | `10` | `14` | `2` | `64` |
| Stage B `25` q8 fused Swift | `12.60` | `24` | `81` | `19` | `13` | `6` | `6` | `4` | `56` |
| Stage B `25` q6 fused Swift | `13.44` | `21` | `69` | `31` | `25` | `10` | `7` | `2` | `56` |

Exact-match drift:

| pair | exact | changed |
| --- | ---: | ---: |
| q6 vs Stage B q8 | `13` | `87` |
| q6 vs old Granite `425` | `12` | `88` |
| q6 vs q5 current | `0` | `100` |

Swift timing:

| artifact | avg generation ms | median ms | max ms |
| --- | ---: | ---: | ---: |
| Stage B `25` q8 fused Swift | `117.1` | `115.5` | `188` |
| Stage B `25` q6 fused Swift | `130.0` | `124.0` | `378` |

Read: q6 is not a viable candidate from this probe. It saves `84M`, but it
drifts heavily from the q8 artifact, worsens the length profile, and is slower
in this Swift run. It also keeps or worsens known bad rows: the PVDF actuator
row still has impossible `20 meters` / `500 newtons`, and the split-federated
learning row still flips the accuracy benefit into "cutting communication and
accuracy."

No blind review should be spent on this q6 artifact.

## Decision

Current best artifacts:

- Conservative q8 fused candidate: Stage A `425`, `fused_round1_q8_g64`.
- Preferred correction candidate: Stage B `25`, `fused_round2_q8_g64`.

Stage B `25` is now recorded as the Granite candidate pointer. This does not
replace the existing app-facing Gemma promoted pointer.

Skip Stage C for now. The blind review did not reveal a narrow repeated Stage B
failure worth targeting, and the later Stage B checkpoints drifted more without
a clear benefit.
