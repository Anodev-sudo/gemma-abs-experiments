# q5 Precision Failure Overlay v1

Goal: test whether the remaining q5 `replay_75` severe modes can be corrected
with a tiny targeted overlay instead of another broad behavior run.

The targeted modes were:

- metric-role binding
- malformed repetition
- polarity or scoped-claim slips

## Data

Materializer:

`scripts/materialize_q5_precision_failure_overlay.py`

Unweighted release:

`hf_dataset/releases/recent_3y_balanced_10746_q5_precision_failure_overlay_v1`

Weighted release:

`hf_dataset/releases/recent_3y_balanced_10746_q5_precision_failure_overlay_weighted_v2`

The unweighted release used 14 confirmed q5 severe repairs plus 72 clean
`replay_75` anchors.

The weighted release repeated the 14 repair rows 6 times and used 56 clean
`replay_75` anchors:

| source | rows |
| --- | ---: |
| `q5_precision_confirmed_severe_repair` | 84 |
| `q5_precision_clean_replay75_anchor` | 56 |

Weighted bucket mix:

| bucket | rows |
| --- | ---: |
| `style_anchor` | 56 |
| `number_role_binding` | 36 |
| `malformed_avoidance` | 30 |
| `scoped_claims` | 12 |
| `polarity_direction` | 6 |

## Training

Unweighted config:

`configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_precision_failure_overlay_v1_lr_1e_7.yaml`

Weighted config:

`configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_precision_failure_overlay_weighted_v2_lr_2e_7.yaml`

Both runs resumed from:

`hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_final_guardrail_overlay_replay_v2_lr_2e_7/adapters/0000075_adapters.safetensors`

Unweighted run:

`hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_precision_failure_overlay_v1_lr_1e_7`

Weighted run:

`hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_precision_failure_overlay_weighted_v2_lr_2e_7`

Weighted validation loss improved inside the tiny weighted distribution:

| step | val loss |
| ---: | ---: |
| 1 | 2.580 |
| 25 | 2.557 |
| 50 | 2.478 |
| 75 | 2.413 |
| 100 | 2.381 |
| 125 | 2.339 |

Final weighted test loss was 2.348, test perplexity 10.463.

## Target-14 Checks

Target fixture:

`hf_bucket/experiments/mlx_lora/review_comparisons/q5_precision_failure_overlay_eval_20260507/target_14_inputs.jsonl`

Unweighted screen:

`hf_bucket/experiments/mlx_lora/review_comparisons/q5_precision_failure_overlay_eval_20260507/target14_checkpoint_screen.md`

| step | looks improved | still bad | needs review | avg words | over15 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 2 | 9 | 3 | 13.79 | 5 |
| 50 | 2 | 8 | 4 | 15.00 | 8 |
| 75 | 2 | 9 | 3 | 14.43 | 7 |
| 100 | 1 | 11 | 2 | 14.57 | 7 |

Weighted screen:

`hf_bucket/experiments/mlx_lora/review_comparisons/q5_precision_failure_overlay_eval_20260507/target14_weighted_checkpoint_screen.md`

| step | looks improved | still bad | needs review | avg words | over15 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | 3 | 11 | 0 | 14.79 | 7 |
| 50 | 4 | 10 | 0 | 14.29 | 6 |
| 75 | 4 | 10 | 0 | 14.21 | 7 |
| 100 | 3 | 11 | 0 | 13.79 | 5 |
| 125 | 6 | 8 | 0 | 13.71 | 4 |

## Read

This did not produce a usable candidate.

The unweighted overlay was too weak and barely moved the exact failure rows.
The weighted overlay moved more rows, but only partially: the best target check
was weighted step 125, with 6 of 14 looking improved and 8 still bad. Some
corrections were also unstable across checkpoints.

The remaining failure modes look sticky at generation time. A small supervised
patch can nudge a subset, but it does not reliably erase the bad continuation
basins. More repetition/weight would risk turning this into a memorization run
with unknown collateral damage.

Recommendation: do not promote either precision overlay. Keep `replay_75` as
the current q5 candidate. If we keep training, use a larger high-quality dataset
of these same modes rather than repeating the same 14 rows harder. In
production, consider a lightweight runtime guard for the most detectable class:
malformed repetition and obviously wrong metric binding.
