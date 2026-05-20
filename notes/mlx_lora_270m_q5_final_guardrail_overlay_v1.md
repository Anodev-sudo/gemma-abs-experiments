# q5 Final Guardrail Overlay v1

This pass harvests high-confidence corrective examples for the remaining q5
len15 failure classes: number-role binding, polarity and direction, scoped
claims, and malformed notation avoidance.

## Dataset

- Release: `hf_dataset/releases/recent_3y_balanced_10746_q5_final_guardrail_overlay_v1`
- MLX dataset: `hf_dataset/releases/recent_3y_balanced_10746_q5_final_guardrail_overlay_v1/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_glanceable_summary_v1`
- Materializer: `scripts/materialize_q5_final_guardrail_overlay.py`
- Rows: 164
- Split: train 148, valid 8, test 8
- Summary length: average 12.59 audit words, max 15

Source mix:

| source | rows |
| --- | ---: |
| `kimi_qwen_verified_rewrite` | 61 |
| `old_q5_review_q8_clean_repair` | 68 |
| `reorient_q8_repair_for_q5_severe` | 13 |
| `reorient_q5_clean_win_anchor` | 13 |
| `reorient_q5_clean_anchor` | 5 |
| `len15_targeted_review_repair` | 4 |

Bucket mix:

| bucket | rows |
| --- | ---: |
| `scoped_claims` | 56 |
| `polarity_direction` | 38 |
| `number_role_binding` | 27 |
| `malformed_avoidance` | 21 |
| `style_anchor` | 14 |
| `entity_binding` | 8 |

Strict local audit rejected 48 rows, mostly for length:

- `too_long`: 33
- `high_title_overlap`: 10
- `target_number_not_in_source`: 5
- `adjacent_repetition`: 1

## Training

- Config: `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_final_guardrail_overlay_v1_lr_5e_7.yaml`
- Parent adapter: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_numeric_hardening_overlay_q4mix_len15_v2_lr_1e_6/adapters/0000125_adapters.safetensors`
- Run root: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_final_guardrail_overlay_v1_lr_5e_7`
- Learning rate: 5e-7
- Steps: 200

Validation kept improving through the run:

| step | val loss |
| ---: | ---: |
| 1 | 4.483 |
| 25 | 4.188 |
| 50 | 4.010 |
| 75 | 3.884 |
| 100 | 3.803 |
| 125 | 3.727 |
| 150 | 3.669 |
| 175 | 3.604 |
| 200 | 3.561 |

Final test loss was 3.643, test perplexity 38.204.

## Mechanical Check

Mechanical checkpoint screen:

`hf_bucket/experiments/mlx_lora/review_comparisons/q5_final_guardrail_overlay_eval_20260507/mechanical_checkpoint_summary.md`

Same 100-paper set as the q8 vs q5 len15 reorient eval:

| variant | avg words | max | over15 | over20 | number not in source | known malformed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `len15_step125` | 12.61 | 19 | 15 | 0 | 3 | 5 |
| `guardrail_25` | 12.59 | 19 | 13 | 0 | 3 | 5 |
| `guardrail_50` | 12.88 | 23 | 16 | 1 | 3 | 4 |
| `guardrail_75` | 12.85 | 21 | 15 | 1 | 3 | 4 |
| `guardrail_100` | 12.90 | 23 | 16 | 1 | 3 | 3 |
| `guardrail_125` | 13.17 | 22 | 21 | 1 | 2 | 4 |
| `guardrail_150` | 13.29 | 20 | 23 | 0 | 3 | 3 |
| `guardrail_200` | 13.31 | 20 | 21 | 0 | 3 | 4 |

## Read

The harvested data is good enough to avoid a fresh Kimi/Qwen generation pass for
now. It covers the failure classes we wanted and stayed inside the hard 15-word
target.

The first training sweep moved the model faster than expected. Validation loss
favored later checkpoints, but behavior did not: later checkpoints fixed some
known malformed outputs while raising length and introducing fresh drift.

Current best behavior candidate from this run is probably `guardrail_25`, not
the final checkpoint. It is shallow, preserves the len15 length center, and has
the best mechanical profile. It still needs a real blind review against
`len15_step125` before we should trust it.

## Replay-Balanced Follow-Up

Follow-up note:

`notes/mlx_lora_270m_q5_final_guardrail_overlay_replay_v2.md`

The follow-up added clean q5 len15 replay anchors and lowered LR to `2e-7`.
This reduced drift substantially. The strongest follow-up candidates are
`replay_75` for conservative length preservation and `replay_175` for slightly
more malformed-output correction.
