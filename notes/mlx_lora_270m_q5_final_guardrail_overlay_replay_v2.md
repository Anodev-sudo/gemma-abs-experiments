# q5 Final Guardrail Overlay replay v2

This run adjusts the first final guardrail overlay to reduce behavioral drift:

- adds clean `q5_len15_step125` replay anchors
- lowers learning rate from `5e-7` to `2e-7`
- keeps the hard 15-word local audit
- still starts from the q5 len15 step-125 adapter

## Dataset

- Release: `hf_dataset/releases/recent_3y_balanced_10746_q5_final_guardrail_overlay_replay_v2`
- MLX dataset: `hf_dataset/releases/recent_3y_balanced_10746_q5_final_guardrail_overlay_replay_v2/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_glanceable_summary_v1`
- Materializer: `scripts/materialize_q5_final_guardrail_overlay.py`
- Rows: 267
- Split: train 241, valid 13, test 13
- Summary length: average 12.27 audit words, max 15

Source mix:

| source | rows |
| --- | ---: |
| `len15_step125_replay_anchor` | 103 |
| `old_q5_review_q8_clean_repair` | 68 |
| `kimi_qwen_verified_rewrite` | 61 |
| `reorient_q8_repair_for_q5_severe` | 13 |
| `reorient_q5_clean_win_anchor` | 13 |
| `reorient_q5_clean_anchor` | 5 |
| `len15_targeted_review_repair` | 4 |

Bucket mix:

| bucket | rows |
| --- | ---: |
| `style_anchor` | 117 |
| `scoped_claims` | 56 |
| `polarity_direction` | 38 |
| `number_role_binding` | 27 |
| `malformed_avoidance` | 21 |
| `entity_binding` | 8 |

## Training

- Config: `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_final_guardrail_overlay_replay_v2_lr_2e_7.yaml`
- Parent adapter: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_numeric_hardening_overlay_q4mix_len15_v2_lr_1e_6/adapters/0000125_adapters.safetensors`
- Run root: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_final_guardrail_overlay_replay_v2_lr_2e_7`
- Learning rate: 2e-7
- Steps: 200

Validation kept improving through step 200:

| step | val loss |
| ---: | ---: |
| 1 | 4.292 |
| 25 | 4.147 |
| 50 | 4.036 |
| 75 | 3.949 |
| 100 | 3.893 |
| 125 | 3.845 |
| 150 | 3.802 |
| 175 | 3.773 |
| 200 | 3.743 |

Final test loss was 3.951, test perplexity 51.974.

## Mechanical Check

Mechanical checkpoint screen:

`hf_bucket/experiments/mlx_lora/review_comparisons/q5_final_guardrail_overlay_replay_v2_eval_20260507/mechanical_checkpoint_summary.md`

Same 100-paper set as the q8 vs q5 len15 reorient eval:

| variant | avg words | max | over15 | over20 | number not in source | known malformed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `len15_step125` | 12.61 | 19 | 15 | 0 | 3 | 5 |
| `replay_25` | 12.67 | 19 | 15 | 0 | 3 | 5 |
| `replay_50` | 12.59 | 19 | 12 | 0 | 3 | 5 |
| `replay_75` | 12.46 | 19 | 12 | 0 | 3 | 5 |
| `replay_100` | 12.80 | 19 | 15 | 0 | 3 | 5 |
| `replay_125` | 12.87 | 19 | 16 | 0 | 3 | 5 |
| `replay_150` | 12.85 | 20 | 17 | 0 | 2 | 4 |
| `replay_175` | 12.74 | 19 | 15 | 0 | 3 | 4 |
| `replay_200` | 12.77 | 21 | 15 | 1 | 3 | 4 |

## Read

The knobs helped. Compared with the first guardrail sweep, the replay-balanced
2e-7 run keeps the length center much more stable and avoids broad late-run
drift. Later checkpoints still are not automatically better: validation loss
keeps improving, but behavior presents a tradeoff.

- `replay_50` and `replay_75` are the best length-preserving checkpoints.
- `replay_150` and `replay_175` reduce known malformed flags but give back some
  length.
- `replay_200` should not be selected mechanically because it reintroduces an
  over-20 output.

The next gate should be blind review, probably comparing `len15_step125`,
`replay_75`, and `replay_175` on targeted contrast rows. `replay_75` is the
conservative candidate; `replay_175` is the higher-correction candidate.

## Three-Way Blind Review

Review packet:

`hf_bucket/experiments/mlx_lora/review_comparisons/q5_final_guardrail_overlay_replay_v2_eval_20260507/three_way_blind_review_v1`

Focused set: 31 rows selected from prior `len15_step125` severe cases plus
mechanically flagged changed outputs.

| source | single wins | severe marks | usable marks |
| --- | ---: | ---: | ---: |
| `len15_step125` | 6 | 14 | 17 |
| `replay_75` | 6 | 12 | 19 |
| `replay_175` | 7 | 13 | 18 |

There were 10 all-bad sticky failures and 2 ties. `replay_175` won one more
single row than `replay_75`, but `replay_75` had the better safety profile:
fewest severe marks and most usable outputs.

Recommendation: treat `replay_75` as the tentative production candidate from
this overlay family. Do not select `replay_175` just because validation loss
and one malformed mechanical flag look slightly better; the blind review did
not show a clean behavioral upgrade.

## Fresh q8 vs replay_75 Check

Fresh comparison:

`hf_bucket/experiments/mlx_lora/review_comparisons/q8_vs_q5_guardrail_replay_75_fresh_50_v1`

Blind review:

`hf_bucket/experiments/mlx_lora/review_comparisons/q8_vs_q5_guardrail_replay_75_fresh_50_v1/blind_review_v1`

This used 50 freshly sampled rows from the recent balanced corpus, excluding the
replay-v2 training rows. Generation was greedy with the same `title_abstract`
prompt path.

| source | wins | severe marks | usable marks | avg words | max | over15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q8_promoted` | 19 | 6 | 44 | 12.08 | 18 | 5 |
| `q5_guardrail_replay_75` | 29 | 8 | 42 | 11.82 | 17 | 5 |

There was 1 tie and 1 both-bad row.

Read: `replay_75` is competitive and wins more raw preferences on this fresh
sample, but q8 still has the cleaner safety profile. This argues against
another immediate corrective overlay; the next useful step is a production-gate
eval focused on severe-rate confidence and runtime behavior.

## Fresh 200 Risk Dig

Fresh comparison:

`hf_bucket/experiments/mlx_lora/review_comparisons/q8_vs_q5_guardrail_replay_75_fresh_200_v1`

Mechanical risk screen:

`hf_bucket/experiments/mlx_lora/review_comparisons/q8_vs_q5_guardrail_replay_75_fresh_200_v1/mechanical_risk_screen.md`

Risk blind review:

`hf_bucket/experiments/mlx_lora/review_comparisons/q8_vs_q5_guardrail_replay_75_fresh_200_v1/risk_blind_review_v1`

Full 200-row mechanical screen:

| side | avg words | max | over15 | over20 | num-not-source rows | malformed rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q8` | 12.03 | 21 | 19 | 1 | 5 | 5 |
| `q5` | 12.23 | 23 | 21 | 4 | 3 | 12 |

The mechanical screen selected a 33-row risk packet, mostly q5 mechanical-risk
rows plus q8-risk contrasts.

| source | wins | severe marks | usable marks | avg words | max | over15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q8_promoted` | 17 | 4 | 28 | 12.91 | 21 | 4 |
| `q5_guardrail_replay_75` | 14 | 6 | 22 | 15.42 | 23 | 18 |

Read: average q5 behavior is still competitive, but the risk slice shows a real
remaining production concern. q5's residual severe cases are concentrated in
metric-role binding and malformed repetition, with a few unsupported or
over-scoped claims. This does not look like a broad style/length failure, so a
broad corrective overlay would likely be the wrong tool. If we train again, it
should be a tiny precision overlay built only from confirmed q5 severe rows.

## Precision Overlay Attempt

Follow-up note:

`notes/mlx_lora_270m_q5_precision_failure_overlay_v1.md`

I tried exactly that tiny precision overlay. The unweighted variant used 14
confirmed q5 severe repairs plus 72 clean replay anchors. A weighted variant
repeated the 14 repairs 6 times and used 56 clean anchors.

Result on the target-14 severe fixture:

| run | best checkpoint | looks improved | still bad | needs review |
| --- | ---: | ---: | ---: | ---: |
| unweighted | 50 | 2 | 8 | 4 |
| weighted | 125 | 6 | 8 | 0 |

Read: the patch can nudge some rows, but it does not reliably erase the sticky
bad continuations. Do not promote either precision overlay. If we keep training
this family, the next data step should be a larger set of high-quality examples
covering the same failure modes, not heavier repetition of the same 14 rows.
