# q5 Numeric Hardening Overlay q4mix len15 v2

This run rebuilds the original q5 numeric hardening overlay with stricter
length sensitivity. It starts again from the pre-overlay q5 checkpoint instead
of trying to repair the earlier numeric overlay after the fact.

## Dataset

- Release: `hf_dataset/releases/recent_3y_balanced_10746_q5_numeric_hardening_overlay_q4mix_len15_v2`
- MLX dataset: `hf_dataset/releases/recent_3y_balanced_10746_q5_numeric_hardening_overlay_q4mix_len15_v2/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_glanceable_summary_v1`
- Rows: 285
- Split: train 257, valid 14, test 14
- Summary length: average 12.74 words, max 15 words

Source mix:

- q5-native mined numeric: 130
- q4 prior numeric hardening: 155

The stricter audit started from the same original q5 numeric candidates and old
q4 numeric rows, but rejected any target over 15 words. Relative to the earlier
numeric overlay, this removed most long numeric targets before training rather
than trying to correct length afterward.

## Training

- Config: `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_numeric_hardening_overlay_q4mix_len15_v2_lr_1e_6.yaml`
- Parent adapter: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_current_lora_first_phase_naive_v1/adapters/0002400_adapters.safetensors`
- Run root: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_numeric_hardening_overlay_q4mix_len15_v2_lr_1e_6`
- Learning rate: 1e-6
- Steps: 250

Validation favored the final checkpoint:

| step | val loss |
| ---: | ---: |
| 1 | 1.071 |
| 25 | 0.968 |
| 50 | 0.914 |
| 75 | 0.877 |
| 100 | 0.865 |
| 125 | 0.854 |
| 150 | 0.849 |
| 175 | 0.833 |
| 200 | 0.840 |
| 225 | 0.835 |
| 250 | 0.826 |

Final test loss was 0.747, test perplexity 2.111.

## Mechanical Eval

Eval root:

`hf_bucket/experiments/mlx_lora/review_comparisons/q5_numeric_len15_v2_eval_20260507`

Mechanical report:

`hf_bucket/experiments/mlx_lora/review_comparisons/q5_numeric_len15_v2_eval_20260507/mechanical_eval_report.md`

### Step 125

Step 125 is the current behavior candidate because it has the best
numeric-risk-first mechanical profile:

| eval set | variant | mean words | over15 | over20 | number-not-in-source |
| --- | --- | ---: | ---: | ---: | ---: |
| numeric severe 194 | numeric overlay v1 | 13.54 | 37 | 5 | 13 |
| numeric severe 194 | style anti-regression | 13.22 | 35 | 3 | 13 |
| numeric severe 194 | len15 v2 step 125 | 13.13 | 31 | 3 | 10 |
| general 200 | numeric overlay v1 | 12.79 | 28 | 2 | 3 |
| general 200 | style anti-regression | 12.35 | 19 | 1 | 4 |
| general 200 | len15 v2 step 125 | 12.34 | 17 | 2 | 3 |

Behavior candidate adapter view:

`hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_numeric_hardening_overlay_q4mix_len15_v2_lr_1e_6/mlx_behavior_candidate_125_adapter_view`

### Later Checkpoints

Step 175 and step 250 continue reducing length, but their numeric suspicion
count rises on the numeric-severe set:

- step 175: numeric-severe number-not-in-source 14
- step 250: numeric-severe number-not-in-source 14

Step 250 has the best validation loss and the lowest over-20 count, but it
looks less numerically safe than step 125 by mechanical checks.

## Spot Check

Step 125 still has real bad cases. The mechanically suspicious list includes
some false positives from formatting, such as `2.2 times` becoming `2.2x`, but
also at least these problems:

- `2405.01210`: superflare numeric text is still mangled.
- `2312.03620`: Golden Gemini binds a reduction to parameters when the abstract
  describes EER/minDCF reductions separately from parameter/FLOP reductions.

## Read

This direction looks better than broad re-centering. It gives us a checkpoint
that improves both length and numeric mechanical metrics relative to the
original numeric overlay and the style anti-regression checkpoint.

## Targeted Blind Review

Review packet:

`hf_bucket/experiments/mlx_lora/review_comparisons/q5_numeric_len15_v2_eval_20260507/targeted_blind_review_v1`

Unblinded summary:

`hf_bucket/experiments/mlx_lora/review_comparisons/q5_numeric_len15_v2_eval_20260507/targeted_blind_review_v1/unblinded_review_summary.md`

The review compared `len15_v2` step 125 against `numeric_overlay_v1` and
`style_antireg` on 60 targeted changed rows.

| source | explicit wins | tie usable credit | severe | usable | unusable |
| --- | ---: | ---: | ---: | ---: | ---: |
| `len15_v2` | 19 | 10 | 8 | 47 | 13 |
| `numeric_overlay_v1` | 10 | 9 | 12 | 36 | 24 |
| `style_antireg` | 17 | 7 | 13 | 38 | 22 |

The blind review supports step 125 as the current best q5 corrective candidate.
It is better than both comparison checkpoints on this targeted slice, especially
when the len15 target removed a bad extra number from the v1 output.

It is still not promotion-clean. The remaining failures are concentrated in
number-role binding and malformed repeated scientific tokens. Broad re-centering
is probably the wrong next move; the useful next correction would be a strict,
small overlay focused on number-role binding and malformed token repetition
while keeping the 15-word target cap hard.

## q8 Reorient Eval

Eval root:

`hf_bucket/experiments/mlx_lora/review_comparisons/q8_vs_q5_len15_step125_reorient_100_v1`

Unblinded review summary:

`hf_bucket/experiments/mlx_lora/review_comparisons/q8_vs_q5_len15_step125_reorient_100_v1/blind_review_packets/unblinded_review_summary.md`

This was a fresh 100-row holdout comparison against promoted q8. Sides were
randomized before review and unblinded after decisions were written.

| source | wins | tie usable credit | severe | usable | unusable |
| --- | ---: | ---: | ---: | ---: | ---: |
| `q8` | 48 | 2 | 19 | 77 | 23 |
| `q5_len15` | 46 | 2 | 21 | 70 | 30 |

Both bad: 4. Ties: 2.

Length was slightly better for q5 len15:

| generated source | avg audit words | max | over15 | over20 |
| --- | ---: | ---: | ---: | ---: |
| `q8_promoted` | 13.16 | 28 | 19 | 1 |
| `q5_len15_step125` | 12.61 | 19 | 15 | 0 |

Read: q5 len15 is close on preference but still behind q8 on severe and usable
counts. This does not look like a length problem anymore; the remaining
production blockers are malformed text, polarity flips, overclaims, and
number/entity binding.

## Follow-up Guardrail Overlay

Follow-up note:

`notes/mlx_lora_270m_q5_final_guardrail_overlay_v1.md`

A high-confidence harvested overlay produced 164 rows covering the remaining
failure classes. The first 5e-7 sweep suggests the useful checkpoint is shallow:
`guardrail_25` has the best mechanical profile, while later checkpoints drift in
length and introduce fresh errors despite better validation loss.
