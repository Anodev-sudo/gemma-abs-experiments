# q5 Broad Re-Centering Overlay v1

This was a short, low-learning-rate re-centering pass from the q5 style
anti-regression step-50 checkpoint. The goal was to see whether a broader
promoted-style dataset could pull behavior back toward the q8 style without
losing the numeric hardening.

## Dataset

- Release: `hf_dataset/releases/recent_3y_balanced_10746_q5_recenter_broad_overlay_v1`
- MLX dataset: `hf_dataset/releases/recent_3y_balanced_10746_q5_recenter_broad_overlay_v1/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_glanceable_summary_v1`
- Rows: 626
- Split: train 564, valid 31, test 31
- Summary length: average 11.97 words, max 15 words

Source mix:

- promoted q8 style anchors: 499
- q4 numeric rehearsal rows: 44
- general style drift rows using pre-q5 targets: 34
- blind-review numeric rehearsal rows: 19
- mechanical numeric rehearsal rows: 15
- blind-review overlay regression rows: 10
- q8 target numeric repair rows: 5

## Training

- Config: `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_recenter_broad_overlay_v1_lr_2e_7.yaml`
- Parent adapter: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_style_antiregression_overlay_v1_lr_5e_7/adapters/0000050_adapters.safetensors`
- Run root: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_recenter_broad_overlay_v1_lr_2e_7`
- Learning rate: 2e-7
- Steps: 75

Validation improved across the short run:

| step | val loss |
| ---: | ---: |
| 1 | 1.500 |
| 25 | 1.492 |
| 50 | 1.474 |
| 75 | 1.470 |

The final checkpoint had test loss 1.580 and test perplexity 4.857 on this
broader re-centering split.

## Mechanical Eval

Eval root:

`hf_bucket/experiments/mlx_lora/review_comparisons/q5_recenter_broad_eval_20260507`

Mechanical checkpoint report:

`hf_bucket/experiments/mlx_lora/review_comparisons/q5_recenter_broad_eval_20260507/checkpoint_25_50_75_mechanical_report.md`

### Best General-Style Checkpoint: Step 50

Step 50 improved general length relative to the style anti-regression checkpoint:

- general mean words: 12.35 -> 12.21
- general over-15: 19 -> 15
- general over-20: 1 -> 1
- general number-not-in-source: 4 -> 3

But it worsened the numeric-severe set:

- numeric-severe mean words: 13.22 -> 13.32
- numeric-severe over-15: 35 -> 42
- numeric-severe over-20: 3 -> 4
- numeric-severe number-not-in-source: 13 -> 14

### Least Numeric-Damaging Checkpoint: Step 25

Step 25 did less damage numerically than steps 50 or 75, but did not improve
general style:

- general over-15: 19 -> 23
- numeric-severe over-15: 35 -> 36
- numeric-severe over-20: 3 -> 5
- numeric-severe number-not-in-source: 13 -> 14

## Read

The broader re-centering pass is useful evidence, but not a better candidate.
It can push the general style profile in the right direction, but the tradeoff
shows up immediately in the numeric-severe set. The current best candidate
remains the q5 style anti-regression step-50 checkpoint, pending targeted blind
review.
