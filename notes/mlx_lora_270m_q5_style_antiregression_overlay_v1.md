# q5 Style Anti-Regression Overlay v1

This run starts from the q5 numeric hardening overlay and applies a smaller
style-preserving correction pass. The goal was to keep the numeric gains while
pulling summaries back toward the promoted model's concise 10-15 word style.

## Dataset

- Release: `hf_dataset/releases/recent_3y_balanced_10746_q5_style_antiregression_overlay_v1`
- MLX dataset: `hf_dataset/releases/recent_3y_balanced_10746_q5_style_antiregression_overlay_v1/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_glanceable_summary_v1`
- Rows: 222
- Split: train 200, valid 11, test 11
- Summary length: average 12.11 words, max 16 words

Source mix:

- promoted q8 style anchors: 110
- general style drift rows using pre-q5 targets: 34
- blind-review numeric rehearsal rows: 24
- q4 numeric rehearsal rows: 23
- mechanical numeric rehearsal rows: 16
- blind-review overlay regression rows: 10
- q8 target numeric repair rows: 5

The materializer rejected 40 candidate rows:

- too long: 19
- target number not found in source by audit regex: 17
- high title overlap: 9

Some rows had more than one reject reason.

## Training

- Config: `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_style_antiregression_overlay_v1_lr_5e_7.yaml`
- Parent adapter: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_numeric_hardening_overlay_q4mix_v1_lr_1e_6/adapters/0000400_adapters.safetensors`
- Run root: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_style_antiregression_overlay_v1_lr_5e_7`
- Learning rate: 5e-7
- Steps: 250

Validation loss favored an early checkpoint:

| step | val loss |
| ---: | ---: |
| 1 | 1.085 |
| 25 | 1.061 |
| 50 | 1.051 |
| 75 | 1.061 |
| 100 | 1.057 |
| 125 | 1.067 |
| 150 | 1.060 |
| 175 | 1.063 |
| 200 | 1.069 |
| 225 | 1.059 |
| 250 | 1.068 |

Best checkpoint:

`hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_style_antiregression_overlay_v1_lr_5e_7/adapters/0000050_adapters.safetensors`

Adapter view:

`hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_style_antiregression_overlay_v1_lr_5e_7/mlx_best_50_adapter_view`

Step-50 test loss with prompt masking was 1.013, test perplexity 2.753.
Final step-250 test loss was 0.990, test perplexity 2.690, but validation
selected step 50 for generation eval.

## Mechanical Eval

Eval root:

`hf_bucket/experiments/mlx_lora/review_comparisons/q5_style_antiregression_eval_20260507`

### Numeric Severe 194

| variant | mean words | median | max | over15 | over20 | number rows | number-not-in-source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pre_q5 | 13.21 | 13 | 36 | 30 | 5 | 92 | 52 |
| numeric_overlay | 13.54 | 13 | 25 | 37 | 5 | 62 | 13 |
| style_antireg | 13.22 | 13 | 25 | 35 | 3 | 68 | 13 |

### General 200

| variant | mean words | median | max | over15 | over20 | number rows | number-not-in-source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pre_q5 | 12.31 | 12 | 20 | 13 | 0 | 28 | 9 |
| numeric_overlay | 12.79 | 12 | 24 | 28 | 2 | 21 | 3 |
| style_antireg | 12.35 | 12 | 21 | 19 | 1 | 21 | 4 |

## Read

The step-50 checkpoint does tighten the length/style drift introduced by the
numeric overlay. It lowers general-set mean word count and over-15/over-20
counts, and it also reduces numeric-severe over-20 cases.

It is not a clean promotion by mechanical checks alone. The numeric-severe set
keeps the improved number-not-in-source count versus pre-q5, but it still has
13 mechanically suspicious numeric rows and introduces a few changed numeric
phrases that need targeted review.

Spot-checking the new mechanically suspicious numeric rows found mixed evidence:

- `2509.11560`: `35 times` became `35x`; this is supported.
- `2603.01415`: `31.40%` became `31.4%`; this is supported.
- `2505.22654`: `2.91x` appears supported by the abstract continuation, but still needs review because the phrasing is awkward.
- `2407.18818`: the style checkpoint produced a bad trailing threshold chain: `10-13-16-19-20-23-24-25-26`.

The next decision should be a small blind review of the changed rows where
style_antireg differs from numeric_overlay, biased toward:

- numeric-severe rows with new or changed numeric literals
- general rows where style_antireg is shorter than numeric_overlay
- the remaining over-20 rows
