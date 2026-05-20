# q5 Numeric Hardening Overlay q4mix v1

This run targets the q5 numeric-failure cluster found in the 6000-row blind
review. It starts from the pre-overlay q5 checkpoint and trains a focused
overlay with q5-native numeric failures plus old q4 numeric hardening rows.

## Dataset

- Release: `hf_dataset/releases/recent_3y_balanced_10746_q5_numeric_hardening_overlay_q4mix_v1`
- MLX dataset: `hf_dataset/releases/recent_3y_balanced_10746_q5_numeric_hardening_overlay_q4mix_v1/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_glanceable_summary_v1`
- Rows: 353
- Split: train 317, valid 18, test 18
- Source mix:
  - q5-native mined numeric: 165
  - q4 prior numeric hardening: 188
- Summary length: average 13.61 words, max 20 words

The q5 target audit started with 194 q5 numeric severe/q8-clean rows and kept
165. It rejected 29 rows:

- high title overlap: 14
- q8 target number not found in source by the audit regex: 11
- too long: 4

Old q4 numeric hardening rows were also sanity-filtered for length/malformed
output. Two were rejected for being too long.

## Training

- Config: `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_numeric_hardening_overlay_q4mix_v1_lr_1e_6.yaml`
- Parent adapter: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_current_lora_first_phase_naive_v1/adapters/0002400_adapters.safetensors`
- Run root: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_numeric_hardening_overlay_q4mix_v1_lr_1e_6`
- Learning rate: 1e-6
- Steps: 400

Validation loss improved through the run:

| step | val loss |
| ---: | ---: |
| 1 | 1.117 |
| 50 | 0.938 |
| 100 | 0.893 |
| 150 | 0.875 |
| 200 | 0.862 |
| 250 | 0.845 |
| 300 | 0.846 |
| 350 | 0.843 |
| 400 | 0.841 |

Best checkpoint by validation loss:

`hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_numeric_hardening_overlay_q4mix_v1_lr_1e_6/adapters/0000400_adapters.safetensors`

Adapter view:

`hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_numeric_hardening_overlay_q4mix_v1_lr_1e_6/mlx_best_400_adapter_view`

Test loss: 0.730, test perplexity: 2.074.

## Next Check

Evaluate this checkpoint against the pre-overlay q5 checkpoint on:

- the q5 numeric severe shortlist,
- the q5 6000-row mined holdout summary metrics,
- a small general feed sample to catch word-count or style drift.
