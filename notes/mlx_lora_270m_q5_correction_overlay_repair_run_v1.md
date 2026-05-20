# q5 Correction Overlay Repair Run v1

## Purpose

Run a small corrective MLX LoRA overlay using the repaired q5 severe-case rows.
This continues from the selected q5 first-phase checkpoint and trains only on
the high-confidence Kimi/Qwen correction overlay release.

## Inputs

- Training release: `hf_dataset/releases/recent_3y_balanced_10746_q5_kimi_qwen_correction_overlay_repair_v1`
- MLX dataset: `hf_dataset/releases/recent_3y_balanced_10746_q5_kimi_qwen_correction_overlay_repair_v1/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_glanceable_summary_v1`
- Training rows: 82
- MLX split: train 74, valid 4, test 4
- Parent checkpoint: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_current_lora_first_phase_naive_v1/adapters/0002400_adapters.safetensors`

The parent run path still contains `q4_current` because the q5 naming happened
after selecting the naive first-phase checkpoint. The selected checkpoint is the
q5 deployment-lineage LoRA we called `q5_best_2400_first_phase_naive`.

## Training Config

- Config: `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_correction_overlay_repair_v1_lr_1e_6.yaml`
- Base model: `hf_bucket/experiments/mlx_lora/quantized_models/google_gemma_3_270m_it/q4_g64`
- LR: `1e-6`
- Iterations: 200
- Effective batch size: 8
- Save every: 50
- Eval every: 25
- LoRA targets: attention projections only, rank 16, scale 10

## Results

Validation curve:

| Step | Val loss |
|---:|---:|
| 1 | 1.858 |
| 25 | 1.799 |
| 50 | 1.733 |
| 75 | 1.708 |
| 100 | 1.701 |
| 125 | 1.705 |
| 150 | 1.709 |
| 175 | 1.725 |
| 200 | 1.715 |

Final test:

- Test loss: 1.405
- Test perplexity: 4.074

Selected checkpoint:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_correction_overlay_repair_v1_lr_1e_6/adapters/0000100_adapters.safetensors`
- Reason: lowest validation loss during the 200-step overlay run.

Convenience adapter view:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_correction_overlay_repair_v1_lr_1e_6/mlx_best_100_adapter_view`

This view contains the selected checkpoint as `adapters.safetensors` plus the
matching `adapter_config.json`, so downstream sidecar/conversion probes can
point at a normal MLX adapter directory without accidentally using the final
step-200 adapter.

## Next Step

Evaluate `0000100` against the q5 first-phase checkpoint and promoted q8 on the
same q5 severe/mining samples. This run fit the repair rows cleanly, but the
real question is whether it reduces q5 severe drift without damaging the broad
summary style or word-count behavior.
