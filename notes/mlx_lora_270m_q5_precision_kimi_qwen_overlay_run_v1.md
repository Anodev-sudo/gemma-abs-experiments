# q5 precision Kimi/Qwen overlay training run v1

Date: 2026-05-07

## Goal

Run one final small q5 precision overlay before pausing, using the strict
Kimi/Qwen-reviewed rows from:

- `hf_dataset/releases/recent_3y_balanced_10746_q5_precision_kimi_qwen_overlay_v1/training_rows.jsonl`

## Dataset

MLX dataset:

- `hf_dataset/releases/recent_3y_balanced_10746_q5_precision_kimi_qwen_overlay_v1/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_glanceable_summary_v1`

Split:

- train: 65
- valid: 8
- test: 8

## Training config

Config:

- `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_precision_kimi_qwen_overlay_v1_lr_1e_7.yaml`

Parent adapter:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_final_guardrail_overlay_replay_v2_lr_2e_7/adapters/0000075_adapters.safetensors`

Run:

- learning rate: `1e-7`
- iterations: 100
- save every: 25
- batch size: 4
- grad accumulation: 2
- LoRA rank: 16
- LoRA scale: 10.0

## Outputs

Run root:

- `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q5_precision_kimi_qwen_overlay_v1_lr_1e_7`

Checkpoints:

- `adapters/0000025_adapters.safetensors`
- `adapters/0000050_adapters.safetensors`
- `adapters/0000075_adapters.safetensors`
- `adapters/0000100_adapters.safetensors`
- `adapters/adapters.safetensors`

## Loss

Validation:

- step 1: 1.374
- step 25: 1.388
- step 50: 1.367
- step 75: 1.369
- step 100: 1.369

Test:

- loss: 1.444
- perplexity: 4.237

## Read

By validation loss, step 50 is the first checkpoint to inspect. Steps 75 and
100 are essentially tied with it, so behavior review should decide. I would not
promote from loss alone; the next step is a small q8/replay_75/this-run behavior
comparison on the known precision-risk rows plus a small fresh slice.

## Quick Behavior Check

Artifacts:

- `hf_bucket/experiments/mlx_lora/review_comparisons/q5_replay75_vs_precision_kq50_fresh50_check_v1`
- `hf_bucket/experiments/mlx_lora/review_comparisons/q5_precision_kq_checkpoint_known_severe_check_v1/checkpoint_known_severe_screen.md`

Step 50 was checked against `q5_replay75` on the existing fresh-50 slice, then
steps 25, 50, 75, and 100 were checked on the 8 known q5-severe rows from that
slice.

Fresh-50 mechanical screen:

- `q5_replay75`: avg 11.86 words, max 17, over15 5, num-not-source 1, malformed 1
- `q5_precision_kq_50`: avg 12.20 words, max 19, over15 5, num-not-source 1, malformed 1
- identical outputs: 34/50

Known q5-severe checkpoint screen:

- step 25: changed 2/8, over15 4, malformed 3, num-not-source 1
- step 50: changed 4/8, over15 4, malformed 2, num-not-source 1
- step 75: changed 3/8, over15 4, malformed 3, num-not-source 1
- step 100: changed 5/8, over15 4, malformed 3, num-not-source 1

Read from the known severe rows:

- Step 50 is the least bad by this small mechanical screen, but it is not good.
- It slightly improves one malformed/repetitive entropy row mechanically, but
  the resulting sentence is still semantically awkward.
- It does not fix the eBPF metric-binding row, the Big Bang polarity row, or the
  outage MTTD/outage conflation row.
- It worsens the class-number row by adding an overbroad "all odd natural
  numbers" claim.
- It worsens the Pluto row by introducing "last century."
- Steps 25, 75, and 100 do not rescue the pattern.

Do not promote this overlay from the quick check. If we resume, the useful
artifact is the reviewed dataset itself, not this trained checkpoint.
