# MLX LoRA 270M q4 Lineage Redo

## Goal

Test whether training LoRA adapters against a raw q4 MLX base can recover enough
quality to make a smaller q4 deployment path viable.

This is deliberately separate from the current promoted q8 lineage. Do not
update the promoted pointer unless a later review explicitly justifies it.

## Source Lineage

- Base model: `google/gemma-3-270m-it`
- Raw q4 base target: `hf_bucket/experiments/mlx_lora/quantized_models/google_gemma_3_270m_it/q4_g64`
- Base-aggressive data: `hf_dataset/releases/recent_3y_balanced_10746_v1/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_glanceable_summary_v1`
- Overlay data: `hf_dataset/releases/recent_3y_balanced_10746_faithfulness_overlay_1400_v2/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_glanceable_summary_v1`

## Planned Runs

1. q4 base-aggressive replay:
   - Config: `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_base_aggressive_v1.yaml`
   - Adapter dir: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_base_aggressive_v1/adapters`
2. q4 overlay sweep:
   - `2.5e-6`: `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_faithfulness_overlay_1400_v2_lr_2_5e_6.yaml`
   - `5e-6`: `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_faithfulness_overlay_1400_v2_lr_5e_6.yaml`
   - `1e-5`: `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_faithfulness_overlay_1400_v2_lr_1e_5.yaml`

## Commands

Convert raw q4 base:

```bash
python3 scripts/prepare_mlx_quantized_models.py \
  --source-hf-model hf_bucket/experiments/mlx_lora/.hf-home/hub/models--google--gemma-3-270m-it/snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3 \
  --output-root hf_bucket/experiments/mlx_lora/quantized_models \
  --label google_gemma_3_270m_it \
  --q-bits 4 \
  --trust-remote-code
```

Train q4 base-aggressive:

```bash
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_base_aggressive_v1.yaml \
  --mask-prompt
```

Train each overlay sweep member:

```bash
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  -m mlx_lm.lora \
  --config <overlay config> \
  --mask-prompt
```

## Results

### Raw q4 Base

- Output: `hf_bucket/experiments/mlx_lora/quantized_models/google_gemma_3_270m_it/q4_g64`
- Manifest: `hf_bucket/experiments/mlx_lora/quantized_models/google_gemma_3_270m_it/mlx_quantization_manifest.json`
- Total bytes: `184,362,855`
- Quantization: `4` bits, group size `64`

### q4 LoRA Smoke

- A 2-iteration throwaway LoRA smoke train against the raw q4 base completed
  successfully outside repo artifacts.
- Result: MLX can train LoRA adapters on the q4 base.

### q4 Base-Aggressive Replay

- Config: `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_base_aggressive_v1.yaml`
- Log: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_base_aggressive_v1/train.log`
- Adapter dir: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_base_aggressive_v1/adapters`
- Final checkpoint: `0003600_adapters.safetensors`
- Best validation checkpoint: `0003600_adapters.safetensors`
- Final validation loss: `1.870`
- Test loss / ppl: `1.848` / `6.346`
- Peak memory: `7.262 GB`
- Checkpoints were written every `200` iterations from `0000200` through
  `0003600`.

### q4 Overlay Sweep

All sweep members resumed from:
`hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_base_aggressive_v1/adapters/0003600_adapters.safetensors`.

| LR | Config | Best checkpoint | Best val | Final val | Test loss / ppl |
|---:|---|---|---:|---:|---:|
| `2.5e-6` | `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_faithfulness_overlay_1400_v2_lr_2_5e_6.yaml` | `0000600` | `1.542` | `1.542` | `1.725` / `5.614` |
| `5e-6` | `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_faithfulness_overlay_1400_v2_lr_5e_6.yaml` | `0000600` | `1.544` | `1.544` | `1.729` / `5.633` |
| `1e-5` | `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_faithfulness_overlay_1400_v2_lr_1e_5.yaml` | `0000200` | `1.559` | `1.565` | `1.748` / `5.744` |

The sweep metric winner is the `2.5e-6` final checkpoint:
`hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_faithfulness_overlay_1400_v2_lr_2_5e_6/adapters/0000600_adapters.safetensors`.

## Generation Sanity

- Preview bundle, low vs mid LR:
  `hf_bucket/experiments/mlx_lora/review_comparisons/previews/q4_lr_2_5e_6_600_vs_q4_lr_5e_6_600_preview20`
- Preview bundle, low LR vs high-LR best checkpoint:
  `hf_bucket/experiments/mlx_lora/review_comparisons/previews/q4_lr_2_5e_6_600_vs_q4_lr_1e_5_200_preview20`
- Empty summaries: `0` in both preview bundles.
- Selected q4-trained candidate:
  `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_faithfulness_overlay_1400_v2_lr_2_5e_6/adapters/0000600_adapters.safetensors`

Rationale: `2.5e-6@600` is the metric winner and looked cleaner than the nearby
`5e-6@600` preview on obvious entity/topic drifts. The `1e-5` run showed worse
validation/test loss and visible numeric/entity corruption in the preview.

## Blinded Reviews

### q4-Trained vs Current q8

| Bundle | Sample | q4-trained side | q8 side | q4-trained wins | q8 wins | Tie | Both bad | Severe q4-trained | Severe q8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `model_e_vs_model_f_holdout_100_q4lineage` | reused fixed holdout | Left | Right | `38` | `46` | `8` | `8` | `11` | `8` |
| `model_g_vs_model_h_fresh_100_q4lineage_seed_20260501` | fresh sample, seed `20260501` | Right | Left | `48` | `44` | `7` | `1` | `13` | `5` |
| `model_m_vs_model_n_fresh_100_q4lineage_seed_2026050102_no_prior` | fresh sample, seed `2026050102`, excluding prior q8/q4-trained comparison ids | Right | Left | `44` | `45` | `7` | `4` | `17` | `10` |

Combined first two runs:

- q4-trained wins: `86`
- q8 wins: `90`
- Tie: `15`
- Both bad: `9`
- Severe q4-trained issues: `24`
- Severe q8 issues: `13`

Combined all three q8 comparison runs:

- q4-trained wins: `130`
- q8 wins: `135`
- Tie: `22`
- Both bad: `13`
- Severe q4-trained issues: `41`
- Severe q8 issues: `23`

The q4-trained model is close on preference votes, but it does not meet the
predeclared competitive bar because severe q4-trained issues are about `1.78x`
the q8 severe count across all three blinded q8 comparisons.

### q4-Trained vs Post-Hoc q4

| Bundle | Sample | q4-trained side | post-hoc q4 side | q4-trained wins | post-hoc q4 wins | Tie | Both bad | Severe q4-trained | Severe post-hoc q4 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `model_i_vs_model_j_holdout_100_q4lineage_posthoc_check` | reused fixed holdout | Right | Left | `71` | `20` | `3` | `6` | `10` | `30` |
| `model_k_vs_model_l_fresh_100_q4lineage_posthoc_check_seed_20260501` | fresh sample, seed `20260501` | Left | Right | `76` | `21` | `0` | `3` | `11` | `35` |

Combined:

- q4-trained wins: `147`
- post-hoc q4 wins: `41`
- Tie: `3`
- Both bad: `9`
- Severe q4-trained issues: `21`
- Severe post-hoc q4 issues: `65`

This clears the predeclared promising bar: severe q4 failures fell by more than
half versus the post-hoc quantized merged model.

## Conclusion

q4-aware training helped a lot and is worth keeping as an experiment lineage.
It is not a replacement candidate for the current q8 promoted model yet because
the severe-error rate is still too high.

No promoted pointer was updated.
