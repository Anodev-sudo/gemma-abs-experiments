# q5 Four-Stress Overfit Diagnostic v1

Diagnostic date: 2026-05-15

Purpose: test whether LoRA-only SFT can overwrite the hardest q5 llama.cpp numeric/value-attractor completions when trained from the promoted checkpoint-700 adapter.

This is not a promotion candidate.

## Inputs

- Starting adapter: `hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/checkpoint-700`
- Runtime base: `hf_bucket/experiments/mlx_lora/runtime_shift/llama_cpp_q4_lora_sidecar_benchmark_20260507/gguf_base/gemma_3_270m_it.Q5_K_M.gguf`
- Eval path: llama.cpp Q5_K_M base plus f16 GGUF LoRA sidecar
- Builder: `scripts/build_q5_four_stress_overfit_diag_v1_dataset.py`
- Eval driver: `scripts/eval_q5_four_stress_overfit_diag_v1.py`
- Four stress rows: `1901.06548`, `2111.10635`, `2303.09722`, `2406.04373`
- Labels: Kimi-authored and Qwen-accepted rows from the hard-attractor booster run

The builder repeated each stress row 50 times and used the exact runtime title-plus-abstract prompt format. The training prompt was checked against `scripts/generate_llama_stream.py`; the rendered prompt strings matched exactly.

## Low-LR Diagnostic

Config: `configs/sft/q5_step2400_sft700_four_stress_overfit_diag_v1_lr2e6.yaml`

Result: not enough pressure.

- Output root: `hf_bucket/experiments/sft/q5_step2400_sft700_four_stress_overfit_diag_v1_lr2e6`
- Learning rate: `2e-6`
- Max steps: `100`
- Full assistant target weight: `8x`
- Crucial span weight: `12x`
- Runtime result at checkpoint 100: `3/4` rows still contained digits; `0/4` exact targets
- PEFT-side checkpoint 100 also failed, so this was not a GGUF conversion-only issue

## Aggressive-LR Diagnostic

Config: `configs/sft/q5_step2400_sft700_four_stress_overfit_diag_v1_lr5e5.yaml`

Result: the hard completions are physically overwriteable by LoRA-only SFT.

- Output root: `hf_bucket/experiments/sft/q5_step2400_sft700_four_stress_overfit_diag_v1_lr5e5`
- Learning rate: `5e-5`
- Max steps: `100`
- Full assistant target weight: `8x`
- Crucial span weight: `12x`
- Loss collapsed below `1.0` by roughly step 25 and near zero by roughly step 40
- Runtime checkpoint 20: `0/4` digit rows, `2/4` exact targets, including `2303.09722`
- Runtime checkpoint 30: `0/4` digit rows, `4/4` exact targets
- Runtime checkpoint 50: `0/4` digit rows, `4/4` exact targets
- Runtime checkpoint 100: `0/4` digit rows, `4/4` exact targets

## Interpretation

The previous broad SFT attempts were likely too diluted and too low-pressure to break the hardest q5 attractors. The q5 llama.cpp deployment path can express the corrected summaries once the LoRA is pushed hard enough; the failure mode is not a fundamental inability of Q5 plus f16 sidecar LoRA.

The next real experiment should use this as a curriculum signal, not as a candidate:

1. Start with a short high-pressure hard-attractor burn-in.
2. Immediately regularize against clean normal/style anchors and broader ideal numeric rows.
3. Evaluate whether the hard flips survive after the regularization phase.
4. Promote only if the broader numeric-risk pool improves without obvious style collapse.
