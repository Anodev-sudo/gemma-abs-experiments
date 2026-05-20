# MLX-Swift q4 Deployment Path Probe

## Summary

Checked two alternatives to the behavior-breaking q4 requantized fused artifact:

1. MLX-Swift loading the q4 base plus a LoRA sidecar.
2. A mixed MLX artifact with untouched tensors kept q4 and LoRA-touched
   attention projections stored as dense fused weights.

Both are much better than the old q4 fused artifact on the 11-row parity
fixture set.

## Results

| Path | Prompt mismatches | Generated token matches | Generated token mismatches | Text mismatches |
| --- | ---: | ---: | ---: | ---: |
| MLX-Swift q4 base + LoRA sidecar | 0 | 9 | 2 | 2 |
| MLX-Swift mixed q4/dense fused | 0 | 9 | 2 | 2 |
| MLX Python mixed q4/dense fused | 0 | 7 | 4 | 4 |
| MLX-Swift old q4 requantized fused | 0 | 0 | 11 | 11 |

The first MLX-Swift sidecar mismatch against canonical was a late reorder:

- paper: `2403.19682`
- first differing generated token index: `15`
- canonical token/text: `528` / ` in`
- sidecar token/text: `684` / ` by`
- shared prefix text:
  `Drude's error of a factor of two and Lorentz correction were removed`

## Sidecar Notes

MLX-Swift-LM has runtime LoRA adapter support through `LoRAContainer` and
`model.load(adapter:)`.

The Python adapter config uses `num_layers: -1` to mean all layers. The Swift
LoRA loader uses `suffix(numLayers)`, so the probe used a copied adapter config
with `num_layers` normalized to `18`, matching this Gemma 270M model's layer
count.

Probe adapter copy:

- `hf_bucket/experiments/mlx_lora/runtime_shift/mlx_swift_parity_root_cause_v1/mlx_swift_lora_sidecar_adapter`
- size: about `5.6M`

## Mixed Artifact Notes

The mixed artifact was built by:

- creating a dequantized fused source from q4 base plus adapter
- copying the q4 base artifact
- replacing only LoRA-touched attention projection weights with dense fused
  weights
- omitting those projections' q4 `scales` and `biases` tensors, so MLX loaders
  leave those modules dense while keeping untouched modules q4

Mixed artifact:

- `hf_bucket/experiments/mlx_lora/runtime_shift/mlx_swift_parity_root_cause_v1/mixed_q4_dense_lora_attention_fused`
- size: about `216M`

## Interpretation

The q4 base plus LoRA sidecar is the best size/behavior candidate. It keeps the
q4 base size and adds only a small adapter. It also avoids the damaging
post-fusion q4 requantization step.

The mixed artifact is a plausible fallback if app-side sidecar loading is
awkward, but it is larger and did not beat sidecar behavior on this fixture set.

## Artifacts

- `hf_bucket/experiments/mlx_lora/runtime_shift/mlx_swift_parity_root_cause_v1/deployment_path_probe_report.md`
- `hf_bucket/experiments/mlx_lora/runtime_shift/mlx_swift_parity_root_cause_v1/deployment_path_probe_report.json`
- `hf_bucket/experiments/mlx_lora/runtime_shift/mlx_swift_parity_root_cause_v1/swift_traces/mlx_swift_q4_base_lora_sidecar.jsonl`
- `hf_bucket/experiments/mlx_lora/runtime_shift/mlx_swift_parity_root_cause_v1/swift_traces/mlx_swift_mixed_q4_dense_lora_attention_fused.jsonl`
- `hf_bucket/experiments/mlx_lora/runtime_shift/mlx_swift_parity_root_cause_v1/python_traces/mlx_python_mixed_q4_dense_lora_attention_fused_stream.jsonl`
