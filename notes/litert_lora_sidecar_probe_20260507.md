# LiteRT LoRA Sidecar Probe

## Summary

Tested whether the current promoted adapter can enter the documented
MediaPipe/LiteRT LoRA sidecar conversion path.

Inputs:

- base checkpoint: `google/gemma-3-270m-it` safetensors snapshot
- adapter: `hf_bucket/experiments/litert_exports/faith_overlay_1400_v2_final_600/peft_adapter`
- converter model type: `GEMMA3_300M`
- backend: `gpu`
- LoRA rank: `16`
- LoRA alpha: `160`

Probe script:

- `scripts/probe_litert_lora_sidecar_conversion.py`

Outputs:

- `hf_bucket/experiments/litert_lora_sidecar_probe_20260507/conversion_result.json`
- `hf_bucket/experiments/litert_lora_sidecar_probe_20260507/packaging_attempts.json`
- `hf_bucket/experiments/litert_lora_sidecar_probe_20260507/weight_bin_summary.json`

## Result

The adapter does pass the Python-side LoRA configuration checks, and the
converter maps the adapter into LiteRT-style LoRA weight bins.

Evidence from the failed conversion work directory before cleanup:

- base weight bins: `364`
- LoRA weight bins: `144`
- LoRA weight-bin payload: `5,898,240` bytes
- sample mapped LoRA names:
  - `params.lm.transformer.x_layers_0.self_attention.q.w_prime_left`
  - `params.lm.transformer.x_layers_0.self_attention.q.w_prime_right`
  - `params.lm.transformer.x_layers_0.self_attention.k.w_prime_left`
  - `params.lm.transformer.x_layers_0.self_attention.k.w_prime_right`

The final flatbuffer packaging fails before producing either the base TFLite or
the LoRA TFLite:

```text
NotImplementedError: Image encoder not supported in this build.
```

I also tried reusing the already-written weight bins with:

- plain `GEMMA3_300M`
- `GEMMA3_300M` with `submodel_type` and `lora_main_model_type`
- a mutated `FUNCTIONGEMMA3_300M` final packaging attempt

All hit the same final-packaging error.

## Interpretation

This is not a simple "our adapter is incompatible" result. The current adapter
shape is good for the documented LoRA path: attention-only, rank 16, PEFT
format. The failure appears lower in the MediaPipe converter's GPU TFLite
packager for `GEMMA3_300M`, where the current pip build routes into an image
encoder path that is not available.

So LiteRT sidecar remains plausible, but not through the current public
MediaPipe pip converter as exercised here.

## Cleanup

The generated `converter_work/` directory was about `263M` and was removed after
recording the JSON evidence. The remaining probe directory is small.
