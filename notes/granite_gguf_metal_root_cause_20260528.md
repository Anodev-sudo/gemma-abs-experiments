# Granite GGUF Metal Offload Root Cause Probe

Date: 2026-05-28

## Summary

The Granite GGUF repeated-`@` failure is not primarily a Q5/Q6 quantization problem. It reproduces on the F16 GGUF with the same layer boundary, and it is fixed by keeping one specific tensor on CPU:

`^blk\.2\.ffn_down\.weight=CPU`

The concrete local root cause is therefore: llama.cpp Metal execution corrupts Granite generation when `blk.2.ffn_down.weight` is offloaded. Full Metal offload hits this path; CPU-only and slightly partial offload avoid it.

## Environment

- llama.cpp: Homebrew build `9020 (a4701c98f)`
- Backend: Metal
- Probe model family: Granite 4.0 350M v5 checkpoint 425 GGUF exports
- Probe artifacts:
  - `hf_bucket/experiments/llama_cpp/granite_4_0_350m_v5_ckpt425_gguf_q6_q5_v1/root_cause_gpu_offload_20260528/`

## Key Evidence

First, broad toggles did not fix Q6_K full offload:

| Run | Result |
| --- | --- |
| CPU-only | normal output |
| Metal default | `@@@@@@@@...` |
| `--no-kv-offload` | `@@@@@@@@...` |
| `--no-op-offload` | `@@@@@@@@...` |
| `--flash-attn off` | `@@@@@@@@...` |
| `--no-repack` | `@@@@@@@@...` |
| `--fit off` | `@@@@@@@@...` |
| `GGML_METAL_FUSION_DISABLE=1` | `@@@@@@@@...` |
| `GGML_METAL_GRAPH_OPTIMIZE_DISABLE=1` | `@@@@@@@@...` |
| `GGML_METAL_CONCURRENCY_DISABLE=1` | `@@@@@@@@...` |
| `GGML_METAL_BF16_DISABLE=1` | `@@@@@@@@...` |

Then the `--n-gpu-layers` threshold made the failure specific:

| GGUF | `ngl=26` | `ngl=27` | `ngl=27` plus `blk.2.ffn_down` on CPU |
| --- | --- | --- | --- |
| Q5_K_M | normal | `@@@@` | normal |
| Q6_K | normal | `@@@@` | normal |
| F16 | normal | `@@@@` | normal |

For Q6_K, exact block overrides showed the failing block:

| Override at `ngl=27` | Result |
| --- | --- |
| `^blk\.2\..*=CPU` | normal |
| `^blk\.3\..*=CPU` | `@@@@` |
| `^blk\.4\..*=CPU` | `@@@@` |
| `^blk\.25\..*=CPU` | `@@@@` |
| `^blk\.26\..*=CPU` | `@@@@` |

Within block 2:

| Override at `ngl=27` | Result |
| --- | --- |
| `^blk\.2\.attn_.*=CPU` | `@@@@` |
| `^blk\.2\.ffn_.*=CPU` | normal |
| `^blk\.2\.ffn_gate\.weight=CPU` | `@@@@` |
| `^blk\.2\.ffn_up\.weight=CPU` | `@@@@` |
| `^blk\.2\.ffn_down\.weight=CPU` | normal |

The tensor metadata is ordinary:

- `blk.2.ffn_down.weight`
- dims: `[2048, 1024]`
- Q6 type in the Q6_K file: `Q6_K`
- All `ffn_down.weight` tensors share the same type in the Q6_K file.

## Interpretation

`--n-gpu-layers 26` offloads the output layer plus 25 repeating layers and works. `--n-gpu-layers 27` offloads one more repeating layer and fails. The exact CPU override proves that the bad boundary is `blk.2.ffn_down.weight`, not output projection, KV cache, flash attention, repacking, generic Metal fusion, graph optimization, bf16, or quantization.

The logs also show llama.cpp resolving fused Gated Delta Net support for Granite, even when the run later works. That remains suspicious as a Granite-specific codepath, but it is not sufficient by itself to explain the failure because CPU-only and lower partial-offload runs also log it and still produce normal text.

## Practical Workarounds

For llama.cpp GGUF Metal:

- Use `--n-gpu-layers 26`, or
- Use full-ish offload with:

```bash
--override-tensor '^blk\.2\.ffn_down\.weight=CPU'
```

The override is more precise than reducing GPU layers, but it should be benchmarked before treating it as a deployment option. It creates extra graph splits and may affect latency.

## Next Steps

- Test a current upstream llama.cpp build before assuming the bug still exists there.
- If still present upstream, file a compact repro using F16 plus the `ngl=26` / `ngl=27` / `blk.2.ffn_down` override matrix.
- If GGUF deployment becomes relevant again, benchmark the exact override against `ngl=26`.
