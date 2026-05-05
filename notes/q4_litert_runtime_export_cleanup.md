# q4 LiteRT and Runtime Export Cleanup

## Scope

This note records the q4 runtime/export investigation around the promoted
`q4_hardening_combined_v1_lr_1e_6_300` model and documents what was kept after
cleanup.

## Source Model

The promoted q4 model is the q4 MLX base plus the promoted q4 hardening adapter:

- q4 MLX base: `hf_bucket/experiments/mlx_lora/quantized_models/google_gemma_3_270m_it/q4_g64`
- adapter dir: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_hardening_combined_v1_lr_1e_6/adapters`
- adapter checkpoint: `0000300_adapters.safetensors`

The important correction was that this adapter should be fused into the q4 MLX
base, not converted back to PEFT and merged into the original full
Transformers base.

## What Worked

Created a fused q4 MLX checkpoint with:

```sh
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python -m mlx_lm.fuse \
  --model hf_bucket/experiments/mlx_lora/quantized_models/google_gemma_3_270m_it/q4_g64 \
  --adapter-path hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_hardening_combined_v1_lr_1e_6/adapters \
  --save-path hf_bucket/experiments/litert_exports/q4_hardening_combined_v1_lr_1e_6_300/q4_fused_mlx
```

Retained artifact:

- `hf_bucket/experiments/litert_exports/q4_hardening_combined_v1_lr_1e_6_300/q4_fused_mlx`
- `model.safetensors` SHA256: `74b715b6337c4dffd8bf8dd38b263e2fc3f8ba0c0978880a1d70e3faf2f6e10f`

The q4 MLX artifact loaded through the MLX Swift probe and produced normal
non-empty text. See `notes/mlx_swift_q4_probe.md`.

## What Failed

### Direct q4 MLX to LiteRT

Trying to pass the fused q4 MLX checkpoint directly into
`litert_torch.generative.export_hf` failed during Transformers model loading:

```text
ValueError: The model's quantization config from the arguments has no `quant_method` attribute.
```

The reason is format compatibility, not missing weights: MLX q4 safetensors use
MLX quantization metadata (`bits`, `group_size`, `mode`) and packed tensors,
while the LiteRT Torch exporter enters through Transformers and expects a
Transformers-supported quantization format.

### PEFT/full-base merge

An earlier attempt converted the q4 adapter back to PEFT and merged it into the
original full Transformers base. That produced a `.litertlm`, but it was the
wrong lineage for this q4 test because the merge base was not the promoted q4
MLX base.

The small logs for that attempt are preserved under:

- `hf_bucket/experiments/litert_exports/q4_hardening_combined_v1_lr_1e_6_300/superseded_full_base_merge/`

## Bridge That Exported

The LiteRT exporter could load a dequantized bridge checkpoint created from the
q4 MLX base:

```sh
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python -m mlx_lm.fuse \
  --model hf_bucket/experiments/mlx_lora/quantized_models/google_gemma_3_270m_it/q4_g64 \
  --adapter-path hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_hardening_combined_v1_lr_1e_6/adapters \
  --save-path hf_bucket/experiments/litert_exports/q4_hardening_combined_v1_lr_1e_6_300/q4_fused_dequantized_hf \
  --dequantize
```

Then LiteRT export succeeded through the Colima Docker image with
`dynamic_wi4_afp32`. The exported `.litertlm` metadata is preserved in:

- `hf_bucket/experiments/litert_exports/q4_hardening_combined_v1_lr_1e_6_300/litert_from_q4_fused_dequantized_prefill1024_result.json`

The bridge checkpoint itself was removed during cleanup because it is a large,
reproducible intermediate. The `.litertlm` binary is also not retained in the
cleaned export directory; the size, checksum, and export settings are preserved
in the result JSON.

## GGUF and Runtime Shift Probes

GGUF conversion was tested from the dequantized bridge checkpoint. Local
llama.cpp CPU runs did not show the LiteRT-style prompt echo or repetition in
small smoke tests. Details are in `notes/llama_cpp_gguf_q4_probe.md`.

A 200-row runtime-shift comparison was then run across:

- canonical MLX q4 adapter
- GGUF `Q4_0` through llama.cpp
- fused q4 MLX through MLX Swift

The lightweight generation outputs, blinded review packets, and aggregations
are preserved under:

- `hf_bucket/experiments/mlx_lora/runtime_shift/q4_runtime_shift_200_v1/`

Summary note:

- `notes/q4_runtime_shift_200_v1.md`

The takeaway was that both alternate deployment paths shifted behavior downward
relative to the canonical MLX adapter, even though they avoided the obvious
LiteRT int4 malformed-output failure.

## Cleanup Policy

Kept:

- source scripts and probe tools
- notes
- small manifests and result JSON
- fused q4 MLX checkpoint
- runtime-shift generated outputs and review logs
- custom LiteRT quantization recipes

Removed:

- GGUF binaries under `gguf_probe/`
- dequantized HF bridge checkpoint
- corrected LiteRT `.litertlm` binary, with metadata retained in JSON
- empty failed direct-q4 LiteRT export scratch directory
- local shell clutter such as `.DS_Store` and Python cache directories

The removed model binaries are reproducible from the commands above and the
preserved source artifacts.
