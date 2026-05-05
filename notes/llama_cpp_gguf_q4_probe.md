# llama.cpp GGUF q4 Probe

## Context

This probe tests whether the q4-hardened Gemma lineage can run through a GGUF + llama.cpp path without the LiteRT int4 repetition failure.

Source model:

- `hf_bucket/experiments/litert_exports/q4_hardening_combined_v1_lr_1e_6_300/q4_fused_dequantized_hf`

Outputs:

- `hf_bucket/experiments/litert_exports/q4_hardening_combined_v1_lr_1e_6_300/gguf_probe/q4_fused_dequantized_hf.f16.gguf`
- `hf_bucket/experiments/litert_exports/q4_hardening_combined_v1_lr_1e_6_300/gguf_probe/q4_fused_dequantized_hf.Q4_K_M.gguf`
- `hf_bucket/experiments/litert_exports/q4_hardening_combined_v1_lr_1e_6_300/gguf_probe/q4_fused_dequantized_hf.Q4_0.gguf`

## Converter Notes

Homebrew `llama.cpp` build `9020` installed successfully, but the packaged converter could not run directly with the existing Python environments:

- system Python was missing `transformers`
- the MLX runtime Python was missing `torch`
- the DPO runtime Python had `torch`, `transformers`, and `safetensors`, but needed `gguf`
- PyPI `gguf==0.18.0` was too old for the Homebrew converter

Using a small temporary sparse checkout of current llama.cpp plus its local `gguf-py` package worked after two temporary converter patches:

1. Ignore one tokenizer entry outside model `vocab_size`.
   - The tokenizer has `262145` entries and config `vocab_size=262144`.
   - The extra entry is `<image_soft_token>` at ID `262144`.
   - The text embedding only has `262144` rows, so the out-of-range image token should not be emitted into the text GGUF.
2. Map the local Gemma 3 text tokenizer hash to llama.cpp's existing `gemma4` BPE pre-tokenizer.
   - Hash: `789696f5946cc0fc59371f39f6097cafed196b3acded6140432f26bbb1ae1669`
   - Rationale: the tokenizer JSON uses BPE, byte fallback, and space-to-`▁` normalization; llama.cpp implements that pre-tokenization under the `gemma4` BPE path.

The resulting GGUF metadata has:

- `general.architecture = gemma3`
- `tokenizer.ggml.model = gpt2`
- `tokenizer.ggml.pre = gemma4`
- `tokenizer.ggml.add_bos_token = true`

## Artifacts

| Artifact | Size | Notes |
| --- | ---: | --- |
| f16 GGUF | 526 MB | Intermediate only |
| Q4_K_M GGUF | 249 MB | 90 tensors fell back to q5/q8 due dimensions |
| Q4_0 GGUF | 238 MB | Only embedding fell back to q8 |

The large embedding matrix dominates both q4 files, so `Q4_0` only saves about 11 MB versus `Q4_K_M`.

## Local llama.cpp Results

Metal could not be tested in this shell because llama.cpp reported a failed Metal command queue. CPU-only runs worked with `--device none --fit off`.

`Q4_K_M` CAbLECAR output:

> Shifting-based spin qubit scheduling improves QLDPC code efficiency and reduces logical error rates.

`Q4_0` CAbLECAR output:

> QLDPC codes can be implemented on a tileable chip with shuttling capabilities to improve code efficiency and reduce error rates.

Three tripwire rows with `Q4_K_M` did not show LiteRT-style prompt echo or repetition. They were coherent but not all review-clean:

- `numeric_0001`: Organic hydrogen sensors with organic materials enable real-time hydrogen monitoring in fuel cell applications.
- `numeric_0002`: A classical solution to Hessian equations is established in 1986.
- `numeric_0003`: Frozen Bardeen-Dirac stars form in a critical region with zero-value magnetic charge.

## CPU Timing

`llama-bench` with CPU-only execution and a 1024-token prompt reported warm steady-state results around:

- prompt processing: about `2000 tok/s`
- generation: about `340 tok/s`

The first decode sample can be much slower, so single cold samples are noisy. The generated-summary path with one CAbLECAR prompt showed roughly `900 tok/s` prompt processing and `280 tok/s` generation in `llama-completion`.

## Initial Read

The GGUF path looks materially better than the LiteRT int4 export for correctness: no prompt echo, no repeating fragments, and no malformed outputs in the small smoke set.

The next real question is iOS integration/runtime cost, not whether the model weights are intrinsically broken.
