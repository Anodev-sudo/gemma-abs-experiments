# q5 llama.cpp vs Granite q8 MLX Swift 100-Row Resource Check

Local coarse benchmark over the same 100-paper Granite blind packet.

This is intentionally simple: run generation, sample the local process tree
with `ps`, and compare wall time, per-row latency, CPU samples, and resident
memory. It is not Instruments, and it does not perfectly account for all GPU
driver or Metal heap behavior.

## Artifacts

- Benchmark root:
  `hf_bucket/experiments/runtime_benchmarks/q5_llama_vs_granite_q8_mlx_swift_100_20260528`
- q5 outputs:
  `q5_llama_max64_outputs_100.jsonl`
- Granite outputs:
  `granite_q8_swift_outputs_100.jsonl`
- Summary metrics:
  `summary_metrics.json`

## Inputs

- Source packet:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/review_samples/blind100_gemma425_vs_granite425_title_abstract_feed_blurb_v5/generations.jsonl`
- Prompt:
  `prompts/mlx_lora/gemma_270m_it_abstract_to_feed_blurb_v5.json`

## Compared Runtimes

q5 llama.cpp:

- Base:
  `hf_bucket/experiments/mlx_lora/runtime_shift/llama_cpp_q4_lora_sidecar_benchmark_20260507/gguf_base/gemma_3_270m_it.Q5_K_M.gguf`
- LoRA:
  `hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1/gguf_adapter/checkpoints/promoted_q8_adapter_q5_sidecar_lora.f16.gguf`
- Runtime:
  `llama-server`, `ctx-size=2048`, `n-gpu-layers=99`, `max_tokens=64`

Granite q8 MLX Swift:

- Model:
  `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q8_fusion_aware_v1/fused_round2_q8_g64`
- Runtime:
  repo-local `mlx-summary-probe`, `maxTokens=64`

## Results

| metric | q5 llama.cpp max64 | Granite q8 MLX Swift |
| --- | ---: | ---: |
| rows ok | `100 / 100` | `100 / 100` |
| total wall | `11.934s` | `13.444s` |
| mean row latency | `106.20 ms` | `125.34 ms` |
| median row latency | `103.28 ms` | `120.00 ms` |
| p95 row latency | `139.41 ms` | `159.10 ms` |
| max row latency | `191.81 ms` | `641.00 ms` |
| mean words | `12.99` | `12.63` |
| max words | `19` | `24` |
| sampled peak tree RSS | `1741.39 MB` | `486.34 MB` |
| sampled median tree RSS | `1045.75 MB` | `483.31 MB` |
| sampled mean CPU | `25.95%` | `61.56%` |
| sampled peak CPU | `69.20%` | `82.50%` |

q5 generated `1,934` completion tokens over the 100 rows. The Swift batch probe
does not emit token counts in normal batch mode.

## Read

This local run does not show a dramatic Granite slowdown. Granite q8 MLX Swift
is slower, but only by about `1.18x` on mean row latency and `1.13x` on total
wall time in this macOS command-line benchmark.

The surprise is memory: sampled RSS is much lower for Granite than for
llama.cpp q5. Treat that cautiously. The sampler sees process resident memory,
not a complete accounting of all Metal/GPU allocations, and llama.cpp's
server/mmap behavior may inflate resident-set readings relative to the Swift
probe.

The practical latency difference here is smaller than expected. If the app is
feeling much slower, likely suspects are app integration costs: model loading,
prompt assembly, per-call process/session overhead, queueing, thread hops,
thermal state, or iPhone-specific MLX behavior rather than the row-generation
kernel alone.
