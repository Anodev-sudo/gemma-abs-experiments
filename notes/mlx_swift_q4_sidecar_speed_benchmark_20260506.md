# MLX Swift q4 LoRA Sidecar Speed Benchmark

## Summary

Benchmarked the q4 base plus LoRA sidecar through the repo-local
`mlx-summary-probe` Release executable on macOS.

Model inputs:

- q4 base: `hf_bucket/experiments/mlx_lora/quantized_models/google_gemma_3_270m_it/q4_g64`
- LoRA sidecar: `hf_bucket/experiments/mlx_lora/runtime_shift/mlx_swift_parity_root_cause_v1/mlx_swift_lora_sidecar_adapter`
- eval input: `hf_dataset/evals/q4_runtime_shift_200_v1/eval_inputs.jsonl`

Runtime settings matched the earlier MLX Swift q4 fused probe:

- prompt mode: `title_abstract`
- max output tokens: 64
- temperature: 0
- top-p: 1
- top-k: 1

## 23-Row Eval Probe

The first 3 rows were treated as warmup. The remaining 20 rows provide the
app-shaped comparison slice.

| Metric | All 23 rows | Warm 20 rows |
| --- | ---: | ---: |
| errors | 0 | 0 |
| load time | 1,387 ms | n/a |
| mean generation | 669.1 ms | 600.6 ms |
| median generation | 573 ms | 544.5 ms |
| p95 generation | n/a | 880 ms |
| min / max generation | 442 / 1,860 ms | 442 / 1,333 ms |
| mean words | n/a | 13.35 |
| median words | n/a | 13 |
| max words | n/a | 20 |

## CAbLECAR Repeated Fixture

Warmup repeats: 2

Measured repeats: 10

| Metric | Value |
| --- | ---: |
| load time | 1,307 ms |
| mean generation | 527.5 ms |
| median generation | 527.5 ms |
| min / max generation | 522 / 533 ms |

Output:

```text
Shifting schedules improve QLDPC coding efficiency and reduce logical error rates.
```

## Comparison

The existing q8 LiteRT app-path benchmark measured:

- 20-row smoke warm mean: 296.4 ms
- 195-row run warm mean: 332.0 ms

Against that reference, the q4 MLX Swift LoRA sidecar path is roughly:

- 1.8x to 2.0x slower on the 23-row app-shaped warm slice
- about 1.6x to 1.8x slower on the repeated CAbLECAR fixture

This is still subsecond warm generation, but the sidecar does appear slower
than both q8 LiteRT and the earlier q4 fused MLX Swift probe. The likely reason
is that the sidecar avoids destructive post-fusion requantization by applying
LoRA deltas at runtime, which costs extra compute on the affected projections.

## Caveat

The q8 LiteRT benchmark was an iOS Simulator app-path benchmark through the app
bridge. This sidecar benchmark is a macOS command-line MLX Swift probe. The
comparison is useful for sizing the tradeoff, but it is not a perfect
same-process runtime shootout.
