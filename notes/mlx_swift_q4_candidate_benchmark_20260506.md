# MLX Swift q4 Candidate Benchmark

## Summary

Reran the q4 fused MLX candidate through the repo-local `mlx-summary-probe`
Release executable on macOS.

Model:

- `hf_bucket/experiments/litert_exports/q4_hardening_combined_v1_lr_1e_6_300/q4_fused_mlx`

The fused model directory is about 184 MB. The main weight file is about 151 MB.

Runtime:

- `mlx-swift-lm` 3.31.3 via `tools/mlx-summary-probe`
- Release build
- macOS arm64
- prompt mode: `title_abstract`
- generation settings: 64 max tokens, temperature 0, top-p 1, top-k 1

## CAbLECAR Fixture

Warmup rows: 2

Measured repeats: 10

| Metric | Value |
| --- | ---: |
| load time | 1,303 ms |
| mean generation | 501.7 ms |
| median generation | 501.5 ms |
| min / max generation | 497 / 508 ms |

Output:

```text
Efficienting QLDPC codes on a tileable spin qubit chip with shuttling offers significant improvements in code performance and error reduction.
```

The output is normal text, though not stylistically ideal.

## 23-Row Eval Probe

Input:

- `hf_dataset/evals/q4_runtime_shift_200_v1/eval_inputs.jsonl`

The first 3 rows were treated as warmup. The remaining 20 rows provide the
app-shaped comparison slice.

| Metric | All 23 rows | Warm 20 rows |
| --- | ---: | ---: |
| errors | 0 | 0 |
| empty outputs | 0 | 0 |
| mean generation | 479.0 ms | 463.0 ms |
| median generation | 462 ms | 454.5 ms |
| p95 generation | 613 ms | 610 ms |
| min / max generation | 384 / 618 ms | 384 / 613 ms |
| mean words | 18.39 | 17.85 |
| median words | 19 | 18.5 |
| max words | 29 | 23 |

Temporary output JSONL:

- `/private/tmp/mlx_swift_q4_candidate_bench_23_20260506.jsonl`

## Interpretation

This q4 MLX Swift path is slower than the q8 LiteRT app-shaped benchmark, but
not wildly so:

- q8 LiteRT warm app-shaped mean: about 296-332 ms
- q4 MLX Swift warm app-shaped mean: about 463 ms

As a frame of reference, MLX Swift q4 is roughly 1.4-1.6x slower than q8 LiteRT
in these macOS/app-shaped probes, while staying far below the multi-second range.
