# q4 Runtime Shift 200 v1

## Summary

This run compares behavior shift across three q4 runtime/export paths on the same 200 held-out papers:

- `canonical_mlx_q4_adapter`: q4 MLX base plus the promoted q4 hardening adapter.
- `gguf_q4_0_llama_cpp`: GGUF `Q4_0` export run through `llama.cpp`.
- `mlx_swift_q4_fused`: fused q4 MLX export run through `mlx-swift-lm`.

The eval set is `hf_dataset/evals/q4_runtime_shift_200_v1/eval_inputs.jsonl`.

The main result: both alternate deployment paths shift behavior downward relative to the canonical MLX adapter. GGUF is much faster locally than MLX Swift, but it loses clearly to canonical in blinded review. MLX Swift is slower and also loses clearly to canonical, though it beats GGUF head-to-head.

## Sample

Built from the recent balanced corpus with prior training rows and prior q4/q8 review samples excluded.

Bucket targets:

| Bucket | Rows |
| --- | ---: |
| numeric | 70 |
| polarity | 50 |
| entity | 40 |
| long_notation | 20 |
| control | 20 |
| total | 200 |

Manifest: `hf_dataset/evals/q4_runtime_shift_200_v1/manifest.json`

## Generation Artifacts

Outputs:

- `hf_bucket/experiments/mlx_lora/runtime_shift/q4_runtime_shift_200_v1/variant_outputs/canonical_mlx_q4_adapter.jsonl`
- `hf_bucket/experiments/mlx_lora/runtime_shift/q4_runtime_shift_200_v1/variant_outputs/gguf_q4_0_llama_cpp.jsonl`
- `hf_bucket/experiments/mlx_lora/runtime_shift/q4_runtime_shift_200_v1/variant_outputs/mlx_swift_q4_fused.jsonl`

MLX Swift was generated in 20 isolated chunks of 10 rows after a prior long single-process run exhausted memory. Chunk files are under:

- `hf_bucket/experiments/mlx_lora/runtime_shift/q4_runtime_shift_200_v1/variant_outputs/mlx_swift_q4_fused_chunks/`

All three outputs have exactly 200 rows and no empty summaries.

## Mechanical Checks

| Variant | Rows | Empty | Mean words | Median words | Max words | Local timing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| canonical MLX adapter | 200 | 0 | 13.00 | 13 | 31 | not recorded |
| GGUF Q4_0 llama.cpp | 200 | 0 | 14.97 | 14 | 45 | 86.7 ms mean wall time |
| MLX Swift q4 fused | 200 | 0 | 16.91 | 16 | 49 | 440.8 ms mean generation time |

Local format scan found no prompt echoes, empty outputs, triple-word repetition, or triple-bigram repetition in any of the three 200-row output sets.

Exact output matches:

- canonical vs GGUF: 2/200
- canonical vs MLX Swift: 0/200
- GGUF vs MLX Swift: 0/200

## Blinded Pairwise Reviews

Review packets:

- `hf_bucket/experiments/mlx_lora/runtime_shift/q4_runtime_shift_200_v1/review_packets/canonical_mlx_q4_adapter_vs_gguf_q4_0_llama_cpp`
- `hf_bucket/experiments/mlx_lora/runtime_shift/q4_runtime_shift_200_v1/review_packets/canonical_mlx_q4_adapter_vs_mlx_swift_q4_fused`
- `hf_bucket/experiments/mlx_lora/runtime_shift/q4_runtime_shift_200_v1/review_packets/gguf_q4_0_llama_cpp_vs_mlx_swift_q4_fused`

Each pair was reviewed as 8 blinded packets of 25 rows. Reviewers did not receive side maps. Aggregated summaries are in each packet directory under `aggregation/review_summary.json`.

### Canonical MLX Adapter vs GGUF

| Metric | canonical MLX | GGUF |
| --- | ---: | ---: |
| wins | 127 | 60 |
| severe | 19 | 44 |
| mangled | 0 | 6 |

Other outcomes: 7 ties, 6 both-bad.

### Canonical MLX Adapter vs MLX Swift

| Metric | canonical MLX | MLX Swift |
| --- | ---: | ---: |
| wins | 116 | 73 |
| severe | 22 | 40 |
| mangled | 0 | 4 |

Other outcomes: 1 tie, 10 both-bad.

### GGUF vs MLX Swift

| Metric | GGUF | MLX Swift |
| --- | ---: | ---: |
| wins | 80 | 107 |
| severe | 29 | 33 |
| mangled | 8 | 5 |

Other outcomes: 4 ties, 9 both-bad.

## Interpretation

The canonical q4 MLX adapter remains the behavior reference. The GGUF export is fast and not obviously malformed by simple local filters, but blinded review shows it loses substantially to canonical and has over twice as many severe flags. MLX Swift is slower than GGUF in this macOS harness and also loses substantially to canonical, though it looks somewhat better than GGUF head-to-head.

For deployment decisions, this points away from treating either GGUF or MLX Swift as behavior-equivalent drop-in replacements. If size is still the priority, the next useful work is export/runtime investigation with a canonical-behavior regression harness wired into the conversion step, not another broad model-training pass.

## Validation

- `scripts/run_q4_runtime_shift_eval.py` passed `python3 -m py_compile`.
- Canonical, GGUF, and MLX Swift output files each have 200 rows.
- All 24 blinded packet `decisions.tsv` files were filled.
- All three aggregations report `missing_decisions: 0`.
