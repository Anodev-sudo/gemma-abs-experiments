# llama.cpp q4 LoRA Sidecar Benchmark

## Summary

Tested a llama.cpp sidecar path for the q4 hardening adapter:

- Base: HF `google/gemma-3-270m-it` converted to GGUF and quantized to `Q4_0`
- Adapter: MLX q4 hardening LoRA converted to PEFT, then GGUF LoRA f16
- Runtime: Homebrew llama.cpp build `9020`
- Decode: greedy, `max_tokens=64`, prompt cache disabled for the speed pass

Artifacts live under:

`hf_bucket/experiments/mlx_lora/runtime_shift/llama_cpp_q4_lora_sidecar_benchmark_20260507/`

Kept payload size after deleting the f16 base intermediate: about `241M`.

## Results

Warm 23-row speed slice, first 3 rows discarded:

| Runtime | Warm rows | Mean ms | Median ms | p95 ms | Min / max ms | Mean words |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama.cpp q4 + LoRA sidecar | 20 | 105.65 | 97.9 | 171.43 | 67.09 / 251.7 | 15.55 |
| MLX Python q4 + adapter stream | 20 | 396.69 | 357.56 | 629.98 | 253.52 / 816.5 | 13.55 |
| MLX Swift q4 + LoRA sidecar | 20 | 600.6 | 544.5 | 880 | 442 / 1333 | 13.35 |

Fixture parity proxy:

- llama.cpp exact matches vs MLX Python q4+adapter: `0/11`
- llama.cpp exact matches vs MLX Swift q4 sidecar: `0/11`
- MLX Python exact matches vs MLX Swift q4 sidecar on the same fixtures: `9/11`
- llama.cpp fixture mean word count: `19.09`
- MLX Python fixture mean word count: `15.09`
- MLX Swift fixture mean word count: `15.18`

## Read

The llama.cpp sidecar path works mechanically and is much faster in this local
server benchmark, but behavior is not close enough to treat it as a runtime-only
swap. The outputs are generally coherent, yet several drift longer or into
authorial/formal language.

The largest caveat is that llama.cpp uses a freshly quantized GGUF `Q4_0` base
from the HF base model, while the MLX paths use the MLX `q4_g64` base. This
means the sidecar approach avoids destructive LoRA fusion, but it still changes
the base quantizer and runtime implementation.

Detailed outputs:

- `benchmark_report.md`
- `benchmark_report.json`
- `llama_traces/*.jsonl`
- `mlx_python_traces/*.jsonl`
