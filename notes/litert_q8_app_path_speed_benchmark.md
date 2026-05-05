# LiteRT q8 App-Path Speed Benchmark

## Summary

This benchmark measures the q8 LiteRT export through the same local-summary path used by the app:

- Swift `LiteRTBridgeAdapter` / `LiteRTLMBridge`
- current `SummaryPromptTemplate` trained system + user prompt
- `LiteRTGenerationOptions.summary`
- one sequential generation per paper
- iOS Simulator XCTest harness

The measured model was:

- `hf_bucket/experiments/litert_exports/faith_overlay_1400_v2_final_600/litert_prefill1024/model.litertlm`

The input set was:

- `hf_dataset/evals/q4_runtime_shift_200_v1/eval_inputs.jsonl`

This is not a device benchmark. It is app-shaped simulator evidence for the bridge path and prompt/runtime parameters.

## Runtime Settings

Generation options matched the app summary defaults:

| Setting | Value |
| --- | ---: |
| `topK` | 1 |
| `topP` | 1 |
| `temperature` | 0 |
| `randomSeed` | 0 |
| `maxOutputTokens` | 64 |
| max context tokens | 768 |

## Results

### 20-row smoke

Warmup rows: 3

| Metric | Value |
| --- | ---: |
| measured rows | 20 |
| failed rows | 0 |
| load time | 990 ms |
| measured load time | 1028 ms |
| mean generation | 296.4 ms |
| median generation | 283 ms |
| p95 generation | 436 ms |
| min / max generation | 234 / 446 ms |
| mean words | 11.2 |
| median words | 12 |
| mean prompt tokens | 318.8 |

### 195-row run

Warmup rows: 5

| Metric | Value |
| --- | ---: |
| measured rows | 190 |
| failed rows | 5 |
| load time | 1645 ms |
| measured load time | 1683 ms |
| mean generation | 332.0 ms |
| median generation | 288 ms |
| p95 generation | 564 ms |
| min / max generation | 200 / 966 ms |
| mean words | 12.17 |
| median words | 12 |
| mean prompt tokens | 346.15 |

All five failed rows reported:

- `LiteRT failed to create a conversation.`

The failures were bridge-level setup failures, not slow generations. The benchmark continued after recording them.

## Interpretation

For this simulator app path, q8 LiteRT is roughly `0.3s` per warm generation at the current output length. The larger run shows occasional conversation-creation failures under repeated sequential XCTest use, so the speed result should be treated as mostly representative but not a clean reliability pass.

Relative to the q4 runtime-shift probes, this q8 LiteRT path is slower than local `llama.cpp` GGUF timing but faster than the MLX Swift q4 fused probe in that earlier macOS harness. Those are not identical environments, so the comparison is directional only.

## Validation

The benchmark used a temporary XCTest file in the app repo:

- `SomethingTests/LiteRTQ8SpeedBenchmarkTests.swift`

Commands were run from the app checkout with derived data outside the repo:

```bash
./scripts/test.sh --only-testing SomethingTests/LiteRTQ8SpeedBenchmarkTests/testQ8LiteRTSpeedOnlyBenchmark --derived-data /private/tmp/SomethingQ8BenchDD --strict-test-destination
```

Temporary JSON outputs:

- `/private/tmp/litert_q8_bench_20.json`
- `/private/tmp/litert_q8_bench_195.json`
