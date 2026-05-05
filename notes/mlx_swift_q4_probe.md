# MLX Swift q4 Probe

## Summary

Added a reusable macOS SwiftPM probe under `tools/mlx-summary-probe` to test the
q4 fused MLX summary model through `mlx-swift-lm` before app integration.

## Result

- SwiftPM build succeeded with scratch output under `/private/tmp/mlx-summary-probe-build`.
- Plain `swift run` built the executable but failed at runtime because the command-line SwiftPM build did not provide MLX's `default.metallib`.
- Xcode initially could not compile MLX Metal kernels because the Metal Toolchain component was uninstalled.
- After installing the Metal Toolchain, the Xcode package build succeeded with DerivedData under `/private/tmp/mlx-summary-probe-xcodebuild`.
- The Xcode-built executable loaded the q4 fused MLX model and generated a non-empty CAbLECAR summary.

## Commands

```sh
swift build --package-path tools/mlx-summary-probe --scratch-path /private/tmp/mlx-summary-probe-build

xcodebuild -downloadComponent MetalToolchain

xcodebuild \
  -scheme MLXSummaryProbe \
  -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath /private/tmp/mlx-summary-probe-xcodebuild \
  -skipMacroValidation \
  build

/private/tmp/mlx-summary-probe-xcodebuild/Build/Products/Release/mlx-summary-probe \
  --example cablecar \
  --warmup 1 \
  --repeat 5 \
  --json
```

## Probe Output

Initial cold Debug-style probe:

```json
{
  "generation_time_ms" : 2459,
  "input_mode" : "title_abstract",
  "load_time_ms" : 2892,
  "model_dir" : "hf_bucket/experiments/litert_exports/q4_hardening_combined_v1_lr_1e_6_300/q4_fused_mlx",
  "normalized_output" : "Efficienting QLDPC codes on a tileable spin qubit chip with shuttling offers significant improvements in code performance and error reduction.",
  "ok" : true,
  "prompt_spec" : "prompts/mlx_lora/gemma_270m_it_abstract_to_glanceable_summary_v1.json",
  "prompt_version" : "gemma_270m_it_abstract_to_glanceable_summary_v1",
  "raw_output" : "Efficienting QLDPC codes on a tileable spin qubit chip with shuttling offers significant improvements in code performance and error reduction.",
  "title" : "CAbLECAR: efficiently scheduling QLDPC codes on a tileable spin qubit chip with shuttling"
}
```

Release probe with one warmup generation and five timed generations:

```json
{
  "generation_time_ms" : 505,
  "generation_times_ms" : [
    493,
    493,
    506,
    509,
    505
  ],
  "input_mode" : "title_abstract",
  "load_time_ms" : 1284,
  "model_dir" : "hf_bucket/experiments/litert_exports/q4_hardening_combined_v1_lr_1e_6_300/q4_fused_mlx",
  "normalized_output" : "Efficienting QLDPC codes on a tileable spin qubit chip with shuttling offers significant improvements in code performance and error reduction.",
  "ok" : true,
  "prompt_spec" : "prompts/mlx_lora/gemma_270m_it_abstract_to_glanceable_summary_v1.json",
  "prompt_version" : "gemma_270m_it_abstract_to_glanceable_summary_v1",
  "raw_output" : "Efficienting QLDPC codes on a tileable spin qubit chip with shuttling offers significant improvements in code performance and error reduction.",
  "repeat_count" : 5,
  "title" : "CAbLECAR: efficiently scheduling QLDPC codes on a tileable spin qubit chip with shuttling",
  "warmup_count" : 1
}
```

## Takeaway

The local q4 MLX artifact is loadable by MLX Swift on macOS. The generated text
is not stylistically ideal, but it is a normal model output rather than the
LiteRT int4 repetition/prompt-echo failure. The cold single-generation timing is
not comparable to app runtime. The warmed Release probe generated the fixture in
roughly 0.5 seconds per pass after a 1.3 second model load on macOS. The next
useful check is an iOS Simulator probe using the same model-loading path.
