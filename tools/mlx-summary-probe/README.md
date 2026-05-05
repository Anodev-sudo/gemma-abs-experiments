# MLX Summary Probe

Small macOS probe for checking whether MLX Swift can load the q4 fused MLX
summary model and generate with the trained `title_abstract` prompt contract.

## Build

SwiftPM can compile the executable:

```sh
swift build --package-path tools/mlx-summary-probe --scratch-path /private/tmp/mlx-summary-probe-build
```

For local execution, build with Xcode so MLX's Metal shader bundle is packaged:

```sh
xcodebuild \
  -scheme MLXSummaryProbe \
  -configuration Release \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath /private/tmp/mlx-summary-probe-xcodebuild \
  -skipMacroValidation \
  build
```

If Xcode reports a missing Metal toolchain, install it once:

```sh
xcodebuild -downloadComponent MetalToolchain
```

## Run

From the repo root:

```sh
/private/tmp/mlx-summary-probe-xcodebuild/Build/Products/Release/mlx-summary-probe \
  --example cablecar \
  --json
```

For a steadier latency read, use the Release executable with warmup and repeat
generation:

```sh
/private/tmp/mlx-summary-probe-xcodebuild/Build/Products/Release/mlx-summary-probe \
  --example cablecar \
  --warmup 1 \
  --repeat 5 \
  --json
```

The default model directory is:

```text
hf_bucket/experiments/litert_exports/q4_hardening_combined_v1_lr_1e_6_300/q4_fused_mlx
```

The default prompt spec is:

```text
prompts/mlx_lora/gemma_270m_it_abstract_to_glanceable_summary_v1.json
```
