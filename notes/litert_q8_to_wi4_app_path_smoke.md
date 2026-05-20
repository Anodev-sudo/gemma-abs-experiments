# LiteRT q8-to-WI4 App-Path Smoke

## Summary

Tested the promoted q8 lineage after LiteRT 4-bit export through the same
local-summary app path used for the q8 speed benchmark.

Model artifact:

- `hf_bucket/experiments/litert_exports/faith_overlay_1400_v2_final_600/litert_prefill1024_wi4/model.litertlm`
- size: `238,731,264` bytes
- sha256: `e23665261ed7d3967031ccae93d29a59b3a5a6d8e669b8fc4a35c0d184299cbc`

Input set:

- `hf_dataset/evals/q4_runtime_shift_200_v1/eval_inputs.jsonl`

This was an iOS Simulator XCTest smoke using `LiteRTLMBridge`, the current
`SummaryPromptTemplate`, and the same deterministic app summary defaults used
for the q8 app-path speed benchmark.

## Runtime Settings

| Setting | Value |
| --- | ---: |
| `topK` | 1 |
| `topP` | 1 |
| `temperature` | 0 |
| `randomSeed` | 0 |
| `maxOutputTokens` | 64 |
| max context tokens | 768 |
| warmup rows | 3 |
| measured rows | 20 |

## Result

The WI4 `.litertlm` loaded, but every measured generation failed before
streaming through the iOS app C-bridge path:

| Metric | Value |
| --- | ---: |
| load time | 660 ms |
| measured rows | 20 |
| failed rows | 20 |
| successful rows | 0 |
| degenerate rows | 0 |

Every failed measured row reported:

```text
LiteRT failed to create a conversation.
```

Because there were no successful generations through this bridge path, this
run does not provide meaningful latency or quality metrics.

## Comparison To q8 App Path

The same app-shaped q8 benchmark previously succeeded on the 20-row smoke with
`0` failures and showed only occasional conversation-creation failures in the
larger run. That makes this WI4 result an app-bridge smoke failure, not merely
a slower or lower-quality model result.

This should not be read as proof that the `.litertlm` cannot run through every
LiteRT-LM runtime surface. Earlier q4 WI4 experiments ran far enough to produce
malformed or repetitive output, so the Python/Linux LiteRT-LM path should be
rechecked separately before declaring the exported artifact itself unusable.

It also should not be read as proof that the iOS/device runtime cannot run WI4
LiteRT exports. The earlier mangled q4 LiteRT export reportedly ran on device
and failed at output quality, not at conversation creation. That makes the
current failure more likely to be specific to this q8-to-WI4 artifact, export
recipe, or benchmark bridge setup.

## Validation

The XCTest result bundle for this smoke was:

- `/private/tmp/SomethingWI4BenchDD/Logs/Test/Test-Something-2026.05.06_03-23-17--0500.xcresult`

The exported XCTest attachment contained the 20 failed rows and was inspected
with `xcresulttool export attachments`.

Temporary app-side benchmark code was removed after the run.
