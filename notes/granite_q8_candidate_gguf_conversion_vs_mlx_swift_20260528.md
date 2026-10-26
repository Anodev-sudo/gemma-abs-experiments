# Granite q8 Candidate GGUF Conversion vs MLX-Swift

Date: 2026-05-28

## Summary

Converted the current Granite q8 candidate to GGUF Q8_0 and compared llama.cpp Metal output against the saved MLX-Swift output on the existing 100-row blind packet.

The conversion works, but GGUF is not behavior-identical to MLX-Swift:

- Exact matches: `58/100`
- Changed rows: `42/100`
- No repeated-`@` collapse when using the Metal workaround
- GGUF is slightly wordier and has a few more long/digit rows than MLX-Swift

## Artifacts

- Candidate pointer:
  - `hf_bucket/experiments/mlx_lora/promoted/granite_4_0_350m_title_abstract_candidate_q8.json`
- Source MLX q8 candidate:
  - `hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q8_fusion_aware_v1/fused_round2_q8_g64`
- GGUF Q8_0:
  - `hf_bucket/experiments/llama_cpp/granite_4_0_350m_v5_stage_b25_q8_candidate_gguf_v1/gguf/granite_4_0_350m_v5_stage_b25_q8_candidate.Q8_0.gguf`
- GGUF generations:
  - `hf_bucket/experiments/llama_cpp/granite_4_0_350m_v5_stage_b25_q8_candidate_gguf_v1/generations/granite_stage_b25_q8_0_gguf_metal_blk2ffndown_cpu_blind100_v2.jsonl`
- Comparison report:
  - `hf_bucket/experiments/llama_cpp/granite_4_0_350m_v5_stage_b25_q8_candidate_gguf_v1/reports/q8_0_gguf_metal_vs_mlx_swift_blind100/MECHANICS.md`
  - `hf_bucket/experiments/llama_cpp/granite_4_0_350m_v5_stage_b25_q8_candidate_gguf_v1/reports/q8_0_gguf_metal_vs_mlx_swift_blind100/changed_examples.md`

## Conversion Notes

Direct conversion from the MLX q8 directory to GGUF failed because llama.cpp does not understand MLX quantization side tensors such as:

`model.embed_tokens.biases`

Working route:

1. Dequantize MLX q8 to temporary F16 with `mlx_lm.convert --dequantize`.
2. Convert dequantized F16 to GGUF F16 with `convert_hf_to_gguf.py`.
3. Quantize GGUF F16 to Q8_0 with `llama-quantize`.
4. Delete temporary dequantized F16 and GGUF F16 intermediates.

Final retained run root is about `400M`; final Q8_0 GGUF is `361M`.

## Metal Workaround Check

Without the known Granite Metal workaround, the converted GGUF still collapses:

`@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@`

The 100-row comparison used full offload plus:

```bash
--override-tensor '^blk\.2\.ffn_down\.weight=CPU'
```

## Mechanics

| label | rows | avg | median | max | <=14 | >14 | >15 | <10 | digits | formula | title >=3 | title >=4 | @ collapse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLX-Swift q8 fused Stage B 25 | 100 | 12.60 | 12.0 | 24 | 81 | 19 | 13 | 6 | 6 | 4 | 56 | 41 | 0 |
| GGUF Q8_0 Metal with block-2 override | 100 | 13.00 | 12.0 | 25 | 75 | 25 | 20 | 5 | 8 | 1 | 56 | 38 | 0 |

## Read

The GGUF conversion is close enough to be a plausible runtime variant, but not close enough to call behavior-equivalent.

Mostly harmless changes:

- `Switching cryogenic resistors` -> `Switching cryogenic resistances`
- `hidden symmetries` wording changes around gravitational higher-form symmetries
- small support-word changes like `must` vs `need to`

Concerning changes:

- More long rows: `25` over 14 words vs `19` for Swift.
- More digit rows: `8` vs `6`.
- Some concreteness drift:
  - `Ampere` adds `93.13%`.
  - `Retrograde Hot Jupiter` loses the retrograde point.
  - `Integer Sequence Generation` becomes `integer sequence code generation`.
  - `alpha-poor stars` changes the proposed source mixture.
  - `modal mu-calculus` changes `completed` to `performed`, a small but worse wording shift.

## Takeaway

GGUF Q8_0 can run the Granite q8 candidate with the block-2 CPU override, but MLX-Swift remains the better behavior reference. I would not treat GGUF as a drop-in replacement without a blind review if we seriously consider it.
