# q5 llama.cpp promoted candidate: q5 base plus promoted q8 sidecar

Promoted on 2026-05-19.

This is now the promoted q5 llama.cpp candidate. The current promoted q5 pointer
is:

```text
hf_bucket/experiments/sft/promoted/q5_llama_cpp_current.json
```

Promoted runtime path:

```text
hf_bucket/experiments/mlx_lora/runtime_shift/llama_cpp_q4_lora_sidecar_benchmark_20260507/gguf_base/gemma_3_270m_it.Q5_K_M.gguf
hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1/gguf_adapter/checkpoints/promoted_q8_adapter_q5_sidecar_lora.f16.gguf
```

Adapter source:

```text
hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1
```

Promotion decision:

- Promote q5 base plus promoted q8 sidecar as the current compact q5-class
  llama.cpp candidate.
- Mark broad q8-win checkpoint 50 as the previous leading q5-trained
  continuation candidate.

Evidence against broad q8-win checkpoint 50:

- 200-row preference-only review narrowly favored checkpoint 50: checkpoint 50
  `98`, promoted q8 sidecar `95`, ties `5`, neither `2`.
- 500-row accuracy-first review favored promoted q8 sidecar: promoted q8 sidecar
  `252`, checkpoint 50 `196`, ties `38`, neither `14`.
- 500-row readability-only review favored promoted q8 sidecar: promoted q8
  sidecar `254`, checkpoint 50 `225`, ties `17`, neither `4`.

500-row accuracy split:

- Random 400: promoted q8 sidecar `201`, checkpoint 50 `157`, ties `32`,
  neither `10`.
- Adversarial 100: promoted q8 sidecar `51`, checkpoint 50 `39`, ties `6`,
  neither `4`.

500-row readability split:

- Random 400: promoted q8 sidecar `199`, checkpoint 50 `183`, ties `15`,
  neither `3`.
- Adversarial 100: promoted q8 sidecar `55`, checkpoint 50 `42`, ties `2`,
  neither `1`.

Promotion caveats:

- Checkpoint 50 still narrowly wins the numeric adversarial slice in both 500-row
  reviews.
- This candidate is a compact q5-class deployment path, not a claim that it
  matches or replaces the promoted q8 standalone quality reference.
- The q5 promotion pointer has been moved to this candidate.
