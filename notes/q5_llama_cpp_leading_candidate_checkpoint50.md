# q5 llama.cpp previous q5-trained candidate: broad q8-win checkpoint 50

Marked on 2026-05-19.

This is the previous leading q5-trained llama.cpp continuation candidate. It has
been superseded as the overall compact candidate by the q5 base plus promoted q8
sidecar. The current promoted q5 pointer is:

```text
hf_bucket/experiments/sft/promoted/q5_llama_cpp_current.json
```

Previous candidate runtime path:

```text
hf_bucket/experiments/mlx_lora/runtime_shift/llama_cpp_q4_lora_sidecar_benchmark_20260507/gguf_base/gemma_3_270m_it.Q5_K_M.gguf
hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/gguf_adapter/checkpoints/checkpoint_50_lora.f16.gguf
```

PEFT checkpoint:

```text
hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/checkpoint-50
```

Why it was the leading q5-trained candidate:

- The 500-row q8-vs-q5 blind review gave q8 `253` preference wins and this q5
  checkpoint `217`, with `14` ties and `16` neither.
- This narrowed the q5-vs-q8 gap compared with the prior broad-cleanup
  checkpoint 30 result.
- Follow-up probes did not supersede it: stronger replay got noisier, round-two
  q8-win cleanup lacked a convincing quality signal, broad basin v2 lost to this
  checkpoint in bridge review, and anchored union / DPO / prompt variants did
  not improve the surface.

Latest sidecar comparison:

- Review root:
  `hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1/eval/blind_ab_q8sidecar_vs_q5_leading_q8win50_random200_20260519`
- Judge setup: 200 random recent-3y rows, five `gpt-5.4-mini` blind-review
  shards, preference only, no hard criteria.
- Result: checkpoint 50 won `98`, the q5 base plus promoted q8
  sidecar won `95`, with `5` ties and `2` neither.

Interpretation: checkpoint 50 narrowly beat the promoted-q8-adapter sidecar on
the 200-row preference-only repeat, but this is now historical. The larger
accuracy and readability reviews later favored the q8 sidecar, which is now the
promoted q5 llama.cpp candidate.

Expanded accuracy-first sidecar comparison:

- Review root:
  `hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1/eval/blind_ab_q8sidecar_vs_q5_leading_q8win50_accuracy500_rand400_adv100_20260519`
- Judge setup: 500 rows total: 400 uniform random recent-3y rows plus 100
  adversarial rows, sampled as 20 each from control, entity, long-notation,
  numeric, and polarity hardening splits. Ten `gpt-5.4-mini` blind-review
  shards judged accuracy/support only.
- Overall accuracy result: q5 base plus promoted q8 sidecar won `252`, this q5
  checkpoint won `196`, with `38` ties and `14` neither.
- Random 400: promoted q8 sidecar `201`, this q5 checkpoint `157`, ties `32`,
  neither `10`.
- Adversarial 100: promoted q8 sidecar `51`, this q5 checkpoint `39`, ties `6`,
  neither `4`.
- Notable slice: this q5 checkpoint still won the numeric adversarial slice
  `12` to `6`, with `1` tie and `1` neither.

Updated interpretation: checkpoint 50 was the leading q5-trained candidate from
the earlier q5 lineage work, but the q5 base plus promoted q8 sidecar is the
stronger accuracy candidate in the expanded 500-row review.

Expanded readability-only sidecar comparison:

- Review root:
  `hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1/eval/blind_ab_q8sidecar_vs_q5_leading_q8win50_readability500_rand400_adv100_20260519`
- Judge setup: same 500 rows, same outputs, same side-randomization seed, ten
  `gpt-5.4-mini` blind-review shards. Judges considered readability only:
  clarity, flow, natural wording, compactness, and glanceability.
- Overall readability result: q5 base plus promoted q8 sidecar won `254`, this
  q5 checkpoint won `225`, with `17` ties and `4` neither.
- Random 400: promoted q8 sidecar `199`, this q5 checkpoint `183`, ties `15`,
  neither `3`.
- Adversarial 100: promoted q8 sidecar `55`, this q5 checkpoint `42`, ties `2`,
  neither `1`.
- Numeric adversarial slice again narrowly favored this q5 checkpoint, `10` to
  `9`, with `1` neither.

Current read across the three direct q8-sidecar comparisons: the 200-row
preference-only run narrowly favored checkpoint 50, but the larger 500-row
accuracy and readability runs both favor the q5 base plus promoted q8 sidecar.
The current promoted compact candidate is now tracked in:

```text
notes/q5_llama_cpp_leading_candidate_q8_sidecar.md
```
