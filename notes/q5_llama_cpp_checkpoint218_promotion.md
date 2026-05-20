# q5 llama.cpp checkpoint 218 promotion

Promoted `checkpoint-218` from the residual-repair run as the current q5 llama.cpp sidecar candidate on 2026-05-16.

Promotion pointer:

```text
hf_bucket/experiments/sft/promoted/q5_llama_cpp_current.json
```

Runtime path:

```text
hf_bucket/experiments/mlx_lora/runtime_shift/llama_cpp_q4_lora_sidecar_benchmark_20260507/gguf_base/gemma_3_270m_it.Q5_K_M.gguf
hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7/gguf_adapter/current_q5_candidate_checkpoint_218_lora.f16.gguf
```

Selection basis:

- The 200-row blind review versus checkpoint 700 favored checkpoint 218: `131` preferred wins and `128` accuracy wins.
- Numeric-risk rows moved strongly toward the desired no-digit/value-attachment behavior.
- Fresh-normal rows still slightly favored checkpoint 700, but not enough to outweigh the adversarial improvements.
- Follow-up grounded-shape rewrite probes from checkpoint 218 did not produce a better successor.

This does not replace the q8 MLX-Python standalone promoted model. It records the current q5 llama.cpp sidecar candidate.

Post-promotion q8 comparison:

- Final 500-row blind review: `hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7/eval/blind_ab_q8_mlx_python_vs_residual218_500/BLIND_REVIEW_SUMMARY.md`
- q8 MLX-Python won `356` preferred and `356` accuracy decisions.
- q5 checkpoint 218 won `144` preferred and `144` accuracy decisions.

Interpretation: checkpoint 218 is promoted for the q5 llama.cpp sidecar path, but q8 remains the stronger overall quality reference.
