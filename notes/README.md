# Notes

Lab notes, one per experiment, written as the work happened. They are grouped
here by theme; inside each group they are sorted by file name. Paths like
`hf_bucket/...` point to run artifacts kept in a separate storage bucket that is
not part of this repository.

## First LoRA runs on the 10,746-row summary set

- [Faithfulness Cleanup Overlay v1](faithfulness_cleanup_overlay_v1.md)
- [MLX LoRA 270M 10.7k Checkpoint Notes](mlx_lora_270m_10746_checkpoint_notes.md)
- [MLX LoRA 270M Faithfulness Overlay 1400 v2 Promotion](mlx_lora_270m_faith_overlay_1400_v2_promotion.md)
- [MLX LoRA 270M Run 01](mlx_lora_270m_run_01.md)

## 4-bit hardening and promotion

- [Gemma 270M q4 Hardening Combined v1](mlx_lora_270m_q4_hardening_combined_v1.md)
- [q4 Hardening Distill 6000 v1](mlx_lora_270m_q4_hardening_distill_6000_v1.md)
- [MLX LoRA 270M q4 Hardening Distill v1](mlx_lora_270m_q4_hardening_distill_v1.md)
- [q4 Kimi Correction Salvage No Reasoning v1](mlx_lora_270m_q4_kimi_correction_salvage_no_reasoning_v1.md)
- [MLX LoRA 270M q4 Lineage Redo](mlx_lora_270m_q4_lineage_redo.md)
- [Gemma 270M q4 Promotion](mlx_lora_270m_q4_promotion.md)

## 5-bit correction overlays

- [q5 Correction Overlay Repair Run v1](mlx_lora_270m_q5_correction_overlay_repair_run_v1.md)
- [q5 Final Guardrail Overlay replay v2](mlx_lora_270m_q5_final_guardrail_overlay_replay_v2.md)
- [q5 Final Guardrail Overlay v1](mlx_lora_270m_q5_final_guardrail_overlay_v1.md)
- [q5 Hardening Mine 6000 v1](mlx_lora_270m_q5_hardening_mine_6000_v1.md)
- [q5 Kimi/Qwen Correction Overlay v1](mlx_lora_270m_q5_kimi_qwen_correction_overlay_v1.md)
- [q5 Numeric Hardening Overlay q4mix len15 v2](mlx_lora_270m_q5_numeric_hardening_overlay_q4mix_len15_v2.md)
- [q5 Numeric Hardening Overlay q4mix v1](mlx_lora_270m_q5_numeric_hardening_overlay_q4mix_v1.md)
- [q5 Precision Failure Overlay v1](mlx_lora_270m_q5_precision_failure_overlay_v1.md)
- [q5 precision Kimi/Qwen overlay training run v1](mlx_lora_270m_q5_precision_kimi_qwen_overlay_run_v1.md)
- [q5 precision Kimi/Qwen overlay v1](mlx_lora_270m_q5_precision_kimi_qwen_overlay_v1.md)
- [q5 Broad Re-Centering Overlay v1](mlx_lora_270m_q5_recenter_broad_overlay_v1.md)
- [q5 Style Anti-Regression Overlay v1](mlx_lora_270m_q5_style_antiregression_overlay_v1.md)
- [q5 checkpoint-50 natural-language clean overlay v1](q5_checkpoint50_natural_language_clean_digitfree_overlay_v1.md)
- [q5 Four-Stress Overfit Diagnostic v1](q5_four_stress_overfit_diagnostic_v1.md)
- [q5 residual-218 push run - 2026-05-17](q5_residual218_push_run_20260517.md)

## On-device runtimes: LiteRT, llama.cpp, MLX Swift

- [Granite GGUF Metal Offload Root Cause Probe](granite_gguf_metal_root_cause_20260528.md)
- [Granite q8 Candidate GGUF Conversion vs MLX-Swift](granite_q8_candidate_gguf_conversion_vs_mlx_swift_20260528.md)
- [LiteRT LoRA Sidecar Probe](litert_lora_sidecar_probe_20260507.md)
- [LiteRT q8 App-Path Speed Benchmark](litert_q8_app_path_speed_benchmark.md)
- [LiteRT q8-to-WI4 App-Path Smoke](litert_q8_to_wi4_app_path_smoke.md)
- [LiteRT Weight-Only WI4 Probe](litert_weight_only_wi4_probe_20260506.md)
- [llama.cpp GGUF q4 Probe](llama_cpp_gguf_q4_probe.md)
- [llama.cpp q4 LoRA Sidecar Benchmark](llama_cpp_q4_lora_sidecar_benchmark_20260507.md)
- [MLX Swift q4 Candidate Benchmark](mlx_swift_q4_candidate_benchmark_20260506.md)
- [MLX-Swift q4 Deployment Path Probe](mlx_swift_q4_deployment_path_probe_20260506.md)
- [MLX Swift q4 Probe](mlx_swift_q4_probe.md)
- [MLX Swift q4 LoRA Sidecar Speed Benchmark](mlx_swift_q4_sidecar_speed_benchmark_20260506.md)
- [q4 LiteRT and Runtime Export Cleanup](q4_litert_runtime_export_cleanup.md)
- [q4 Runtime Shift 200 v1](q4_runtime_shift_200_v1.md)
- [q5 llama.cpp checkpoint 218 promotion](q5_llama_cpp_checkpoint218_promotion.md)
- [q5 llama.cpp previous q5-trained candidate: broad q8-win checkpoint 50](q5_llama_cpp_leading_candidate_checkpoint50.md)
- [q5 llama.cpp promoted candidate: q5 base plus promoted q8 sidecar](q5_llama_cpp_leading_candidate_q8_sidecar.md)
- [q5 llama.cpp vs Granite q8 MLX Swift 100-Row Resource Check](runtime_benchmark_q5_llama_vs_granite_q8_mlx_swift_100_20260528.md)

## Safety evaluation

- [q8 RealDB Safety Eval 10000 v1](q8_realdb_safety_eval_10000_v1.md)

## Natural-language teacher data (DeepSeek V4 Pro writer and verifier)

- [Natural Language Teacher 300-Row Milestone](natural_language_teacher_300_milestone.md)
- [Natural Language Teacher 500-Row Milestone](natural_language_teacher_500_milestone.md)
- [Natural-language teacher lane v1](natural_language_teacher_deepseek_v4_pro_v1.md)
- [Natural-Language Teacher Math Path v14](natural_language_teacher_math_path_v14.md)
- [Natural-Language Teacher Math Path v16](natural_language_teacher_math_path_v16.md)
- [Natural-Language Teacher No-Credit Backlog Rescue](natural_language_teacher_no_credit_backlog_rescue.md)
- [Natural-Language Teacher Rescue Lanes v11-v13](natural_language_teacher_rescue_lanes_v11_v13.md)
- [Natural-language teacher v5 checkpoint](natural_language_teacher_v5_checkpoint.md)
- [Natural-language teacher v6 writer-pressure checkpoint](natural_language_teacher_v6_writer_pressure.md)
- [Natural-language teacher v7 writer-pressure checkpoint](natural_language_teacher_v7_writer_pressure.md)
- [Natural-language teacher v8 yield rescue](natural_language_teacher_v8_yield_rescue.md)
- [Natural-language teacher writer-yield rescue](natural_language_teacher_v9_writer_yield_rescue.md)

## Feed-blurb training on the teacher data

- [MLX LoRA 1B natural-language 500-row LoRA v1](mlx_lora_1b_natural_language_500_lora_v1.md)
- [MLX full finetune 270M natural-language 100-row POC v1](mlx_lora_270m_natural_language_100_full_poc_v1.md)
- [MLX LoRA 270M natural-language 100-row POC v1](mlx_lora_270m_natural_language_100_poc_v1.md)
- [MLX LoRA 270M heuristic-3000 burn-in plus gold finisher v1](mlx_lora_270m_natural_language_10746_heuristic3000_goldfinish_v1.md)
- [MLX full finetune 270M natural-language 300-row v1](mlx_lora_270m_natural_language_300_full_v1.md)
- [MLX LoRA 270M natural-language 300-row LoRA v1](mlx_lora_270m_natural_language_300_lora_v1.md)
- [MLX LoRA 270M natural-language 500-row LoRA v1](mlx_lora_270m_natural_language_500_lora_v1.md)
- [MLX LoRA 270M natural-language 800-row high-LR LoRA v1](mlx_lora_270m_natural_language_800_highlr_v1.md)
- [MLX LoRA 270M natural-language 994-row long LoRA v1](mlx_lora_270m_natural_language_994_lora_lr1e5_long_v1.md)
- [MLX full finetune 270M 994 title+abstract feed-blurb v5 control](mlx_lora_270m_natural_language_994_title_abstract_feed_blurb_v5_full_control_v1.md)
- [MLX LoRA 270M natural-language 994-row title+abstract feed-blurb v5 v1](mlx_lora_270m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr1e5_long_v1.md)
- [MLX LoRA 270M natural-language 994-row title+abstract v1](mlx_lora_270m_natural_language_994_title_abstract_lora_lr1e5_long_v1.md)
- [MLX LoRA 270M natural-language 994-row title+abstract prompt-v3 v1](mlx_lora_270m_natural_language_994_title_abstract_promptv3_lora_lr1e5_long_v1.md)
- [MLX LoRA 270M natural-language 994-row title+abstract prompt-v4 v1](mlx_lora_270m_natural_language_994_title_abstract_promptv4_lora_lr1e5_long_v1.md)
- [MLX LoRA 270M 994-gold plus heuristic-3000 replay v1](mlx_lora_270m_natural_language_994gold_heuristic3000_replay_v1.md)
- [MLX full finetune 270M natural-language fullmix gold4 support v1](mlx_lora_270m_natural_language_fullmix_gold4_support_v1.md)
- [MLX full finetune 270M natural-language fullmix gold6 selected-support v2](mlx_lora_270m_natural_language_fullmix_gold6_selectedsupport_v2.md)
- [q5 promoted q8 sidecar curated200 DPO v1](q5_promoted_q8_sidecar_curated200_dpo_v1.md)
- [q5 promoted sidecar natural-language 500 aggressive diagnostic](q5_promoted_q8_sidecar_natural_language_500_aggressive_v1.md)
- [q5 Promoted Q8 Sidecar Natural-Language Anchor Mix v1](q5_promoted_q8_sidecar_natural_language_500_anchor_mix_v1.md)
- [q5 promoted sidecar natural-language 500 low-LR no-anchor diagnostic](q5_promoted_q8_sidecar_natural_language_500_lowlr_noanchors_v1.md)
- [q5 promoted sidecar natural-language 500 no-anchor creep v2](q5_promoted_q8_sidecar_natural_language_500_lowlr_noanchors_v2.md)
- [q5 Promoted Q8 Sidecar NL500 Low-LR No-Anchor v3](q5_promoted_q8_sidecar_natural_language_500_lowlr_noanchors_v3.md)
- [q5 Promoted Q8 Sidecar Natural-Language 604 Anchor Mix v1](q5_promoted_q8_sidecar_natural_language_604_anchor_mix_v1.md)
- [q5 promoted sidecar natural-language burn-in32 stabilizer v1](q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v1.md)
- [q5 Promoted q8 Sidecar: Burn-In 32 Stabilizer v2](q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v2.md)
- [q5 Promoted q8 Sidecar: Burn-In 32 Stabilizer v3](q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v3.md)
- [q5 Promoted q8 Sidecar: Burn-In 32 Stabilizer v4](q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v4.md)
- [q5 promoted sidecar natural-language 994-gold burn-in v1](q5_promoted_q8_sidecar_natural_language_994_gold_burnin_v1.md)
- [q5 Promoted Q8 Sidecar Natural-Language 994 Token-Weighted Gold v1](q5_promoted_q8_sidecar_natural_language_994_token_weighted_gold_v1.md)
- [q5 promoted sidecar natural-language token-weighted gold v2](q5_promoted_q8_sidecar_natural_language_994_token_weighted_gold_v2.md)
- [q5 Promoted q8 Sidecar: Clean Digit-Free Fresh v1](q5_promoted_q8_sidecar_natural_language_clean_digitfree_10to14_fresh_v1.md)
- [q5 Promoted q8 Sidecar: Clean Digit-Free Fresh v2](q5_promoted_q8_sidecar_natural_language_clean_digitfree_10to14_fresh_v2.md)
- [q5 Promoted q8 Sidecar: Clean Digit-Free Fresh v3](q5_promoted_q8_sidecar_natural_language_clean_digitfree_10to14_fresh_v3.md)
- [q5 promoted q8 sidecar natural-language guarded support v1](q5_promoted_q8_sidecar_natural_language_guarded_support_v1.md)
- [q5 promoted q8 sidecar natural-language numeric guard v1](q5_promoted_q8_sidecar_natural_language_numeric_guard_v1.md)
- [q5 promoted q8-sidecar natural-language specificity support v1](q5_promoted_q8_sidecar_natural_language_specificity_support_v1.md)
- [q5 promoted q8 sidecar natural-language under14 high-support v1](q5_promoted_q8_sidecar_natural_language_under14_highsupport_v1.md)

## Granite 4.0 350M comparison

- [MLX LoRA Granite 350M 994-gold x8 plus strict-10746 support v1](mlx_lora_granite_350m_natural_language_994_gold8_strict10746_support_v1.md)
- [MLX LoRA Granite 350M natural-language 994-row title+abstract feed-blurb v5 v1](mlx_lora_granite_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1.md)
- [Granite 350M v5 q6 Fusion-Aware v1](mlx_lora_granite_350m_v5_q6_fusion_aware_v1.md)
- [Granite 350M v5 q8 Fusion-Aware v1](mlx_lora_granite_350m_v5_q8_fusion_aware_v1.md)
