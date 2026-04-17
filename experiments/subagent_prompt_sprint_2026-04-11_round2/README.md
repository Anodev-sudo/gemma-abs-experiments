# Round 2 Prompt Sprint

Run artifacts now stage under the mirrored `hf_bucket/experiments/subagent_prompt_sprint_2026-04-11_round2/` tree. Legacy local links may still exist at the old paths during migration.

Current baseline:
- generator: prompts/synthetic_summary/deepseek_v32_generator_v5_hybrid_sentence_case_minimal.json
- reviewer: prompts/synthetic_summary/qwen3_235b_a22b_2507_reviewer_classifier_v2_headline_baseline.json

Goal for this sprint:
- Explore prompt directions that preserve hook/payoff quality while pushing toward plainer, less title-like, less notation-heavy prose.
- Each branch should run two live 100-row trials to reduce outlier risk.
- Use distinct output dirs per run.
