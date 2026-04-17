# Subagent Prompt Sprint

This folder is reserved for a five-agent prompt-design sprint focused on the
synthetic `abstract -> summary` pipeline.

Run artifacts now stage under the mirrored `hf_bucket/experiments/subagent_prompt_sprint_2026-04-09/` tree. Legacy local links may still exist at the old paths during migration.

Intended structure:

- `agent_1/`
- `agent_2/`
- `agent_3/`
- `agent_4/`
- `agent_5/`

Each agent run should be comparable against the current baseline:

- generator prompt: `deepseek_v32_generator_v3`
- reviewer prompt: `qwen3_235b_a22b_2507_reviewer_classifier_v1`
- reviewer mode: classifier
- local hard filters enabled before review
- smoke sample size: `100`

This directory is used for collating results after the delegated runs report back.
