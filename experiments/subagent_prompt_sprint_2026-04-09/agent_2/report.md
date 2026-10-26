# Agent 2 Report

## Hypothesis
If the generator spends more of its limited words on the payoff or result instead of restating the topic, the kept summaries should become more feed-useful and title-overlap failures should drop.

## Prompt Changes
- Added [deepseek_v32_generator_v4_payoff_first.json](/experiments/subagent_prompt_sprint_2026-04-09/agent_2/deepseek_v32_generator_v4_payoff_first.json).
- Added [qwen3_235b_a22b_2507_reviewer_classifier_v2_payoff_first.json](/experiments/subagent_prompt_sprint_2026-04-09/agent_2/qwen3_235b_a22b_2507_reviewer_classifier_v2_payoff_first.json).
- Generator direction:
  - choose the strongest open-worthy payoff first
  - treat title words as already spent
  - prefer result, improvement, limitation, or use case over topic restatement
  - keep the 12-15 word target and hard 15-word cap
  - include payoff ladder examples and explicit bad-to-good title-restatement examples
- Reviewer direction:
  - keep the classifier architecture unchanged
  - sharpen title-overlap wording so the title can be background, not the main content of the summary
  - keep accept/drop semantics and the same structured JSON schema

## Command Run
```bash
python3 scripts/run_synthetic_summary_pipeline.py \
  --sample-size 100 \
  --output-dir experiments/subagent_prompt_sprint_2026-04-09/agent_2/run_payoff_first_100_v1 \
  --generator-prompt experiments/subagent_prompt_sprint_2026-04-09/agent_2/deepseek_v32_generator_v4_payoff_first.json \
  --reviewer-prompt experiments/subagent_prompt_sprint_2026-04-09/agent_2/qwen3_235b_a22b_2507_reviewer_classifier_v2_payoff_first.json \
  --generator-model deepseek/deepseek-v3.2 \
  --reviewer-model qwen/qwen3-235b-a22b-2507
```

## Key Metrics
- The live 100-row run was not completed from this shell.
- The shell here could not resolve `openrouter.ai`, so no final manifest or row-level metrics were produced locally.
- Partial artifacts were written before the network failure, including sampled source rows and generation requests.
- The central run should be executed against the files in this agent folder to collect the actual keep/drop and title-overlap metrics.

## Judgment
The prompt direction is a good fit for this sprint because it attacks the main remaining generator weakness directly: spending words on payoff instead of topic restatement. The central live run will tell us whether the payoff-first framing actually lowers title overlap and produces more open-worthy summaries, but the prompt variant itself is ready.
