# Agent 1 Report

## Hypothesis

Make the generator start from the payoff instead of a generic article-style lead-in. The idea was that stricter opening guidance would reduce `A ...` and `New ...` summaries while still keeping enough rows alive under the existing hard filters.

## Prompt Changes

- Added a generator variant at `experiments/subagent_prompt_sprint_2026-04-09/agent_1/prompts/deepseek_v32_generator_agent1_result_first_v1.json`.
- Kept the Qwen classifier reviewer as the control at `experiments/subagent_prompt_sprint_2026-04-09/agent_1/prompts/qwen3_235b_a22b_2507_reviewer_classifier_v1.json`.
- Generator changes focused on:
  - forcing a direct, result-first opening
  - discouraging article-style lead-ins like `A method`, `A model`, and `A framework`
  - keeping the same 12-15 word target and 15-word hard cap
  - preserving one-payoff-only compression

## Command Run

```bash
python3 scripts/run_synthetic_summary_pipeline.py \
  --sample-size 100 \
  --seed 20260407 \
  --generator-prompt experiments/subagent_prompt_sprint_2026-04-09/agent_1/prompts/deepseek_v32_generator_agent1_result_first_v1.json \
  --reviewer-prompt experiments/subagent_prompt_sprint_2026-04-09/agent_1/prompts/qwen3_235b_a22b_2507_reviewer_classifier_v1.json \
  --generator-model deepseek/deepseek-v3.2 \
  --reviewer-model qwen/qwen3-235b-a22b-2507 \
  --output-dir experiments/subagent_prompt_sprint_2026-04-09/agent_1/runs/result_first_v1
```

## Key Metrics

- The run completed successfully.
- Final outcome: `53` kept / `47` dropped.
- Local hard filters removed `27` rows before review, all for `over 15 words`.
- Of the `53` kept rows, `9` still used a generic opening pattern.
- Kept summaries averaged `13.4` words, with `0` over the `15` word cap.
- Reviewer decisions were `53 accept` / `47 drop`.
- Compared with the prior baseline on the same sample, keep rate stayed flat while generic openings among kept rows dropped from `16` to `9`.
- The tradeoff is that generator-side `too_long` leakage got worse: `27` local drops here versus `16` in the baseline.

## Judgment

This direction is promising but not a clean win. It did reduce generic openings in the final kept set while preserving the keep rate, which is the main thing I wanted to test. But it also increased overlong generator outputs a lot, so the prompt needs a length-control follow-up if we want to keep this result without paying for it in upstream waste.
