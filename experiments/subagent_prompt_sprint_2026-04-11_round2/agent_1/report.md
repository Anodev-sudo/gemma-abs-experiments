# Agent 1 Report

## Prompt idea

Generator variant: [deepseek_v32_generator_agent1_plain_language_anti_notation_v1.json](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_1/deepseek_v32_generator_agent1_plain_language_anti_notation_v1.json)

Direction:
- keep the current payoff-first shape
- explicitly prefer plain-language paraphrases over symbolic notation
- avoid formula-shaped summaries and notation-heavy openings

Reviewer:
- unchanged baseline reviewer: [qwen3_235b_a22b_2507_reviewer_classifier_v2_headline_baseline.json](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/prompts/synthetic_summary/qwen3_235b_a22b_2507_reviewer_classifier_v2_headline_baseline.json)

## Run results

### Run A

Seed: `20260411`

Artifacts:
- [run_a/run_manifest.json](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_1/run_a/run_manifest.json)
- [run_a/training_rows.jsonl](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_1/run_a/training_rows.jsonl)

Manifest summary:
- kept: `77`
- dropped: `23`
- locally filtered before review: `2` (`over 15 words`)
- reviewer drops:
  - `misses the main payoff`: `19`
  - `too vague`: `2`

Kept-set read:
- generic openings: `13 / 77`
- notation-heavy kept summaries: `4 / 77`
- average kept length: `11.91` words

### Run B

Seed: `20260412`

Artifacts:
- [run_b/run_manifest.json](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_1/run_b/run_manifest.json)
- [run_b/training_rows.jsonl](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_1/run_b/training_rows.jsonl)

Manifest summary:
- kept: `83`
- dropped: `17`
- locally filtered before review: `2` (`over 15 words`)
- reviewer drops:
  - `misses the main payoff`: `12`
  - `mostly repeats the title`: `1`
  - `too vague`: `2`

Kept-set read:
- generic openings: `13 / 83`
- notation-heavy kept summaries: `0 / 83`
- average kept length: `11.73` words

## Two-run read

- Average kept rows: `80 / 100`
- Combined drop reasons:
  - `misses the main payoff`: `31`
  - `over 15 words`: `4`
  - `too vague`: `4`
  - `mostly repeats the title`: `1`
- Combined generic openings in kept rows: `26 / 160`
- Combined notation-heavy kept rows: `4 / 160`

What worked:
- The prose is genuinely plainer than the current baseline direction when the prompt lands.
- Run B in particular nearly eliminated symbolic surface form from the kept set.
- The line style stayed compact and mostly readable.

What did not work:
- Yield is only moderate.
- The main failure mode is still semantic weakness, not notation: the reviewer kept dropping summaries for missing the main payoff.
- Generic openings did not improve.
- A few notation-heavy lines still slipped through in Run A, including `μRWELL`, `Π¹₂`, `z=14.2`, and `O(log...)`.

## Recommendation

Do not promote this branch as the new baseline.

The anti-notation idea is directionally useful, but in this form it trades away too much payoff strength for only a partial readability gain. I would keep the best parts:
- explicit preference for plain-language paraphrase over symbolic notation
- explicit ban on formula-shaped openings

But I would merge those rules into a stronger payoff-first prompt instead of using this variant on its own.

Note: the JSONL artifacts are append-only and can contain stale rows from resumed progress, so the kept-set read above was computed from the final per-paper review decisions rather than raw line counts alone.
