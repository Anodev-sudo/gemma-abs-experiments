# Agent 5 Report

## Prompt Idea
Keep the current payoff-first baseline, but make the sentence itself easier to read:
- prefer one clean clause over stacked phrasing
- prefer strong everyday verbs over dense noun piles
- keep sentence case and direct openings
- prefer plainer wording over notation when both are faithful

I kept the reviewer unchanged:
- [baseline reviewer](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/prompts/synthetic_summary/qwen3_235b_a22b_2507_reviewer_classifier_v2_headline_baseline.json)

Generator variant:
- [deepseek_v32_generator_agent5_plain_verbs_v1.json](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_5/deepseek_v32_generator_agent5_plain_verbs_v1.json)

## Runs
### Run A
- seed: `20260411`
- manifest: [run_a/run_manifest.json](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_5/run_a/run_manifest.json)
- result: `77 kept / 23 dropped`
- locally filtered before review: `3`
- model reviewed: `97`
- `captures_main_point=false`: `19`
- `title_overlap_ok=false`: `4`
- main drop reasons from [run_a/dropped_rows.jsonl](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_5/run_a/dropped_rows.jsonl):
  - `misses the main payoff`
  - `over 15 words`
  - `too vague`

### Run B
- seed: `20260412`
- manifest: [run_b/run_manifest.json](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_5/run_b/run_manifest.json)
- result: `80 kept / 20 dropped`
- locally filtered before review: `5`
- model reviewed: `95`
- `captures_main_point=false`: `14`
- `title_overlap_ok=false`: `5`
- main drop reasons from [run_b/dropped_rows.jsonl](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_5/run_b/dropped_rows.jsonl):
  - `misses the main payoff`
  - `mostly repeats the title`
  - `over 15 words`

## Two-Run Read
- average result across the two manifests: `78.5 kept / 21.5 dropped`
- the main failure mode in both runs was still semantic, not formatting: the reviewer usually dropped rows for `misses the main payoff`
- local hard filtering stayed modest: `3` rows in run A, `5` rows in run B, all from obvious constraints like length

Readability did improve some. Many accepted lines are easier to parse at a glance than the denser baseline style:
- [run_a/training_rows.jsonl](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_5/run_a/training_rows.jsonl)
  - `Production monitoring generates realistic mocks for unit tests automatically.`
  - `DarkHorse uses UDP to cut onion service latency by up to 3.6x.`
  - `Domain knowledge cuts CFST column capacity prediction error by over 50%.`
- [run_b/training_rows.jsonl](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_5/run_b/training_rows.jsonl)
  - `LicenseGPT cuts license analysis time by 94% for commercial AI datasets.`
  - `AI spots Mars habitability clues from NASA mass spectrometry data.`
  - `Editing LLMs to keep both old and new facts avoids catastrophic forgetting.`

But the simplification came with a tradeoff. The syntax is often cleaner, yet the line sometimes loses the strongest hook and falls back to a safer topic statement. A second issue is that generic/article-led openings still show up too often:
- `A new μRWELL-PICOSEC detector achieves sub-24 picosecond timing for particle physics.`
- `A hybrid 6G protocol stack transmits both semantic and non-semantic data simultaneously.`
- `A WE economy model reduces inequality by linking moral responsibility and risk vulnerability.`

## Final Recommendation
This direction is useful, but I would not promote it over the current baseline.

Why:
- it does produce plainer, more readable sentences
- but it underdelivers on payoff often enough that both runs lost too many rows to `misses the main payoff`
- the anti-generic-opening discipline also weakened more than I’d want

My take:
- keep the idea as a secondary constraint for future prompt work
- do not adopt this exact variant as the new default
- if we revisit it, the next version should keep the plain-verb bias but restore harder pressure on direct non-article openings and the single strongest reason to open
