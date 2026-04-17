# Agent 2 Report

## Direction

This branch pushes harder against generic openings than the current baseline generator.

Main idea:
- keep the current payoff-first, sentence-case style
- attack article-led and label-led openings more aggressively
- force the first few words to carry the hook
- reject openings that merely classify the work instead of naming what changes

Generator prompt:
- [deepseek_v32_generator_agent2_opening_attack_v1.json](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_2/deepseek_v32_generator_agent2_opening_attack_v1.json)

Reviewer used unchanged:
- [qwen3_235b_a22b_2507_reviewer_classifier_v2_headline_baseline.json](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/prompts/synthetic_summary/qwen3_235b_a22b_2507_reviewer_classifier_v2_headline_baseline.json)

## Runs

### Run A

- Seed: `20260411`
- Output dir: [run_a](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_2/run_a)
- Manifest: [run_manifest.json](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_2/run_a/run_manifest.json)

Manifest summary:
- kept: `75`
- dropped: `25`
- locally filtered before review: `4`
- model-reviewed: `96`
- `captures_main_point = false`: `21`
- `title_overlap_ok = false`: `5`

Row-level opening read:
- accepted summaries with article-led openings: `6`
- accepted summaries with label-led openings like `A method`, `A model`, `A framework`, `Reviewing`, `Modeling`: `0`
- average accepted length: `11.97` words

Main drop reasons from final review decisions:
- `misses the main payoff`: `15`
- `over 15 words`: `4`
- `too vague`: `3`
- `mostly repeats the title`: `2`

### Run B

- Seed: `20260412`
- Output dir: [run_b](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_2/run_b)
- Manifest: [run_manifest.json](/Users/alanwren/myApps/model-experiments/gemma-abstract-finetuning/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_2/run_b/run_manifest.json)

Manifest summary:
- kept: `78`
- dropped: `22`
- locally filtered before review: `5`
- model-reviewed: `95`
- `captures_main_point = false`: `17`
- `title_overlap_ok = false`: `6`

Row-level opening read:
- accepted summaries with article-led openings: `3`
- accepted summaries with label-led openings like `A method`, `A model`, `A framework`, `Reviewing`, `Modeling`: `0`
- average accepted length: `11.90` words

Main drop reasons from final review decisions:
- `misses the main payoff`: `14`
- `over 15 words`: `5`
- `mostly repeats the title`: `1`
- `too vague`: `1`
- `too narrow, misses the main payoff of the paper`: `1`

## Combined Read

Average official result across the two manifests:
- kept: `76.5 / 100`
- dropped: `23.5 / 100`
- locally filtered: `4.5 / 100`

Combined opening read across the two accepted sets:
- article-led openings: `9`
- label-led openings: `0`
- average accepted length: about `11.9` words

Important note:
- the raw JSONL artifacts contain duplicate rows from checkpoint/resume behavior
- the manifest counts are the trustworthy totals
- the opening counts above come from final per-paper review decisions after deduping by `paper_id`

## Qualitative Read

What improved:
- the branch did succeed at killing the ugliest label-led starts
- I saw no accepted lines starting with patterns like `A method`, `A model`, `A framework`, `A review`, or `Modeling ...`
- the better outputs open directly and still carry a hook, for example:
  - `LicenseGPT cuts license analysis time by 94% for software IP lawyers.`
  - `SEAL estimates per-prompt carbon emissions to guide sustainable LLM use.`
  - `BibAgent verifies citations at scale, even for paywalled papers, using an adaptive agentic framework.`

What did not improve enough:
- article-led openings still survive in some accepted rows, especially for math and stats papers
- examples that still feel too generic at the front:
  - `A convex condition predicts success for generalized demixing problems.`
  - `A hybrid protocol stack lets 6G networks mix semantic and traditional transmissions.`
  - `A dual model of liquids quantifies thermophysical properties by tracking excited degrees of freedom.`

Main downside:
- the branch lost too many rows to `misses the main payoff`
- that suggests the stronger anti-generic pressure sometimes pushes the model into cleaner openings without actually sharpening the best reason to open the paper

## Final Recommendation

Not recommended as the new baseline.

My read:
- this direction is partially successful
- it clearly improves the worst label-led opening patterns
- but it costs too much in yield and still does not fully solve article-led starts
- the strongest reusable part is the explicit anti-label opening language, not the full prompt as written

If this branch is mined for ideas, I would keep:
- the ban on label-led openings
- the rule that the first few words must carry the hook
- the test `if the opening could fit many papers, rewrite`

I would not replace the current baseline with this full prompt.
