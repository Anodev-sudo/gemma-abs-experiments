# Agent 4 Report

## Prompt idea

Generator prompt: [deepseek_v32_generator_agent4_theory_calm_v1.json](/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_4/deepseek_v32_generator_agent4_theory_calm_v1.json)

Reviewer: unchanged baseline [qwen3_235b_a22b_2507_reviewer_classifier_v2_headline_baseline.json](/prompts/synthetic_summary/qwen3_235b_a22b_2507_reviewer_classifier_v2_headline_baseline.json)

Direction:
- Keep the generator payoff-first.
- For math, theory, statistics, and survey-style papers, prefer calm, literal statements of the actual gain.
- Avoid splashy verbs when the real value is a proof, bound, characterization, or clarified condition.
- Try to improve trustworthiness without drifting into bland topic labels.

## Run results

### Run A

Manifest: [run_manifest.json](/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_4/run_a/run_manifest.json)

- Seed: `20260411`
- Kept: `81`
- Dropped: `19`
- Locally filtered before review: `6`, all `over 15 words`
- Model-reviewed rows: `94`
- `captures_main_point=false`: `13`
- `title_overlap_ok=false`: `0`
- Kept average length: `12.31` words
- Generic openings in kept rows: `15`
- Main drop reasons: `misses the main payoff` `12`, `over 15 words` `6`, `too vague` `1`

### Run B

Manifest: [run_manifest.json](/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_4/run_b/run_manifest.json)

- Seed: `20260412`
- Kept: `80`
- Dropped: `20`
- Locally filtered before review: `5`, all `over 15 words`
- Model-reviewed rows: `95`
- `captures_main_point=false`: `15`
- `title_overlap_ok=false`: `4`
- Kept average length: `12.31` words
- Generic openings in kept rows: `11`
- Main drop reasons: `misses the main payoff` `10`, `over 15 words` `5`, `mostly repeats the title` `3`, `too vague` `2`

## Two-run read

- Combined kept rows: `161 / 200`
- Average kept rows per run: `80.5`
- Combined kept average length: `12.31` words
- Combined generic openings: `26 / 161`
- The branch is stable across seeds. It did not spike high or crater.
- The main failure mode is consistent: the calmer style sometimes gives up too much hook, so the reviewer drops the line for `misses the main payoff`.
- Hard-length control stayed decent. Local drops were only `6` and `5`, both from overlong candidates.

## Theory-heavy read

The theory-friendly nudge helped in some places.

Good examples:
- [run_a/training_rows.jsonl](/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_4/run_a/training_rows.jsonl)
  `Local feedback and open-loop Nash equilibria share optimality conditions in general zero-sum games.`
- [run_a/training_rows.jsonl](/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_4/run_a/training_rows.jsonl)
  `Measurable dilators isolate the structural core of Martin's large-cardinal proof of Π¹₂-determinacy.`
- [run_a/training_rows.jsonl](/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_4/run_a/training_rows.jsonl)
  `Geometrically finite ball quotients are Stein when the critical exponent is below 2.`

These feel more literal and trustworthy than the more aggressive headline style.

But the branch still has some weak spots:
- It sometimes stays artifact-led, for example `A third-order dynamical system solves...`
- A few theory lines are still too interpretive or awkward, for example the Tarski summary in [run_b/training_rows.jsonl](/experiments/subagent_prompt_sprint_2026-04-11_round2/agent_4/run_b/training_rows.jsonl)
- The calmer tone did not automatically reduce generic openings as much as I wanted

## Recommendation

I would not promote this as the new full baseline.

My read:
- The theory-friendly rule is directionally good.
- It does improve some math/theory/stat summaries by making them sound more literal and less overhyped.
- But as a whole prompt, it gives back too much payoff pressure and loses too many rows to `misses the main payoff`.

Best use of this branch:
- Keep it as a reference prompt for future theory-aware tuning.
- Borrow the calm-theory instruction into the main baseline more selectively, instead of adopting this version outright.
