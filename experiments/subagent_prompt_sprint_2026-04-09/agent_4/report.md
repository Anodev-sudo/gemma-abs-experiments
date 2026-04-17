# Agent 4 Report

## Hypothesis

An example-driven generator prompt will do a better job than a rule-heavy prompt at teaching the model to sound feed-native, especially on the failure mode we have been seeing: artifact-led or method-led openings that read like paper abstracts instead of mobile blurbs.

## Prompt Changes

- Added [`deepseek_v32_generator_v4_examples_v1.json`](./prompt_variants/deepseek_v32_generator_v4_examples_v1.json).
- Kept the reviewer architecture unchanged by copying the current Qwen classifier prompt into [`qwen3_235b_a22b_2507_reviewer_classifier_v1.json`](./prompt_variants/qwen3_235b_a22b_2507_reviewer_classifier_v1.json).
- Shifted the generator prompt toward bad-to-good demonstrations and pattern cues instead of adding more standalone rules.
- Added [`run_pipeline_with_resolver.py`](./run_pipeline_with_resolver.py) as a local-only DNS shim because the shell environment could not resolve OpenRouter hosts directly.

## Command Run

```bash
python3 experiments/subagent_prompt_sprint_2026-04-09/agent_4/run_pipeline_with_resolver.py \
  --sample-size 100 \
  --output-dir experiments/subagent_prompt_sprint_2026-04-09/agent_4/runs/example_driven_generator_v1b \
  --generator-prompt experiments/subagent_prompt_sprint_2026-04-09/agent_4/prompt_variants/deepseek_v32_generator_v4_examples_v1.json \
  --reviewer-prompt experiments/subagent_prompt_sprint_2026-04-09/agent_4/prompt_variants/qwen3_235b_a22b_2507_reviewer_classifier_v1.json
```

## Key Metrics

- Baseline kept `53` / dropped `47`.
- Branch kept `50` / dropped `50`.
- Baseline local-filtered `16`; branch local-filtered `23`.
- Baseline model-reviewed `84`; branch model-reviewed `77`.
- Baseline average kept length `12.64` words; branch average kept length `12.88` words.
- Baseline generic openings in kept set `16`; branch generic openings in kept set `17`.
- Branch kept set had `0` summaries over `15` words.
- Branch kept set had `25` `captures_main_point = false` rows and `16` `title_overlap_ok = false` rows.

## Judgment

The example-heavy branch is a mixed result, not a clear win.

It did improve the mechanical side of the generator:
- more rows were filtered locally before review
- fewer rows needed model review
- no kept row exceeded the hard 15-word cap

But the feed-native quality did not clearly improve:
- the kept set was slightly smaller
- average length actually ticked up a bit
- generic openings did not go down
- the kept set still had a lot of `captures_main_point = false` and `title_overlap_ok = false` rows

My read is that the prompt examples helped the model stay within the house style, but they did not yet teach it to choose better openings consistently. I would not recommend this branch as a replacement baseline. It is worth keeping as a direction, but it needs another pass with stronger bad-to-good examples that target generic openings more directly.
