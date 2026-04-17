# Agent 3 Report

## Hypothesis

Now that deterministic failures are filtered before review, the reviewer should focus only on semantic judgment. My bet was that a semantic-only gate would make the classifier less noisy by concentrating it on four checks: `faithful`, `captures_main_point`, `style_ok`, and `title_overlap_ok`.

## Prompt Changes

I created a new reviewer variant at `experiments/subagent_prompt_sprint_2026-04-09/agent_3/qwen3_235b_a22b_2507_reviewer_semantic_v1.json`.

Main changes:
- removed `one_sentence` and `wordcount_ok` from the reviewer schema
- explicitly told the model that length and sentence-count hygiene are handled elsewhere
- tightened the decision rule to `accept` only when all semantic checks are true
- kept `reason_span` short and plain-English for drops
- framed `faithful` as an evidence check and `captures_main_point` as the strongest-payoff test

## Command Run

```bash
python3 scripts/run_synthetic_summary_pipeline.py \
  --sample-size 100 \
  --seed 20260407 \
  --source-split agent_3_semantic_only_v1 \
  --generator-prompt prompts/synthetic_summary/deepseek_v32_generator_v3.json \
  --reviewer-prompt experiments/subagent_prompt_sprint_2026-04-09/agent_3/qwen3_235b_a22b_2507_reviewer_semantic_v1.json \
  --generator-model deepseek/deepseek-v3.2 \
  --reviewer-model qwen/qwen3-235b-a22b-2507 \
  --output-dir experiments/subagent_prompt_sprint_2026-04-09/agent_3/run_semantic_only_v1 \
  --max-retries 10 \
  --retry-backoff-seconds 10
```

## Key Metrics

The live run did not complete from this sandbox. The reviewer call kept failing with `URLError` / DNS lookup failures before the 100-row pass could finish.

Partial artifacts in `experiments/subagent_prompt_sprint_2026-04-09/agent_3/run_semantic_only_v1`:
- `sample_source_rows.jsonl`: `100` rows
- `generation_outputs.jsonl`: `100` rows, seeded from the cached baseline generation set
- `review_requests.jsonl`: `84` rows, meaning `16` candidates were filtered locally before review
- `review_outputs.jsonl`: `1` row written before the sandbox network stall
- `dropped_rows.jsonl`: `1` row written before the sandbox network stall

## Judgment

The prompt direction itself is sensible: local filters should own length and sentence hygiene, while the reviewer handles only semantic quality. I did not get a complete live run from this shell, so I would not treat these partial artifacts as evidence for the prompt quality one way or another. The centralized rerun should use this same prompt file and compare the final keep/drop mix against the current Qwen classifier baseline.
