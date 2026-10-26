# Natural-Language Teacher Rescue Lanes v11-v13

## Goal

Test the non-math yield improvements suggested by the 300-row analysis:

- targeted too-long repair
- normal-verifier reason repair
- strict-audit jargon repair

This deliberately does not address the separate math/category-ratio problem.

## New Reusable Pieces

- `scripts/build_natural_language_rescue_lanes.py`
- `prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_writer_v11_length_rescue.json`
- `prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_writer_v12_verifier_reason_rescue.json`
- `prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_writer_v13_jargon_repair.json`

## Batch3 Results

### v11 Too-Long Rescue

Source:

- `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v7_batch3/local_eligible_for_review/excluded.jsonl`

Selection:

- `writer_decision=rewrite`
- `writer_local_filter_flags == ["too_long"]`
- previous draft at most `17` words

Results:

| stage | rows |
| --- | ---: |
| selected source rows | 40 |
| writer rewrites | 40 |
| local eligible | 30 |
| normal verifier accepts | 17 |
| strict-audit accepts | 10 |

Read:

- This is a strong standard rescue lane.
- It converts near-miss length failures into strict-good rows at `25%` of
  selected source rows.
- The main remaining local losses were `too_many_writer_technical_terms` and
  `forbidden_prefix`, which suggests it should run after deterministic opening
  cleanup and before broader LLM rescue.

### v12 Verifier-Reason Rescue

Source:

- `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v7_batch3/materialized_release/excluded_review_rows.jsonl`

Selection:

- normal verifier rejects tagged with one of:
  - `too_vague`
  - `too_generic`
  - `not_reason_to_open`
  - `title_rehash`
- excludes rows carrying wrong or unsupported tags

Results:

| stage | rows |
| --- | ---: |
| selected source rows | 90 |
| writer rewrites | 90 |
| local eligible | 38 |
| normal verifier accepts | 20 |
| strict-audit accepts | 5 |

Read:

- This version is too broad for the standard cascade.
- The writer was too willing to rewrite rows that the local gates and strict
  audit still did not like.
- Keep it as an experimental lane until selection is narrower or the prompt has
  a stronger drop bias.

### v13 Strict-Jargon Rescue

Source:

- `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v7_batch3_audit/materialized_release/excluded_review_rows.jsonl`

Initial selection:

- strict-audit rejects tagged with:
  - `too_technical`
  - `jargon_stack`
- excludes rows carrying wrong or unsupported tags

Results:

| stage | rows |
| --- | ---: |
| selected source rows | 30 |
| writer rewrites | 30 |
| local eligible | 18 |
| normal verifier accepts | 8 |
| strict-audit accepts | 5 |

Read:

- This is useful but should be narrowed.
- The strict-good rows came from rows tagged only `too_technical` or
  `too_technical + jargon_stack`.
- Rows that also carried `not_reason_to_open`, `too_vague`, or `title_rehash`
  did not contribute strict-good rows in this probe.
- The builder default now excludes those mixed tags for future strict-jargon
  rescue lanes.

## Combined Probe Release

Release:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_320_rescue_probe/`

Composition:

| component | added rows |
| --- | ---: |
| 300-row milestone | 300 |
| v11 too-long rescue | 10 |
| v12 verifier-reason rescue | 5 |
| v13 strict-jargon rescue | 5 |

Validation:

- rows: `320`
- unique paper IDs: `320`
- max summary length: `14`
- over-14-word summaries: `0`
- empty summaries: `0`

Current umbrella spread:

| umbrella | rows |
| --- | ---: |
| cs | 137 |
| physics | 103 |
| math | 6 |
| other | 74 |

## Recommendation

Updated non-math cascade:

1. Primary writer.
2. Local gates.
3. Deterministic opening cleanup.
4. Too-long-only repair for short over-limit drafts.
5. Normal verifier.
6. Strict audit.
7. Narrow strict-jargon repair only for `too_technical` or
   `too_technical + jargon_stack` strict rejects.
8. Treat verifier-reason repair as experimental until it has better selection.

Do not relax the final strict audit to improve yield. The strict audit is still
catching the exact failure modes these rescue prompts tend to introduce.
