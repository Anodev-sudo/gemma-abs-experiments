# Natural-Language Teacher Math Path v14

Historical note: this path fixed math underproduction, but it has been
superseded as the default by v16/v6 because v14/v5 accepted some faithful but
overly dense named-object summaries. Keep this note for measured v14 results.

## Goal

Fix the structural underproduction of math rows without weakening the general
natural-language workflow.

The general v7 writer treated pure math and theorem-heavy abstracts as drop
signals. That made sense for broad non-math readability, but it was incompatible
with the target dataset mix.

## Math Natural-Language Definition

For `primary_category` values starting with `math.`, a useful feed summary does
not need a practical or real-world payoff.

A math summary should describe the result shape in plain prose:

- condition
- bound
- algorithm
- proof
- classification
- equivalence
- counterexample
- explanation
- connection between mathematical objects

Allowed math words include:

- proof
- theorem
- bound
- condition
- classification
- counterexample
- algorithm
- curve
- graph
- surface
- space
- equation
- tiling
- dimension
- operator
- algebra
- `L-function`

The summary should still avoid:

- formulas and LaTeX
- parameter strings
- symbol-heavy notation
- dense named-object chains
- vague claims like `new theorem improves understanding`
- unsupported solved-conjecture or real-world claims

## New Artifacts

Prompts:

- `prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_writer_v14_math.json`
- `prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_verifier_v5_math.json`

Builder:

- `scripts/build_natural_language_rescue_lanes.py --lane math`

The math lane applies one broad category rule:

- include rows where `primary_category` starts with `math.`

For math-only configs, the builder relaxes local gates to:

- `max_writer_technical_terms = 4`
- `max_long_words = 3`
- `max_acronyms = 2`

This is still paired with the math verifier; the local gate should catch shape
failures, not be the main math-quality judge.

## Probe

Source:

- `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v7_batch3/writer_run/writer_outputs_ordered.jsonl`

Selection:

- `primary_category` starts with `math.`
- old `writer_decision` was not `rewrite`
- first `60` selected rows

Config:

- `configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v14_math_batch3_drop_probe.json`
- `configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v14_math_batch3_drop_probe_audit.json`

Results:

| stage | rows |
| --- | ---: |
| selected old math drops | 60 |
| math writer rewrites | 54 |
| math writer drops | 6 |
| local eligible with math gate | 35 |
| normal math verifier accepts | 32 |
| strict math audit accepts | 30 |

Read:

- The old writer was heavily over-dropping math.
- The math definition produced a `50%` strict-good yield from old math drops.
- The remaining local losses were mostly still shape-related:
  - `too_long = 10`
  - `too_many_writer_technical_terms = 9`
  - `empty_summary = 6`
  - `forbidden_prefix = 2`

## Probe Release

Strict-good math release:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v14_math_batch3_drop_probe_audit/`

Combined with the 320-row rescue probe:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_350_math_probe/`

Validation:

- rows: `350`
- unique paper IDs: `350`
- max summary length: `14`
- over-14-word summaries: `0`
- empty summaries: `0`

Umbrella spread after merge:

| umbrella | rows |
| --- | ---: |
| cs | 137 |
| physics | 103 |
| math | 36 |
| other | 74 |

## Historical Recommendation

The original recommendation was to use this math path for all `math.*` rows:

1. Select math rows with a broad category gate.
2. Run v14 math writer.
3. Run math local gates.
4. Run v5 math verifier.
5. Run v5 math verifier again as strict audit.
6. Merge only strict-good math rows.

Do not split by math subcategory yet. The broad gate is already enough to recover
substantial coverage.

Next useful improvement:

- add a math-specific local-reject rescue for `too_long` and simple over-jargon
  near misses, because many rejected v14 rewrites are close rather than
  semantically bad.

Current replacement:

- use v16 math writer
- use v6 math verifier/audit
- keep v14/v5 yields as historical measurements only
