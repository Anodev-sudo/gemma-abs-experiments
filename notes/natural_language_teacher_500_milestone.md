# Natural Language Teacher 500-Row Milestone

## Summary

The 500-row milestone is a strict-audited, ratio-selected natural-language
teacher release. It keeps the 350-row math-probe base as the starting point,
then fills the remaining target gaps with strict-good continuation rows.

On `2026-05-22`, the release was manually pruned for math readability: `10`
math rows with dense named-object summaries were removed and replaced with `10`
strict-good math overflow rows from the v15 math audit pool.

Release:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_500_milestone/`

## Target Mix

The release matches the current corpus policy:

| collapsed umbrella | target rows | actual rows |
| --- | ---: | ---: |
| cs | 185 | 185 |
| physics | 140 | 140 |
| math | 80 | 80 |
| other combined | 95 | 95 |

Rows: `500`
Unique paper IDs: `500`
Max summary words: `14`
Over-14-word summaries: `0`
Empty summaries: `0`

## Source Components

| component | source strict-good rows | selected rows |
| --- | ---: | ---: |
| 350-row math probe base | 350 | 345 |
| v10 non-math continuation strict audit | 131 | 106 |
| v15 math continuation strict audit | 86 | 49 |

The selection policy is:

1. Keep rows from the 350-row strict-good base except manually pruned math
   readability offenders.
2. Fill cs, physics, and other gaps from the v10 non-math strict-audited
   continuation release in release order.
3. Fill the math gap from the v15 math strict-audited continuation release,
   skipping manually pruned math readability offenders and using audited
   overflow rows for backfill.
4. Stop at exact `185/140/80/95` collapsed umbrella counts.

## Continuation Runs

### v10 Non-Math Continuation

Source:

- `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v10_ratio_fill_500/source_candidates/natural_language_candidates.jsonl`

Selection:

- exclude `primary_category` values starting with `math.`
- reuse the already completed non-math writer outputs from the paused v10 run

Results:

| stage | rows |
| --- | ---: |
| selected source rows | 600 |
| cached writer outputs reused | 44 |
| writer rewrites | 464 |
| writer drops | 136 |
| local eligible | 254 |
| normal verifier accepts | 180 |
| strict-audit accepts | 131 |

Strict-good yield from selected non-math rows: `131/600 = 21.8%`.

### v15 Math Continuation

Source:

- same paused v10 ratio-fill source pool

Selection:

- `primary_category` starts with `math.`
- first `150` math rows from that source pool
- route through v14 math writer and v5 math verifier

Results:

| stage | rows |
| --- | ---: |
| selected source rows | 150 |
| math writer rewrites | 146 |
| math writer drops | 4 |
| local eligible with math gate | 112 |
| normal math verifier accepts | 92 |
| strict math audit accepts | 86 |

Strict-good yield from selected math rows: `86/150 = 57.3%`.

Read this as a real measured result for this continuation slice, not a permanent
forecast for every future math batch. It is stronger than the old-drop probe
because these were fresh math source rows, not only rows the old general writer
had rejected.

## Quality Read

The milestone keeps the strict-audit rule: only rows accepted by the final audit
enter the release.

Math rows are intentionally allowed to stay result-shaped. Some accepted math
summaries still contain necessary terms such as `K-theory`, `Fourier`, or
`operator`, but the post-prune bar is stricter about stacked named-object
summaries. This is the current coverage tradeoff: math should read like a plain
description of the result shape, not like an applied payoff when the paper has
none, and not like a compressed theorem title.

Example rows:

- `Scenario-aware bin packing keeps performance guarantees under uncertain demand.`
- `Conjectured eigenvalue maximum is confirmed for all parallelogram shapes.`
- `Dynamic early exit cuts point cloud registration compute without hurting accuracy.`
- `Correcting for rater severity produces fairer estimates of AI output quality.`

Pruned example failures:

- `Fundamental sequence systems for two ordinal notation systems satisfy the Bachmann property.`
- `Cluster-tilted algebra and cluster repetitive category share the same Krull-Gabriel dimension.`
- `Fourier paraproduct operator norms between Hardy spaces match the symbol's Hardy norm.`

## Cost Snapshot

OpenRouter credits before the continuation were about `6.69` remaining. After
the 500 milestone, credits were about `3.34` remaining. The continuation spent
about `3.35` credits.
