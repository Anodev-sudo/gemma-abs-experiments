# Natural-language teacher v7 writer-pressure checkpoint

## What changed

`writer_v7` keeps the self-check fields from `writer_v6`:

- `plain_takeaway`
- `technical_terms_used`
- `naturalness_risk`

The prompt is less severe than v6. It still drops jargon chains and theorem-title prose, but it allows one named technical object when the surrounding sentence is plain and the payoff is concrete.

Local filtering still enforces:

- `writer_naturalness_risk == "none"`
- non-empty `writer_plain_takeaway`
- at most `2` writer-declared specialist terms

## First 100 comparison

Against the same first `100` source candidates:

| run | local eligible | normal accepts | strict-audit accepts |
| --- | ---: | ---: | ---: |
| v5 | 71 | 36 | 14 |
| v6 | 17 | 12 | 9 |
| v7 | 41 | 21 | 15 |

Against the first `50` source candidates:

| run | local eligible | normal accepts | strict-audit accepts |
| --- | ---: | ---: | ---: |
| v5 | 34 | 20 | 7 |
| v6 | 9 | 8 | 7 |
| v7 | 23 | 11 | 8 |

## Current read

v7 is the best writer-side shape so far. It recovers more useful rows than v6 without returning to v5's downstream cost.

The gain is not huge in raw accepted rows, but the direction is good:

- fewer rows sent to the normal verifier than v5
- more strict-audit rows than v5 and v6
- fewer over-drops than v6

## Current outputs

- v7 run root: `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v7`
- v7 release: `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v7`
- v7 strict-audit run root: `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v7_audit`
- v7 strict-audit release: `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v7_audit`

## 100-row proof set

The v7 strict-audit release was extended to `200` source candidates, yielding `24` strict-audited rows.

A separate v7 batch2 source pool excluded the original v1 candidate pool and used the same writer, normal verifier, and strict audit:

- batch2 source candidates: `800`
- batch2 writer limit run so far: `650`
- batch2 local eligible rows: `243`
- batch2 normal accepts: `168`
- batch2 strict-audit accepts: `89`

The combined proof release uses the first `100` unique `paper_id` rows from:

- `24` rows from `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v7_audit/training_rows.jsonl`
- `76` rows from `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v7_batch2_audit/training_rows.jsonl`

Final proof release:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v7_100_good/training_rows.jsonl`
- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v7_100_good/manifest.json`

## Examples from the v7 strict-audit release

- `Mesoscopic pits in graphene scatter electrons more strongly than diffusive models predict.`
- `Five criteria help reviewers judge qualitative HCI studies without forcing quantification.`
- `Inexpensive GPS spoofing tricks Wi-Fi 6E access points into causing harmful interference.`
- `LLMs can generate and tweak custom chip layouts from natural language descriptions.`
- `Encrypted prompt ensembling makes private model queries more robust with little slowdown.`
- `LLM agents spontaneously formed groups instead of solving the bar problem optimally.`

## Next prompt idea

The next improvement should target false accepts, not just yield:

- Be more skeptical of unsupported numbers and percentages even when source-supported.
- Keep the v7 writer pressure.
- Consider a cheap local pre-audit rule for summaries whose payoff is mostly a metric.
- Consider adding a final local report that compares strict-audit accepts against writer-declared `technical_terms_used`.
