# Natural Language Teacher 300-Row Milestone

## Summary

The 300-row natural-language teacher milestone was built from strict-audited
DeepSeek V4 Pro rows. The exact 300-row release keeps the first 300 rows from a
305-row strict-good pool.

## Releases

- Exact milestone:
  `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_300_milestone/`
- Overflow pool:
  `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_300_milestone_pool_305/`

## Source Components

| component | strict-good rows |
| --- | ---: |
| Existing v7/v8/v9/article-strip release | 155 |
| Batch3 v7 primary strict audit | 116 |
| Batch3 article-strip rescue strict audit | 34 |
| Total pool | 305 |

No duplicate paper IDs were skipped during the 305-row merge.

## Batch3 Yield

Batch3 started from 850 fresh source abstracts with no overlap against the
earlier source pools or the existing 155-row release.

| stage | rows |
| --- | ---: |
| v7 writer outputs | 850 |
| local eligible for normal review | 298 |
| normal verifier accepts | 177 |
| strict-audit accepts | 116 |
| article-strip local eligible rescue rows | 80 |
| article-strip normal verifier accepts | 51 |
| article-strip strict-audit accepts | 34 |

Combined batch3 strict-good yield was 150 / 850, or 17.6%.

## Validation

- Exact milestone rows: 300
- Unique paper IDs: 300
- Max summary length: 14 words
- Over-14-word summaries: 0
- Empty summaries: 0
- Overflow pool rows: 305
- Overflow pool unique paper IDs: 305
