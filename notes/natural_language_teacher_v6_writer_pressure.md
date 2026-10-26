# Natural-language teacher v6 writer-pressure checkpoint

## What changed

`writer_v6` adds explicit self-check fields:

- `plain_takeaway`
- `technical_terms_used`
- `naturalness_risk`

The prompt tells the writer to drop rows when `naturalness_risk` is not `none`, and local filtering enforces:

- `writer_naturalness_risk == "none"`
- at most one writer-declared specialist term
- non-empty `writer_plain_takeaway`

This moves pressure upstream. Rows that are faithful but likely unreadable should die after the writer call instead of after normal verification plus strict audit.

## First 100 comparison

Against the same first `100` source candidates:

| run | local eligible | normal accepts | strict-audit accepts |
| --- | ---: | ---: | ---: |
| v5 | 71 | 36 | 14 |
| v6 | 17 | 12 | 9 |

Against the first `50` source candidates:

| run | local eligible | normal accepts | strict-audit accepts |
| --- | ---: | ---: | ---: |
| v5 | 34 | 20 | 7 |
| v6 | 9 | 8 | 7 |

## Current read

The writer-side pressure works. It sharply reduces downstream verifier and audit spend while preserving the first-50 audited yield.

The tradeoff is recall. v6 dropped some rows that v5 strict audit accepted, including rows about satellite photosynthesis records, stopwatch automata, segmentation masks, medical-report hallucinations, prompt ensembling, and LLM serving throughput.

## Current outputs

- v6 run root: `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v6`
- v6 release: `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v6`
- v6 strict-audit run root: `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v6_audit`
- v6 strict-audit release: `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v6_audit`

## Next prompt idea

The next writer should be less binary about technical terms:

- Still drop jargon chains and theorem-title prose.
- Allow one named technical object when the surrounding sentence is plain and the payoff is concrete.
- Do not drop solely because a domain term remains.
- Keep the self-check fields and local gates, but tune the prompt so `naturalness_risk=none` is possible for rows with one clear named object.

That likely lands between v5's recall and v6's cost control.
