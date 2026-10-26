# Natural-language teacher writer-yield rescue

## Goal

Improve yield before the reviewer by targeting rows the v7 writer already
rewrote, but the local gates discarded.

The important finding is that many writer-step losses were not semantic drops.
They were valid-looking rewrites with shape failures, especially summaries that
started with `A`, `An`, `The`, or `New`.

## Source Bucket

Source:

- `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v7_batch2/local_eligible_for_review/excluded.jsonl`

Subset:

- `123` rows with `writer_decision=rewrite`

Original local-filter failure counts in that subset:

| flag | count |
| --- | ---: |
| `forbidden_prefix` | 91 |
| `too_long` | 37 |
| `too_many_acronyms` | 11 |
| `unsupported_number_string` | 9 |
| `too_many_numbers` | 6 |
| `title_overlap` | 4 |

## v9 Format-Rescue Writer

Prompt:

- `prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_writer_v9_format_rescue.json`

Config:

- `configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v9_format_rescue_local_rejects.json`
- `configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v9_format_rescue_local_rejects_audit.json`

The v9 rescue writer sees:

- title
- abstract
- previous failed draft
- previous local-filter flags

It is asked to repair the sentence shape or drop the row.

Results:

| stage | count |
| --- | ---: |
| source local-rejected rewrites | 123 |
| v9 writer rewrites | 122 |
| v9 local eligible | 102 |
| normal verifier accepts | 42 |
| strict-audit accepts | 16 |

Credit use:

| stage | credits |
| --- | ---: |
| v9 writer | ~0.2313 |
| normal verifier | ~0.1808 |
| strict audit | ~0.0861 |
| total | ~0.4982 |

Read:

- v9 is a useful rescue lane, but not a replacement for v7.
- It improves local eligibility strongly.
- It sometimes rewrites too aggressively, which creates drift or extra jargon.
- Strict rejects were mostly `too_technical`, `title_rehash`, and `too_vague`.

## Deterministic Article-Strip Rescue

Experiment:

- Strip only an opening `A`, `An`, `The`, or `New` from v7 local-rejected
  rewrites.
- Re-run local gates.
- Send passing rows through the normal verifier and strict audit.

Artifacts:

- `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v7_article_strip_rescue_local_rejects/`
- `configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v7_article_strip_rescue_local_rejects.json`
- `configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v7_article_strip_rescue_local_rejects_audit.json`

Results:

| stage | count |
| --- | ---: |
| opening-repair candidates | 91 |
| local eligible after strip | 68 |
| normal verifier accepts | 37 |
| strict-audit accepts | 15 |

Read:

- Mechanical repair nearly matched v9 final yield with no writer-call cost.
- Normal-review yield was better than v9 because the original v7 semantic choice
  was preserved.
- This suggests the local-filter rescue should happen before any LLM rewriter
  rescue.

## Combined Release

New combined release:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v7_v8_v9_strip_rescue_155/training_rows.jsonl`
- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v7_v8_v9_strip_rescue_155/manifest.json`

Composition:

| component | input rows | added rows |
| --- | ---: | ---: |
| existing v7+v8 strict release | 129 | 129 |
| v9 format-rescue strict accepts | 16 | 16 |
| article-strip strict accepts | 15 | 10 |

The article-strip lane overlaps v9 on `5` paper IDs, so it adds `10` unique rows.

Validation:

- `155` rows
- `155` unique paper IDs
- max summary length: `14` words
- over `14` words: `0`

## Current Recommendation

For the next full run, the writer-side cascade should be:

1. Run v7 primary writer.
2. Run local gates.
3. For local-rejected rewrites with only simple opening/length failures, try a
   deterministic article-opening repair first.
4. Review and strict-audit those deterministic repairs.
5. Use v9-style LLM format rescue only for remaining local-rejected rewrites
   where mechanical repair does not pass.
6. Keep v8 as a separate rescue lane for normal-review rejects.

The big lesson: when the writer already found a good content anchor, preserve it.
Use LLM repair only when a tiny deterministic edit cannot make the row eligible.
