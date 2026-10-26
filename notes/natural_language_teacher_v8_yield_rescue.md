# Natural-language teacher v8 yield rescue

## Goal

Improve strict-audit yield without lowering the natural-language quality bar.

The main v7 batch2 strict-audit rejects were mostly:

- `too_vague`
- `title_rehash`
- `too_technical`

So v8 was designed as a writer-side pressure change:

- require one concrete non-title detail
- prefer a result, condition, finding, effect, or setting over a method name
- drop rows where the only faithful summary is a method label plus a generic benefit
- simplify specialist terms earlier

Prompt:

- `prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_writer_v8.json`

## Replacement Probe

First test: run v8 as a direct replacement for v7 on the same first `150` batch2 source rows.

| run | source rows | writer rewrites | local eligible | normal accepts | strict accepts |
| --- | ---: | ---: | ---: | ---: | ---: |
| v7 first 150 | 150 | 91 | 55 | 32 | 17 |
| v8 first 150 | 150 | 69 | 50 | 31 | 16 |

Read:

- v8 was more selective, but not better as a replacement.
- It slightly improved strict accepts per local-eligible row.
- It did not improve strict accepts per source row.

## Cascade Probe

Second test: use v8 only as a rescue pass for rows that v7 batch2 sent to the
normal reviewer but the normal reviewer rejected.

Input:

- `75` v7 batch2 normal-review rejects

Generated rescue source:

- `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v8_rescue_normal_rejects/source_candidates/natural_language_candidates.jsonl`

Generated configs:

- `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v8_rescue_normal_rejects/config.json`
- `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v8_rescue_normal_rejects/audit_config.json`

Results:

| stage | count |
| --- | ---: |
| rescue source rows | 75 |
| v8 local eligible | 37 |
| normal accepts | 24 |
| strict-audit accepts | 16 |

Read:

- v8 is useful as a targeted second writer, not as the primary writer.
- The cascade adds `16` strict-audited rows from `75` targeted retries.
- That improves batch2 strict-audited yield from `89/650` to `105/650` source rows, with `75` additional writer calls.

## Combined Release

Release:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v7_plus_v8_rescue_129/training_rows.jsonl`
- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v7_plus_v8_rescue_129/manifest.json`

Composition:

- `24` original v7 strict-audited rows
- `89` v7 batch2 strict-audited rows
- `16` v8 rescue strict-audited rows
- `129` unique rows total

Validation:

- `129` unique paper IDs
- max summary length: `14` words
- over `14` words: `0`

## Credit Use

Credit usage before v8 work:

- `71.761959214`

Credit usage after v8 replacement probe plus rescue cascade:

- `72.464270419`

Approximate cost:

- `0.7023` credits

## Next Step

The best current yield workflow is:

1. Run v7 primary writer.
2. Run local gates.
3. Run normal verifier.
4. Run strict audit on normal accepts.
5. Run v8 rescue only on v7 normal-review rejects.
6. Run local gates, normal verifier, and strict audit for rescued rows.

The next improvement should target v7 audit rejects separately. The v8 rescue
worked best on v7 normal rejects; its marginal rate on v7 audit rejects was lower
in the first-150 probe.
