# Natural-Language Teacher No-Credit Backlog Rescue

## Goal

Inventory what can be recovered without new writer-model spend while credits are
low.

There are two different no-credit buckets:

- already strict-good rows outside the 500-row milestone
- deterministic opening cleanup rows that become local-eligible, but still need
  verifier and strict-audit calls before entering a training release

## Already Strict-Good Overflow

These rows are already through strict audit and are not in the current 500-row
milestone:

| source release | unused rows |
|---|---:|
| `v15_math_ratio_fill_500_audit` | `37` |
| `v10_ratio_fill_500_nonmath_audit` | `25` |
| `v7_batch3_article_strip_rescue_local_rejects_audit` | `5` |
| `v14_math_batch3_drop_probe_audit` | `5` |

Unique strict-good rows outside the 500 milestone: `72`.

Spread:

| umbrella | rows |
|---|---:|
| math | `42` |
| cs | `12` |
| physics | `5` |
| q-bio | `4` |
| econ | `4` |
| stat | `2` |
| eess | `2` |
| q-fin | `1` |

These can be merged into a larger pool without model calls, but the math overflow
should still get a quick manual readability skim because the 500 milestone
manually pruned dense math rows.

## Deterministic Opening Cleanup

The script removes only opening `A`, `An`, `The`, or `New`, then reruns local
gates. No writer-model call is made.

Generated current-milestone cleanup lanes:

| source backlog | selected for cleanup | local-eligible after cleanup | still locally excluded |
|---|---:|---:|---:|
| `v10_ratio_fill_500_nonmath` | `153` | `133` | `20` |
| `v15_math_ratio_fill_500` | `2` | `1` | `1` |

Generated artifacts:

- `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v10_ratio_fill_500_nonmath_start_cleanup/`
- `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v15_math_ratio_fill_500_start_cleanup/`
- `configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v10_ratio_fill_500_nonmath_start_cleanup.json`
- `configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v10_ratio_fill_500_nonmath_start_cleanup_audit.json`
- `configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v15_math_ratio_fill_500_start_cleanup.json`
- `configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v15_math_ratio_fill_500_start_cleanup_audit.json`

The `134` local-eligible cleanup rows were verified after credits returned:

| source backlog | local-eligible | normal verifier accepts | strict-audit accepts |
|---|---:|---:|---:|
| `v10_ratio_fill_500_nonmath` | `133` | `63` | `32` |
| `v15_math_ratio_fill_500` | `1` | `1` | `0` |

The `32` strict-good cleanup rows are unique against both the 500-row milestone
and the `72` already strict-good overflow rows.

Current strict-good running pool: `604` unique rows.

Merged expansion-pool release:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_604_expansion_pool/training_rows.jsonl`
- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_604_expansion_pool/manifest.json`

Composition:

| bucket | unique rows |
|---|---:|
| 500-row milestone | `500` |
| already strict-good overflow | `72` |
| deterministic cleanup strict-good rows | `32` |

New cleanup strict-good spread:

| umbrella | rows |
|---|---:|
| cs | `15` |
| physics | `13` |
| stat | `2` |
| eess | `1` |
| nlin | `1` |

All extra strict-good rows outside the 500 milestone:

| umbrella | rows |
|---|---:|
| math | `42` |
| cs | `27` |
| physics | `18` |
| q-bio | `4` |
| econ | `4` |
| stat | `4` |
| eess | `3` |
| q-fin | `1` |
| nlin | `1` |

Materialized strict-audit releases:

- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v10_ratio_fill_500_nonmath_start_cleanup_audit/`
- `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v15_math_ratio_fill_500_start_cleanup_audit/`

## Older Backlog Also Available

Other deterministic cleanup candidates outside the current 500-milestone fill:

| source backlog | local-eligible after cleanup | not in 500 |
|---|---:|---:|
| `v7_batch3` | `94` | `65` |
| `v7_batch2` | `76` | `55` |
| `v13_batch3_strict_jargon_rescue` | `8` | `8` |

I did not generate these lanes yet because they are older/backfill pools and may
duplicate older rescue experiments. They are available if we want a broader
review queue.

## Not Free Yet

Too-long-only rows are easy to select, but the v11 repair itself needs a writer
call, so this is not a no-credit recovery path:

| source backlog | too-long-only rows at most 17 words |
|---|---:|
| `v10_ratio_fill_500_nonmath` | `40` |
| `v15_math_ratio_fill_500` | `10` |
| `v7_batch3` | `40` |
| `v7_batch2` | `14` |

## Read

Best immediate no-credit value:

1. Treat the `104` strict-good rows outside the 500 milestone as the current
   expansion pool.
2. Keep the remaining deterministic cleanup rejects out of the training set.
3. Do not treat local-eligible cleanup rows as strict-good. Local gates cannot
   catch unsupported value claims, wrong emphasis, or bland reasons to open.
