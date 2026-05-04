# q4 Kimi Correction Salvage No Reasoning v1

## Summary

This is the compact record of the correction pass that survived the q4 repair
detour. The useful output is the `92`-row Kimi/Qwen verified release that was
later merged into `recent_3y_balanced_10746_q4_hardening_combined_v1`.

The abandoned DeepSeek/reasoning repair attempts, Qwen 3.5/3.6 repair probes,
manual override scaffolding, and intermediate retry directories were removed
from the source tree or pruned from `hf_bucket`.

## Kept Artifacts

- Release:
  `hf_dataset/releases/recent_3y_balanced_10746_q4_kimi_correction_salvage_no_reasoning_v1`
- Verification input:
  `hf_bucket/experiments/mlx_lora/q4_kimi_correction_salvage_no_reasoning_v1/local_eligible_for_qwen_verification/corrections_ordered.jsonl`
- Qwen verification output:
  `hf_bucket/experiments/mlx_lora/q4_kimi_correction_salvage_no_reasoning_v1/qwen_verification_run/verifications_ordered.jsonl`
- Materialized selected/excluded rows:
  `hf_bucket/experiments/mlx_lora/q4_kimi_correction_salvage_no_reasoning_v1/materialized_release`

## Counts

| Item | Count |
| --- | ---: |
| q4-severe candidate rows entering correction work | 278 |
| Local-clean correction rows sent to Qwen verification | 254 |
| Qwen-accepted training rows | 92 |
| Qwen-rejected rows | 162 |

Accepted-row buckets:

| Bucket | Rows |
| --- | ---: |
| numeric | 45 |
| polarity | 29 |
| entity | 10 |
| long_notation | 6 |
| control | 2 |

## What To Reuse

For future correction passes, reuse only the compact pattern:

1. Build a small candidate pool from confirmed q4-only severe failures.
2. Generate concise no-reasoning rewrites with strict local validation.
3. Verify with an independent model.
4. Materialize only verified rows into a separate release.

Do not reuse the abandoned reasoning-model branch as-is. It spent too much time
and token budget on hard leftovers and produced poor operational yield.
