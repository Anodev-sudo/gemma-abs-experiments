# Natural-Language Teacher Math Path v16

## Why This Exists

The v14 math path fixed math underproduction, but the accepted rows showed a
readability failure: some summaries were faithful and short while still reading
like compressed theorem titles.

The main failure pattern was a dense named-object stack:

- several specialist objects joined by `satisfy`, `share`, `match`,
  `stratify`, or `characterize`
- little ordinary-language result shape
- often high title overlap
- still technically true enough for the v5 math verifier to accept

## Current Default

For new math runs, use:

- writer: `prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_writer_v16_math.json`
- verifier/audit: `prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_verifier_v6_math.json`

The builder default in `scripts/build_natural_language_rescue_lanes.py` now
points the `math` lane at v16/v6.

## Policy Shift

Math coverage is still required. Pure theory is not a drop reason by itself.

The stricter rule is:

- keep a technical term only when it is needed for faithfulness
- use ordinary grammar to carry the result
- prefer at most three technical terms
- allow a fourth term only when the sentence remains easy to parse
- drop if the only faithful summary is a named-object chain

The writer now has to provide `plain_takeaway`. If that field cannot describe
the result shape without copying the same specialist terms, the row should be
dropped.

## Examples To Reject

- `Fundamental sequence systems for two ordinal notation systems satisfy the Bachmann property.`
- `Free Araki-Woods factors satisfy a strong rigidity property called omega-solidity.`
- `Cluster-tilted algebra and cluster repetitive category share the same Krull-Gabriel dimension.`
- `Fourier paraproduct operator norms between Hardy spaces match the symbol's Hardy norm.`
- `Survey classifies dimension-free Nullstellensatze for noncommutative polynomials by vanishing type.`

## 500-Row Milestone Prune

The current 500-row release was pruned in place:

- dropped: `10` dense math summaries
- backfilled: `10` strict-good overflow math rows from the v15 audit pool
- final rows: `500`
- math rows: `80`
- collapsed umbrella counts: `cs=185`, `physics=140`, `math=80`, `other=95`

Reproducible command:

```bash
python3 scripts/prune_natural_language_math_release.py \
  --release-dir hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_500_milestone \
  --replacement-jsonl hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v15_math_ratio_fill_500_audit/training_rows.jsonl \
  --drop-paper-id 2410.15953 \
  --drop-paper-id 1406.6160 \
  --drop-paper-id 2403.06270 \
  --drop-paper-id 2508.21403 \
  --drop-paper-id 2504.02098 \
  --drop-paper-id 2210.11798 \
  --drop-paper-id 2312.11717 \
  --drop-paper-id 2006.06904 \
  --drop-paper-id 2308.16695 \
  --drop-paper-id 2402.13084 \
  --backfill-paper-id 2206.06777 \
  --backfill-paper-id 2403.15997 \
  --backfill-paper-id 2304.06423 \
  --backfill-paper-id 2303.12547 \
  --backfill-paper-id 2301.07655 \
  --backfill-paper-id 2603.03395 \
  --backfill-paper-id 2501.11941 \
  --backfill-paper-id 2502.11542 \
  --backfill-paper-id 2408.06230 \
  --backfill-paper-id 1411.0183 \
  --reason "manual math natural-language prune for dense named-object summaries" \
  --overwrite
```
