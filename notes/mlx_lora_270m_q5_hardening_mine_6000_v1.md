# q5 Hardening Mine 6000 v1

## Purpose

Return to the pre-overlay q5 checkpoint and widen the mining surface. The goal
is to identify rows where promoted q8 stays usable while pre-overlay q5 fails,
then use those rows for a larger corrective dataset.

## Artifacts

- Eval set: `hf_dataset/evals/q5_hardening_mine_6000_v1`
- Paired generations: `hf_bucket/experiments/mlx_lora/review_comparisons/q8_vs_q5_hardening_mine_6000_v1`
- Review packets: `hf_bucket/experiments/mlx_lora/q5_hardening_distill_6000_v1/review_packets`

## Sample Construction

The 6000-row sample reused the same deterministic hardening bucket logic as the
q4 mine, scaled to:

| Bucket | Rows |
| --- | ---: |
| numeric | 2100 |
| polarity | 1500 |
| entity | 1200 |
| long_notation | 600 |
| control | 600 |

Important exclusions:

- Main 10746-row training release
- Faithfulness overlay release
- Prior q5 1000-row mining comparison
- q5 Kimi/Qwen correction overlay releases

The first attempt also excluded old q4 review surfaces, but that left too little
recent-corpus surface area for 6000 rows. For this q5 mine, the old q4 review
surfaces were allowed back in because the failure target is q5-specific.

## Generation

Compared:

- Left: promoted q8 standalone MLX model, `faith_overlay_1400_v2_final_600/q8_g64`
- Right: pre-overlay q5 lineage, q4 MLX base plus `q5_best_2400_first_phase_naive` adapter view

Generation settings:

- input mode: `title_abstract`
- max tokens: 40
- temperature: 0
- top-p: 0
- batch size: 8

Generation checks:

- Paired rows: 6000
- Unique paper IDs: 6000
- Empty q8 summaries: 0
- Empty q5 summaries: 0
- Exact matching summaries: 36

Length check:

| Side | Mean words | Median | Min | Max | Over 15 | Under 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q8 | 12.05 | 12 | 6 | 28 | 537 | 58 |
| q5 | 11.73 | 11 | 1 | 26 | 352 | 55 |

## Review Prep

Prepared 240 blinded packets of 25 rows each:

- `hf_bucket/experiments/mlx_lora/q5_hardening_distill_6000_v1/review_packets/primary/packet_000` through `packet_239`
- Side map uses `q8` and `q5` aliases.
- Review IDs use prefix `q5_hardening_6000`.

Next step is primary blind review, then aggregate to isolate rows where q8 is
clean/usable and q5 has a severe failure.
