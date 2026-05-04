# q4 Hardening Distill 6000 v1

## Summary

Expanded the pre-hardening q4 severe-case mine from `1,000` to `6,000` held-out papers, generated deterministic q8/q4 pairs, reviewed them blinded with `gpt-5.4-mini` workers, and materialized a QA-gated distill release.

No promoted q8 pointer, app integration, LiteRT export, or model promotion was changed.

## Artifact Roots

- Eval set: `hf_dataset/evals/q4_hardening_mine_6000_v1`
- Paired generations: `hf_bucket/experiments/mlx_lora/review_comparisons/q8_vs_q4_hardening_mine_6000_v1`
- Review and aggregation root: `hf_bucket/experiments/mlx_lora/q4_hardening_distill_6000_v1`
- QA-gated release: `hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_distill_6000_v1`
- Primary-only snapshot: `hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_distill_6000_v1_primary`

## Sample Construction

The 6k sample used the same deterministic q4-hardening bucket logic as the 1k mine, scaled to:

| Bucket | Rows |
| --- | ---: |
| numeric | 2,100 |
| polarity | 1,500 |
| entity | 1,200 |
| long_notation | 600 |
| control | 600 |

Additional exclusions included the prior 1k hardening mine, tripwire comparisons, current training releases, and prior q8/q4 review samples.

## Generation

Compared the current q8 promoted standalone export against the pre-hardening q4 lineage adapter:

```bash
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  scripts/sample_mlx_holdout_generations.py \
  --reuse-sample-jsonl hf_dataset/evals/q4_hardening_mine_6000_v1/eval_inputs.jsonl \
  --output-dir hf_bucket/experiments/mlx_lora/review_comparisons/q8_vs_q4_hardening_mine_6000_v1 \
  --left-label q8_promoted \
  --right-label q4_lineage \
  --left-model-path hf_bucket/experiments/mlx_lora/quantized_models/faith_overlay_1400_v2_final_600/q8_g64 \
  --right-model-path hf_bucket/experiments/mlx_lora/quantized_models/google_gemma_3_270m_it/q4_g64 \
  --right-adapter-dir hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_faithfulness_overlay_1400_v2_lr_2_5e_6/adapters \
  --right-checkpoint hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_faithfulness_overlay_1400_v2_lr_2_5e_6/adapters/0000600_adapters.safetensors \
  --input-mode title_abstract \
  --max-tokens 40 \
  --temp 0.0 \
  --top-p 0.0 \
  --batch-size 8
```

Generation checks:

- `6,000` paired rows
- `6,000` unique paper IDs
- `0` empty q8 summaries
- `0` empty q4 summaries
- `223` exact matching summaries

## Review

Primary review used `240` blinded packets of `25` rows. QA review used `736` follow-up rows: every primary q4-severe/q8-clean candidate plus clean controls.

All review workers used `gpt-5.4-mini`. A few TSV formatting issues were mechanically normalized after review:

- abbreviated winners such as `L/R/T/B`
- `true/false` booleans
- extra blank TSV columns before target-usability fields
- issue-tag separators such as `|` and `;`

Final schema checks passed:

- Primary decisions: `6,000 / 6,000` valid
- QA decisions: `736 / 736` valid

## Results

Primary-only aggregate:

| Metric | Count |
| --- | ---: |
| q8 wins | 2,843 |
| q4 wins | 2,381 |
| ties | 722 |
| both bad | 54 |
| q4 severe | 238 |
| q8 severe | 187 |
| q4 severe / q8 clean | 173 |
| primary-only distill rows | 172 |

QA-gated aggregate:

| Metric | Count |
| --- | ---: |
| q8 wins | 2,847 |
| q4 wins | 2,389 |
| ties | 703 |
| both bad | 61 |
| q4 severe | 194 |
| q8 severe | 233 |
| q4 severe / q8 clean | 117 |
| primary/QA conflicts excluded | 417 |
| QA-gated distill rows | 74 |

QA-gated distill bucket counts:

| Bucket | Rows |
| --- | ---: |
| numeric | 38 |
| polarity | 17 |
| long_notation | 10 |
| entity | 5 |
| control | 4 |

## Release

Prepared MLX dataset:

```bash
python3 scripts/prepare_mlx_lora_dataset.py \
  --source hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_distill_6000_v1 \
  --input-mode title_abstract
```

Split counts:

| Split | Rows |
| --- | ---: |
| train | 66 |
| valid | 4 |
| test | 4 |

The final release is intentionally small because QA was conservative and primary/QA conflicts were excluded from training.
