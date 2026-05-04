# MLX LoRA 270M q4 Hardening Distill v1

## Goal

Reduce q4-only severe failures in the current q4-trained lineage by mining a
large q8/q4 paired comparison set, using clean q8 outputs as distillation
targets only when q4 fails severely and q8 remains faithful.

This experiment is separate from the promoted q8 lineage. Do not update the
promoted pointer from this run.

## Planned Artifacts

- Eval set: `hf_dataset/evals/q4_hardening_mine_1000_v1`
- Paired generations: `hf_bucket/experiments/mlx_lora/review_comparisons/q8_vs_q4_hardening_mine_1000_v1`
- Review and aggregation root: `hf_bucket/experiments/mlx_lora/q4_hardening_distill_v1`
- Distill release: `hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_distill_v1`

## Commands

Build the 1,000-paper mining set:

```bash
python3 scripts/build_q4_hardening_eval_set.py --overwrite
```

Generate paired q8/q4 summaries:

```bash
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  scripts/sample_mlx_holdout_generations.py \
  --reuse-sample-jsonl hf_dataset/evals/q4_hardening_mine_1000_v1/eval_inputs.jsonl \
  --left-label q8_promoted \
  --left-model-path hf_bucket/experiments/mlx_lora/quantized_models/faith_overlay_1400_v2_final_600/q8_g64 \
  --right-label q4_lineage \
  --right-model-path hf_bucket/experiments/mlx_lora/quantized_models/google_gemma_3_270m_it/q4_g64 \
  --right-adapter-dir hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_faithfulness_overlay_1400_v2_lr_2_5e_6/adapters \
  --right-checkpoint hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_faithfulness_overlay_1400_v2_lr_2_5e_6/adapters/0000600_adapters.safetensors \
  --output-dir hf_bucket/experiments/mlx_lora/review_comparisons/q8_vs_q4_hardening_mine_1000_v1 \
  --sample-size 1000 \
  --batch-size 8 \
  --temp 0.0 \
  --top-p 0.0 \
  --max-tokens 40 \
  --blinded
```

Prepare blinded review packets:

```bash
python3 scripts/prepare_q4_hardening_review_packets.py --overwrite
```

Aggregate primary reviews and write QA follow-up:

```bash
python3 scripts/aggregate_q4_hardening_reviews.py --write-qa-followup
```

After QA/adjudication, materialize the distill release:

```bash
python3 scripts/aggregate_q4_hardening_reviews.py --require-qa-for-distill
python3 scripts/prepare_mlx_lora_dataset.py \
  --source hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_distill_v1 \
  --input-mode title_abstract
```

Train the hardening sweep:

```bash
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_hardening_distill_v1_lr_5e_7.yaml \
  --mask-prompt
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_hardening_distill_v1_lr_1e_6.yaml \
  --mask-prompt
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_hardening_distill_v1_lr_2_5e_6.yaml \
  --mask-prompt
```

## Results

### Mining Set

- Eval set: `hf_dataset/evals/q4_hardening_mine_1000_v1`
- Rows: `1,000`
- Bucket counts:
  - numeric: `350`
  - polarity: `250`
  - entity: `200`
  - long_notation: `100`
  - control: `100`
- Paired generations:
  `hf_bucket/experiments/mlx_lora/review_comparisons/q8_vs_q4_hardening_mine_1000_v1`
- Empty q8 summaries: `0`
- Empty q4 summaries: `0`

### Review And Distill Materialization

- Primary review packets: `40` packets of `25` rows
- QA review rows: `214`
- QA review packets: `9`
- Final QA-gated distill rows: `127`
- Release:
  `hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_distill_v1`
- MLX dataset:
  `hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_distill_v1/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_glanceable_summary_v1`
- Split counts: train `101`, valid `13`, test `13`

QA-gated review summary:

| Metric | Count |
|---|---:|
| Reviewed rows | `1,000` |
| q4 severe | `258` |
| q8 severe | `223` |
| q4 severe / q8 clean | `145` |
| Primary/QA conflicts excluded | `42` |
| Distill rows | `127` |

### Training Sweep

All runs resumed from:
`hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_faithfulness_overlay_1400_v2_lr_2_5e_6/adapters/0000600_adapters.safetensors`.

| LR | Best checkpoint | Best val | Final val | Test loss / ppl | Note |
|---:|---|---:|---:|---:|---|
| `5e-7` | `0000300` | `0.646` | `0.646` | `0.562` / `1.754` | Smoothest validation curve |
| `1e-6` | `0000200` | `0.643` | `0.649` | `0.570` / `1.769` | Best validation checkpoint |
| `2.5e-6` | `0000050` | `0.643` | `0.924` | `0.814` / `2.256` | Overfit after step `50` |

Selected tripwire candidate:
`hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_hardening_distill_v1_lr_1e_6/adapters/0000200_adapters.safetensors`.

### Held-Out Tripwire Evaluation

Tripwire set:
`hf_dataset/evals/q4_hardening_tripwire_100_v1`.

This set uses the same bucket proportions as the mining set and excludes the
1,000-row mining set plus the distill release rows.

| Bundle | q4 wins | q8 wins | Tie | Both bad | q4 severe | q8 severe | q4-only severe |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q8_vs_q4_lineage_tripwire_100_v1` | `36` | `41` | `11` | `12` | `28` | `18` | `16` |
| `q8_vs_q4_hardened_lr_1e_6_200_tripwire_100_v1` | `42` | `37` | `9` | `12` | `24` | `19` | `12` |

The hardening overlay improved preference count and reduced q4-only severe
failures from `16` to `12` on this held-out tripwire set. That is a `25%`
q4-only severe reduction, which is useful but does not meet the predeclared
`40%` severe-reduction bar.

## Conclusion

The q4 hardening distill pass helped, but not enough to promote or replace the
current q4 candidate. Keep it as an experiment. A next attempt should likely
use more QA-gated hardening rows, a larger fresh tripwire, or a loss/objective
that more directly penalizes severe fact drift rather than a tiny SFT overlay.

No promoted pointer was updated.
