# MLX LoRA 270M Faithfulness Overlay 1400 v2 Promotion

## Decision

Promote `faith_overlay_1400_v2_final_600` as the current preferred 270M
title+abstract adapter.

Promoted checkpoint:

```text
hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_faithfulness_overlay_1400_v2/adapters/0000600_adapters.safetensors
```

Stable pointer:

```text
hf_bucket/experiments/mlx_lora/promoted/gemma_3_270m_it_title_abstract_current.json
```

## Lineage

- Base model: `google/gemma-3-270m-it`
- Parent adapter: `base_aggressive_3600`
- Parent checkpoint: `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_from_base_aggressive_v1/adapters/0003600_adapters.safetensors`
- Overlay dataset: `recent_3y_balanced_10746_faithfulness_overlay_1400_v2`
- Overlay mix: `1000` faithfulness cleanup rows plus `400` specificity-preserving rows
- Training config: `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_faithfulness_overlay_1400_v2.yaml`

## Training Metrics

- Steps: `600`
- Best validation loss: `1.426` at checkpoints `250` and `350`
- Final validation loss: `1.430`
- Final test loss: `1.620`
- Final test perplexity: `5.055`
- Peak memory: `5.922 GB`

The final checkpoint was not the numerically best validation checkpoint. It was
promoted because it was the checkpoint selected for qualitative review, and the
blinded fixed-holdout review showed a clear improvement over both baselines.

## Review Evidence

Fixed holdout: `100` papers from
`hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_v1/review_samples/holdout_100_20260419-075505/generations.jsonl`.

Comparison against `attn_1400`:

- Bundle: `hf_bucket/experiments/mlx_lora/review_comparisons/attn_1400_vs_faith_overlay_1400_v2_final_600_holdout_100_20260428-013807`
- Reviewer votes: candidate `116`, baseline `66`, both bad `13`, tie `5`
- Consensus rows: candidate `53`, baseline `29`, split left/right `7`
- Severe faithfulness votes: baseline `90`, candidate `41`

Comparison against `base_aggressive_3600`:

- Bundle: `hf_bucket/experiments/mlx_lora/review_comparisons/base_aggressive_3600_vs_faith_overlay_1400_v2_final_600_holdout_100_20260428-013834`
- Reviewer votes: candidate `107`, baseline `74`, both bad `10`, tie `9`
- Consensus rows: candidate `48`, baseline `30`, split left/right `10`
- Severe faithfulness votes: baseline `70`, candidate `38`

## Interpretation

This promotion is not just a readability win. The final overlay reduced severe
faithfulness failures versus both `attn_1400` and `base_aggressive_3600` while
remaining preferred overall by independent blinded reviewers.

Remaining caveat: the model still has semantic compression failures, especially
missed-point and overstrong-claim cases. Future overlay work should keep testing
for grounded specificity rather than simply pushing the model toward safer,
blander summaries.
