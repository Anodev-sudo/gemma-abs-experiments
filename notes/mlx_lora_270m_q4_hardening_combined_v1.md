# Gemma 270M q4 Hardening Combined v1

## Summary

Merged the strict 6k QA-gated q4 hardening rows with Qwen-accepted Kimi
corrections, preferring Kimi/Qwen rows on overlap, then trained a conservative
q4 hardening overlay sweep on top of the current q4 lineage adapter.

After two fresh blinded reviews and the fixed tripwire replay, the selected q4
checkpoint was promoted as the current MLX title+abstract adapter on
2026-05-04.

## Artifacts

- Combined release:
  `hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_combined_v1`
- MLX dataset:
  `hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_combined_v1/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_glanceable_summary_v1`
- Base q4 adapter:
  `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_faithfulness_overlay_1400_v2_lr_2_5e_6/adapters/0000600_adapters.safetensors`

## Merge

```bash
python3 scripts/materialize_q4_hardening_combined_release.py
python3 scripts/prepare_mlx_lora_dataset.py \
  --source hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_combined_v1 \
  --prompt-spec prompts/mlx_lora/gemma_270m_it_abstract_to_glanceable_summary_v1.json \
  --dataset-format chat \
  --input-mode title_abstract
```

| Source | Rows |
| --- | ---: |
| Strict 6k QA-gated | 74 |
| Kimi/Qwen accepted | 92 |
| Overlap, Kimi preferred | 18 |
| Combined unique rows | 148 |

Combined bucket counts:

| Bucket | Rows |
| --- | ---: |
| numeric | 73 |
| polarity | 40 |
| entity | 15 |
| long_notation | 15 |
| control | 5 |

MLX split counts:

| Split | Rows |
| --- | ---: |
| train | 134 |
| valid | 7 |
| test | 7 |

## Training Sweep

All runs resume from the current q4 adapter listed above.

```bash
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_hardening_combined_v1_lr_5e_7.yaml \
  --mask-prompt

hf_bucket/experiments/mlx_lora/runtime/py312/bin/python -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_hardening_combined_v1_lr_1e_6.yaml \
  --mask-prompt

hf_bucket/experiments/mlx_lora/runtime/py312/bin/python -m mlx_lm.lora \
  --config configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_hardening_combined_v1_lr_2_5e_6.yaml \
  --mask-prompt
```

| LR | Best checkpoint | Best val | Final val | Test loss / ppl | Note |
| ---: | --- | ---: | ---: | ---: | --- |
| `5e-7` | `0000250` | `0.768` | `0.774` | `0.999` / `2.716` | Smooth, slight final drift |
| `1e-6` | `0000300` | `0.768` | `0.768` | `1.002` / `2.723` | Selected; best checkpoint is final |
| `2.5e-6` | `0000050` | `0.770` | `0.832` | `1.090` / `2.975` | Overfit after early checkpoint |

## Evaluation

Planned candidate selection:

1. Select the lowest-validation checkpoint from each LR run.
2. Run a 20-row tripwire sanity generation pass for each selected checkpoint.
3. Choose the lowest-validation sanity-passing checkpoint.
4. Compare against q8 on the fixed tripwire and one fresh 100-paper set.

Sanity generation bundles:

- `hf_bucket/experiments/mlx_lora/review_comparisons/q4_hardening_combined_lr_5e_7_250_tripwire_20_sanity_v1`
- `hf_bucket/experiments/mlx_lora/review_comparisons/q4_hardening_combined_lr_1e_6_300_tripwire_20_sanity_v1`
- `hf_bucket/experiments/mlx_lora/review_comparisons/q4_hardening_combined_lr_2_5e_6_50_tripwire_20_sanity_v1`

All three sanity candidates produced 20 non-empty q4 summaries with no obvious
repetition or non-answer failures. The selected candidate is:

`hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_hardening_combined_v1_lr_1e_6/adapters/0000300_adapters.safetensors`

Final blinded review bundles:

- Fixed tripwire generations:
  `hf_bucket/experiments/mlx_lora/review_comparisons/q8_vs_q4_hardening_combined_lr_1e_6_300_tripwire_100_v1`
- Fixed tripwire review packets:
  `hf_bucket/experiments/mlx_lora/q4_hardening_combined_v1/review_packets_tripwire`
- Fresh eval set:
  `hf_dataset/evals/q4_hardening_combined_fresh_100_v1`
- Fresh generations:
  `hf_bucket/experiments/mlx_lora/review_comparisons/q8_vs_q4_hardening_combined_lr_1e_6_300_fresh_100_v1`
- Fresh review packets:
  `hf_bucket/experiments/mlx_lora/q4_hardening_combined_v1/review_packets_fresh`
- Repeat fresh eval set:
  `hf_dataset/evals/q4_hardening_combined_repeat_100_v1`
- Repeat fresh generations:
  `hf_bucket/experiments/mlx_lora/review_comparisons/q8_vs_q4_hardening_combined_lr_1e_6_300_repeat_100_v1`
- Repeat fresh review packets:
  `hf_bucket/experiments/mlx_lora/q4_hardening_combined_v1/review_packets_repeat`

Each final bundle has 100 q8 outputs, 100 q4 outputs, 100 review rows, 100
side-map rows, and no empty generated summaries.

## Review Results

Primary blinded review was completed in four 25-row packets for each eval set.
The aggregation roots are:

- Tripwire:
  `hf_bucket/experiments/mlx_lora/q4_hardening_combined_v1/eval_aggregation_tripwire`
- Fresh:
  `hf_bucket/experiments/mlx_lora/q4_hardening_combined_v1/eval_aggregation_fresh`

| Eval | q4 wins | q8 wins | Tie | Both bad | q4 severe | q8 severe | q4-only severe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed tripwire 100 | 56 | 30 | 11 | 3 | 7 | 4 | 4 |
| Fresh 100 | 54 | 37 | 6 | 3 | 6 | 10 | 3 |
| Repeat fresh 100 | 44 | 51 | 4 | 1 | 5 | 5 | 5 |

Compared with the prior hardening tripwire result, q4-only severe failures fell
from `12` to `4` on the same fixed tripwire set. This clears the predeclared
"promising" bar for the tripwire and is comfortably competitive on the fresh
100-paper review.

A second fresh 100-paper repeat was more mixed on preference (`44` q4 wins vs
`51` q8 wins) but still replicated the severe-failure improvement: q4-only
severe stayed low at `5`, with q4 and q8 tied at `5` severe flags each.

## Promotion

Promoted pointer:

```text
hf_bucket/experiments/mlx_lora/promoted/gemma_3_270m_it_title_abstract_current.json
```

Promoted checkpoint:

```text
hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_hardening_combined_v1_lr_1e_6/adapters/0000300_adapters.safetensors
```

Quantized q4 base:

```text
hf_bucket/experiments/mlx_lora/quantized_models/google_gemma_3_270m_it/q4_g64
```

The previous q8 pointer details are preserved inside the promoted JSON under
`previous_promoted`. LiteRT export and app integration were not updated in this
promotion step.
