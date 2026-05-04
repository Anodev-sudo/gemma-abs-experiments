# Gemma 270M q4 Promotion

## Summary

Promoted the q4 hardening combined checkpoint as the current MLX
title+abstract adapter on 2026-05-04. This replaces the q8
`faith_overlay_1400_v2_final_600` pointer for local MLX summary generation, but
does not update LiteRT export or app integration.

## Promoted Artifacts

- Promotion pointer:
  `hf_bucket/experiments/mlx_lora/promoted/gemma_3_270m_it_title_abstract_current.json`
- Quantized q4 base:
  `hf_bucket/experiments/mlx_lora/quantized_models/google_gemma_3_270m_it/q4_g64`
- Promoted adapter checkpoint:
  `hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_hardening_combined_v1_lr_1e_6/adapters/0000300_adapters.safetensors`
- Source config:
  `configs/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_hardening_combined_v1_lr_1e_6.yaml`
- Training release:
  `hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_combined_v1`

## Why This Was Promoted

The combined hardening overlay materially reduced the severe q4-only failures
that motivated this experiment while keeping preference close to q8 across
fresh blinded reviews.

| Eval | q4 wins | q8 wins | Tie | Both bad | q4 severe | q8 severe | q4-only severe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed tripwire 100 | 56 | 30 | 11 | 3 | 7 | 4 | 4 |
| Fresh 100 | 54 | 37 | 6 | 3 | 6 | 10 | 3 |
| Repeat fresh 100 | 44 | 51 | 4 | 1 | 5 | 5 | 5 |

The previous q4 hardening tripwire result had `12` q4-only severe failures; the
promoted combined candidate cut that to `4` on the same fixed tripwire.

## Training Summary

| LR | Best checkpoint | Best val | Final val | Test loss / ppl | Decision |
| ---: | --- | ---: | ---: | ---: | --- |
| `5e-7` | `0000250` | `0.768` | `0.774` | `0.999` / `2.716` | Passed sanity |
| `1e-6` | `0000300` | `0.768` | `0.768` | `1.002` / `2.723` | Promoted |
| `2.5e-6` | `0000050` | `0.770` | `0.832` | `1.090` / `2.975` | Early overfit |

## Caveats

- The second fresh review was preference-close but favored q8 by `51` to `44`.
- The current pointer now targets an MLX q4 base plus LoRA adapter, not the old
  merged q8 standalone export.
- LiteRT export and app integration were intentionally left untouched.
- Keep watching numeric binding, polarity flips, unsupported entities, and
  overstrong mechanism claims in later launch checks.
