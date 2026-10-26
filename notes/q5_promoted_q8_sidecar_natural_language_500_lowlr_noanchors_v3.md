# q5 Promoted Q8 Sidecar NL500 Low-LR No-Anchor v3

## Goal

Accidental no-anchor side runs from a clarified request. The actual intended
experiment is LR creep with the low-weight anchor mix, captured in
`notes/q5_promoted_q8_sidecar_natural_language_500_anchor_mix_v1.md`.

This note preserves the accidental no-anchor outputs so they are not confused
with the main line.

The previous no-anchor
sweep showed:

- `1e-6`: too cold
- `2e-6`: marginal
- `3e-6`: first real llama.cpp movement, but semantically fragile
- `1e-5`: too hot

These configs test intermediate points while keeping the same early-stop shape:
no anchors, `50` max steps, checkpoints every `10`.

## Runs

| config | LR | output |
|---|---:|---|
| `configs/sft/q5_promoted_q8_sidecar_natural_language_500_lowlr_noanchors_v3_lr4e6_steps50.yaml` | `4e-6` | `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_lowlr_noanchors_v3/lr4e6_steps50` |
| `configs/sft/q5_promoted_q8_sidecar_natural_language_500_lowlr_noanchors_v3_lr5e6_steps50.yaml` | `5e-6` | `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_lowlr_noanchors_v3/lr5e6_steps50` |
| `configs/sft/q5_promoted_q8_sidecar_natural_language_500_lowlr_noanchors_v3_lr6e6_steps50.yaml` | `6e-6` | `hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_lowlr_noanchors_v3/lr6e6_steps50` |

## Planned Eval

Do not prioritize eval unless the anchored LR creep needs a direct no-anchor
control at the same LR.

## Completed Side Runs

- `4e-6`: completed, train loss `11.631`, runtime `86.3s`
- `5e-6`: completed, train loss `11.569`, runtime `84.5s`
