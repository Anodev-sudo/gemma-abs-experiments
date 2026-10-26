# Run: gemma_v5_multitask_summary10k_brief994_v1  — NEGATIVE RESULT

**Date:** 2026-06-13 · **Config:** `training/configs/gemma_v5_multitask_summary10k_brief994_v1.yaml`

## Hypothesis
gemma has brief *style* but lacks *grounding*. Teach grounding from the wide ~10k
summary DB behind a 'summary' system prompt; keep brief *voice* on the brief prompt
(used at inference). Style separated into a label, not blended. ("Teach correctness
without teaching style.")

## Setup
- Data: 12,875 train rows = 10,193 summary-mode (deduped 10746, 994 brief
  train/valid/test papers excluded — no leakage) + 894 briefs ×3 upsample (21%).
  valid/test = 50/50 brief-mode only. MLP LoRA recipe, 2400 iters (<1 epoch),
  cosine decay, dropout 0.05.
- Brief val loss bottomed **2.116** (vs single-task 2.185); test ppl **8.807**
  (vs single-task 9.926). Token metrics improved.

## Result — blind 5-way re-judge (fresh pass; within-pass ordering only)

| system | faith/3 | hook/2 | total/7 | wins | hard halluc |
|---|---|---|---|---|---|
| granite (shipped q8) | 2.22 | 1.42 | 5.56 | 19 | 3 |
| mt ckpt1700 (best mt) | 2.06 | 1.30 | 5.22 | 8 | 4 |
| mlp ckpt400 (single-task) | 2.02 | 1.24 | 5.10 | 9 | 3 |
| mt ckpt2100 | 1.94 | 1.26 | 5.00 | 7 | 5 |
| mt ckpt2400 | 1.74 | 1.18 | 4.70 | 7 | 6 |

## Read — the transfer did not work
1. **Best multitask ≈ single-task, within noise.** faith 2.06 vs 2.02, halluc 4 vs
   3, wins 8 vs 9. The 11× summary grounding bought nothing decisive on brief
   faithfulness.
2. **Token metrics improved, judged faithfulness did not.** Lower val loss / ppl,
   same hallucination rate. Loss is not the product (again).
3. **Errors redistributed, not reduced.** Multitask fixed papers 17 & 48 but broke
   1, 9, 31. Net wash.
4. **More training on the mix degrades briefs** (ckpt1700→2400: faith 2.06→1.74).
   Summary register / overfit drags brief faithfulness down with exposure. Early
   stop if this family is revisited.
5. **Capability-ceiling signature:** paper 45 fails in *every* gemma variant and
   recipe; 31/17 fail in most. Errors that survive attn-only → +MLP → +11× grounding
   look like a base-model limit on hard technical/causal abstracts, not a coverage gap.

## Verdict
gemma plateaus at full precision around faith ~2.0 / total ~5.1–5.2, ~0.3–0.5 below
shipped granite, and the plateau held against both levers tried (MLP coverage in
round 2 helped vs attn-only; multitask grounding here did not stack further). The
"teach correctness without teaching style" framing was sound, but the correctness gap
here is not a separable style-entanglement problem — it reads as a base-model
capability limit. Remaining options: (a) DPO targeting the specific failure modes
(causal inversion, fabricated numbers) — directly punishes the errors instead of
hoping SFT coverage helps; (b) accept the plateau and weigh it against granite, with
the quantization cliff still ahead for gemma on MLX-Swift.
