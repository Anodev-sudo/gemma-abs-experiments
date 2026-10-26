# Conclusion: gemma-3-270m IS viable for the briefs task — ship `faithful_only_q5` (concluded 2026-06-17)

## The question
Can gemma-3-270m be made faithful enough — and small enough — to ship the arXiv "feed brief"
task? After eight rounds: **yes.** Reversing the round-3 shelving below.

## The answer
**`faithful_only_q5` (97 MB) is statistically TIED with shipped Granite q8 (374 MB) on
faithfulness** — same-runtime, blind, double-judged, 240 repr + 150 stress papers, bootstrap CIs
(faith−granite −0.11 repr / −0.14 stress, both NS; hallucination tied). The founding 0.3–0.5
faithfulness gap is closed at **1/4 the size**. It reads acceptably (readability within noise of
Granite). The one dimension Granite keeps is **hook** (+0.26, sig) — and we have strong, multi-angle
evidence that gap is **capacity-pinned**, not a data deficit we can close.

## How the faith gap closed (what reopened the round-3 shelving)
1. **Faithfulness-first distillation.** Regenerated all grounding targets *plain and faithful*
   (gemma-4-26b teacher, fresh-context verify gate) instead of hook-led. The hook-led objective
   was teaching a 270M to fabricate confidently; plain targets removed that.
2. **Drop the voice stage.** Stage-2 voice-drilling re-injected the fabrication. Grounding-only won
   by +0.31/+0.35 faith over the prior candidate.
3. **Vocab-strip + q5.** 54.9% of the vocab is never used on arXiv; DF≥100 strip + q5 lands 97 MB
   with q8/q5 both holding ≈bf16 (the cliff is q5→q4, not here).

## Why hook is closed (capacity-pinned, three independent confirmations)
- **cond_hook_v1** — isolated the *safe* half of Granite's hook (grounded subject-fronting), gated
  clean to 98% audited precision. Judged hook moved **~0.00** vs faithful_only (pre-registered null).
  The model learned the surface form (~74% briefs front a named specific) but blind judges rate them
  no higher. Granite's hook edge stayed detectable (positive control) → the null is real.
- **voice stage-2 overlay** — −0.6 faith, no hook.
- **voice as sole single-stage data (style_only_v1)** — −0.49 faith, 3.4× hard-hallucination, hook
  −0.10 (NS, no gain).

Granite's hook comes from a *contrast/surprise clause* that asserts a payoff the abstract often
doesn't supply — the fabrication vector. At 270M, recovering it costs faith for no hook gain. The
safe half is inert. So the hook gap is a real capacity/safety frontier, conceded on purpose.

## Standing recommendation
**Ship `faithful_only_q5` (97 MB)** for the briefs task. It matches Granite on the axis that matters
(faithfulness) at 1/4 the size and reads acceptably. Hook is not worth re-opening without a larger
base — which would be a different product, not a fix to this one.

## Artifacts / receipts
- Faith tie: `quant_probe/aggregate_hg.py`, `quant_probe/VOCAB_STRIP_FINDINGS.md` (top).
- Ship model: `quant_probe/faithful_only_q5/` (97 MB).
- Hook closed: `training/runs/gemma_v5_curriculum_df100_cond_hook_v1/NOTE.md`,
  `quant_probe/judge3way_condhook/`, `quant_probe/judge3way_styleonly/`,
  `VOCAB_STRIP_FINDINGS.md` (cond_hook + style_only sections).
- Eval harness (reusable): `eval_v5_vs_granite/`, `quant_probe/gen_hg.py` + `build_3way_*` + `aggregate_3way_*`.

---
<details><summary>Superseded round-1→3 record (shelving verdict, kept for history)</summary>

Rounds 1–3 concluded gemma plateaued ~0.3–0.5 below Granite and recommended shelving / shipping
Granite. That rested on the full-precision plateau + gemma's prior quantization-collapse history,
*without* a fresh quant probe. Both premises were overturned: q5 holds ≈bf16 here, and
faithfulness-first data broke the plateau to parity. The reasoning was sound on its evidence; the
evidence was incomplete. (See the three notes under `training/runs/` for rounds 1–3.)
</details>
