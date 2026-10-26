# Run: gemma_v5_curriculum_v1  — best gemma recipe so far

**Date:** 2026-06-13 · Configs: `gemma_v5_curriculum_stage1_grounding_v1.yaml`,
`gemma_v5_curriculum_stage2_voice_v1.yaml`

## Idea
Two-stage curriculum to teach correctness without (permanently) teaching summary
style — the sequential alternative to the failed multitask run (round 3).
- **Stage 1 (grounding):** MLP-LoRA on ~10.2k deduped, leakage-excluded summaries,
  rendered UNDER THE BRIEF PROMPT (grounding built in the inference mode, not behind
  a quarantined 'summary' token). 2000 iters. Stage-1 brief test ppl 10.4 (summary
  voice → high, expected).
- **Stage 2 (voice):** resume stage-1 adapter, continue on the 894 briefs only,
  400 iters @ lr 5e-6 (half). Brief val bottomed **2.079 @ iter 300** — best of ALL
  runs (single-task 2.185, multitask 2.116). Test ppl **8.687** (best).

## Result — blind 4-way re-judge (fresh pass; granite anchors ≈ round 2)

| system | faith/3 | hook/2 | total/7 | wins | hard halluc |
|---|---|---|---|---|---|
| granite (shipped q8) | 2.32 | 1.56 | 5.74 | 23 | 2 |
| **curriculum ckpt300** | **2.06** | 1.50 | **5.38** | 12 | **3** |
| mlp_ckpt400 (round-2 best) | 1.88 | 1.42 | 5.06 | 11 | 5 |
| curriculum ckpt400 | 1.82 | 1.40 | 5.04 | 4 | 4 |

## Read
1. **First lever to beat round 2.** In-pass, curr300 > mlp_ckpt400 on faith
   (2.06 vs 1.88), total (5.38 vs 5.06), hallucinations (3 vs 5), wins (12 vs 11).
   Modest but consistent across all four axes. The curriculum sequencing worked
   where mixed multitask (round 3) did not.
2. **ckpt300 > ckpt400** — best val = best product again; stop early, stage-2 LR low.
3. **Hard core persists** (31 cortical-folding, 45 stellar-evolution, 48 privacy
   inversion) — same capability-ceiling papers. curr300 *did* fix 17 and 25 that
   mlp_ckpt400 missed. Granite gets 45/48; its only misses (10,11) are invented-number.
4. **Gap to granite at full precision: ~0.26 faith / ~0.36 total, hallucinations 3 vs 2.**
   Narrowed but not closed.

## Why this reopens gemma
gemma now sits within ~0.3 of granite on faithfulness AND is ~49% smaller after
vocab-strip (embeddings 168M→~36M; 268M→~136M; only 21% of vocab touches arXiv).
Size-adjusted, that trade is compelling — IF faithfulness survives quantization
(gemma's historical failure point, the whole reason for the original Granite switch).

## Decisive next gate
**Quantize curr_ckpt300 (fused) to q8/q5/q4 and re-judge on the 50.** If it holds →
vocab-strip + ship-class gemma is real, pursue. If it collapses → the size win is
moot, shelve confirmed. Then (if it holds) prototype the vocab-strip.

## Best artifact
`training/runs/gemma_v5_curriculum_v1/stage2/adapters/0000300_adapters.safetensors`
(base google/gemma-3-270m-it, MLP-LoRA r16).
