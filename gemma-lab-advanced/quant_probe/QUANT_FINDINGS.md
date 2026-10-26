# Quantization probe — curriculum ckpt300 (2026-06-13)

Fused `gemma_v5_curriculum_v1` stage-2 ckpt300 into the base, quantized (group size 64),
generated brief-mode on the held-out 50, blind 4-way judge across levels.

## Faithfulness vs bits (blind, 3 judges, 50 papers)

| level | faith/3 | total/7 | Δtotal vs bf16 | hard halluc | fused size |
|---|---|---|---|---|---|
| bf16 | 2.08 | 5.38 | — | 4 | 543 MB |
| q8_g64 | 2.14 | 5.54 | +0.16 | 1 | 304 MB |
| q5_g64 | 2.22 | 5.56 | +0.18 | 3 | 208 MB |
| q4_g64 | 1.14 | 3.28 | **−2.10** | **14** | 176 MB |

## Finding
**The cliff is q5→q4, not q8→q5.** bf16 ≈ q8 ≈ q5 (all ~2.1 faith / ~5.5 total;
differences within judge noise). q5 reworded every brief (0/50 identical to bf16) but
the rewordings are scored equally faithful — lateral drift, not degradation. q4
collapses: faithfulness halves, hallucinations 3→14, readability tanks. The judges
clearly discriminate (q4 caught hard), so "q5 holds" is real, not lenient scoring.

This is BETTER than the prior expectation (old gemma collapsed at q5). Plausibly the
grounding pretrain left more quantization-robust representations.

## Size prize (with vocab-strip, ~49% param cut, mostly embeddings)
- q8 stripped ≈ ~145 MB ; **q5 stripped ≈ ~100 MB** ; vs shipped Granite q8 374 MB.
- → vocab-stripped q5 gemma ≈ ~3.7× smaller than shipped Granite, faithfulness within ~0.2.

## Operating point
- **q8 = safe** (≈bf16, near-zero risk).
- **q5 = aggressive but viable** here; confirm on a larger/fresh holdout before trusting
  over q8 in production (q5≈bf16 is within noise on n=50, and it rewrites every output).
- **q4 = dead, do not use.**

## Artifacts
`quant_probe/{fused_bf16,q8_g64,q5_g64,q4_g64}/` ; generations
`eval_v5_vs_granite/out/qp_{bf16,q8,q5,q4}.jsonl` ; judging `eval_v5_vs_granite/judge5/`.

## Next
Prototype the vocab-strip on the fused model (slice embedding+head to the
arXiv-reachable token set + byte alphabet, rebuild tokenizer), re-quantize to q5/q8,
re-judge to confirm the size win doesn't cost faithfulness.
