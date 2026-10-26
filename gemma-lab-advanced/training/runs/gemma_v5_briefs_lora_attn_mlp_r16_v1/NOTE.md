# Run: gemma_v5_briefs_lora_attn_mlp_r16_v1

**Date:** 2026-06-13 · **Config:** `training/configs/gemma_v5_briefs_lora_attn_mlp_r16_v1.yaml`

## What this run is
A controlled repeat of the shipped-era v5 failure-shape test. **Same data**
(v5 briefs, 894/50/50). **One deliberate change:** LoRA targets expanded from
attention-only (`q/k/v/o`) to attention + MLP (`+gate/up/down`). Plus mild dropout
0.05 and cosine-decay LR (warmup 50) to guard overfit. Everything else held to the
shipped run (rank 16, scale 10, lr 1e-5, batch 4, 600 iters, mask-prompt).

Trainable params: **3.80M** (vs 1.48M attention-only). Best val **2.185 @ iter 400**.
Final test ppl **9.926** (shipped-era attention-only run: 10.217).

## Result — blind 5-way re-judge on the held-out 50 (identical conditions)

3 independent Opus judges, briefs anonymized A–E and shuffled per paper.

| system | faith/3 | read/2 | hook/2 | total/7 | best wins | hard halluc | flags |
|---|---|---|---|---|---|---|---|
| **granite (shipped q8)** | **2.28** | 1.88 | **1.58** | **5.74** | **24** | **2** | 26 |
| **mlp_ckpt400 (this run)** | **2.08** | 1.82 | 1.36 | **5.26** | 12 | 4 | 31 |
| mlp_ckpt600 | 1.88 | 1.82 | 1.28 | 4.98 | 4 | 5 | 33 |
| mlp_ckpt500 | 1.78 | 1.82 | 1.24 | 4.84 | 6 | 4 | 37 |
| gemma425_attn (shipped-era) | 1.74 | 1.78 | 1.20 | 4.72 | 4 | 7 | 37 |

## Read

1. **The MLP recipe is a genuine improvement over attention-only.** Best checkpoint
   (ckpt400): faithfulness 1.74→2.08 (+20%), hard hallucinations 7→4 (−43%), best-pick
   wins 4→12 (3×). It closes **~63% of the faithfulness gap** to shipped granite
   (gap 0.54→0.20) and ~53% of the total-score gap.
2. **Best val == best product checkpoint here (iter 400).** This *reverses* round-1's
   "val loss mis-ranks checkpoints" lesson — that was specific to the flat-LR,
   attention-only recipe. With cosine decay + dropout, val loss tracks generation
   quality, and 500/600 overfit toward *more* confident fabrication (ckpt600 adds a
   hallucination ckpt400 doesn't have). Early-stop ~400.
3. **Residual gemma failures are concentrated and characterizable.** All MLP
   checkpoints still fail the same 4 papers: 17 (quantum detailed-balance), 31
   (cortical folding — the "model folds babies' sulci" misattribution), 45 (stellar
   evolution / "most massive BH in universe"), 48 (privacy embeddings — inverts the
   goal). Pattern: **technical physics/math abstractions and causal inversions.**
   Granite gets all four right.

## Verdict update
Round-1 read "further gemma work hard to justify." This run softens that: the deficit
is **partly a training-coverage artifact, not purely a capacity wall** — touching the
MLP blocks recovered a real chunk of faithfulness. gemma at full precision now trails
shipped-q8 granite by a *narrow* margin. Remaining levers worth trying:
- faithfulness-targeted data (evidence-span-anchored targets; the gold rows carry
  `reviewer_evidence_span`), especially up-weighting physics/math and causal-direction cases;
- the quantization cliff is still the gating risk before any of this can ship on MLX-Swift.

## Repro
```
./.venv/bin/python -m mlx_lm.lora --config training/configs/gemma_v5_briefs_lora_attn_mlp_r16_v1.yaml --train --mask-prompt
./.venv/bin/python training/generate_for_run.py --checkpoint training/runs/gemma_v5_briefs_lora_attn_mlp_r16_v1/adapters/0000400_adapters.safetensors --label mlp_r16_ckpt400
python3 eval_v5_vs_granite/round2_shards.py   # then 3 blind judges -> judge2/scores2_*.json
python3 eval_v5_vs_granite/aggregate2.py
```
