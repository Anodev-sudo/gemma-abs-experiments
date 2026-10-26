# gemma-3-270m vs shipped Granite on the new "briefs" data (v5)

**Date:** 2026-06-13
**Question:** Does the new briefs gold set (994-row feed-blurb v5) make gemma-3-270m
viable enough to justify further work?

## Setup

- **Data:** bundled `gold_pipeline/.../mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5`,
  50-row held-out **test** split (never seen in training).
- **gemma:** `google/gemma-3-270m-it` + the pre-existing v5 LoRA adapter
  (rank 16, lr 1e-5, 500 iters), **full precision**, checkpoints 425 and 500.
- **granite:** the **actually shipped** artifact — `granite_4_0_350m_v5_ckpt425`,
  fused, **q8** (restored from the `peering-latent/gemma-abs-experiments` bucket).
- **Decoding:** greedy (temp 0), max_tokens 40, each model's own chat template.
- **Judging:** 3 independent Opus sub-agents, **blind** (briefs relabeled A/B/C,
  order shuffled per paper), scoring faithful 0-3 / readable 0-2 / hook 0-2 against
  the abstract. Unblinded afterward via `judge/key.json`.

## Result

| system | faithful /3 | readable /2 | hook /2 | total /7 | best wins | hard hallucinations | flags |
|---|---|---|---|---|---|---|---|
| **granite (shipped q8)** | **2.14** | 1.78 | **1.58** | **5.50** | **32/50** | **3/50** | 28 |
| gemma425 (full prec) | 1.56 | 1.72 | 1.16 | 4.44 | 13/50 | 9/50 | 39 |
| gemma500 (full prec) | 1.50 | 1.70 | 1.08 | 4.28 | 4/50 | 7/50 | 43 |

Head-to-head gemma425 vs granite per paper: granite higher 30, gemma higher 15, tie 4.

## Read

1. **gemma's problem is not fluency or length — it's faithfulness.** Readability
   is essentially tied (1.72 vs 1.78) and length behavior is fine (~40/50 ≤14
   words). The gap is grounding: gemma hard-hallucinates 9/50 vs granite's 3/50,
   and trails on faithfulness by a wide, consistent margin.
2. **This is the worst-case failure mode for the product:** fluent, on-length,
   confident — and wrong. A reader gets a plausible reason to open a paper that
   the paper doesn't support.
3. **The gap exists at gemma's *best* precision, against granite's *shipped q8*.**
   Quantizing gemma (the whole point of "minimize it further") has historically
   made faithfulness worse, not better. So the precision deck is stacked in
   gemma's favor here and it still loses.
4. **ckpt425 > ckpt500**, confirming the prior note's instinct: the lower
   validation-loss checkpoint (500) is the weaker product checkpoint. Validation
   loss is not the right selector for this task.
5. gemma keeps up on clean/easy abstracts and even wins a few (e.g. paper 11,
   where granite invented precision/recall numbers). It falls apart on
   technical/physics/math abstracts and on subtle takeaways — exactly where it
   confidently fabricates.

### Illustrative failures (gemma, full precision)
- **[31] cortical folding:** gemma425 *"Self-supervised learning helps babies fold
  their sulci more accurately during pregnancy"* (the model folds babies' brains?)
  vs granite *"Self-supervised model reveals a cortical folding pattern linked to
  prematurity."*
- **[17] detailed balance:** gemma425 *"Quantum systems can be broken down by a
  magnetic field, leading to thermal collisions"* (nonsense) vs granite *"Quantum
  systems can heat up even when detailed balance is violated."*
- **[8] sharded blockchain:** gemma425 invented *"...by only 5% in a real Ethereum
  dataset"* — a fabricated number.

## Verdict

The new briefs data did **not** fix gemma-3-270m's core weakness. The data is fine
(granite learned it cleanly); gemma learns the *shape* (length, tone, single
sentence) but not the *grounding*. Further work aimed at shrinking gemma is hard to
justify **until the faithfulness gap is closed at the data/training level**, because
quantization will only widen it.

## Caveats
- Single judge per paper, no cross-validation; faithfulness scoring on technical
  abstracts is itself error-prone (judges occasionally misread). Treat magnitudes as
  directional, the direction as solid.
- n=50, greedy decoding, one adapter recipe (the prior v5 run). A fresh train was
  not run — these are the existing checkpoints.

## Repro
```
eval_v5_vs_granite/generate.py   # per-variant generation
eval_v5_vs_granite/merge.py      # -> out/combined.{jsonl,md}
eval_v5_vs_granite/make_judge_shards.py
eval_v5_vs_granite/aggregate.py  # -> judge/aggregate_out.json
```
