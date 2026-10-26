# 🏁 FAITHFUL (97 MB) ≈ GRANITE (374 MB) ON FAITHFULNESS — same-runtime head-to-head (2026-06-16)

> Note added Oct 2026: "tied" in this section means no significant difference, not shown
> equal. The 95% intervals (−0.26 to +0.03 typical, −0.35 to +0.05 hard) leave room for a gap
> of about a quarter point, and would fail the −0.10 non-inferiority bar used for cond_hook.

The founding question answered. `faithful_only_q5` (97 MB gemma q5) vs shipped-era Granite 4.0 350m q8
(374 MB) — both through the IDENTICAL `gen_threeway.py` harness (same BRIEF prompt, temp 0, max 40 tok,
MLX load+generate), regenerated in one pass so prompt version is provably identical. Blind 2-way (X/Y),
double-judged, 390 papers, bootstrap 95% CIs. Inter-rater faithful 80% exact/MAD 0.20 (repr), 71%/0.29 (stress).

| dim | faith (97 MB) | granite (374 MB) | faith − granite |
|---|---|---|---|
| repr faithful | 2.083 | 2.196 | **−0.113 [−0.260,+0.031] NS (tied)** |
| repr halluc<=0.5 | 3.3% | 3.8% | tied |
| repr readable | 1.710 | 1.800 | −0.090 *sig* |
| repr hook | 1.085 | 1.369 | −0.283 *sig* |
| stress faithful | 2.091 | 2.235 | **−0.144 [−0.346,+0.054] NS (tied)** |
| stress halluc<=0.5 | 9.4% | 8.1% | tied |
| stress readable | 1.789 | 1.782 | +0.007 (tied) |
| stress hook | 1.003 | 1.265 | −0.262 *sig* |

**Result: the 97 MB gemma is statistically TIED with the 3.9×-larger Granite on faithfulness (both sets)
and on hard-hallucination rate.** Granite's only durable edge is hook (+0.27/+0.26 sig) — the one dimension
proven capacity-pinned and deliberately not chased. Readability ~wash (−0.09 sig repr only, tied stress).
Closes the 5-round ~0.3–0.5 faithfulness gap to Granite; the faithfulness-first distillation got there at
1/4 the size. Harness: `gen_faith_vs_granite.sh`, `build_hg_shards.py`, `aggregate_hg.py`, `judge_hg/`.

---

# ⛔ STYLE VOICE-STAGE — DEAD END, hook is capacity-pinned (2026-06-16)

Tried to buy back the small hook deficit with a *fresh, faithfulness-gated* stage-2 style set
(Gemma-26B, hook-led, gates: length<=136c/18w + quality>=2 + strict faithfulness PASS; 2,500 rows,
audited clean). Drilled it on top of the faithful grounding → `faithful_style_q5`. 3-way blind
double-judge (390 papers, 26 runs, inter-rater 82-84%):

| repr / stress | faith_only (champ) | faith_style | candidate |
|---|---|---|---|
| faithful | **2.06 / 2.10** | 1.44 / 1.35 | 1.71 / 1.71 |
| halluc <=0.5 | **4.6% / 4.0%** | 18% / 21% | 12% / 13% |
| readable | **1.83 / 1.90** | 1.69 / 1.58 | 1.63 / 1.58 |
| hook | 0.99 / 0.96 | 0.99 / 0.99 | 1.02 / 1.03 |

faith_style −0.63/−0.75 faithful vs faith_only (sig), 4–5× hallucination, worse readable — and
**+0.00/+0.03 hook (flat, NS): it bought literally no hook.** Two clean conclusions:
1. **The stage-2 voice drill is poison for a 270M, regardless of data quality.** Human gold caused
   drift; a gated gemma style set caused it *worse*. Any "be punchy" drill teaches the model to
   *reach* for pull; at 270M the reach = fabrication + garble, not hook.
2. **Hook is capacity-pinned at ~1.0/2.0** — all three models, every intervention, sit at hook≈1.0.
   The −0.07 "deficit" isn't headroom, it's the ceiling. Faithfulness is the only movable lever.

**Decision: `faithful_only_q5` is the artifact (grounding only, NO voice stage). Voice-drilling closed.**

**Low-LR probe (2026-06-16) — confirms it's the drill, not the dose.** Reran the SAME style stage at
peak LR 1.5e-6 (~3.3× gentler, all else identical) to test "maybe the drill is just too strong."
License-clean gemma-verifier PASS rate on the 60-paper held-out faith set:
faith_only **53.3%** (32/60) · style full-LR(5e-6) 30.0% (18/60) · style low-LR(1.5e-6) 33.3% (20/60).
The gentler drill recovered **+3.3pts = 2 papers (noise)** and stayed ~20pts below faith_only — at 1/3
the LR it landed *on top of* the full-strength run, not halfway back. **Damage is not proportional to
dose: a whisper of voice-drill imports nearly all the overreach.** The stage teaches the wrong thing
and this student learns it fast. Eyeball matched (kept the punch + the overreach). Voice-drilling now
dead 3 ways: gold voice, full-LR style, low-LR style. Don't retry LR/curation/per-token — same stage,
same failure. (Note absolute rates here are on a different 60-paper sample than the earlier 70-paper
66% vs 40% cross-check; the ~20pt faith_only-over-style gap is the consistent signal.)

---

# ✅ FAITHFULNESS-FIRST DISTILLATION — WIN (2026-06-16)

**First sub-100MB gemma variant to significantly BEAT the q5 candidate on faithfulness.**
Reverses the Round-6 backfire and softens the "capacity wall" read.

**What changed vs Round 6 (which lost −0.21):** two fixes, both pointed at the brief *form*.
1. **Faithfulness-first data.** Regenerated ALL 5,314 stage-1 targets with the 26B teacher under a
   faithfulness-first prompt (plain > catchy, feed guardrails: no method-names/LaTeX/jargon) + a strict
   claim-decomposition verify gate (repair ≤2x, drop unsalvageable → 5,182 kept, 2.5% dropped). Blind
   before/after audit: data faithfulness **2.62 → 2.92** (+0.30 sig, 92% strongly faithful, 0% halluc).
   Note the old "hooky" data was already 2.62 at the data level — so data faithfulness was a *partial*,
   not total, lever.
2. **Dropped the voice stage.** Ablation (held-out 60) showed the stage-2 gold voice drill SIGNIFICANTLY
   hurts faithfulness (faithful-only beat faithful+voice by +0.35 sig) — it re-injects the hookiness the
   faithful grounding removed. So the shipped variant is **stage-1 grounding only** (`faithful_only_q5`).

**Full eval — A=faithful_only_q5 vs candidate=recovered_df100_q5, 390 papers, blind double-judge, 26 runs,
inter-rater 77–80% exact MAD 0.21–0.24:**

| | A faithful-only | candidate | delta |
|---|---|---|---|
| faithful repr   | **2.110** | 1.798 | **+0.312 *sig*** |
| faithful stress | **2.074** | 1.728 | **+0.346 *sig*** |
| halluc ≤0.5 repr/stress | **4.6% / 6.0%** | 12.9% / 13.4% | ~½–⅓ the rate |
| readable repr/stress | **1.83 / 1.87** | 1.57 / 1.61 | +0.25 *sig* |
| hook repr/stress | 1.02 / 0.99 | 1.10 / 1.06 | −0.07 *sig* (small) |

**Costs:** hook down ~0.07 (sig but small); length discipline slightly worse (28/32% >14w vs cand 22/26%)
— the voice stage's one real benefit was brevity. Net: A is the new best artifact.

**Reframe of the capacity-wall conclusion:** it's not a pure wall. Faithful data + removing the voice
stage moved faithfulness +0.31/+0.35 significantly AND cut hallucination 2–3×. The Round-6 −0.21 was the
*combination* of hooky data and the voice stage; fix both and it flips positive. Open: A vs Granite head-
to-head (not in this eval — judge calibration differs across rounds, no cross-round absolute compare); and
recovering brevity/hook without a faithfulness-killing voice stage (faithfulness-filtered gold for stage 2?).

Artifact: `quant_probe/faithful_only_q5/` (97 MB; weights byte-identical to recovered_df100_q5 — was
briefly 103 MB only because `mlx_lm convert` pretty-printed tokenizer.json, +5 MB whitespace; minified
to compact JSON, lossless). Data: `curriculum_stage1_gemma4xl_faithful_v1`.

---

# ⛔ ROUND 6 — 26B-teacher brief-distillation BACKFIRED (2026-06-15)

**Bet:** the 5-round plateau (strip/quant/LoRA/prompt all move faith <0.15) is a *grounding-objective*
problem — the stage-1 grounding data was summaries rendered under a brief prompt, not real briefs.
Fix: distill genuine briefs from a strong teacher. Teacher = **gemma-4-26b-a4b-qat** (LMStudio HTTP,
reasoning off), prompt-only few-shot (6 gold exemplars + brief-vs-summary rules block, no finetune —
license-clean Gemma→Gemma). Pulled **12k recent VPS papers** (feed-representative, sorted by
created_date — NOT arxiv_id, which floats 1990s prefixed IDs to the top), generated, then
self-verified faithfulness with the same teacher (fresh context). **12k → 5,314 kept (44%)**, all
rejections genuine. Retrained DF≥100 curriculum (stage1 = 5,314 verified briefs, stage2 = 894 gold)
→ fuse → q5 (`gemma4xl_df100_q5`, 93 MB, stage2 val loss 2.080 == baseline).

**Result (3-way blind double-judge, 26 runs, repr-240 + stress-150, inter-rater 84–85% exact MAD 0.15):**

| faithful | gemma4xl (new) | baseline (recovered_df100_q5) | granite |
|---|---|---|---|
| repr   | 1.458 | **1.669** (+0.211 *sig*) | 2.046 |
| stress | 1.477 | **1.713** (+0.237 *sig*) | 2.117 |
| hard-halluc ≤0.5 (repr/stress) | 14.6% / 18.0% | 12.6% / 12.0% | 3.3% / 4.7% |

readable: tie. hook: tie (new slightly worse). **The new model is strictly dominated by the
baseline** — significantly *less* faithful, heavier hallucination tail, no hook gain.

**Why it backfired (concrete failure mode):** the brief objective taught the 270M student to emit a
punchy, *specific-sounding* claim — and at 270M it fills that slot with **invented or wrong
specifics**. Examples: *"Binary search is now a single Python solution"* (paper is a 5-variant
analysis); *"...without running out of memory"* (invented; abstract was about Runge instabilities);
*"lets rare disease patients share their medical records"* (misreads a representation-learning paper).
The old **summary-grounding was protective** — it kept the model anchored to abstract content. A
bigger/better teacher does NOT help because the bottleneck is **student capacity**, not teacher
quality (and self-verification by the same teacher rubber-stamps its own confident style).

**Decision:** abandon brief-distillation. **`recovered_df100_q5` remains the ship artifact.** This
*confirms* "briefs ≠ summaries" — but in the opposite-of-hoped direction: a 270M model cannot carry
the hook-led objective without hallucinating. Pipeline kept at `quant_probe/distill/` for the record.

---

# ⚠️ THREE-WAY SHIP EVAL (2026-06-14) — Granite still wins faithfulness, decisively

The most rigorous eval in this lab to date. **Corrects the earlier "near-lossless"
framing**, which compared stripped/quantized gemma to *lossless gemma* — not to the
incumbent. Against shipped **Granite 4.0 350M q8 (374 MB)**, both small gemma
artifacts are significantly *less faithful*.

**Method:** 240 feed-representative recent arXiv papers (NOT mindf-filtered; the real
feed) + 150 strip-stress papers (min-docf<25, worst case). Three-way blind A/B/C,
**double-judged** (2 independent LLM judges/shard, 26 judge runs), faithfulness 0–3
scored only against the abstract. Bootstrap 95% CIs, paired per-paper deltas.
Inter-rater: exact 78%, within-1 99–100%, MAD 0.22 → judges reliable; gaps are real.
Identical BRIEF prompt + temp 0 across all three. Data: `judge3way/RESULTS.txt`.

### Headline — representative feed (237 papers w/ all 3 scored)
| model | size | faithful | 95% CI | hard-halluc (≤0.5: inverted/fabricated) | best-of-3 |
|---|---|---|---|---|---|
| **Granite q8 (shipped)** | 374 MB | **2.333** | [2.236, 2.428] | **2.5%** | 49.6% |
| gemma q5 + DF≥100 recovery | 98 MB | 1.926 | [1.812, 2.040] | 8.4% | 28.1% |
| gemma q4-QLoRA + DF≥100 | 88 MB | 1.755 | [1.637, 1.876] | 13.1% | 22.4% |

Paired (per-paper, bootstrap CI): q5−Granite **−0.407** [−0.549,−0.268] *sig*;
q4q−Granite **−0.578** [−0.726,−0.428] *sig*; q4q−q5 **−0.171** [−0.314,−0.021] *sig*.

### Worst case — strip-stress papers (149)
Granite 2.356 · q5 1.842 · q4q 1.815. q4q vs q5 **ties** here (−0.027, n.s.); q5 only
pulls ahead on typical papers. Both still ~0.5 below Granite (*sig*), hard-halluc
~13% vs Granite 3.4%.

### What this means
- **Faithfulness gap to Granite is large, significant, and not a strip/quant artifact.**
  The stress set (papers the strip actually touches) is no worse than the representative
  set — so the gap is gemma's base-model ceiling (consistent with round-4: gemma
  plateaus ~0.3 below Granite even at full precision), widened a little by q4/q5.
- **Hard hallucinations (inverted/fabricated results): gemma 8–13% vs Granite 2.5–3.4%**
  — 3–5×. For a faithfulness-critical feed brief, the original reason for switching to
  Granite, this is the decisive number.
- **q5 (98 MB) ≥ q4-QLoRA (88 MB).** The 10 MB q4 saving costs a significant 0.17
  faithfulness on typical papers and lifts the hard-halluc tail 8.4%→13.1%. If a small
  gemma ships, it's q5, not q4-QLoRA.
- **The size win was real; the quality parity with Granite was not.** Shipping any of
  these gemma artifacts trades ~3.8× smaller download for a material faithfulness
  regression. Decision is a genuine size-vs-faithfulness tradeoff — but on the metric
  that drove the original Granite switch, **Granite still wins.**
- **Framing (corrected 2026-06-14 after user clarification):** the product is **gemma** —
  Granite is NOT a ship candidate, it's a fixed reference yardstick. So "−0.4 vs Granite"
  is a ruler mark, not a blocker. The live question is **which gemma variant ships**, and
  the round-2 result below identifies the real lever (extra curriculum training, not strip
  level). Earlier "shrink Granite instead" note is moot — disregard.

## ROUND 2 (2026-06-14): DF≥25-q5 vs DF≥100-q5 vs Granite — strip level flips by population, and the real lever is the recovery fine-tune
Same method (240 repr + 150 stress, blind A/B/C, double-judged, bootstrap CIs).
Data: `judge3way_df25/RESULTS.txt`.

| set | df25-q5 (115 MB) | df100-q5 (98 MB) | Granite | df25−df100 |
|---|---|---|---|---|
| representative (238) | 1.706 | **1.855** | 2.313 | **−0.149 *sig*** |
| strip-stress (150) | **1.973** | 1.803 | 2.317 | **+0.170 *sig*** |

- **The two strips trade places by population.** DF≥25 wins on rare-token stress papers
  (kept the tokens DF≥100 dropped); DF≥100 wins on the representative feed. The feed is
  what ships, so as-built **DF≥25-q5 (115 MB) is dominated by DF≥100-q5 (98 MB)** — bigger
  *and* less faithful on typical papers.
- **Why DF≥100 wins on representative is a CONFOUND, not the strip.** DF≥100-q5 received a
  recovery fine-tune (fresh stage1+stage2 curriculum on the stripped base); DF≥25-q5 did
  **not** (post-hoc strip of the curriculum model, no retrain — correct, since DF≥25 is
  tokenization-lossless on typical papers). So DF≥25-q5 on representative IS the original
  curriculum gemma @ q5 (1.706); DF≥100-q5 is that + extra training (1.855). Decisive tell:
  DF≥100 re-segments **37% (88/240)** of representative papers vs DF≥25's **12% (28/240)** —
  so on tokenization fidelity DF≥100 is markedly worse on the feed, yet it wins anyway. The
  win is the recovery training overcoming its own (larger) strip damage.
- **~~New free lever~~ (WALKED BACK by round 3 — see below):** I claimed extra curriculum
  training is a vocab-independent ~+0.15 gain. Round 3 (DF≥50 + recovery) shows it's
  **redistribution, not gain** — it buys ~+0.14 on typical papers but *costs* ~−0.17 on
  rare-token papers (robbing stress to pay representative). Net near-wash. The "free lever"
  framing was too strong.
- **Size tension:** lossless strips that avoid re-segmentation (DF≥1 124 MB, DF≥25 115 MB
  at q5) are **>100 MB**; sub-100 MB requires an aggressive strip (DF≥50 103, DF≥100 ~98)
  that re-segments. (Note: DF≥50 came out 103 MB, *over* 100 — the closure-complete strip
  keeps 74,430 tokens, more than the 66,621 the old keep-curve projected.)

## ROUND 3 (2026-06-14): DF≥50 + recovery (the sub-100 candidate) — 4-way, settles the lever question
Built DF≥50 stripped base (74,430 tok) + the *same* recovery curriculum DF≥100 got, fused,
q5 → 103 MB. Judged 4-way (df50 / df100 / df25 / Granite), same 240+150 papers, blind A/B/C/D,
double-judged, one calibration. Data: `judge4way_df50/RESULTS.txt`. Inter-rater exact 78–80%.

| set | df50-q5 (103 MB) | df100-q5 (98 MB) | df25-q5 (115 MB) | Granite |
|---|---|---|---|---|
| representative (239) | 1.828 | **1.897** | 1.759 | 2.310 |
| strip-stress (150) | 1.813 | 1.813 | **1.983** | 2.243 |

Pairwise (per-paper bootstrap CI): df50−df100 repr **−0.069 ns**, stress **0.000**;
df50−df25 repr +0.069 ns, stress **−0.170 sig**; df100−df25 repr **+0.138 sig**, stress
**−0.170 sig**.

- **DF≥50 + recovery ≈ DF≥100 + recovery — statistically identical on both sets** (−0.069 ns
  / 0.000). DF≥50 is 103 MB (*over* 100); DF≥100 is 98 MB. So **DF≥50 is strictly dominated —
  same quality, bigger, breaches 100 MB.** No niche for it. **DF≥100-q5 (98 MB) stays the
  best sub-100 artifact.**
- **The predicted DF≥50 win didn't happen.** Hypothesis: DF≥50+recovery keeps DF≥25's
  rare-token edge AND gains the recovery lift → beats DF≥100 everywhere. Reality: the
  recovery fine-tune *erased* the stress edge (DF≥50 stress 1.813 = DF≥100 1.813, both −0.170
  below DF≥25's 1.983). Recovery on an aggressive strip makes it behave like the heavier
  strip, not like the light one.
- **Recovery = redistribution, not gain (the corrected lever).** Across both rounds: recovery
  models win typical papers by ~+0.14 (barely sig) and lose rare-token papers by ~−0.17 (sig).
  It moves faithfulness between populations; it doesn't add it. My round-2 "free +0.15 lever"
  was an overclaim — it ignored the stress cost (which was visible in round-2's own df25 stress
  win, I just under-weighted it).
- **The three gemma variants are clustered ~1.76–1.90 on the feed; strip level + recovery move
  things <0.15, mostly within noise.** The dominant determinant of gemma faithfulness is the
  base model, not these knobs. Further strip/recovery tuning has diminishing returns — the
  variants are ~equivalent on the feed and the choice is **size-driven**:
  - **sub-100 MB →** DF≥100-q5 (98 MB): repr 1.90 / stress 1.81. Ship this.
  - **≤~115 MB, max quality →** DF≥25-q5 (115 MB): best on rare-token papers, tied on the feed,
    cleanest (near-lossless strip, no retrain). The quality play if the size bar relaxes.
  - DF≥50 (103 MB): dominated, drop it.

## ROUND 4 (2026-06-14): improve DF≥100 — cheap-knobs batch (#4 full-FT, #5 quant, #6 prompt)
Diagnosis first (DF≥100 judge flags, n=770): losses split between **"misses the point"**
(saliency, the #1 flag) + grounding errors (inverted/fabricated/misbind) in the hard tail,
and **vague** in the 2→3 gap. **Garbling is NOT strip-induced** (9.7% repr vs 9.9% stress,
flat) → DF≥100's errors are the *model*, not the compression. Then a 4-way blind double-judge
(baseline-q5 / full-FT-q5 / bf16-lossless / faithfulness-prompt), 240+150 papers. Data:
`judge_df100imp/RESULTS.txt`.

| variant | repr | Δ vs base | stress | Δ vs base |
|---|---|---|---|---|
| baseline q5 | 1.883 | — | 1.863 | — |
| #4 full-FT q5 | 1.699 | **−0.184 *sig*** | 1.767 | −0.096 ns |
| #5 bf16 lossless | 2.010 | +0.128 (CI[−0.02,+0.27]) | 1.949 | +0.086 ns |
| #6 faithfulness prompt | 1.933 | +0.050 ns | 1.853 | −0.010 ns |

- **#4 full-FT LOSES — significantly.** Despite *better* token metrics (stage1 ppl 11.5 vs
  LoRA 13.1; stage2 val 2.123), full FT is −0.184 *worse* on generated faithfulness and lifts
  the hard tail (13.0% vs 10.9%). Classic overfit: full FT distorts the instruct model's
  general competence; LoRA's low-rank constraint *regularizes* and preserves faithfulness.
  **LoRA rank-16 is the right choice; full FT rejected.** (LR untuned for full FT, but the
  lower-loss/worse-generation signature is unambiguous.)
- **#6 prompt does nothing** (+0.05 / −0.01, ns). Strengthening the faithfulness instruction
  at inference doesn't move it. Rejected.
- **#5 there's ~0.13 of real quant headroom** (bf16 2.010 vs q5 1.883 on repr, borderline
  sig; hard tail 8.8% vs 10.9%) — but it's **uncapturable under 100 MB.** After the strip the
  transformer is ~70% of params, so raising *its* precision (where the headroom lives) busts
  the budget (q6 uniform ≈ 117 MB; bf16 = 281 MB). q5 stays the sub-100 frontier. A risky
  q6-transformer / q3-embeddings mix (~99.5 MB) *might* reclaim part of it — marginal, untested.
- **Net: none of the cheap knobs improves DF≥100 within 100 MB.** This is the third independent
  confirmation that strip/quant/training-config knobs don't move gemma faithfulness — the base
  model is the ceiling (bf16 tops out at ~2.0 on the feed, ≈round-4's "~0.3 below Granite").
  **The only remaining lever is the data track** (distillation / grounding-objective fix /
  hard-negatives) — the one thing that raises the ceiling rather than approaching it.

---

# Vocab-strip feasibility — objective full-arXiv distribution (2026-06-13)

Tokenized **all 3,073,292 non-deleted arXiv papers** (title+abstract, 643.6M tokens)
through the gemma tokenizer on `composer-vps` (the live FirstPass DB). Recorded
per-token occurrence count + document frequency. Tokenizer is BPE with
`byte_fallback: True` and all 256 byte tokens present → keeping the byte alphabet
makes any input representable (no `<unk>` risk; dropped tokens re-segment).

## Coverage of gemma's 262,145 vocab over all of arXiv
- **seen: 118,191 (45.1%) · NEVER seen: 143,954 (54.9%)**
- Embeddings are tied and are 167.8M of 268.1M params (62.6%) → vocab dead-weight is
  almost all removable embedding rows.

## Keep-curve (kept vocab → occurrence coverage → projected footprint)
| keep rule | kept | occ-cov | params | q8 MB | q5 MB |
|---|---|---|---|---|---|
| none (current) | 262,145 | — | 268.1M | 304 | 208 |
| **DF>=1 (seen+bytes+special)** | 124,788 | 100.0000% | 180.2M | 191 | 124 |
| DF>=10 | 91,642 | 99.977% | 159.0M | 169 | 109 |
| DF>=50 | 66,621 | 99.851% | 143.0M | 152 | 98 |
| DF>=100 | 55,835 | 99.691% | 136.1M | 144 | 94 |
| _shipped Granite q8_ | 100,352 | — | ~350M | 374 | — |

## Operating points
- **DF>=1 = free & behaviorally identical.** Every real arXiv input already tokenizes
  into kept tokens, so outputs are bit-identical on the historical corpus; no retrain.
  268M->180M params; q5 208->124MB, q8 304->191MB. The no-brainer baseline.
- **DF>=50 hits the ~100MB moonshot** (q5 ~98MB, 99.85% coverage). Drops only tokens
  in <50 of 3M papers; needs the short re-fine-tune (re-segmentation) + re-judge.
- Combine with q5 (shown viable in QUANT_FINDINGS) for the smallest shippable artifact.

## Artifacts
`quant_probe/token_stats.npz` (per-token total+DF over full arXiv) ·
`quant_probe/keep_ids_df1.json` (lossless keep-list) ·
`quant_probe/analyze_coverage.py` (re-run sweep / emit other thresholds) ·
remote job: `composer-vps:~/vocab_strip/`.

## Next
Build the strip: slice embedding/head to the keep-list, rebuild the HF tokenizer with
remapped ids (drop non-kept merges, retain byte fallback), re-quantize, re-judge on
the 50. Start with DF>=1 (zero-risk, confirms the pipeline), then push to DF>=50.

---
## DF>=1 strip — BUILT & VERIFIED (2026-06-14)
Pipeline: multi-parent merge-closure keep-set → slice tied embed_tokens → remap all
ids (vocab, merges, post_processor, added_tokens_decoder, config). Code: `strip_vocab.py`.

- **Keep-set: 129,588** (seen 118,191 + bytes/special + 4,801 merge-closure intermediates).
  Note: closure was load-bearing — naive "keep seen tokens" re-segmented 341 texts
  (e.g. `▁avenue` builds via the dropped intermediate `▁aven`). Multi-parent closure fixes it.
- **Tokenizer identity:** 0 mismatches on 19,720 in-distribution texts (`test_strip_tokenizer.py`).
  430/20k weighted-corpus texts touch served-DB-unseen tokens → re-segment via byte fallback (expected).
- **Model identity:** stripped bf16 generates **50/50 identical briefs** to unstripped bf16. Lossless.
- **Footprint (model.safetensors):** bf16 536→347 MB. Stripped q8 ~196 MB, **stripped q5 ~128 MB**
  (+~9 MB tokenizer). vs unstripped q5 208 MB / q8 304 MB; vs shipped Granite q8 374 MB.
  → lossless DF>=1 + q5 ≈ ~130 MB deployable, ~2.9x smaller than shipped Granite, ZERO quality cost.

Next: DF>=50 for the ~100 MB moonshot (re-segments the rare tail → add short re-fine-tune,
re-judge). Optionally union keep-set with the training/eval corpus tokens to cover the
~2% weighted-corpus OOV before shipping.

---
## Degradation sweep on a POWERED enriched eval (2026-06-14)
The 50-paper held-out set couldn't measure strip degradation (only ~6 re-segment).
Pulled 3,000 recent papers from the live FirstPass DB (composer-vps, read-only),
excluded training ids, kept the **353 that re-segment at DF>=25** (deployment-like;
no gold needed — judge scores faithfulness vs the abstract). Generated lossless/
DF>=25/DF>=100 at q5, blind faith-only judge (4 agents, 352 papers scored).

| variant | mean faith/3 | halluc | Δ vs lossless | size |
|---|---|---|---|---|
| lossless DF>=1 q5 | 2.142 | 11 | — | 120 MB |
| **DF>=25 q5** | **2.173** | 8 | **+0.03** | 104 MB |
| DF>=100 q5 | 2.088 | 15 | -0.05 | 96 MB |

Paired (papers the cut actually changed):
- DF>=25 (169 changed): 44 worse / 55 better / 70 same — symmetric reshuffle, net +0.07.
- DF>=100 (243 changed): 81 worse / 72 better / 90 same — sign flips negative, +4 halluc.

**Verdict:** DF>=25 (~104MB) is FREE — indistinguishable from lossless even on an
all-re-segmenting enriched set, no re-finetune. Hits the ~100MB moonshot. DF>=100
(~96MB) is where degradation starts (small, would need re-finetune); saves only 8MB.
Size span is tiny (120->104->96), so the sweet spot is DF>=1 (zero-risk) or DF>=25
(validated-free moonshot); don't go below ~96MB without retraining.

---
## Sub-100MB milestone: DF>=100 + special-trim + recovery curriculum + q5 (2026-06-14)
- **Dead-special trim** (drop 6,242 <unusedN> slots) verified FREE independently:
  DF>=1+trim = 50/50 identical to lossless, 0 tokenizer mismatches. (First over-trim
  attempt broke 18 papers by dropping newline-run tokens used in the prompt — caught by
  the independent test; fixed to drop <unusedN> only.) Now default on all strips.
- **DF>=100 recovery:** stripped the BASE (not the fused model) at DF>=100+trim, re-ran
  the full curriculum on it (stage1 grounding 10k + stage2 briefs, all on re-segmented
  tokenization). Brief val 2.077 == full-vocab curriculum's 2.079.
- **Enriched eval (352 papers that all re-segment at DF>=25), blind faith judge:**

  | variant | mean faith | halluc(0) | bundle |
  |---|---|---|---|
  | lossless DF>=1 q5 | 2.119 | 6 | ~130 MB |
  | DF>=100 q5 no-recovery | 2.060 (-0.060) | 13 | ~102 MB |
  | **DF>=100 q5 RECOVERED** | **2.105 (-0.014)** | 11 | **98.1 MB** |

  Recovery erased ~76% of the strip degradation; mean faith within noise of lossless.
  Residual: slightly heavier hallucination tail (11 vs 6) — near-lossless on average,
  marginally riskier at the edges.

**Result: sub-100MB (98.1MB total bundle) at near-lossless quality.** ~3.8x smaller
than shipped Granite q8 (374MB). Artifact: quant_probe/recovered_df100_q5/.
Next: q4 (longshot) for deeper compression.

---
## q4 exploration — NEGATIVE (2026-06-14)
Tested whether the fresh recovery model quantizes to q4 better than the original
curr300 did (it didn't). Blind faith judge on the 349-paper enriched eval:

| variant | size | mean faith | halluc(0) | >=2 faith |
|---|---|---|---|---|
| q5 (ref) | 98 MB | 2.146 | 16 | 76% |
| mixed_4_6 | 84 MB | 1.567 (-0.58) | 36 | 50% |
| q4 g32 | 90 MB | 1.542 (-0.61) | 36 | 49% |
| q4 g64 | 81 MB | (not judged; avg 15.5w, 51% <=14 — broken length discipline) | | |

**q4 collapses faithfulness ~0.58-0.61 and doubles hallucinations**, for all of
straight-q4, q4-g32, and mixed_4_6. Finer groups + mixed precision do NOT rescue it.
**Lesson:** length mechanics misled — q4-g32 had *better* length discipline than q5
(11.8w/90% vs 13.1w/74%) but worse faithfulness (tight, confident, wrong). Judge, don't
trust mechanics. **q5 is the quantization floor for this model.**

## FINAL shippable artifact
`quant_probe/recovered_df100_q5/` — **98.1 MB bundle**, near-lossless faith
(Δ-0.014 vs lossless on enriched eval), ~3.8x smaller than shipped Granite q8.
Quantization floor reached; further size only via deeper vocab cut (grows the
hallucination tail) against the ~69MB q5 transformer floor.

---
## q4 RE-EXAMINED — QLoRA makes it work (2026-06-14) [supersedes the q4-NEGATIVE above]
The earlier q4 negative tested only POST-HOC q4 (train curriculum at full precision,
then quantize). The fix is the same principle that rescued vocab-strip: train ON the
degraded representation. Quantize the stripped base to q4 FIRST, then run the
curriculum as QLoRA (LoRA adapter trains on the frozen q4 base, learning to correct
quantization error). Blind faith judge, 347-paper enriched eval:

| variant | size | mean faith | halluc(0) | >=2 |
|---|---|---|---|---|
| q5 recovery | 98 MB | 2.098 | 15 | 75% |
| q4 post-hoc | 81 MB | 1.326 (-0.77) | 55 | 39% |
| **q4 QLoRA** | **~88 MB** | **2.066 (-0.032)** | 16 | 72% |

QLoRA recovers q4 from catastrophic collapse to within-noise of q5. Deployment:
ship q4-base + adapter SEPARATELY (MLX-Swift applies adapter at load) — fusing +
requantizing would re-inject the error the adapter corrects. **Principle: train on the
degraded representation (stripped tokenizer OR quantized weights) and it recovers.**

## NEW best artifact
q4-QLoRA: `quant_probe/stripped_base_df100_q4/` (base) + `quant_probe/q4qlora_adapter/`
(adapter) — **~88 MB bundle**, near-lossless faith, ~4.2x smaller than shipped Granite q8.

---

## cond_hook_v1 — conditional subject-fronting (NEGATIVE, pre-registered) — 2026-06-17

Tested whether the *groundable subject-fronting* half of Granite's hook recovers on
`faithful_only_q5` via teacher-relabeled, gate-verified targets (79.4% frontable; cross-family
Claude audit 98.3% precision, 0 ungrounded-subject leaks). q5, 97 MB, size-controlled. Blind
3-way vs faith + granite, 240 repr + 150 stress, double-judged, bootstrap CIs.

| dim | cond_hook−faith (repr) | cond_hook−faith (stress) | cond_hook−granite (repr) |
|---|---|---|---|
| hook | −0.031 [−0.121,+0.052] | −0.010 [−0.133,+0.123] | **−0.263 sig** |
| faithful | +0.056 [−0.060,+0.169] | +0.150 [−0.010,+0.310] | +0.035 ns |
| readable | −0.071 ns | −0.047 ns | −0.094 sig |

**Hook did not move (~0, vs +0.15 bar); faith held (non-inferior both sets); repr hard-halluc
3.8%→6.2%.** Granite's hook edge still detected sig (positive control → null is real, not judge
noise). **Decision: subject-fronting hook is capacity-pinned at 270M; keep `faithful_only_q5`,
do not ship cond_hook.** The safe half of the hook is inert; the half that moves it is the
contrast-clause fabrication vector, scoped out on purpose. First named limit in the arc that held.

---

## style_only_v1 — voice set as SOLE single-stage data (NEGATIVE) — 2026-06-17

Q: does the faith-gated voice/style set (2,500 rows) behave differently as the SOLE training
objective (single stage from stripped_base) vs the prior stage-2 overlay-on-faithful drill?
Same recipe as cond_hook; ckpt400 (val-min, overfits early on the smaller set); q5, 97 MB.
LIGHT eval: blind 3-way vs faith+granite, repr-240 only, single-pass, 12 Haiku judges.

| dim | style_only−faith (repr) | style_only−granite |
|---|---|---|
| faithful | **−0.492 sig** | −0.421 sig |
| readable | **−0.329 sig** | −0.333 sig |
| hook | −0.096 ns | −0.175 sig |

Hard-halluc 2.9%→**10.0%** (3.4×), misleading 16.7%→37.5%. **Strictly dominated: faith craters,
readability drops, hook does NOT gain.** Sole-data is ~as damaging as the stage-2 overlay (−0.49 vs
−0.6 faith) for zero hook gain → the voice objective teaches a 270M confident fabrication regardless
of how it's injected; hook stays capacity-pinned. Keep faithful_only_q5. (Header label in
aggregate_3way_styleonly.py prints a stale "cond_hook_q5" string — cosmetic; styleonly column IS
style_only_q5 via the secret key.)
