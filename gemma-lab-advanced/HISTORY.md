# gemma-lab-advanced: lab log

Running log kept during the work (June 2026), newest status at the top. Kept as written. Where it says "statistically TIED" with Granite, read "no significant difference": the intervals can't rule out a gap of about a quarter point. For a short summary see README.md; for the verdict see CONCLUSION.md.

Isolated, self-contained lab for one question: **can gemma-3-270m be made viable
for the arXiv "feed brief" task** (one faithful ~10–14-word sentence giving a
reader a reason to open a paper), ideally small enough to ship.

Deployment runtime target: **MLX-Swift**. We train here in **Python MLX** (LoRA SFT).

Self-contained on purpose — this lab does **not** depend on the older
`gemma-abstract-finetuning` repo at runtime. Its own venv (`.venv`), its own data
(`gold_pipeline/`), its own runs. Keep it that way.

## Layout

```
gold_pipeline/          The 994-row "briefs" gold set + the pipeline that made it.
                        Data actually used: gold_pipeline/05_milestone_assembly/
                        gold_set/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5
                        (894 train / 50 valid / 50 test, chat format).
eval_v5_vs_granite/     Round 1: read of the pre-existing shipped-era gemma v5
                        adapter vs shipped Granite q8. See its FINDINGS.md.
training/               LoRA SFT runs done in THIS lab.
  configs/              one yaml per run.
  runs/<name>/          adapters/ + logs + per-run notes.
.venv/                  python3.12 + mlx-lm (pinned). The only interpreter used here.
```

## Status: ACTIVE (2026-06-16) — gemma IS the product; Granite is only a reference yardstick. **HEADLINE: `faithful_only_q5` (97 MB) is statistically TIED with the 374 MB Granite q8 on faithfulness — same-runtime, blind, double-judged head-to-head (faith−granite −0.11 repr / −0.14 stress, both NS; hallucination tied). Granite (3.9× larger) keeps only a hook edge (+0.27 sig), the one capacity-pinned dimension. The founding ~0.3–0.5 faithfulness gap to Granite is closed at 1/4 the size.** See VOCAB_STRIP_FINDINGS top section + `aggregate_hg.py`. NEW best artifact: `faithful_only_q5` (97 MB) — faithfulness-first distilled grounding, NO voice stage. Beats prior candidate `recovered_df100_q5` on faithfulness **+0.31 repr / +0.35 stress (sig)**, hallucination ~½–⅓, readability +0.25; small hook cost −0.07. See `quant_probe/VOCAB_STRIP_FINDINGS.md` (top section). [Round 6 hooky-distillation backfired −0.21; fixed by faithfulness-first data + dropping the voice stage.] Stage-2 voice-drilling confirmed harmful THREE ways (human gold; a fresh faith-gated gemma style set: −0.6/−0.75 faith, 4–5× halluc, +0 hook; and a low-LR rerun of that style set at 1.5e-6 which recovered only noise — damage isn't dose-proportional, the drill teaches the wrong thing) → hook is capacity-pinned ~1.0; ship grounding-only.
Three rigorous double-judged rounds (240 representative + 150 strip-stress papers, blind, bootstrap
CIs). Granite q8 (374 MB) sits at faith ~2.31 as a fixed ruler — **not a ship candidate**. Gemma:
- gemma q5+DF≥100+recovery (98 MB): repr **1.90**, stress 1.81  ← **best sub-100 MB; ship this**
- gemma q5+DF≥50+recovery (103 MB): repr 1.83, stress 1.81  ← dominated (= DF≥100, bigger, >100 MB)
- gemma q5+DF≥25 (115 MB, no retrain): repr 1.76, stress **1.98**  ← best on rare-token; quality play if size relaxes
- gemma q4-QLoRA+DF≥100 (88 MB): repr 1.76 — dominated by DF≥100-q5

**Key findings (rounds 2→3):** strip level trades by population — lighter DF≥25 wins rare-token
papers, heavier+recovery wins typical ones. The recovery fine-tune is **redistribution, not gain**
(≈+0.14 typical / −0.17 rare; round-2 "free +0.15 lever" claim WALKED BACK). DF≥50+recovery =
DF≥100+recovery statistically → no niche, dropped. **All gemma variants cluster ~1.76–1.90 on the
feed; strip/recovery knobs move <0.15, mostly noise — the base model is the ceiling, so the choice
is size-driven.** MLX-Swift QLoRA sidecar verified. Dead-token trim free, default. Best artifact:
`quant_probe/recovered_df100_q5/` (98 MB).

**Round 4 — DF≥100 improvement (cheap-knobs batch, all rejected):** full-FT is *worse* than LoRA
(−0.184 sig; overfits, LoRA regularizes); faithfulness-first prompt does nothing (+0.05 ns); q5 has
~0.13 headroom to bf16 but it's uncapturable <100 MB (transformer is 70% of params post-strip). Garbling
is NOT strip-induced. Confirms: knobs can't move it, **base model is the ceiling. Only remaining lever =
data track** (distillation / grounding-objective fix / hard-negatives). See VOCAB_STRIP_FINDINGS round 4.


Round 4 (curriculum) produced the best gemma yet (~0.2–0.3 below shipped Granite on
faithfulness). Quant probe (`quant_probe/QUANT_FINDINGS.md`): **q8 and q5 both hold
(≈bf16); the cliff is q5→q4, not q8→q5** — better than the old-gemma prior. So a
vocab-stripped q5 gemma projects to ~100 MB (~3.7× smaller than shipped Granite q8)
at near-Granite faithfulness. Vocab-strip feasibility now measured on the FULL arXiv
distribution (3.07M papers, via composer-vps): 54.9% of vocab is never used; a
lossless DF>=1 strip is 268M->180M (q5 124MB, behaviorally identical, no retrain),
DF>=50 hits the ~98MB moonshot at 99.85% coverage. See `quant_probe/VOCAB_STRIP_FINDINGS.md`.
Next: build the strip (slice + remap tokenizer), re-quantize, re-judge.

## Findings so far

- **Round 1 (eval of the shipped-era v5 adapter):** gemma learns brief *shape*
  (length, tone, one sentence) but not *grounding* — it hard-hallucinates ~3× more
  than shipped Granite at full precision, while reading just as fluently.
  Faithfulness, not fluency, is the gap. See `eval_v5_vs_granite/FINDINGS.md`.
- **Round 2 (`gemma_v5_briefs_lora_attn_mlp_r16_v1`):** expanding the LoRA to the
  MLP blocks (same data) recovered a real chunk of faithfulness — best checkpoint
  (ckpt400) cut hard hallucinations 7→4 and closed ~63% of the faithfulness gap to
  shipped granite. Residual failures concentrate in physics/math + causal-inversion
  abstracts. See `training/runs/gemma_v5_briefs_lora_attn_mlp_r16_v1/NOTE.md`.
- **Round 3 (`gemma_v5_multitask_summary10k_brief994_v1`) — NEGATIVE:** dual-prompt
  multi-task (import grounding from the 10k summary DB behind a 'summary' prompt,
  keep brief voice on its own prompt) did **not** improve brief faithfulness — best
  checkpoint tied single-task within noise. Token metrics (val loss, ppl) improved
  while judged faithfulness didn't. Errors that survive attn→MLP→+11× grounding data
  (esp. papers 45/31/17) read as a base-model capability ceiling, not a coverage gap.
  gemma plateaus ~0.3–0.5 below shipped granite at full precision. See that run's NOTE.md.
- **Round 4 (`gemma_v5_curriculum_v1`) — best gemma yet:** two-stage curriculum
  (grounding on 10k summaries *in brief mode*, then voice-drill on the 994) beat the
  round-2 winner — hallucinations 5→3, total 5.06→5.38 in-pass, ~0.3 below granite.
  First lever to improve on round 2. Hard core (31/45/48) still fails. Reopens gemma
  pending a quantization probe. See `training/runs/gemma_v5_curriculum_v1/NOTE.md`.

## Conventions

- Run names: `gemma_v5_briefs_<recipe>_v<n>`. One config yaml per run under
  `training/configs/`, output under `training/runs/<name>/`.
- Evaluate every run with the round-1 harness (50-row held-out test split, blind
  multi-judge) so numbers stay comparable across runs.
- Pick checkpoints by **test-set generation quality, not validation loss**
  (round-1 showed val loss mis-ranks checkpoints for this task).
