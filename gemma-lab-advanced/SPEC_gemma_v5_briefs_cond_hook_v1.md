# SPEC — `gemma_v5_briefs_cond_hook_v1`

**Goal (one line):** Recover the *subject-fronting* half of Granite's hook edge on
`faithful_only_q5` **without** re-introducing the unsupported-specificity that
poisoned faithfulness in every prior voice run — by making the generic hedge
*conditional* instead of removing it.

**Status:** PLANNED. Incumbent to beat = `faithful_only_q5` (97 MB). Fixed ruler =
Granite q8 (374 MB), not a ship candidate. Same idiom/harness as `VOCAB_STRIP_FINDINGS`.

> **Working set & teacher (corrected 2026-06-16 — this supersedes the earlier "gold set"
> framing).** The incumbent `faithful_only_q5` was trained on
> `training/data/curriculum_stage1_gemma4xl_faithful_v1` — **5,182 train / 50 valid /
> 50 test**, chat format, strict-verify-gated, audited at 2.92/3 faithful. That is the set
> we relabel here, *not* the 894-row v5 gold split. The only teacher we have is the
> LMStudio-served **`google/gemma-4-26b-a4b-qat`** at **http://127.0.0.1:1234** (OpenAI-
> compatible HTTP; reasoning disabled), driven through `quant_probe/distill/client.py`.
> **Division of labor.** The **teacher does the in-pipeline work** — generating candidate
> briefs and running the faithfulness filter/gate that decides what enters training data
> (the same pipeline that built the faithful set, reused; independence within it is a
> *fresh-context claim-decomposition* gate, as in `quant_probe/distill/regenerate.py`).
> **Claude(s) do the evaluation** — judging data and results: the double-judged Stage-D
> eval, the verifier-precision audits in Stage A, and the §1 abstract-verified adjudication.
> So Claude is the *evaluator/auditor*, never the inline data filter; the teacher is the
> *generator/filter*, never the headline judge.

---

## 1. Motivating finding

Side-by-side of `faithful_only_q5` vs Granite on 40 most-recent papers (2026-06-16),
plus abstract-verified adjudication of three faithful-suspect briefs:

- **Hook delta is one concentrated tic, not diffuse flatness.** ~⅓ of faithful briefs
  open with a generic subject — *"A new framework / method / dataset …"*. Granite almost
  never does; it fronts the **specific named mechanism** as the subject ("Cache-aware
  eviction helps…", "Saturation-aware guidance avoids…") and leads with the payoff.
- **The generic hedge is doing real defensive work.** Abstract-verified:
  - Paper 10 (asteroids, `2606.17047`): faithful *did not* hedge, reached for a specific
    subject, and **fabricated** one — "Cygnus-1 system" (pure invention; these are
    Sun-orbiting NEAs). Granite hedged toward the grounded headline and was clean.
  - Paper 37 (axion, `2606.17019`): faithful grabbed the correct specific
    ("early-time runaway exponential potential") then **over-compressed** "erasure of
    isocurvature perturbations" → "axion erasure" (flips what's erased) + nonsense
    "inflationary QCD". Granite kept it intact.
  - Paper 22 (DP-FL backdoor, `2606.17035`): wash — faithful nailed the mechanism
    ("masks malicious contributions"), inverted only the tail clause ("to improve
    robustness"); Granite misframed the whole paper as info-leakage.
- **Interpretation.** Faithful breaks *exactly when it reaches for a specific subject
  anchor it can't ground* — it either fabricates (10) or garbles by compression (37).
  That is the **same instinct** that makes Granite hooky (front a specific subject),
  degrading at 270M when grounding is absent. So the hedge isn't pure flatness: on
  ungroundable papers, hedging to generic is the *correct* move. Kill the hedge
  globally → manufacture more Cygnus-1s.

**Thesis:** The recoverable hook is the **groundable-subject-fronting** lever only. The
model currently over-applies the generic hedge *everywhere*; the fix is teaching it
*when the hedge is unnecessary* — i.e. front the specific mechanism **iff** it is
verifiably in the abstract, else keep the hedge.

---

## 2. Hypothesis (falsifiable, pre-registered)

> Rewriting the grounding targets so the subject is fronted **only when the specific
> mechanism is abstract-grounded** (generic hedge retained otherwise), trained as a
> single faithfulness-first stage, will improve judged **hook ≥ +0.15** (repr set, CI
> excludes 0) over `faithful_only_q5`, while keeping **faithfulness statistically
> non-inferior** (lower CI bound of faith Δ ≥ −0.10 on *both* repr and stress) and
> **hallucination not increased** (no sig rise).

**Null / kill outcome is itself valuable:** if hook fails to move ≥ noise floor (~0.15)
under this gate, that is clean evidence the subject-fronting hook is *also*
capacity-pinned at 270M, not a data artifact — which closes the question your last three
rounds left open. Pre-register and report either way.

---

## 3. Scope decision (read before building)

Granite's hook comes from **two** moves: (a) specific-mechanism **subject-fronting**, and
(b) a **contrast clause** ("…even when thrust is limited", "…without cost spikes").

- **In scope:** (a) only. Subject-fronting is grounded-retrieval — gateable against the
  abstract.
- **Explicitly OUT of scope:** (b). The contrast clause is the fabrication vector — it
  rewards asserting a surprise the abstract often doesn't supply. This is what poisoned
  faith in rounds 5–6 / the voice drills. **Do not target it. Do not let the relabeler
  produce it.** We concede the contrast-clause hook to capacity on purpose.

Target brief template: `[grounded specific subject] [payoff verb] [grounded object]`.
No "even when / without X" surprise tail.

---

## 4. Method

Single stage. No voice stage (consistent with `faithful_only` "ship grounding-only").
The intervention is a **surface-form relabel of the existing 5,182 grounding targets**,
not a new stage and not new papers.

### Stage A — Relabel the faithful working set (the crux)

Input is the faithful set's **source artifacts**, not the chat rows (title/abstract are
embedded inside the user turn there, so don't parse them back out — go to source):

- `quant_probe/distill/regenerated.jsonl` — the **5,182** faithfulness-first briefs
  (`{arxiv_id, brief}`) that *are* the assistant targets in
  `training/data/curriculum_stage1_gemma4xl_faithful_v1/train.jsonl`. This `brief` is
  `target_v1`.
- `quant_probe/distill/pool.json` — the paper pool (`{arxiv_id, title, abstract}`,
  12k papers) joined by `arxiv_id` to recover the grounding abstract.

So each relabel item is a `(title, abstract, target_v1)` tuple reconstructed by join.
(The 50/50 valid/test stay as the existing brief split — they only track val/test loss,
and we pick checkpoints by test-set generation quality anyway. Relabel **train only**.)

Two-pass, faithfulness-gated, **both passes by the gemma-4-26b teacher**
(`quant_probe/distill/client.py`). **The verify gate is the firewall — it is why this
won't re-poison**; it is the identical mechanism that gated the incumbent faithful set.

1. **Generate (relabeler).** New script `quant_probe/distill/relabel_cond_hook.py`,
   structured like `regenerate.py` (teacher `RSYS`/`VSYS`, `reasoning_effort=none`). For
   each `(title, abstract, target_v1)` produce `target_v2`:
   - Extract candidate specific-mechanism noun phrases present in the abstract (named
     method/dataset/object/result, e.g. "RING", "TriCCS video photometry", "scaling
     attractor").
   - If ≥1 candidate is **clearly supported** in the abstract → rewrite to front it as
     subject + grounded payoff, template per §3.
   - If none is clearly supported → **keep the generic-hedge form** (= return `target_v1`
     unchanged, which is already the grounded generic brief).
   - Tag the row `frontable: true|false` and record the chosen subject NP.
   - Hard constraint to relabeler: front **only** specifics that are verbatim-or-clearly
     entailed by the abstract; never introduce a contrast clause; when in doubt, hedge.
   - Note the relabeler's own `RSYS` in `regenerate.py` currently *forbids* naming the
     method/acronym (rule 4). For this run that rule is **inverted for groundable subjects
     only**: naming the specific mechanism is the entire point — but still gated downstream.

2. **Verify (independent gate — same teacher, fresh context).** Reuse the
   `quant_probe/distill/verify.py` / `regenerate.py` `VSYS` claim-decomposition checker:
   the teacher sees **only** `abstract + target_v2` (never the relabeler's reasoning or
   `target_v1`), lists each factual claim SUPPORTED/UNSUPPORTED, and emits `VERDICT:
   PASS|FAIL`. Independence here is **fresh-context + claim-decomposition**, not a second
   model — that is all we have, and it is exactly what audited the faithful set to 2.92/3.
   Add one cond-hook-specific check to `VSYS`: flag any **subject NP absent from the
   abstract** and any **"surprise/contrast" clause** as automatic FAIL.
   - Fail → demote that row to the generic-hedge form (`target_v1`). Log the failure reason.
   - **Claude audit (do this, not optional):** have Claude evaluate ~50 PASS rows tagged
     `frontable:true` against their abstracts to estimate the teacher-verifier's *precision*
     on the new subject-NP check. This is exactly Claude's role here — evaluating the data
     the teacher filtered, not re-filtering it. The gate is load-bearing; measure it.

3. Emit relabeled set + a `relabel_report.md` under `quant_probe/distill/cond_hook/`:
   `frontable` rate, % demoted by verifier, failure-reason histogram. **Sanity gate before
   training:** if `frontable` rate is tiny (<~25%) the lever has no surface to act on —
   stop and reconsider.

> Disambiguation: the **relabeler/verifier is the gemma-4-26b *teacher*** (`@127.0.0.1:1234`).
> The **gemma-3-270m *student*** never sees the relabel process — only the verified
> `target_v2` rows that survive the gate. "Gemma" means two different models here; keep
> them straight.

### Stage A′ — Assemble the chat set

Run a `build_dataset.py`-style assembler (clone it) over the verified `target_v2` +
`pool.json` to emit `training/data/curriculum_stage1_gemma4xl_cond_hook_v1/`
(train.jsonl from `target_v2`; copy valid.jsonl/test.jsonl unchanged from the faithful
set; write `manifest.json`). Same `BRIEF_SYSTEM`/`BRIEF_USER` envelope as the incumbent so
the *only* delta vs `faithful_v1` is the assistant target surface form.

### Stage B — SFT

- Rebuild the single grounding stage **from `quant_probe/stripped_base_df100`** (the
  DF≥100 vocab-stripped base the incumbent trained from — **not** raw HF gemma-3-270m;
  size-control requires the same base) on the relabeled set. Clean rebuild is the cleanest
  read.
  - *Cheap alt probe:* continue-SFT `faithful_only`'s pre-quant LoRA
    (`training/runs/gemma_v5_curriculum_df100_faithful_v1/stage1/adapters`) on the
    relabeled set at low LR. Faster, but risks drift; treat as a probe, not the headline run.
- LoRA on **attn + MLP** (round-2 established MLP carries grounding for this task;
  attn-only under-recovers). This matches `faithful_df100_stage1.yaml` exactly (rank 16,
  scale 10, 2000 iters, LR 1e-5 cosine) — copy it and swap only the `data:` path.
- Pick checkpoints by **test-set generation quality, not val loss** (round-1 rule).
- Config: `training/configs/cond_hook_df100_stage1.yaml`; output
  `training/runs/gemma_v5_curriculum_df100_cond_hook_v1/`.

### Stage C — Quantize + strip (match incumbent size)

Mirror `train_faithful.sh`'s `q5()` *exactly* so the artifact is size-controlled against
`faithful_only_q5` (97 MB): fuse the Stage-B adapter onto `quant_probe/stripped_base_df100`
(`mlx_lm fuse`), then `mlx_lm convert -q --q-bits 5 --q-group-size 64`. The base is already
DF≥100 vocab-stripped, so no separate strip/recovery fine-tune is in the incumbent recipe —
**do not add one** (recovery was an earlier, dropped variant; adding it would break the
size/recipe match and confound the hook read). Verify the MLX-Swift sidecar loads, per the
existing pipeline.

### Stage D — Eval (reuse harness exactly)

- **Same** 240 representative + 150 strip-stress papers, blind, double-judged, bootstrap
  CIs. Same rubric dimensions: faith (repr / stress), hook, hallucination, readability.
- Run head-to-head via `quant_probe/aggregate_hg.py`, same-runtime, blind:
  - `cond_hook_q5` vs `faithful_only_q5` (primary: did hook move, did faith hold).
  - `cond_hook_q5` vs Granite q8 (does it narrow the +0.27 hook gap; faith stays tied).

---

## 5. Pre-registered decision matrix

| Hook Δ (vs faithful_only, repr) | Faith Δ lower-CI (repr & stress) | Hallucination | Decision |
|---|---|---|---|
| ≥ +0.15, CI excl. 0 | ≥ −0.10 both | no sig rise | **SHIP `cond_hook_q5`** — hook recovered, faith held |
| ≥ +0.15 | < −0.10 either, OR | any sig rise | **REJECT** — re-poisoned; the gate leaked, audit verifier fails |
| < +0.15 (≈noise) | ≥ −0.10 | no rise | **NEGATIVE result, valuable** — subject-hook is also capacity-pinned; keep `faithful_only`, close the question |

Hallucination rising *at all significantly* = kill regardless of hook gain. Non-negotiable.

---

## 6. Ablations / supporting runs

- **Gate-strength sweep:** strict vs lenient "clearly supported" threshold in the Stage A
  verify `VSYS`. Predicts the faith↔hook frontier — strict = less hook moved, safer faith.
- **Gate-off control = round-6 redux:** run `relabel_cond_hook.py` fronting specifics
  *without* the verify pass (emit `target_v2` raw). Expect faith to drop / halluc to rise —
  confirms the verifier (not the fronting) is what saves faith. Cheap, high-information; run it.
- (Optional) contrast-clause-on arm, to empirically reconfirm §3's out-of-scope call if
  challenged later. Expect it to poison; only run if someone disputes the scoping.

---

## 7. Risks / open questions

- **Verifier is load-bearing — and it is the gemma-4-26b teacher, not a stronger judge.**
  If it under-catches unsupported specifics (especially the *new* subject-NP check), faith
  leaks. Treat verifier precision as a first-class metric; have Claude audit a sample of its
  PASS rows, per Stage A step 2.
- **`frontable` may be capacity-correlated.** The papers where a specific is groundable
  may be exactly the easy ones gemma already does well → hook gain concentrates where it
  matters least. Check hook Δ split by `frontable` and by repr/stress population.
- **Compression garble (paper-37 mode) is orthogonal to fronting** and may persist — it's
  a length-pressure failure, not a hedge failure. Watch it; may need a separate fix.
- **Single-teacher collusion.** Generate and verify share one model; fresh context +
  claim-decomposition is the firewall, but it can't catch a blind spot the teacher has in
  *both* roles. The **Claude audit** is the only independent check on that floor — which is
  precisely why the evaluator is a different model family from the generator/filter.

---

## 8. Artifacts to produce

- `quant_probe/distill/relabel_cond_hook.py` — the relabeler (+ reused `verify.py` gate)
- `quant_probe/distill/cond_hook/` — relabeled `target_v2` + `relabel_report.md`
- `training/data/curriculum_stage1_gemma4xl_cond_hook_v1/` — assembled chat set
- `training/configs/cond_hook_df100_stage1.yaml`
- `training/runs/gemma_v5_curriculum_df100_cond_hook_v1/` — adapters, logs, `NOTE.md`
- `quant_probe/cond_hook_q5/` — stripped/quantized artifact (~97 MB)
- head-to-head row appended to `quant_probe/VOCAB_STRIP_FINDINGS.md`

---

## Appendix — relabeler & verifier prompt sketches

These instantiate as the gemma-4-26b `RSYS`/`VSYS` in `relabel_cond_hook.py` and
`verify.py` — `reasoning_effort=none`, OpenAI-compatible call via `client.chat(...)`.

**Relabeler (generate):**
> You rewrite one-sentence arXiv "feed briefs". Given TITLE + ABSTRACT + current BRIEF,
> output a brief that leads with the **specific named mechanism** (method, dataset,
> object, or result) as its subject, followed by its payoff — **only if** that specific
> is verbatim or clearly entailed by the ABSTRACT. If no specific is clearly supported,
> return the current BRIEF unchanged (generic form). Never invent entities, numbers,
> systems, or contrast clauses ("even when…", "without…"). 10–14 words, one sentence.
> Also output JSON: `{frontable: bool, subject_np: str|null, source_span: str|null}`.

**Verifier (gate):**
> Given ABSTRACT + candidate BRIEF, flag every element of the brief not supported by the
> abstract: invented entities/numbers, a subject NP absent from the abstract, polarity
> inversions, or any "surprise/contrast" clause. Output `{pass: bool, fails: [..]}`.
> Err toward flagging. A brief passes only if every specific is grounded and no contrast
> clause is present.
