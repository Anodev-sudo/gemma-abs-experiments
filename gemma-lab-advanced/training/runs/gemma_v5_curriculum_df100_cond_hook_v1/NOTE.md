# cond_hook_v1 — NEGATIVE result (pre-registered)

**Question:** Can the *groundable subject-fronting* half of Granite's hook be recovered on
`faithful_only_q5` without re-poisoning faithfulness, by conditionally relabeling targets to
front a specific named mechanism iff abstract-grounded?

**Pipeline:** 5,182 faithful targets relabeled by the gemma-4-26b teacher (79.4% frontable,
fresh-context gate, cross-family Claude audit 98.3% precision, 0 ungrounded-subject leaks) →
stage-1 LoRA from `stripped_base_df100` (ckpt1000 picked by test-gen quality) → q5, 97 MB,
size-controlled vs incumbent.

**Eval:** blind 3-way (cond_hook / faith / granite), 240 repr + 150 stress, double-judged
(Haiku), bootstrap CIs. `aggregate_3way_condhook.py`.

**Result — hook did NOT move vs faith:**
- hook Δ(cond_hook−faith): repr **−0.031** [−0.121,+0.052], stress **−0.010** [−0.133,+0.123]
  → ~0, far under the pre-registered +0.15 bar.
- faithful Δ: repr +0.056 [−0.060,+0.169], stress +0.150 [−0.010,+0.310] → **non-inferior**
  (lower-CI ≥ −0.10 both). Faith held / nominally up.
- readable Δ: repr −0.071, stress −0.047 (both NS, small cost).
- hallucination (hard, ≤0.5): repr 3.8%→**6.2%** (ticked up), stress 4.7%→4.7% (tied).
- **Positive control:** cond_hook−granite hook = −0.263 repr / −0.283 stress (**sig**) — judges
  detect hook when it exists; they found none cond_hook vs faith. Null is real, not judge noise.

**Decision (§5 matrix row 3):** subject-fronting hook is ALSO capacity-pinned at 270M. The model
learned the surface form (~74% briefs front a named specific) but blind judges rate them no
higher on hook than the generic hedge — and it costs a little faith/readability on repr.
**Keep `faithful_only_q5`. Do not ship cond_hook. Question closed.**

**Why:** the recoverable half of Granite's hook (grounded subject-fronting) is inert for perceived
hook; the half that moves it (the contrast/surprise clause) is the fabrication vector we scoped
out on purpose (§3). So the hook gap to Granite is real capacity, not a data choice — the FIRST
named limit in this project arc that held up as a genuine wall.

Artifacts: relabel `quant_probe/distill/cond_hook/`, data `curriculum_stage1_gemma4xl_cond_hook_v1`,
q5 `quant_probe/cond_hook_q5/` (97 MB), eval `quant_probe/judge3way_condhook/`.
