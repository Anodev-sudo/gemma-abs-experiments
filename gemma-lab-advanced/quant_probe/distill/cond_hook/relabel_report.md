# cond_hook relabel report

**Input:** 5,182 faithfulness-first targets (`regenerated.jsonl`, = the assistant targets in
`curriculum_stage1_gemma4xl_faithful_v1`) joined to abstracts in `pool.json`.
**Relabeler + gate:** gemma-4-26b-a4b-qat teacher (`relabel_cond_hook.py`), fresh-context
claim-decomposition verifier + deterministic surprise-clause pre-demote.
**Run:** resumed across one Mac-sleep outage; completed 2026-06-17 02:22, 5182/5182, 0 missing,
all final briefs unique.

## Distribution

| status | count | % | meaning |
|---|---|---|---|
| fronted | 3,909 | 75.4% | grounded specific subject fronted, passed gate first try |
| fronted_repaired | 205 | 4.0% | passed gate after one repair |
| demoted | 524 | 10.1% | teacher proposed a front, gate rejected → reverted to generic hedge |
| kept_generic | 497 | 9.6% | teacher judged no groundable specific → hedge retained |
| parse_fail | 47 | 0.9% | relabeler JSON unparseable → hedge retained (fail-safe) |
| **frontable_final** | **4,114** | **79.4%** | |

- **Sanity gate (spec §4.3): PASS** — frontable 79.4% ≫ 25% floor; the lever has ample surface.
- **Demote rate: 11.3% of proposed fronts** (524 / 4,638), stable across the entire run — the
  gate is doing consistent work, not waving everything through.
- **Row count preserved:** demotions revert to `target_v1` rather than dropping, so the only
  delta vs the faithful set is surface form (5,182 → 5,182).

## Demotion fail-reason histogram (n=524)

| reason (bucketed) | count |
|---|---|
| other / free-text | 260 |
| unsupported-claim | 181 |
| missed-main-point | 39 |
| ungrounded-subject | 39 |
| verifier-no-verdict (fail-safe) | 3 |
| surprise-clause (deterministic) | 2 |

(Buckets are keyword-matched against the verifier's free-text reason; "other" is large because
the gemma verifier's prose doesn't always key cleanly — not a defect, just coarse bucketing.)

## Caveat — this is the teacher grading itself

79.4% frontable from a **same-model** generate+verify loop is exactly the rubber-stamp shape
the project has been burned by before (self-verification grades on its own curve). The 11.3%
demote rate says the gate isn't *pure* theater, but it does **not** establish that the 4,114
fronts are actually grounded. That is settled only by the **cross-family Claude audit**
(`audit_*.jsonl`), which judges a blind sample of fronted PASS rows against their abstracts.
Do not trust the frontable rate as a quality signal until the audit precision is in.

## Open flag for Stage B

79.4% is high enough that the training signal is "front a specific almost always." Whether a
270M student learns the *conditional* (front iff groundable) or collapses to *always-front*
(→ fabrication on ungroundable inputs) is the pre-registered question. The gate-strength sweep
ablation (stricter threshold → more kept-generic) is the lever if always-front shows up in eval.

## Cross-family Claude audit (result)

Blind, adversarial audit of **60 randomly-sampled fronted rows** by a Claude general-purpose
agent (not told the teacher had passed them), judging each subject + claims against its abstract.

- **Precision = 59/60 = 98.3%.**
- **Ungrounded-subject leaks: 0.** Every named subject (TFMD, GT-MCP, Gumbel-BEARD, …) literally
  appears in / is the named contribution of its abstract. The "plausible-but-absent acronym"
  failure mode — the one that would poison faithfulness — is **absent**.
- **1 leak:** `2606.07952`, a polarity/aggregation slip (a 3-conjunct subject flattened a
  "broken exponential symmetry gives NO Goldstone modes" result into "produce … modes"). Not a
  fabrication. **Patched out** (demoted to generic, status `demoted_audit`) before assembly.
- Calibration notes (not scored as leaks): the gate occasionally fronts a paper's true *secondary*
  result over its headline, and can mishandle multi-result mixed-polarity subjects. Rare,
  content-specific; a targeted filter could push precision →1.0 if needed later.

**Verdict: trustworthy to train on.** The 79.4% frontable rate is high but the fronts are
overwhelmingly grounded — the same-teacher gate is not rubber-stamping the dangerous class.
~1.7% residual leak rate (wide CI on n=60) is within the faithful set's own ~2.92/3 audit noise.
