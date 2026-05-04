# Synthetic Summary Prompts

This directory stores versioned prompt specs for the generator and reviewer used in the synthetic `abstract -> summary` pipeline.

Current prompt pair:

- `deepseek_v32_generator_v5_hybrid_sentence_case_minimal.json`
  - Intended teacher input: `title + abstract`
  - Intended student target: `abstract -> summary`
  - Goal: keep the strong `v5` payoff behavior while adding a minimal sentence-case / non-title-phrasing constraint.
- `qwen3_235b_a22b_2507_reviewer_classifier_v2_headline_baseline.json`
  - Intended reviewer input: `title + abstract + candidate summary`
  - Goal: cheaply classify feed-sentence candidates as `accept` or `drop` without rewriting.

Experimental generator variants:

- `deepseek_v32_generator_v4_headline_baseline.json`
  - Intended teacher input: `title + abstract`
  - Intended student target: `abstract -> summary`
  - Goal: previous headline-style baseline, retained for comparison.
- `deepseek_v32_generator_v5_hybrid_1_5.json`
  - Intended teacher input: `title + abstract`
  - Intended student target: `abstract -> summary`
  - Goal: stronger payoff-first hybrid baseline that tested well but showed run-to-run variance.
- `deepseek_v32_generator_v6_sentence_case_balanced.json`
  - Intended teacher input: `title + abstract`
  - Intended student target: `abstract -> summary`
  - Goal: keep sentence-case flow while restoring stronger payoff pressure than the softer sentence-case trial.
- `deepseek_v32_generator_v5_hybrid_sentence_case_minimal.json`
  - Intended teacher input: `title + abstract`
  - Intended student target: `abstract -> summary`
  - Goal: current baseline; keep the strong `v5` payoff behavior while adding only a minimal sentence-case / non-title-phrasing constraint.

Alternate reviewer prompts:

- `qwen3_235b_a22b_2507_reviewer_classifier_v3_production.json`
  - Intended reviewer input: `title + abstract + candidate summary`
  - Goal: faster production reviewer variant that keeps the same semantic bar while dropping checks already enforced by local filters.
- `nemotron_3_nano_30b_a3b_reviewer_classifier_v1.json`
  - Intended reviewer input: `title + abstract + candidate summary`
  - Goal: cheaply classify candidates as `accept` or `drop` without rewriting.
- `deepseek_v32_reviewer_v3.json`
  - Intended reviewer input: `title + abstract + candidate summary`
  - Goal: accept strong candidates, rewrite usable weak ones, and drop bad rows.
- `deepseek_v32_hard_negative_generator_v1.json`
  - Intended input: `title + abstract + gold summary + requested failure axis`
  - Goal: generate one plausible near-miss summary that is wrong on exactly one requested axis.
- `deepseek_v32_hard_negative_generator_v2_precision.json`
  - Intended input: `title + abstract + gold summary + requested failure axis`
  - Goal: current recommended hard-negative generator; pairs with the structured numeric verifier and tighter axis-specific construction rules.
- `qwen3_235b_a22b_2507_hard_negative_verifier_v6_precision_numeric.json`
  - Intended input: `title + abstract + gold summary + bad summary + requested failure axis`
  - Goal: current recommended verifier for hard negatives; keeps the precision-first consistency rules and explicitly calibrates clean one-slot `wrong_number_binding` swaps.
- `qwen3_235b_a22b_2507_hard_negative_verifier_v3_balanced.json`
  - Intended input: `title + abstract + gold summary + bad summary + requested failure axis`
  - Goal: earlier balanced verifier that preserved smoke-run yield before the structured numeric path landed.
- `qwen3_235b_a22b_2507_hard_negative_verifier_v5_precision.json`
  - Intended input: `title + abstract + gold summary + bad summary + requested failure axis`
  - Goal: earlier precision-first verifier that removed contradiction overrides and rejected acceptable-as-written rows more aggressively, but still underfilled `wrong_number_binding`.
- `qwen3_235b_a22b_2507_hard_negative_verifier_v2.json`
  - Intended input: `title + abstract + gold summary + bad summary + requested failure axis`
  - Goal: yield-aware experimental variant that relaxes some checks to recover brittle numeric and entity axes.
- `qwen3_235b_a22b_2507_hard_negative_verifier_v4_conservative.json`
  - Intended input: `title + abstract + gold summary + bad summary + requested failure axis`
  - Goal: precision-first experimental variant that rejects borderline negatives more aggressively.
- `qwen3_235b_a22b_2507_hard_negative_verifier_v1.json`
  - Intended input: `title + abstract + gold summary + bad summary + requested failure axis`
  - Goal: earlier structured verifier baseline retained for comparison with the newer balanced and experimental variants.
- `qwen3_235b_a22b_2507_faithfulness_cleanup_rewriter_v1.json`
  - Intended input: `title + abstract + existing summary + heuristic risk flags`
  - Goal: draft safer SFT targets for high-risk accepted rows by removing unsupported numbers, entity labels, mechanisms, causal claims, and overstrong wording.
- `kimi_k2_6_q4_correction_salvage_rewriter_v1.json`
  - Intended input: `title + abstract + q4 bad summary + q4 review tags + previous failed correction`
  - Goal: generate concise no-reasoning q4 correction targets for locally ineligible rows.
- `kimi_k2_6_q4_correction_salvage_short_rewriter_v1.json`
  - Intended input: same as the Kimi salvage prompt, with a stricter compression target
  - Goal: salvage a small number of overlong correction rows without reopening the reasoning-model branch.
- `qwen3_235b_a22b_2507_q4_correction_verifier_generic_v1.json`
  - Intended input: `title + abstract + q4 bad summary + corrected summary`
  - Goal: independently verify Kimi correction targets before materializing q4 hardening rows.

Design notes:

- The generator is allowed to be creative within a narrow style envelope.
- The current baseline generator targets `12-15` words with a hard cap at `15`.
- The current baseline generator keeps the stronger payoff-first `v5` behavior but asks for sentence case and less title-like phrasing.
- The current baseline generator pushes harder against generic `A ...` and label-first openings.
- The default reviewer is a Qwen classifier that records `accept` or `drop` without rewriting.
- The `qwen3_235b_a22b_2507_reviewer_classifier_v3_production.json` variant is the fast-path reviewer for large production runs; it removes only checks that are already handled in code before review.
- The DeepSeek rewrite reviewer remains available when a heavier second pass is wanted.
- The hard-negative generator and verifier prompts are for second-stage correction-data construction, not the main 5k synthetic-summary pipeline.
- The faithfulness cleanup rewriter is for SFT overlay construction, not DPO. It drafts candidate replacements that still require manual agent review before release materialization.
- The compact q4 correction path is Kimi no-reasoning rewrite plus Qwen
  verification. The abandoned DeepSeek/reasoning and Qwen 3.5/3.6 repair probes
  were pruned after the combined q4 hardening release was promoted.
- The current default hard-negative pair in `scripts/build_hard_negative_pack.py` is `deepseek_v32_hard_negative_generator_v2_precision.json` plus `qwen3_235b_a22b_2507_hard_negative_verifier_v6_precision_numeric.json`.
- That default keeps the precision-first consistency rules, removes contradiction admissions, and restores `wrong_number_binding` coverage by using structured one-slot numeric swaps before falling back to freeform generation.
- The `v3`, `v5`, `v2`, and `v4` hard-negative verifiers are older experiment variants retained for comparison and troubleshooting.
- Hard local filters now run before review and skip the reviewer entirely for deterministic failures such as overlong summaries, banned openings, non-single-sentence outputs, empty/trivially short outputs, near-duplicate title restatements, and obviously formula-heavy notation that reads poorly in the feed.
- Both prompts assume the app already shows the title, author list, and date, so the summary should focus on the payoff.
- The preferred target remains one sentence and roughly `10-15` words.

Canonical paths after the storage split:

- source corpora live under `hf_dataset/corpora/`
- run artifacts live under `hf_bucket/runs/`
- the repo itself keeps prompts, scripts, and docs

Run the smoke pipeline with:

```bash
python3 scripts/run_synthetic_summary_pipeline.py \
  --prepare-only \
  --sample-size 100 \
  --output-dir hf_bucket/runs/deepseek_v32_smoke_100
```

For a live OpenRouter run, fill in `.env` first, then run:

```bash
python3 scripts/run_synthetic_summary_pipeline.py \
  --sample-size 100 \
  --output-dir hf_bucket/runs/openrouter_smoke_100
```

For a full 20k production run, use deterministic shards plus the slim reviewer:

```bash
python3 scripts/run_synthetic_summary_pipeline.py \
  --full-corpus \
  --shard-count 6 \
  --shard-index 0 \
  --generation-workers 4 \
  --review-workers 4 \
  --generator-max-output-tokens 64 \
  --reviewer-max-output-tokens 48 \
  --reviewer-prompt prompts/synthetic_summary/qwen3_235b_a22b_2507_reviewer_classifier_v3_production.json \
  --output-dir hf_bucket/runs/full_20k_shard_0
```

Repeat with `--shard-index 1` through `5` for the rest of the corpus. In full-corpus mode, `sample_source_rows.jsonl` contains the selected shard rows rather than a random smoke sample.

The script automatically loads `.env` from the repo root.

Faithfulness cleanup overlay workflow:

```bash
python3 scripts/prepare_faithfulness_cleanup_candidates.py

python3 scripts/run_faithfulness_cleanup_rewriter.py \
  --prepare-only

python3 scripts/run_faithfulness_cleanup_rewriter.py \
  2>&1 | tee hf_bucket/experiments/faithfulness_cleanup/recent_3y_balanced_10746_faithfulness_overlay_800_v1/rewrite_run/rewrite.log

python3 scripts/prepare_faithfulness_cleanup_review_bundle.py
```

After manual primary and QA review decisions are filled in, materialize the overlay release:

```bash
python3 scripts/prepare_faithfulness_cleanup_qa_followup.py \
  --review-bundle-dir hf_bucket/experiments/faithfulness_cleanup/recent_3y_balanced_10746_faithfulness_overlay_800_v1/manual_review

python3 scripts/materialize_faithfulness_cleanup_overlay.py \
  --review-bundle-dir hf_bucket/experiments/faithfulness_cleanup/recent_3y_balanced_10746_faithfulness_overlay_800_v1/manual_review
```
