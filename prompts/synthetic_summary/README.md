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

Design notes:

- The generator is allowed to be creative within a narrow style envelope.
- The current baseline generator targets `12-15` words with a hard cap at `15`.
- The current baseline generator keeps the stronger payoff-first `v5` behavior but asks for sentence case and less title-like phrasing.
- The current baseline generator pushes harder against generic `A ...` and label-first openings.
- The default reviewer is a Qwen classifier that records `accept` or `drop` without rewriting.
- The `qwen3_235b_a22b_2507_reviewer_classifier_v3_production.json` variant is the fast-path reviewer for large production runs; it removes only checks that are already handled in code before review.
- The DeepSeek rewrite reviewer remains available when a heavier second pass is wanted.
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
