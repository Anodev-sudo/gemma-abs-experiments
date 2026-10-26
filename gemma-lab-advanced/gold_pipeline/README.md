# Gold-Set Generation Pipeline — Essential Bundle

> **In this repo, only the docs are here.** The pipeline code is identical to scripts the
> parent repo already tracks, so it was not copied twice: `scripts/summary_generation/run_pipeline.py`,
> `scripts/synthetic_summary_runtime.py`, `scripts/build_natural_language_teacher_pool_v1.py`,
> `scripts/promote_summary_run_release.py`, `scripts/merge_summary_releases.py` and
> `scripts/materialize_natural_language_gold_milestone.py`. Prompts are under `prompts/synthetic_summary/`
> and configs under `configs/synthetic_summary_generation/`. The corpus and gold-set data live in a
> private storage bucket.

A self-contained snapshot of the **natural-language summary** pipeline that produced
the **994-row "gold set"** used to fine-tune `ibm-granite/granite-4.0-350m`.

Extracted from the parent repo on 2026-06-13.
This bundle holds the *essential, most up-to-date* parts of all five stages, the
data at both ends, and the exact prompt versions the milestone actually drew from.

> **No third-party dependencies.** The whole pipeline is pure Python stdlib
> (HTTP is hand-rolled in `synthetic_summary_runtime.py`). Python 3.12+.
> Live generation stages need `OPENROUTER_API_KEY` in the environment.

---

## The flow

```
01_corpus            20k weighted arXiv abstracts (the raw input)
       │
02_pool_builder      sample a category-balanced candidate pool
       │
03_pipeline_engine   writer → local-gates → reviewer → retry → materialize
       │             (run_pipeline.py, driven by a JSON config)
       │
04_release_tooling   promote / merge per-run outputs into named releases
       │
05_milestone_assembly  stitch + dedup releases into the balanced 994 gold set
                       └── gold_set/  ← the final artifact + the MLX splits Granite trained on
```

---

## Stage by stage

### 01_corpus
- `arxiv_abstract_corpus_weighted_20000.jsonl` — 20,000 abstracts, category-weighted (28 MB).
- `*_manifest.json`, `*_breakdown.csv` — how the corpus was balanced.
- This is the single input to the whole pipeline.

### 02_pool_builder
- `build_natural_language_teacher_pool_v1.py` — samples a balanced candidate pool
  from the corpus into the writer's input JSONL. (Only `v1` exists; it is the latest.)

### 03_pipeline_engine — the heart
- `run_pipeline.py` — generic, config-driven runner. Stages:
  `prepare → probe-routes → write → local-filter → verify → retry-failed → materialize → report`.
- `synthetic_summary_runtime.py` — **required** shared module (route ladder, HTTP, concurrency).
- `PIPELINE_README.md` — the original engine docs (config schema, output layout).
- `configs/` — **6** configs:
  - `..._v1.json` — clean canonical base / template (uses the original `writer_v1`/`verifier_v1`).
  - `..._v24_legacy_*_start_cleanup_current.json` (×5) — the **final-state** lanes
    (nonmath, math ×2, physics ×2) as of the last milestone build.
- `prompts/writer/` (11) + `prompts/verifier/` (4) — see provenance below.

Run one lane (live):
```bash
OPENROUTER_API_KEY=... python run_pipeline.py \
  --config configs/natural_language_teacher_deepseek_v4_pro_v24_legacy_v10_nonmath_start_cleanup_current.json \
  --stage all
```
> Configs are kept **verbatim**, so their `input.path` / `output.*` fields point at the
> original repo's `hf_bucket/` and `hf_dataset/` paths. Re-point those (or run from the
> original repo) before a live run. The prompt files they reference are all present here.

### 04_release_tooling
- `promote_summary_run_release.py` — promote a single run's accepted rows into a named release.
- `merge_summary_releases.py` — merge multiple releases.
- `synthetic_summary_runtime.py` — bundled again here (both tools import it).

### 05_milestone_assembly
- `materialize_natural_language_gold_milestone.py` — selects the balanced milestone
  (base-gold-first, then diverse round-robin within each category bucket), dedups, writes manifest.
- `gold_set/` — **the deliverable**:
  - `training_rows.jsonl` — the **994 gold rows** (full provenance per row).
  - `manifest.json` — counts, category balance, full source lineage.
  - `selected_additions.jsonl`, `overflow_rows.jsonl`, `duplicate_rows.jsonl` — assembly audit.
  - `mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5/`
    — the MLX chat-format splits **Granite actually trained on** (train 894 / valid 50 / test 50).

---

## Prompt provenance — what the 994 rows were *actually* generated with

Distinct prompt-version stamps read directly off `gold_set/training_rows.jsonl`:

**Writers** (`generator_prompt_version`)
| rows | prompt |
|-----:|--------|
| 412 | `writer_v10_ratio_fill` |
| 229 | `writer_v7` |
| 116 | `writer_v14_math` |
|  33 | `writer_v16_math` |
|  23 | `writer_v8` |
|  16 | `writer_v9_format_rescue` |
|  10 | `writer_v11_length_rescue` |
|   7 | `writer_v5` |
|   5 | `writer_v12_verifier_reason_rescue` |
|   5 | `writer_v13_jargon_repair` |
|   2 | `writer_v6` |
| 92 + 34 | *deterministic* `start_cleanup` / `article_strip_rescue` — **no LLM prompt** (local transforms) |

All 11 LLM writer prompts above are in `prompts/writer/`.

**Verifiers** (`review_prompt_version`)
| rows | prompt |
|-----:|--------|
| 845 | `verifier_v4` |
| 116 | `verifier_v5_math` |
|  33 | `verifier_v6_math` |

`prompts/verifier/` holds these three **plus `verifier_v3`**, which the final `v24_current`
cleanup configs reference (no row carries a `v3` review stamp — those lanes' accepted rows
ended up verified by `v4`).

> The 126 `deterministic_*` rows came from non-LLM cleanup lanes (the `_start_cleanup` /
> `_article_strip_rescue` config variants apply deterministic edits instead of calling the
> writer). There is no prompt file for them by design.

---

## What was deliberately left out
- The ~90 historical config variants (`v2`–`v23`, batch/rescue/audit twins) — superseded by
  the `v24_current` final-state lanes kept here. The full set remains in the source repo.
- Specialized prompt families for other tracks (q4/q5 hardening, kimi/qwen overlays,
  numeric drills) — unrelated to the natural-language gold set.
- Intermediate per-run release directories — their *tooling* (stage 04) and their *product*
  (stage 05 `gold_set/`) are here; the raw intermediates stay in the source repo's `hf_bucket/`.

## Source pointers (original repo)
- corpus: `hf_dataset/corpora/recent_3y_balanced/`
- pool builder + assembler + release tools: `scripts/`
- engine: `scripts/summary_generation/`
- configs: `configs/synthetic_summary_generation/`
- prompts: `prompts/synthetic_summary/`
- gold set: `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/`
