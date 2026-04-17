# AGENTS.md

Guidance for coding agents working in this repository.

## Working Style

- Be casual and direct. Do not be overly flattering.
- Explain what you are about to do before major actions such as repo scans, long-running exports, prompt rewrites, or bulk data generation.
- Share short progress updates while working, especially during long database scans or dataset generation jobs.
- Prefer making reversible, reproducible changes over one-off manual edits.
- Ask clarifying questions when the product goal is ambiguous and a bad assumption could materially change the dataset or prompt style.

## Project Snapshot

- Project: synthetic-data and prompt-design workspace for Gemma summary fine-tuning.
- Current objective: generate high-quality `abstract -> one-sentence summary` training data for a small deployment model.
- Product constraint: summaries are intended for an iPhone paper feed and should be glanceable.
- Current summary target:
  - broad readability over expert density
  - roughly `10-15` words
  - one sentence
  - no fluff like `"This paper"` or `"We propose"`
  - should add value beyond the title, author list, and date shown in the UI

## Repo Map

- `scripts/export_weighted_abstract_sample.py`
  - Exports weighted abstract corpora from the Composer SQLite database.
  - Current defaults are tuned for a recent-only balanced training corpus.
- `hf_dataset/corpora/all_time_weighted_20000/`
  - Older all-time weighted export and metadata.
- `hf_dataset/corpora/recent_3y_balanced/`
  - Main recent-only balanced corpus and metadata.
- `hf_dataset/corpora/smoke/`
  - Small smoke-test export artifacts.
- `hf_dataset/releases/recent_3y_balanced_5000_pilot_v2/`
  - Promoted dataset snapshot from the completed 5k pilot.
- `hf_bucket/runs/`
  - Raw top-level pipeline run artifacts and request/response logs.
- `hf_bucket/experiments/`
  - Machine-generated experiment payloads that do not belong in the main git repo.

## Storage Split

- Treat this repository as source control for scripts, prompts, notes, and lightweight docs.
- Stage publishable corpora and promoted dataset snapshots under `hf_dataset/`.
- Stage raw run exhaust and machine-generated experiment payloads under `hf_bucket/`.
- Local `abstracts/` and `runs/` compatibility links may exist during migration, but the canonical paths are under `hf_dataset/` and `hf_bucket/`.

## Data Source

- The source database lives outside this repo in the Composer project.
- Do not hardcode machine-specific assumptions into documentation or comments.
- Prefer explicit CLI flags like `--db /path/to/arxiv.sqlite` over hidden local assumptions.

## Current Dataset Policy

- Prefer recent papers over full-history arXiv unless the user explicitly wants otherwise.
- The current recommended export uses:
  - `created_date >= 2023-04-07`
  - `20,000` papers
  - umbrella targets of:
    - `cs = 37%`
    - `physics = 28%`
    - `math = 16%`
    - all remaining umbrellas combined = `19%`
  - per-primary-category cap of `275`
  - minimum abstract length of `250` characters
- Preserve diversity. Large categories such as `cs.CV`, `cs.LG`, and `cs.CL` should not dominate.

## Summary Quality Bar

- Summaries should answer: `Why might someone open this paper?`
- The summary should usually contain:
  - the subject or object
  - the main takeaway, finding, or contribution
  - why it matters, if that can fit naturally
- The summary should usually avoid:
  - repeating title phrasing
  - author or date references
  - academic filler
  - vague verbs with no payoff such as `"presents"`, `"explores"`, or `"introduces"` unless the benefit is explicit
  - unsupported claims or invented numbers

## Prompt Design Guidance

- Treat prompt writing as product writing, not generic summarization.
- Optimize for feed usefulness, not abstract compression.
- If a teacher model sees the title, use that to reduce title overlap in the target summary.
- Be careful with small student models:
  - they may copy title text if title is present at inference
  - do not assume fine-tuning alone will fix title repetition
- When changing prompt strategy, note whether the intended student input is:
  - `abstract only`
  - or `title + abstract`
- Keep generation prompts and review prompts aligned with the actual UI and deployment constraints.

## Planned Synthetic Pair Pipeline

- Default pipeline:
  - select source abstracts from the recent balanced export
  - generate a candidate summary with the teacher model
  - review and rewrite with a stricter reviewer prompt
  - filter out bad rows
  - write the final training JSONL
- Current recommended teacher setup:
  - teacher sees `title + abstract`
  - reviewer sees `title + abstract + candidate summary`
  - student training should default to `abstract only` unless the deployment plan changes
- The reviewer should be stricter than the generator.
- Prefer rewrite-over-accept behavior so the final dataset is stylistically consistent.
- Keep intermediate artifacts if they help auditing or error analysis.

## Suggested Training Row Shape

- Keep rows simple and explicit.
- A good default schema is:
  - `paper_id`
  - `title`
  - `abstract`
  - `summary`
  - `primary_category`
  - `categories`
  - `created_date`
  - `teacher_model`
  - `generator_prompt_version`
  - `review_prompt_version`
  - `source_split`
- If storing conversation-style finetune examples, also keep the raw structured source row somewhere in the repo or output directory for later reformatting.

## Filtering Guidance

- Drop rows if the summary:
  - exceeds the word limit
  - repeats title wording too closely
  - uses filler openings or academic boilerplate
  - is too vague to be useful in a scrolling feed
  - contains unsupported metrics or claims
  - sounds unnatural or fragmented
- If a paper abstract is too vague to support a strong summary, prefer dropping it over forcing a weak target.

## Editing Guidelines

- Prefer small, focused changes.
- Keep scripts dependency-light; use the standard library unless a new dependency is clearly justified.
- Do not hand-edit large generated `.jsonl` or `.csv` exports unless the user explicitly asks for that.
- If the export logic changes, update the relevant README in `hf_dataset/corpora/` in the same change.
- Avoid embedding sensitive details, local usernames, local emails, or machine-specific paths in docs or comments.

## Validation

- When changing export logic:
  - run a small smoke export first when practical
  - verify final row count with `wc -l`
  - inspect the manifest and breakdown before calling the export done
- When changing prompt specs:
  - sanity-check for overlap with title
  - sanity-check length constraints
  - prefer examples that reflect the actual mobile feed use case

## Definition Of Done

- Changes are implemented and reproducible.
- Output files or prompt specs match the current product constraints.
- Any changed sampling logic is reflected in README notes.
- Assumptions are stated clearly, especially around date windows, input format, and summary style.
