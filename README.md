# Gemma Abstract Finetuning

This repository now treats source control and data storage as separate concerns.

## Layout

- `scripts/`, `prompts/`, `tools/`, and hand-written experiment notes stay in the main git repo.
- `hf_dataset/` stages publishable corpora and promoted training-set snapshots for a separate Hugging Face dataset repo.
- `hf_bucket/` stages raw run artifacts and machine-generated experiment payloads for a separate Hugging Face bucket.

## Canonical Paths

- Main input corpus: `hf_dataset/corpora/recent_3y_balanced/arxiv_abstract_corpus_weighted_20000.jsonl`
- Top-level run outputs: `hf_bucket/runs/`
- Promoted training snapshot: `hf_dataset/releases/recent_3y_balanced_5000_pilot_v2/`

## Migration Notes

- Local `abstracts/` and `runs/` compatibility aliases may exist so older notes still open during the transition.
- The main repo ignores staged dataset and bucket contents on purpose. Wire those directories to their own remotes instead of trying to push them through the main git repo.
