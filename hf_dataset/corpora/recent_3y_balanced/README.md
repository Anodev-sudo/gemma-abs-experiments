# Recent Balanced Finetune Corpus

This directory contains a recent-only arXiv abstract export intended for summary-pair generation and Gemma finetuning.

It lives under `hf_dataset/` because it is a publishable dataset input, not main-repo source code.

Settings:

- Created-date cutoff: `2023-04-07`
- Target size: `20,000`
- Target umbrella mix:
  - `cs`: `37%`
  - `physics`: `28%`
  - `math`: `16%`
  - all remaining umbrellas combined: `19%`
- Primary-category cap: `275` records per category
- Minimum abstract length: `250` characters

Files:

- `arxiv_abstract_corpus_weighted_20000.jsonl`
- `arxiv_abstract_corpus_weighted_20000_manifest.json`
- `arxiv_abstract_corpus_weighted_20000_breakdown.csv`

Reproduction:

```bash
python3 scripts/export_weighted_abstract_sample.py \
  --db /path/to/arxiv.sqlite \
  --output-dir hf_dataset/corpora/recent_3y_balanced \
  --created-since 2023-04-07
```
