# Legacy All-Time Abstract Corpus

This directory contains an older all-time weighted export of arXiv abstracts intended as source material for synthetic summary generation and downstream fine-tuning work.

It is kept as a historical snapshot. The current recommended corpus for this project is `../recent_3y_balanced/`.

Files:

- `arxiv_abstract_corpus_weighted_20000.jsonl`
  - One JSON object per paper.
  - Fields: `paper_id`, `title`, `abstract`, `primary_category`, `umbrella_category`, `created_date`, `updated_date`, `categories`.
- `arxiv_abstract_corpus_weighted_20000_manifest.json`
  - Export settings and umbrella-level/category-level counts.
- `arxiv_abstract_corpus_weighted_20000_breakdown.csv`
  - Per-primary-category availability, quota, and selected count.

Sampling notes:

- The exporter uses primary categories for quota assignment.
- The top umbrella targets are `cs=25%`, `physics=15%`, and `math=10%`, with the remaining 50% spread across the other umbrella groups.
- Non-top umbrella groups are capped at `8.5%` of the final dataset so they do not overtake math in the mix.
- Individual primary categories are capped at `275` papers so no single category dominates the export.
- Within `cs`, ML-adjacent categories such as `cs.LG`, `cs.CV`, `cs.CL`, and `cs.AI` are favored during quota assignment.
- Abstracts shorter than 250 characters are excluded.

Reproduction of a comparable all-time export:

```bash
python3 scripts/export_weighted_abstract_sample.py \
  --db /path/to/arxiv.sqlite \
  --output-dir hf_dataset/corpora/all_time_weighted_20000
```

If exact historical reproduction matters, use the manifest in this directory as the source of truth for the snapshot rather than assuming the current script defaults are byte-identical.
