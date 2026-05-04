# Safety Representative Eval Set 2500 v1

This directory contains a deterministic input set for checking whether a local
paper-summary model introduces unsavory content during normal summary use.

The set has:

- `2,000` normal recent-balanced corpus rows
- `500` edge-topic rows selected by auditable keyword patterns

The edge rows are not adversarial jailbreak prompts. They are ordinary arXiv
titles and abstracts containing vocabulary around safety-sensitive topics such
as medical harm, violence or weapons, biosecurity, hate or bias, privacy and
surveillance, misinformation or politics, cybersecurity, and sexual content.

Files:

- `eval_inputs.jsonl`: selected title/abstract inputs with metadata
- `paper_ids.json`: paper IDs in eval order
- `manifest.json`: generation settings, candidate counts, topic patterns, and
  category breakdowns

Reproduction:

```bash
python3 scripts/build_safety_eval_set.py
```

The script samples from:

```text
hf_dataset/corpora/recent_3y_balanced/arxiv_abstract_corpus_weighted_20000.jsonl
```

Use `--overwrite` to regenerate this directory in place.
