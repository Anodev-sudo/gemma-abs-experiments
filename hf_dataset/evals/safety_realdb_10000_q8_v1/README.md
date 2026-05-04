# RealDB q8 Safety Eval v1

This eval set samples recent papers from the Composer SQLite database and excludes paper IDs already present in prior dataset release/eval JSONL files.

## Contents

- `eval_inputs.jsonl`: model inputs for q8 safety generation.
- `paper_ids.json`: sampled eval paper IDs, including adversarial synthetic IDs.
- `manifest.json`: reproducibility metadata without machine-specific database paths.

## Counts

- Total rows: 10000
- Normal rows: 6000
- Sensitive-edge rows: 2000
- Adversarial rows: 2000

## Rebuild

```bash
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \
  scripts/build_realdb_safety_eval_set.py \
  --db <COMPOSER_DB> \
  --output-dir hf_dataset/evals/safety_realdb_10000_q8_v1 \
  --normal-count 6000 \
  --edge-count 2000 \
  --adversarial-count 2000 \
  --created-since 2023-04-07 \
  --min-abstract-chars 250 \
  --seed 20260501
```
