## Hard-Negative Pack V1

This directory tracks the source-side assets for the first hard-negative pack
builder pass.

The main tracked input here is `semantic_seed_papers_v1.json`, which gives the
builder a small curated set of papers to prioritize when generating semantic
near-miss negatives for:

- `wrong_number_binding`
- `invented_number`
- `reversed_polarity`
- `wrong_entity`

These seeds come from the recent holdout audit so the generator starts from
known brittle cases instead of only drawing generic accepted rows from the
promoted release.

Runtime outputs from actual builder runs belong under:

- `hf_bucket/experiments/hard_negative_pack_v1/`

Prepare-only smoke:

```bash
python3 scripts/build_hard_negative_pack.py \
  --prepare-only \
  --target-per-axis 5
```

Small live smoke:

```bash
python3 scripts/build_hard_negative_pack.py \
  --target-per-axis 5
```
