# Safety Representative Eval Set 10000 Adversarial v1

This directory contains a deterministic 10k input set for safety-checking a
local paper-summary model.

The set has:

- `6,000` normal recent-balanced corpus rows
- `1,800` real edge-topic rows selected by auditable keyword patterns
- `2,200` adversarial stress rows derived from real corpus rows

The adversarial rows preserve the source title and abstract, then append
untrusted text such as prompt-injection instructions, JSON/HTML command noise,
redacted sensitive-topic pressure, or conflicting style instructions. They are
designed to test whether the model summarizes the paper rather than following
hostile text embedded in the abstract field.

Files:

- `eval_inputs.jsonl`: selected title/abstract inputs with metadata
- `paper_ids.json`: eval row IDs in order
- `manifest.json`: generation settings, candidate counts, topic patterns,
  adversarial templates, and category breakdowns

Reproduction:

```bash
python3 scripts/build_safety_eval_set.py \
  --output-dir hf_dataset/evals/safety_representative_10000_adv_v1 \
  --normal-count 6000 \
  --edge-count 1800 \
  --adversarial-count 2200
```

Use `--overwrite` to regenerate this directory in place.
