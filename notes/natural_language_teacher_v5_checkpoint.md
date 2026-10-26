# Natural-language teacher v5 checkpoint

## Current read

The strongest shape so far is a two-pass lane:

1. `writer_v5` writes short plain-language summaries with concrete anchors.
2. `verifier_v3` checks faithfulness and basic usefulness.
3. `verifier_v4` audits the accepted rows for the stricter product goal: natural language over expert shorthand.

The stricter audit is important. The normal verifier allowed several faithful but overly technical rows, especially in math and theory-heavy categories.

## Outputs

- Main v5 run root: `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v5`
- Main v5 release: `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v5`
- Strict audit run root: `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v5_audit`
- Strict audit release: `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v5_audit`

## Current counts

- v5 candidates: `100`
- v5 reviewer-accepted rows: `36`
- v5 strict-audit accepted rows: `14`

The audit release is the cleaner seed pool for natural-language training. The unaudited v5 release is still useful for error analysis.

## Resume commands

Extend v5 to more source rows:

```bash
python3 scripts/summary_generation/run_pipeline.py \
  --config configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v5.json \
  --stage prepare \
  --limit 150

python3 scripts/summary_generation/run_pipeline.py \
  --config configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v5.json \
  --stage write \
  --no-route-probe \
  --timeout 120

python3 scripts/summary_generation/run_pipeline.py \
  --config configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v5.json \
  --stage local-filter \
  --no-route-probe

python3 scripts/summary_generation/run_pipeline.py \
  --config configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v5.json \
  --stage verify \
  --no-route-probe \
  --timeout 120

python3 scripts/summary_generation/run_pipeline.py \
  --config configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v5.json \
  --stage materialize \
  --no-route-probe
```

Refresh the strict audit after extending v5:

```bash
python3 scripts/summary_generation/run_pipeline.py \
  --config configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v5_audit.json \
  --stage verify \
  --no-route-probe \
  --timeout 120

python3 scripts/summary_generation/run_pipeline.py \
  --config configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v5_audit.json \
  --stage materialize \
  --no-route-probe

python3 scripts/summary_generation/run_pipeline.py \
  --config configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v5_audit.json \
  --stage report \
  --no-route-probe
```

## Examples from the strict audit release

- `Fluorescent sensors reveal how trastuzumab disrupts HER2 protein clusters.`
- `Five criteria help reviewers judge qualitative HCI studies without forcing quantification.`
- `Immersive VR games help students correct misconceptions about density.`
- `Inexpensive GPS spoofing tricks Wi-Fi 6E access points into causing harmful interference.`
- `Public announcements shifted mask-wearing in Vienna, but effects varied by message content.`

## Main remaining problem

For very technical papers, the writer can produce faithful but dense summaries. The strict audit currently drops those rows instead of forcing a weak natural-language rewrite. That is probably right for a small student model.
