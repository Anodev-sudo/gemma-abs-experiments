# Natural-language teacher lane v1

Fresh teacher-output lane for plain-language mobile feed summaries.

## Scope

- Source pool: `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v1/source_candidates/`
- Run root: `hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v1/`
- Release target: `hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_v1/`
- Writer prompt: `prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_writer_v1.json`
- Verifier prompt: `prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_verifier_v1.json`
- Pipeline config: `configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v1.json`

## Target Shape

- One natural sentence.
- 8 to 14 words.
- Reason to open the paper, not complete abstract compression.
- Plain words unless a technical term is necessary.
- No title restatement, empty paper openings, jargon stacks, dense numbers, or formulas.
- Preserve polarity, value binding, uncertainty, and scope.

## Commands

Build a small source pool:

```bash
python3 scripts/build_natural_language_teacher_pool_v1.py --sample-size 200
```

Prepare and inspect request payloads without API spend:

```bash
python3 scripts/summary_generation/run_pipeline.py \
  --config configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v1.json \
  --stage all \
  --dry-run \
  --limit 5
```

Run only the writer for a tiny live smoke:

```bash
python3 scripts/summary_generation/run_pipeline.py \
  --config configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v1.json \
  --stage prepare \
  --limit 20

python3 scripts/summary_generation/run_pipeline.py \
  --config configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v1.json \
  --stage probe-routes

python3 scripts/summary_generation/run_pipeline.py \
  --config configs/synthetic_summary_generation/natural_language_teacher_deepseek_v4_pro_v1.json \
  --stage write \
  --limit 20
```

Then inspect:

```bash
sed -n '1,20p' \
  hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v1/writer_run/writer_outputs_ordered.jsonl
```
