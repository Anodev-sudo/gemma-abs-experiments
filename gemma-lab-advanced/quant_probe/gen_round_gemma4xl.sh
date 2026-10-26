#!/bin/bash
# Generate briefs for the gemma4xl-vs-baseline-vs-granite verdict, all 3 models x 2 eval sets,
# identical BRIEF prompt + temp 0. Clean labels: r6_{model}_{set}.
set -euo pipefail
cd "$(dirname "$0")/.."   # lab root
PY=.venv/bin/python
NEW=quant_probe/gemma4xl_df100_q5
BASE=quant_probe/recovered_df100_q5
GRAN=${GRAN:-../hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_ckpt425_mlx_swift_q8_export_v1/fused_q8_g64}   # Granite export in the parent repo (data not in git)

gen () { $PY quant_probe/gen_threeway.py "$1" "$2" --model "$3"; }

echo "=== [$(date '+%F %T')] generate 6 ==="
gen quant_probe/eval_repr_240.json   r6_gemma4xl_repr   "$NEW"
gen quant_probe/eval_stress_150.json r6_gemma4xl_stress "$NEW"
gen quant_probe/eval_repr_240.json   r6_baseline_repr   "$BASE"
gen quant_probe/eval_stress_150.json r6_baseline_stress "$BASE"
gen quant_probe/eval_repr_240.json   r6_granite_repr    "$GRAN"
gen quant_probe/eval_stress_150.json r6_granite_stress  "$GRAN"
echo "=== [$(date '+%F %T')] GEN DONE ==="
cd quant_probe/eval_out && wc -l r6_*.jsonl