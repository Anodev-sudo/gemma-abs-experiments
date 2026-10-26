#!/bin/bash
# Head-to-head: faithful_only_q5 (97MB gemma) vs Granite 4.0 350m q8 (374MB), SAME RUNTIME.
# Both via gen_threeway.py: identical BRIEF prompt, temp 0, max 40 tok, MLX load+generate.
# Regenerated fresh in one pass so the prompt version is provably identical.
set -euo pipefail
cd /Users/alanwren/myApps/model-experiments/gemma-lab-advanced
PY=.venv/bin/python
FAITH=quant_probe/faithful_only_q5
GRAN=$HOME/myApps/model-experiments/gemma-abstract-finetuning/hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_ckpt425_mlx_swift_q8_export_v1/fused_q8_g64

gen () { $PY quant_probe/gen_threeway.py "$1" "$2" --model "$3"; }

echo "=== [$(date '+%F %T')] gen faithful (97MB q5) ==="
gen quant_probe/eval_repr_240.json   hg_faith_repr     "$FAITH"
gen quant_probe/eval_stress_150.json hg_faith_stress   "$FAITH"
echo "=== [$(date '+%F %T')] gen granite (374MB q8) ==="
gen quant_probe/eval_repr_240.json   hg_granite_repr   "$GRAN"
gen quant_probe/eval_stress_150.json hg_granite_stress "$GRAN"
echo "=== [$(date '+%F %T')] GEN DONE ==="
cd quant_probe/eval_out && wc -l hg_*.jsonl
