#!/bin/bash
# Stage-2 style voice drill on faithful grounding -> faithful_style_q5, then generate on eval sets.
set -euo pipefail
cd "$(dirname "$0")/.."   # lab root
PY=.venv/bin/python
BASE=quant_probe/stripped_base_df100
RUN=training/runs/gemma_v5_curriculum_df100_faithful_style_v1

echo "=== [$(date '+%F %T')] STAGE 2 style voice ==="
mkdir -p $RUN/stage2
$PY -m mlx_lm lora --config training/configs/faithful_style_df100_stage2.yaml 2>&1 | tee $RUN/stage2/train.log

echo "=== [$(date '+%F %T')] fuse + q5 ==="
rm -rf quant_probe/faithful_style_fused quant_probe/faithful_style_q5
$PY -m mlx_lm fuse --model $BASE --adapter-path $RUN/stage2/adapters --save-path quant_probe/faithful_style_fused
$PY -m mlx_lm convert --hf-path quant_probe/faithful_style_fused --mlx-path quant_probe/faithful_style_q5 -q --q-bits 5 --q-group-size 64
du -sh quant_probe/faithful_style_q5

echo "=== [$(date '+%F %T')] generate on eval sets ==="
$PY quant_probe/gen_threeway.py quant_probe/eval_repr_240.json   fs_repr   --model quant_probe/faithful_style_q5
$PY quant_probe/gen_threeway.py quant_probe/eval_stress_150.json fs_stress --model quant_probe/faithful_style_q5
echo "=== [$(date '+%F %T')] FS DONE ==="