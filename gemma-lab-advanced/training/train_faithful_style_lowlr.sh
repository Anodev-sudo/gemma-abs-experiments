#!/bin/bash
# Low-LR (1.5e-6) style voice drill on faithful grounding -> faithful_style_lowlr_q5.
# Eval cheaply on the 60-paper held-out faith testset for a 3-way vs faith_only & full-LR style.
set -euo pipefail
cd "$(dirname "$0")/.."   # lab root
PY=.venv/bin/python
BASE=quant_probe/stripped_base_df100
RUN=training/runs/gemma_v5_curriculum_df100_faithful_style_lowlr_v1
TESTSET=quant_probe/eval_testset_faith.json

echo "=== [$(date '+%F %T')] STAGE 2 low-LR style voice (1.5e-6) ==="
mkdir -p $RUN/stage2
$PY -u -m mlx_lm lora --config training/configs/faithful_style_lowlr_df100_stage2.yaml 2>&1 | tee $RUN/stage2/train.log

echo "=== [$(date '+%F %T')] fuse + q5 ==="
rm -rf quant_probe/faithful_style_lowlr_fused quant_probe/faithful_style_lowlr_q5
$PY -m mlx_lm fuse --model $BASE --adapter-path $RUN/stage2/adapters --save-path quant_probe/faithful_style_lowlr_fused
$PY -m mlx_lm convert --hf-path quant_probe/faithful_style_lowlr_fused --mlx-path quant_probe/faithful_style_lowlr_q5 -q --q-bits 5 --q-group-size 64
du -sh quant_probe/faithful_style_lowlr_q5

echo "=== [$(date '+%F %T')] generate on 60-paper cheap testset ==="
$PY quant_probe/gen_threeway.py $TESTSET tf_faithstyle_lowlr --model quant_probe/faithful_style_lowlr_q5
# full-LR style on the SAME 60 papers, so faith_only / full-LR / low-LR are apples-to-apples
$PY quant_probe/gen_threeway.py $TESTSET tf_faithstyle      --model quant_probe/faithful_style_q5
echo "=== [$(date '+%F %T')] LOWLR DONE ==="
