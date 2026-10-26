#!/bin/bash
# Two ablation variants from one chain on the faithful data:
#   A faithful-only  = fuse stage-1 grounding adapter -> q5
#   B faithful->style = train stage-2 voice on top, fuse stage-2 adapter -> q5
set -euo pipefail
cd /Users/alanwren/myApps/model-experiments/gemma-lab-advanced
PY=.venv/bin/python
BASE=quant_probe/stripped_base_df100
RUN=training/runs/gemma_v5_curriculum_df100_faithful_v1

q5 () { # $1 adapter dir, $2 out name
  local fused="quant_probe/$2_fused" q5="quant_probe/$2_q5"
  rm -rf "$fused" "$q5"
  $PY -m mlx_lm fuse --model $BASE --adapter-path "$1" --save-path "$fused"
  $PY -m mlx_lm convert --hf-path "$fused" --mlx-path "$q5" -q --q-bits 5 --q-group-size 64
  du -sh "$q5"
}

echo "=== [$(date '+%F %T')] STAGE 1 (faithful grounding) ==="
mkdir -p $RUN/stage1
$PY -m mlx_lm lora --config training/configs/faithful_df100_stage1.yaml 2>&1 | tee $RUN/stage1/train.log

echo "=== [$(date '+%F %T')] build A: faithful_only_q5 (stage-1 only) ==="
q5 $RUN/stage1/adapters faithful_only

echo "=== [$(date '+%F %T')] STAGE 2 (voice on top) ==="
mkdir -p $RUN/stage2
$PY -m mlx_lm lora --config training/configs/faithful_df100_stage2.yaml 2>&1 | tee $RUN/stage2/train.log

echo "=== [$(date '+%F %T')] build B: faithful_voice_q5 (stage-2) ==="
q5 $RUN/stage2/adapters faithful_voice

echo "=== [$(date '+%F %T')] TRAIN+BUILD DONE ==="
