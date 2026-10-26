#!/bin/bash
# Fuse the gemma4xl curriculum stage2 adapter into the stripped DF100 base, then q5.
# Mirrors the recovered_df100_q5 recipe (LoRA fuse carries the compact tokenizer).
set -euo pipefail
cd /Users/alanwren/myApps/model-experiments/gemma-lab-advanced
PY=.venv/bin/python
BASE=quant_probe/stripped_base_df100
ADAPT=training/runs/gemma_v5_curriculum_df100_gemma4xl_v1/stage2/adapters
FUSED=quant_probe/gemma4xl_df100_fused
Q5=quant_probe/gemma4xl_df100_q5

echo "=== [$(date '+%F %T')] fuse ==="
rm -rf $FUSED $Q5
$PY -m mlx_lm fuse --model $BASE --adapter-path $ADAPT --save-path $FUSED

echo "=== [$(date '+%F %T')] convert q5 g64 ==="
$PY -m mlx_lm convert --hf-path $FUSED --mlx-path $Q5 -q --q-bits 5 --q-group-size 64

echo "=== [$(date '+%F %T')] BUILD DONE ==="
du -sh $Q5
$PY -c "
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('$Q5'); print('vocab', len(t))
b=AutoTokenizer.from_pretrained('$BASE'); print('base vocab', len(b))
"
ls -lh $Q5/model.safetensors