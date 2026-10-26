#!/bin/bash
# Full distillation run: generate -> verify -> build_dataset, against the LMStudio gemma-4 teacher.
# Resumable: generate.py --resume skips arxiv_ids already in cand.jsonl.
set -euo pipefail
cd /Users/alanwren/myApps/model-experiments/gemma-lab-advanced
D=quant_probe/distill
NAME=gemma4xl_briefground_v1

echo "=== [$(date '+%F %T')] generate ==="
python3 $D/generate.py --papers $D/pool.json --shots 6 --temp 0.3 --resume --out $D/cand.jsonl

echo "=== [$(date '+%F %T')] verify ==="
python3 $D/verify.py --papers $D/pool.json --cand $D/cand.jsonl --out $D/verified.jsonl

echo "=== [$(date '+%F %T')] build_dataset ==="
python3 $D/build_dataset.py --papers $D/pool.json --verified $D/verified.jsonl --name $NAME

echo "=== [$(date '+%F %T')] DONE ==="
wc -l $D/cand.jsonl $D/verified.jsonl
