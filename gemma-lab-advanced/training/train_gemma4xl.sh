#!/bin/bash
# Two-stage curriculum on the gemma-4-26b brief-grounding data:
#   stage1 = brief-OBJECTIVE grounding (5314 verified briefs) on stripped_base_df100
#   stage2 = voice drill on 894 gold (resumes stage1 adapter)
set -euo pipefail
cd /Users/alanwren/myApps/model-experiments/gemma-lab-advanced
PY=.venv/bin/python
RUN=training/runs/gemma_v5_curriculum_df100_gemma4xl_v1

echo "=== [$(date '+%F %T')] STAGE 1 grounding ==="
mkdir -p $RUN/stage1
$PY -m mlx_lm lora --config training/configs/gemma4xl_df100_stage1.yaml 2>&1 | tee $RUN/stage1/train.log

echo "=== [$(date '+%F %T')] STAGE 2 voice ==="
mkdir -p $RUN/stage2
$PY -m mlx_lm lora --config training/configs/gemma4xl_df100_stage2.yaml 2>&1 | tee $RUN/stage2/train.log

echo "=== [$(date '+%F %T')] TRAIN DONE ==="
tail -3 $RUN/stage2/train.log
