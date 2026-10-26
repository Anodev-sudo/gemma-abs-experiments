#!/bin/bash
# Resilient driver for the cond_hook relabel: survives teacher (LMStudio) outages.
# Relaunches relabel_cond_hook.py --resume until all 5182 rows are done. If the teacher
# endpoint is down, it WAITS and re-probes rather than burning failed launches. Safe to
# start while the teacher is offline -- it will auto-resume the moment it comes back.
set -uo pipefail
cd /Users/alanwren/myApps/model-experiments/gemma-lab-advanced
PY=.venv/bin/python
D=quant_probe/distill
OUT=$D/cond_hook/relabeled.jsonl
LOG=$D/cond_hook/relabel_full.log
TARGET=5182
URL=http://127.0.0.1:1234/v1/models

done_rows() { wc -l < "$OUT" 2>/dev/null | tr -d ' '; }

while [ "$(done_rows)" -lt "$TARGET" ]; do
  if curl -s -m 8 "$URL" >/dev/null 2>&1; then
    echo "=== [$(date '+%F %T')] teacher up; resuming at $(done_rows)/$TARGET ===" >> "$LOG"
    $PY $D/relabel_cond_hook.py --regenerated $D/regenerated.jsonl --papers $D/pool.json \
        --out "$OUT" --resume >> "$LOG" 2>&1 || true
    echo "=== [$(date '+%F %T')] relabel exited at $(done_rows)/$TARGET ===" >> "$LOG"
    sleep 5
  else
    echo "=== [$(date '+%F %T')] teacher unreachable; waiting 30s ($(done_rows)/$TARGET done) ===" >> "$LOG"
    sleep 30
  fi
done
echo "=== [$(date '+%F %T')] RELABEL COMPLETE: $(done_rows)/$TARGET ===" >> "$LOG"
