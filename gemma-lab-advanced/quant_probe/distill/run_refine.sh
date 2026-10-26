#!/bin/bash
# Refine the 5314 verified briefs to strong faithfulness, then rebuild the training dataset.
set -euo pipefail
cd "$(dirname "$0")/../.."   # lab root
PY=.venv/bin/python
D=quant_probe/distill

echo "=== [$(date '+%F %T')] refine ==="
$PY $D/refine.py --verified $D/verified.jsonl --papers $D/pool.json --resume --out $D/refined.jsonl

echo "=== [$(date '+%F %T')] build_dataset (faithful) ==="
$PY $D/build_dataset.py --papers $D/pool.json --verified $D/refined.jsonl --name gemma4xl_faithful_v1

echo "=== [$(date '+%F %T')] REFINE DONE ==="
python3 -c "import json,collections;c=collections.Counter(json.loads(l)['status'] for l in open('$D/refined.jsonl'));print('status',dict(c),'total',sum(c.values()))"
