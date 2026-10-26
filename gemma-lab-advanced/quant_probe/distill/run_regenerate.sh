#!/bin/bash
# Regenerate ALL 5314 targets faithfulness-first (v2 prompt) + verify gate, then rebuild dataset.
set -euo pipefail
cd "$(dirname "$0")/../.."   # lab root
PY=.venv/bin/python
D=quant_probe/distill

echo "=== [$(date '+%F %T')] regenerate (faithfulness-first, all rows) ==="
$PY $D/regenerate.py --verified $D/verified.jsonl --papers $D/pool.json --resume --out $D/regenerated.jsonl

echo "=== [$(date '+%F %T')] build_dataset (faithful) ==="
$PY $D/build_dataset.py --papers $D/pool.json --verified $D/regenerated.jsonl --name gemma4xl_faithful_v1

echo "=== [$(date '+%F %T')] REGEN DONE ==="
python3 -c "import json,collections;c=collections.Counter(json.loads(l)['status'] for l in open('$D/regenerated.jsonl'));print('status',dict(c),'total',sum(c.values()))"
