#!/usr/bin/env python3
"""Independent faithfulness pass by the SAME teacher (fresh context: sees only abstract+brief).
Keeps only briefs the teacher rates FAITHFUL. Heuristic pre-filter first (cheap rejects).

  python verify.py --papers pool.json --cand cand.jsonl --out verified.jsonl

Same teacher (gemma-4-26b-a4b-qat over HTTP), fresh context — see client.py.
"""
import sys, json, argparse, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat, MODEL

VSYS = (
 "You check a one-sentence blurb against a paper abstract for an editor. The blurb is FAITHFUL "
 "only if BOTH hold: (1) every claim it makes is directly supported by the abstract — no "
 "inversion, overstatement, or invented specifics; (2) it captures the paper's MAIN "
 "contribution, not a minor side point. Reply with exactly one line: 'FAITHFUL' or "
 "'UNFAITHFUL: <short reason>'."
)
FILLER = ("this paper", "we propose", "we present", "in this work", "this work", "this study")

def heuristic_ok(b):
    w = b.split()
    if not b or len(w) < 4 or len(w) > 16: return False
    if any(f in b.lower() for f in FILLER): return False
    return True

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=MODEL)
ap.add_argument("--papers", required=True)
ap.add_argument("--cand", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

papers = {p["arxiv_id"]: p for p in json.load(open(a.papers))}
cands = [json.loads(l) for l in open(a.cand)]
kept = pre = unf = 0
seen = set()
with open(a.out, "w") as fh:
    for c in cands:
        b = c["brief"].strip(); aid = c["arxiv_id"]
        if not heuristic_ok(b) or b.lower() in seen or aid not in papers:
            pre += 1; continue
        ab = papers[aid].get("abstract", "")
        msgs = [{"role": "system", "content": VSYS},
                {"role": "user", "content": f"Abstract:\n{ab}\n\nBlurb: {b}\n\nVerdict:"}]
        verdict = chat(msgs, temp=0.0, max_tokens=40, model=a.model)
        if re.match(r"\s*FAITHFUL\b", verdict, re.I):
            seen.add(b.lower()); kept += 1
            fh.write(json.dumps({"arxiv_id": aid, "brief": b}) + "\n")
        else:
            unf += 1
        if (kept + unf) % 250 == 0: print(f"  verified {kept+unf}, kept {kept}", flush=True)
print(f"[verify] kept {kept} | heuristic-rejected {pre} | unfaithful {unf} -> {a.out}")
