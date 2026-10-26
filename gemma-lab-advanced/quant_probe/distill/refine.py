#!/usr/bin/env python3
"""Faithfulness refinement of the verified briefs (Gemma-26B teacher, all data ops by the teacher).
For each (paper, brief): strict claim-decomposition verify; if it fails, repair (faithfulness-first,
plain > catchy) up to 2x and re-verify; drop if still unfaithful. Output = strongly-faithful targets.

  python refine.py --verified verified.jsonl --papers pool.json --out refined.jsonl [--resume]
"""
import sys, json, argparse, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat

VSYS=("You are a meticulous fact-checker. Given an ABSTRACT and a one-sentence BLURB, check the blurb for "
"faithfulness. First, on separate lines, list each distinct factual claim the blurb makes; after each claim "
"write SUPPORTED (quote the abstract phrase) or UNSUPPORTED. Also state whether the blurb captures the paper's "
"MAIN contribution or emphasizes a side point. Then a final line 'VERDICT: PASS' (every claim SUPPORTED and "
"main contribution captured) or 'VERDICT: FAIL - <reason>'. Be strict: invented specifics, overstatement, "
"vague claims not grounded in the abstract, inversion, or a misread of the main point all mean FAIL.")
RSYS=("You rewrite a one-sentence blurb for an arXiv paper so it is STRICTLY FAITHFUL to the abstract. "
"State the paper's MAIN contribution; every word must be supported by the abstract; do NOT invent specifics, "
"overstate, or add an ungrounded hook. A plain accurate sentence is far better than a catchy wrong one. "
"14 words or fewer. Do NOT begin with 'This paper', 'This review', 'This study', 'We', or similar filler. "
"Output only the sentence.")

def verify(ab,b):
    out=chat([{"role":"system","content":VSYS},{"role":"user","content":f"ABSTRACT:\n{ab}\n\nBLURB: {b}"}],
             temp=0.0,max_tokens=320)
    m=re.search(r"VERDICT:\s*(PASS|FAIL)", out, re.I)
    return (m.group(1).upper() if m else "FAIL")
def repair(ab,b):
    return chat([{"role":"system","content":RSYS},
                 {"role":"user","content":f"ABSTRACT:\n{ab}\n\nCurrent blurb (may be inaccurate): {b}\n\nFaithful blurb:"}],
                temp=0.2,max_tokens=48).strip().strip('"')

ap=argparse.ArgumentParser()
ap.add_argument("--verified",required=True); ap.add_argument("--papers",required=True)
ap.add_argument("--out",required=True); ap.add_argument("--resume",action="store_true")
a=ap.parse_args()
papers={p["arxiv_id"]:p for p in json.load(open(a.papers))}
items=[json.loads(l) for l in open(a.verified)]
done=set(); mode="w"
if a.resume and Path(a.out).exists():
    for l in open(a.out): done.add(json.loads(l)["arxiv_id"])
    mode="a"
kept=repaired=dropped=0
with open(a.out,mode) as fh:
    for it in items:
        aid=it["arxiv_id"]
        if aid in done or aid not in papers: continue
        ab=papers[aid]["abstract"]; b=it["brief"].strip()
        if verify(ab,b)=="PASS":
            status="kept"; final=b; kept+=1
        else:
            final=None
            for _ in range(2):
                rb=repair(ab,b)
                if rb and 4<=len(rb.split())<=16 and verify(ab,rb)=="PASS":
                    final=rb; break
                b=rb or b
            if final: status="repaired"; repaired+=1
            else: status="dropped"; dropped+=1
        if final:
            fh.write(json.dumps({"arxiv_id":aid,"brief":final,"status":status})+"\n"); fh.flush()
        if (kept+repaired+dropped)%200==0:
            print(f"  {kept+repaired+dropped}/{len(items)}  kept={kept} repaired={repaired} dropped={dropped}",flush=True)
print(f"[refine] kept={kept} repaired={repaired} dropped={dropped} -> {a.out}")
