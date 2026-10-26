#!/usr/bin/env python3
"""License-clean gemma-verifier PASS-rate cross-check over the 60-paper faith testset.
Scores faith_only vs full-LR style vs low-LR style. No Claude in the loop."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill.client import chat

VSYS=("You are a meticulous fact-checker. Given an ABSTRACT and a one-sentence BLURB, check the blurb for "
"faithfulness. First, on separate lines, list each distinct factual claim the blurb makes; after each claim "
"write SUPPORTED (quote the abstract phrase) or UNSUPPORTED. Also state whether the blurb captures the paper's "
"MAIN contribution or emphasizes a side point. Then a final line 'VERDICT: PASS' (every claim SUPPORTED and "
"main contribution captured) or 'VERDICT: FAIL - <reason>'. Be strict: invented specifics, overstatement, "
"vague claims not grounded in the abstract, inversion, or a misread of the main point all mean FAIL.")
import re
def verify(ab,b):
    out=chat([{"role":"system","content":VSYS},{"role":"user","content":f"ABSTRACT:\n{ab}\n\nBLURB: {b}"}],
             temp=0.0, max_tokens=512)
    m=re.search(r"VERDICT:\s*(PASS|FAIL)", out, re.I); return (m.group(1).upper() if m else "FAIL")

papers={p["arxiv_id"]:p for p in json.load(open("quant_probe/eval_testset_faith.json"))}
MODELS={"faith_only":"tf_faithonly","faith_style_fullLR":"tf_faithstyle","faith_style_lowLR":"tf_faithstyle_lowlr"}
def load(f): return {json.loads(l)["arxiv_id"]:json.loads(l)["brief"] for l in open(f"quant_probe/eval_out/{f}.jsonl")}
gens={m:load(f) for m,f in MODELS.items()}
ids=[a for a in papers if all(a in gens[m] for m in MODELS)]
print(f"scoring {len(ids)} papers x {len(MODELS)} models")
res={m:[] for m in MODELS}
for i,a in enumerate(ids):
    ab=papers[a]["abstract"]
    for m in MODELS:
        res[m].append(verify(ab, gens[m][a]))
    if (i+1)%15==0: print(f"  {i+1}/{len(ids)}")
print("\n==== gemma-verifier PASS rates (60-paper faith testset) ====")
for m in MODELS:
    p=sum(1 for v in res[m] if v=="PASS"); n=len(res[m])
    print(f"  {m:<20} {p}/{n}  {100*p/n:.1f}%")
json.dump({m:res[m] for m in MODELS}, open("quant_probe/xcheck_lowlr_verdicts.json","w"))
