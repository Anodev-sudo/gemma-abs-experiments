#!/usr/bin/env python3
"""STRONG manipulation: regenerate EVERY stage-1 target faithfulness-first (v2 prompt: faithful +
feed-rule-compliant), then a strict verify gate. Keeps passes, repairs stragglers once, drops the
unsalvageable. All ops by the Gemma-26B teacher.

  python regenerate.py --verified verified.jsonl --papers pool.json --out regenerated.jsonl [--resume]
"""
import sys, json, argparse, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat

# faithfulness-first rewrite, with the feed guardrails restored (no names/LaTeX/jargon, verb-led)
RSYS=("You rewrite a one-sentence feed blurb for an arXiv paper so it is STRICTLY FAITHFUL to the abstract, "
"while staying readable for a NON-SPECIALIST. Rules: (1) state the paper's MAIN contribution; every claim must "
"be supported by the abstract — no invented specifics, overstatement, inversion, or ungrounded hook; (2) a plain "
"accurate sentence beats a catchy wrong one; (3) write ONE complete sentence with a verb, 14 words or fewer; "
"(4) do NOT name the system/method/dataset (no proper-noun acronyms like 'MAVIS'); (5) NO mathematical symbols, "
"equations, LaTeX, or dense numbers; (6) avoid jargon and filler ('This paper', 'We'). Output only the sentence.")
# strict claim-decomposition verifier (visible reasoning, reasoning_effort=none)
VSYS=("You are a meticulous fact-checker. Given an ABSTRACT and a one-sentence BLURB, check the blurb for "
"faithfulness. First, on separate lines, list each distinct factual claim the blurb makes; after each claim "
"write SUPPORTED (quote the abstract phrase) or UNSUPPORTED. Also state whether the blurb captures the paper's "
"MAIN contribution or emphasizes a side point. Then a final line 'VERDICT: PASS' (every claim SUPPORTED and "
"main contribution captured) or 'VERDICT: FAIL - <reason>'. Be strict: invented specifics, overstatement, "
"vague claims not grounded in the abstract, inversion, or a misread of the main point all mean FAIL.")

def rewrite(ab,b):
    return chat([{"role":"system","content":RSYS},
        {"role":"user","content":f"ABSTRACT:\n{ab}\n\nCurrent blurb (may be inaccurate): {b}\n\nFaithful readable blurb:"}],
        temp=0.2,max_tokens=48).strip().strip('"')
def verify(ab,b):
    out=chat([{"role":"system","content":VSYS},{"role":"user","content":f"ABSTRACT:\n{ab}\n\nBLURB: {b}"}],
             temp=0.0,max_tokens=320)
    m=re.search(r"VERDICT:\s*(PASS|FAIL)", out, re.I); return (m.group(1).upper() if m else "FAIL")
def ok_len(b): return bool(b) and 4<=len(b.split())<=16

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
regen=repaired=dropped=0
with open(a.out,mode) as fh:
    for it in items:
        aid=it["arxiv_id"]
        if aid in done or aid not in papers: continue
        ab=papers[aid]["abstract"]; orig=it["brief"].strip()
        cand=rewrite(ab,orig)
        if ok_len(cand) and verify(ab,cand)=="PASS":
            final,status=cand,"regen"; regen+=1
        else:
            cand2=rewrite(ab, cand if ok_len(cand) else orig)
            if ok_len(cand2) and verify(ab,cand2)=="PASS":
                final,status=cand2,"regen_repaired"; repaired+=1
            else:
                final=None; dropped+=1
        if final:
            fh.write(json.dumps({"arxiv_id":aid,"brief":final,"status":status,"orig":orig})+"\n"); fh.flush()
        n=regen+repaired+dropped
        if n%200==0: print(f"  {n}/{len(items)}  regen={regen} repaired={repaired} dropped={dropped}",flush=True)
print(f"[regenerate] regen={regen} repaired={repaired} dropped={dropped} -> {a.out}")
