#!/usr/bin/env python3
"""Fresh STYLE set (stage-2 voice) — Gemma-26B, prompt-only, hook-led but hard-gated for faithfulness.
Gates (cheapest first): length (<=136 chars & <=18 words) -> quality (>=2 of 3) -> faithfulness (PASS,
strict claim-decomposition). All gen/gating by the teacher; no faithfulness drift, no flat filler.

  python style.py --papers pool.json --out style.jsonl --n 9000 --target 2500 [--resume]
"""
import sys, json, re, random, argparse
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat

GOLD=("gold_pipeline/05_milestone_assembly/gold_set/mlx_lora/"
      "title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5/train.jsonl")
STYLE_SYS=(
"You are the best science writer on an arXiv reading app — your blurbs make people stop scrolling. Write ONE "
"feed blurb for the paper.\n"
"The #1 failure is being BORING. A blurb that merely names the topic or method has failed. Find the single most "
"surprising, counterintuitive, or consequential thing the abstract actually claims, and lead with THAT — the "
"result, the payoff, the 'wait, really?'.\n"
"Hard rules (breaking these makes it unusable):\n"
"- The hook must be a real, abstract-supported claim. Never invent, overstate, or invert to sound punchier — a "
"true plain line beats a catchy false one.\n"
"- ONE sentence; 12-17 words; never exceed 18 words or 136 characters.\n"
"- No LaTeX, equations, symbols, or dense numbers; no method/dataset/system proper-noun names; no jargon a smart "
"non-specialist wouldn't know; no filler ('This paper', 'We propose').\n"
"Write it so a curious outsider thinks 'I want to read that.' Output only the blurb.")
VSYS=("You are a meticulous fact-checker. Given an ABSTRACT and a one-sentence BLURB, check faithfulness. List each "
"factual claim; mark SUPPORTED/UNSUPPORTED vs the abstract; note if it misses the MAIN contribution. Final line "
"'VERDICT: PASS' or 'VERDICT: FAIL - <reason>'. Strict: invented specifics, overstatement, vague ungrounded claims, "
"inversion, or misreading the main point = FAIL.")
HSYS=("Rate a one-sentence paper blurb's quality as a feed item, 0-3:\n"
"3 = compelling AND specific; a curious reader clearly wants to open it.\n"
"2 = solid; conveys a concrete, real takeaway clearly (even if not flashy).\n"
"1 = weak; vague, generic, or mostly restates the topic/title.\n"
"0 = empty or garbled; says nothing specific.\n"
"Reply one line: 'SCORE: <0-3> - <short reason>'.")

def exemplars(k, seed):
    rows=[json.loads(l) for l in open(GOLD)]; random.Random(seed).shuffle(rows); ex=[]
    for r in rows[:k]:
        u=next(m["content"] for m in r["messages"] if m["role"]=="user"); a=next(m["content"] for m in r["messages"] if m["role"]=="assistant")
        m=re.search(r"Title:\n(.*?)\n\nAbstract:\n(.*)$",u,re.S); t,ab=(m.group(1),m.group(2)) if m else ("",u)
        ex.append((t.strip(), ab.strip()[:500], a.strip()))
    return ex
def gen(ex, title, ab):
    msgs=[{"role":"system","content":STYLE_SYS}]
    for t,a,b in ex: msgs+=[{"role":"user","content":f"Title: {t}\nAbstract: {a}"},{"role":"assistant","content":b}]
    msgs.append({"role":"user","content":f"Title: {title}\nAbstract: {ab}"})
    return chat(msgs, temp=0.55, max_tokens=56).strip().strip('"')
def quality(ab,b):
    o=chat([{"role":"system","content":HSYS},{"role":"user","content":f"ABSTRACT:\n{ab}\n\nBLURB: {b}"}],temp=0,max_tokens=40)
    m=re.search(r"SCORE:\s*([0-3])",o); return int(m.group(1)) if m else 0
def faith(ab,b):
    o=chat([{"role":"system","content":VSYS},{"role":"user","content":f"ABSTRACT:\n{ab}\n\nBLURB: {b}"}],temp=0,max_tokens=320)
    m=re.search(r"VERDICT:\s*(PASS|FAIL)",o,re.I); return bool(m) and m.group(1).upper()=="PASS"

ap=argparse.ArgumentParser()
ap.add_argument("--papers",required=True); ap.add_argument("--out",required=True)
ap.add_argument("--n",type=int,default=9000); ap.add_argument("--target",type=int,default=2500)
ap.add_argument("--seed",type=int,default=20260616); ap.add_argument("--resume",action="store_true")
a=ap.parse_args()
pool=json.load(open(a.papers)); random.Random(a.seed).shuffle(pool); pool=pool[:a.n]
ex=exemplars(8, 20260615)
done=set(); keep=0; mode="w"
if a.resume and Path(a.out).exists():
    for l in open(a.out): done.add(json.loads(l)["arxiv_id"]); keep+=1
    mode="a"
seen=0; rej=Counter()
with open(a.out, mode) as fh:
    for p in pool:
        if keep>=a.target: break
        aid=p["arxiv_id"]
        if aid in done: continue
        seen+=1
        b=gen(ex, p.get("title",""), p.get("abstract","")); ch=len(b); w=len(b.split())
        if not b or ch>136 or w>18 or w<6: rej["len"]+=1; continue
        if re.search(r'[\\${}^~]', b) or re.search(r'[A-Za-z0-9]_[A-Za-z0-9]', b): rej["latex"]+=1; continue
        q=quality(p["abstract"], b)
        if q<2: rej["quality"]+=1; continue
        if not faith(p["abstract"], b): rej["faith"]+=1; continue
        fh.write(json.dumps({"arxiv_id":aid,"brief":b,"words":w,"chars":ch,"quality":q})+"\n"); fh.flush()
        keep+=1
        if keep%100==0: print(f"  kept {keep}/{a.target} (seen {seen}, rej {dict(rej)})",flush=True)
print(f"[style] kept {keep} from {seen} seen | rejects {dict(rej)} -> {a.out}")
