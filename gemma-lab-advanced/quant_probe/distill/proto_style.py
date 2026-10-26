import json, re, random, sys
sys.path.insert(0, "quant_probe/distill")
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
rows=[json.loads(l) for l in open(GOLD)]; random.Random(20260615).shuffle(rows)
ex=[]
for r in rows[:8]:
    u=next(m["content"] for m in r["messages"] if m["role"]=="user"); a=next(m["content"] for m in r["messages"] if m["role"]=="assistant")
    m=re.search(r"Title:\n(.*?)\n\nAbstract:\n(.*)$",u,re.S); t,ab=(m.group(1),m.group(2)) if m else ("",u)
    ex.append((t.strip(),ab.strip()[:500],a.strip()))
def gen(title,ab):
    msgs=[{"role":"system","content":STYLE_SYS}]
    for t,a,b in ex: msgs+=[{"role":"user","content":f"Title: {t}\nAbstract: {a}"},{"role":"assistant","content":b}]
    msgs.append({"role":"user","content":f"Title: {title}\nAbstract: {ab}"})
    return chat(msgs,temp=0.55,max_tokens=56).strip().strip('"')
def faith(ab,b):
    o=chat([{"role":"system","content":VSYS},{"role":"user","content":f"ABSTRACT:\n{ab}\n\nBLURB: {b}"}],temp=0,max_tokens=320)
    m=re.search(r"VERDICT:\s*(PASS|FAIL)",o,re.I); return m.group(1).upper() if m else "FAIL"
def hook(ab,b):
    o=chat([{"role":"system","content":HSYS},{"role":"user","content":f"ABSTRACT:\n{ab}\n\nBLURB: {b}"}],temp=0,max_tokens=40)
    m=re.search(r"SCORE:\s*([0-3])",o); return int(m.group(1)) if m else 0
pool=json.load(open("quant_probe/distill/pool.json")); random.Random(99).shuffle(pool)
from collections import Counter
qd=Counter(); pf=keep=0
for p_ in pool[:20]:
    b=gen(p_["title"],p_["abstract"]); ch=len(b); w=len(b.split())
    f=faith(p_["abstract"],b); q=hook(p_["abstract"],b); qd[q]+=1
    pf+=f=="PASS"
    k = (f=="PASS" and q>=2 and ch<=136 and w<=18); keep+=k
    flag="" if (ch<=136 and w<=18) else " <<OVER"
    print(f"[{f} q{q}{' KEEP' if k else '     '}] {w}w {ch}c{flag}  {b}", flush=True)
print(f"\nPASS {pf}/20  quality dist {dict(sorted(qd.items()))}  KEEP(PASS&q>=2&len) {keep}/20", flush=True)
