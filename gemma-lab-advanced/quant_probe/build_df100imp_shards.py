#!/usr/bin/env python3
"""Blind 4-way shards for the DF100-improvement batch:
baseline-q5 / fullft-q5 / bf16-lossless / faithfulness-prompt. Dir judge_df100imp/."""
import json, random
from pathlib import Path
OUT = Path("quant_probe/judge_df100imp"); OUT.mkdir(parents=True, exist_ok=True)
SHARD=30; rng=random.Random(20260617); SLOTS=["A","B","C","D"]
def load(p): return {json.loads(l)["arxiv_id"]:json.loads(l)["brief"] for l in open(p)}
def build(setn, pf, gens):
    papers=json.load(open(pf)); briefs={m:load(f"quant_probe/eval_out/{f}.jsonl") for m,f in gens.items()}
    models=list(gens); key={}; rows=[]
    for p in papers:
        a=p["arxiv_id"]
        if not all(a in briefs[m] for m in models): continue
        order=models[:]; rng.shuffle(order); sm=dict(zip(SLOTS,order)); key[a]=sm
        rows.append({"arxiv_id":a,"title":p.get("title",""),"abstract":p.get("abstract",""),**{s:briefs[sm[s]][a] for s in SLOTS}})
    sh=[rows[i:i+SHARD] for i in range(0,len(rows),SHARD)]
    for n,s in enumerate(sh,1): json.dump({"set":setn,"shard":n,"papers":s},open(OUT/f"{setn}_shard{n:02d}.json","w"),indent=0)
    print(f"[{setn}] {len(rows)} papers -> {len(sh)} shards"); return key
keys={}
keys["repr"]=build("repr","quant_probe/eval_repr_240.json",
    {"baseline":"repr_q5recover","fullft":"repr_df100_fullft","bf16":"repr_df100_bf16","prompt":"repr_df100_prompt"})
keys["stress"]=build("stress","quant_probe/eval_stress_150.json",
    {"baseline":"ev_df100_recover_q5","fullft":"stress_df100_fullft","bf16":"stress_df100_bf16","prompt":"stress_df100_prompt"})
json.dump(keys,open(OUT/"key_SECRET.json","w"),indent=0); print("wrote key")
