import json, random
from pathlib import Path
OUT=Path("quant_probe/judge_r6"); OUT.mkdir(parents=True,exist_ok=True)
SHARD=30; rng=random.Random(20260615)
def load(p): return {json.loads(l)["arxiv_id"]:json.loads(l)["brief"] for l in open(p)}
def build(setn,pf,gens):
    papers=json.load(open(pf)); briefs={m:load(f"quant_probe/eval_out/{f}.jsonl") for m,f in gens.items()}; models=list(gens)
    key={};rows=[]
    for p in papers:
        a=p["arxiv_id"]
        if not all(a in briefs[m] for m in models): continue
        o=models[:]; rng.shuffle(o); sm=dict(zip("ABC",o)); key[a]=sm
        rows.append({"arxiv_id":a,"title":p.get("title",""),"abstract":p.get("abstract",""),**{s:briefs[sm[s]][a] for s in "ABC"}})
    sh=[rows[i:i+SHARD] for i in range(0,len(rows),SHARD)]
    for n,s in enumerate(sh,1): json.dump({"set":setn,"shard":n,"papers":s},open(OUT/f"{setn}_shard{n:02d}.json","w"),indent=0)
    print(f"[{setn}] {len(rows)} -> {len(sh)} shards"); return key
keys={}
keys["repr"]=build("repr","quant_probe/eval_repr_240.json",{"gemma4xl":"r6_gemma4xl_repr","baseline":"r6_baseline_repr","granite":"r6_granite_repr"})
keys["stress"]=build("stress","quant_probe/eval_stress_150.json",{"gemma4xl":"r6_gemma4xl_stress","baseline":"r6_baseline_stress","granite":"r6_granite_stress"})
json.dump(keys,open(OUT/"key_SECRET.json","w"),indent=0); print("key written; SHARD=%d"%SHARD)
