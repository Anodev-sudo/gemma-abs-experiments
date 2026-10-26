import json, random
from pathlib import Path
OUT=Path("quant_probe/judge_styleeval"); (OUT/"scores").mkdir(parents=True,exist_ok=True)
SHARD=30; rng=random.Random(20260616)
def load(p): return {json.loads(l)["arxiv_id"]:json.loads(l)["brief"] for l in open(p)}
def build(setn,pf,gens):
    papers=json.load(open(pf)); briefs={m:load(f"quant_probe/eval_out/{f}.jsonl") for m,f in gens.items()}; models=list(gens)
    key={};rows=[]
    for p in papers:
        a=p["arxiv_id"]
        if not all(a in briefs[m] for m in models): continue
        o=models[:]; rng.shuffle(o); sm=dict(zip("ABC",o)); key[a]=sm
        rows.append({"arxiv_id":a,"abstract":p.get("abstract",""),**{s:briefs[sm[s]][a] for s in "ABC"}})
    sh=[rows[i:i+SHARD] for i in range(0,len(rows),SHARD)]
    for n,s in enumerate(sh,1): json.dump({"set":setn,"shard":n,"papers":s},open(OUT/f"{setn}_shard{n:02d}.json","w"),indent=0)
    print(f"[{setn}] {len(rows)} -> {len(sh)} shards"); return key
keys={}
keys["repr"]=build("repr","quant_probe/eval_repr_240.json",{"faith_only":"fa_A_repr","faith_style":"fs_repr","candidate":"r6_baseline_repr"})
keys["stress"]=build("stress","quant_probe/eval_stress_150.json",{"faith_only":"fa_A_stress","faith_style":"fs_stress","candidate":"r6_baseline_stress"})
json.dump(keys,open(OUT/"key_SECRET.json","w")); print("key written")
