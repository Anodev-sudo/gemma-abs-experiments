import json, random
from pathlib import Path
OUT=Path("quant_probe/judge_testfaith"); (OUT/"scores").mkdir(parents=True,exist_ok=True)
SHARD=30; rng=random.Random(20260616)
def load(p): return {json.loads(l)["arxiv_id"]:json.loads(l)["brief"] for l in open(p)}
gens={"faithonly":"tf_faithonly","faithvoice":"tf_faithvoice","baseline":"tf_baseline"}
papers=json.load(open("quant_probe/eval_testset_faith.json"))
briefs={m:load(f"quant_probe/eval_out/{f}.jsonl") for m,f in gens.items()}; models=list(gens)
key={}; rows=[]
for p in papers:
    a=p["arxiv_id"]
    if not all(a in briefs[m] for m in models): continue
    o=models[:]; rng.shuffle(o); sm=dict(zip("ABC",o)); key[a]=sm
    rows.append({"arxiv_id":a,"abstract":p["abstract"],**{s:briefs[sm[s]][a] for s in "ABC"}})
sh=[rows[i:i+SHARD] for i in range(0,len(rows),SHARD)]
for n,s in enumerate(sh,1): json.dump({"shard":n,"papers":s},open(OUT/f"shard{n:02d}.json","w"),indent=0)
json.dump(key,open(OUT/"key_SECRET.json","w"))
print(f"{len(rows)} papers -> {len(sh)} shards")
