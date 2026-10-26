import json, random
from pathlib import Path
OUT=Path("quant_probe/audit_faith"); (OUT/"scores").mkdir(parents=True,exist_ok=True)
rng=random.Random(20260616); SHARD=30; N=90
pool={p["arxiv_id"]:p for p in json.load(open("quant_probe/distill/pool.json"))}
old={json.loads(l)["arxiv_id"]:json.loads(l)["brief"] for l in open("quant_probe/distill/verified.jsonl")}
new={json.loads(l)["arxiv_id"]:json.loads(l)["brief"] for l in open("quant_probe/distill/regenerated.jsonl")}
ids=[a for a in new if a in old and a in pool]
rng.shuffle(ids); ids=ids[:N]
key={}; rows=[]
for a in ids:
    o=["hooky","faithful"]; rng.shuffle(o); sm=dict(zip("XY",o)); key[a]=sm
    b={"hooky":old[a],"faithful":new[a]}
    rows.append({"arxiv_id":a,"abstract":pool[a]["abstract"],"X":b[sm["X"]],"Y":b[sm["Y"]]})
sh=[rows[i:i+SHARD] for i in range(0,len(rows),SHARD)]
for n,s in enumerate(sh,1): json.dump({"shard":n,"papers":s},open(OUT/f"shard{n:02d}.json","w"),indent=0)
json.dump(key,open(OUT/"key_SECRET.json","w"))
print(f"{len(rows)} papers -> {len(sh)} shards")
