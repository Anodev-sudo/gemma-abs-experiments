#!/usr/bin/env python3
"""Round-2 blind shards: df25-q5 / df100-q5 / granite. Same blind A/B/C method,
new output dir judge3way_df25/ so round-1 (df100 vs q4q vs granite) is untouched."""
import json, random
from pathlib import Path

OUT = Path("quant_probe/judge3way_df25"); OUT.mkdir(parents=True, exist_ok=True)
SHARD = 30
rng = random.Random(20260615)

def load_jsonl(p):
    return {json.loads(l)["arxiv_id"]: json.loads(l)["brief"] for l in open(p)}

def build(set_name, papers_file, gens):
    papers = json.load(open(papers_file))
    briefs = {m: load_jsonl(f"quant_probe/eval_out/{f}.jsonl") for m, f in gens.items()}
    models = list(gens.keys())
    key, rows = {}, []
    for p in papers:
        aid = p["arxiv_id"]
        if not all(aid in briefs[m] for m in models):
            continue
        order = models[:]; rng.shuffle(order)
        slot_map = dict(zip(["A", "B", "C"], order))
        key[aid] = slot_map
        rows.append({"arxiv_id": aid, "title": p.get("title", ""), "abstract": p.get("abstract", ""),
                     **{s: briefs[slot_map[s]][aid] for s in "ABC"}})
    shards = [rows[i:i+SHARD] for i in range(0, len(rows), SHARD)]
    for n, sh in enumerate(shards, 1):
        json.dump({"set": set_name, "shard": n, "papers": sh}, open(OUT/f"{set_name}_shard{n:02d}.json", "w"), indent=0)
    print(f"[{set_name}] {len(rows)} papers -> {len(shards)} shards")
    return key

keys = {}
keys["repr"] = build("repr", "quant_probe/eval_repr_240.json",
                     {"df25q5": "repr_df25q5", "df100q5": "repr_q5recover", "granite": "repr_granite"})
keys["stress"] = build("stress", "quant_probe/eval_stress_150.json",
                       {"df25q5": "stress_df25q5", "df100q5": "ev_df100_recover_q5", "granite": "stress_granite"})
json.dump(keys, open(OUT/"key_SECRET.json", "w"), indent=0)
print("wrote key -> quant_probe/judge3way_df25/key_SECRET.json")
