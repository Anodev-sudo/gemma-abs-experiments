#!/usr/bin/env python3
"""Round-3 blind shards: 4-way A/B/C/D over
df50recover / df100recover / df25 / granite. Output dir judge4way_df50/."""
import json, random
from pathlib import Path

OUT = Path("quant_probe/judge4way_df50"); OUT.mkdir(parents=True, exist_ok=True)
SHARD = 30
rng = random.Random(20260616)
SLOTS = ["A", "B", "C", "D"]

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
        slot_map = dict(zip(SLOTS, order))
        key[aid] = slot_map
        rows.append({"arxiv_id": aid, "title": p.get("title", ""), "abstract": p.get("abstract", ""),
                     **{s: briefs[slot_map[s]][aid] for s in SLOTS}})
    shards = [rows[i:i+SHARD] for i in range(0, len(rows), SHARD)]
    for n, sh in enumerate(shards, 1):
        json.dump({"set": set_name, "shard": n, "papers": sh}, open(OUT/f"{set_name}_shard{n:02d}.json", "w"), indent=0)
    print(f"[{set_name}] {len(rows)} papers -> {len(shards)} shards")
    return key

keys = {}
keys["repr"] = build("repr", "quant_probe/eval_repr_240.json",
    {"df50": "repr_df50recover", "df100": "repr_q5recover", "df25": "repr_df25q5", "granite": "repr_granite"})
keys["stress"] = build("stress", "quant_probe/eval_stress_150.json",
    {"df50": "stress_df50recover", "df100": "ev_df100_recover_q5", "df25": "stress_df25q5", "granite": "stress_granite"})
json.dump(keys, open(OUT/"key_SECRET.json", "w"), indent=0)
print("wrote key -> quant_probe/judge4way_df50/key_SECRET.json")
