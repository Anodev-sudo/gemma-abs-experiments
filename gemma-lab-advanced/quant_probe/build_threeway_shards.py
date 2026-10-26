#!/usr/bin/env python3
"""Build blind A/B/C judge shards for the three-way (q4qlora / q5recover / granite).

Per paper: the three model briefs are shuffled into slots A/B/C (independent per
paper), the slot->model mapping is kept in a SECRET key file the judges never see.
Judge input files carry only title+abstract+A/B/C briefs. Two passes share the same
shards (double-judging), so we build shards once.

  python build_threeway_shards.py
"""
import json, random
from pathlib import Path

OUT = Path("quant_probe/judge3way"); OUT.mkdir(parents=True, exist_ok=True)
SHARD = 30
rng = random.Random(20260614)

def load_jsonl(p):
    return {json.loads(l)["arxiv_id"]: json.loads(l)["brief"] for l in open(p)}

def build(set_name, papers_file, gens):
    papers = json.load(open(papers_file))
    briefs = {m: load_jsonl(f"quant_probe/eval_out/{f}.jsonl") for m, f in gens.items()}
    models = list(gens.keys())
    key = {}
    rows = []
    for p in papers:
        aid = p["arxiv_id"]
        if not all(aid in briefs[m] for m in models):
            continue
        slots = ["A", "B", "C"]
        order = models[:]
        rng.shuffle(order)
        slot_map = dict(zip(slots, order))            # A->model
        key[aid] = slot_map
        rows.append({"arxiv_id": aid,
                     "title": p.get("title", ""),
                     "abstract": p.get("abstract", ""),
                     **{s: briefs[slot_map[s]][aid] for s in slots}})
    # shard
    shards = [rows[i:i+SHARD] for i in range(0, len(rows), SHARD)]
    for n, sh in enumerate(shards, 1):
        json.dump({"set": set_name, "shard": n, "papers": sh},
                  open(OUT/f"{set_name}_shard{n:02d}.json", "w"), indent=0)
    print(f"[{set_name}] {len(rows)} papers -> {len(shards)} shards of <= {SHARD}")
    return key

keys = {}
keys["repr"] = build("repr", "quant_probe/eval_repr_240.json",
                     {"q4qlora": "repr_q4qlora", "q5recover": "repr_q5recover", "granite": "repr_granite"})
keys["stress"] = build("stress", "quant_probe/eval_stress_150.json",
                       {"q4qlora": "ev_df100_q4qlora", "q5recover": "ev_df100_recover_q5", "granite": "stress_granite"})
json.dump(keys, open(OUT/"key_SECRET.json", "w"), indent=0)
print("wrote secret key -> quant_probe/judge3way/key_SECRET.json")
