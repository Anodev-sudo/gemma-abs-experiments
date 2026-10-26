#!/usr/bin/env python3
"""Blind 3-way judge shards: cond_hook_q5 vs faithful_only_q5 vs Granite q8.
Reuses the existing hg_faith_* / hg_granite_* eval generations; adds hg_styleonly_*.
A/B/C slots are shuffled per-paper; the model<->slot map is written to a SECRET key the
judges never see. Mirrors build_hg_shards.py but 3-way (A/B/C), SHARD=20 (60 briefs/shard)."""
import json, random
from pathlib import Path

OUT = Path("quant_probe/judge3way_styleonly"); (OUT / "scores").mkdir(parents=True, exist_ok=True)
SHARD = 20
rng = random.Random(20260618)


def load(label):
    return {json.loads(l)["arxiv_id"]: json.loads(l)["brief"] for l in open(f"quant_probe/eval_out/{label}.jsonl")}


def build(setn, pf, gens):
    papers = json.load(open(pf))
    briefs = {m: load(f) for m, f in gens.items()}
    models = list(gens)
    key = {}; rows = []
    for p in papers:
        a = p["arxiv_id"]
        if not all(a in briefs[m] for m in models):
            continue
        o = models[:]; rng.shuffle(o); sm = dict(zip("ABC", o)); key[a] = sm
        rows.append({"arxiv_id": a, "abstract": p.get("abstract", ""),
                     **{s: briefs[sm[s]][a] for s in "ABC"}})
    sh = [rows[i:i + SHARD] for i in range(0, len(rows), SHARD)]
    for n, s in enumerate(sh, 1):
        json.dump({"set": setn, "shard": n, "papers": s}, open(OUT / f"{setn}_shard{n:02d}.json", "w"), indent=0)
    print(f"[{setn}] {len(rows)} papers -> {len(sh)} shards")
    return key


keys = {}
keys["repr"] = build("repr", "quant_probe/eval_repr_240.json",
                     {"styleonly": "hg_styleonly_repr", "faith": "hg_faith_repr", "granite": "hg_granite_repr"})
json.dump(keys, open(OUT / "key_SECRET.json", "w"))
print("key written ->", OUT / "key_SECRET.json")
