#!/usr/bin/env python3
"""Unblind judge scores via key.json and tally per-system results."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

JDIR = Path(__file__).resolve().parent / "judge"
SYSTEMS = ["gemma425", "gemma500", "granite"]

key = {int(k): v for k, v in json.loads((JDIR / "key.json").read_text()).items()}

papers = {}
for s in range(3):
    data = json.loads((JDIR / f"scores_{s}.json").read_text())
    for p in data["papers"]:
        papers[p["idx"]] = p

agg = {s: defaultdict(float) for s in SYSTEMS}
wins = defaultdict(int)
hallucinations = {s: [] for s in SYSTEMS}
flags_count = defaultdict(int)
per_paper = []

for idx in sorted(papers):
    p = papers[idx]
    mapping = key[idx]  # letter -> system
    row = {"idx": idx}
    for letter, sysname in mapping.items():
        sc = p[letter]
        tot = sc["faithful"] + sc["readable"] + sc["hook"]
        agg[sysname]["faithful"] += sc["faithful"]
        agg[sysname]["readable"] += sc["readable"]
        agg[sysname]["hook"] += sc["hook"]
        agg[sysname]["total"] += tot
        agg[sysname]["n"] += 1
        if sc["faithful"] == 0:
            hallucinations[sysname].append(idx)
        flags_count[sysname] += len(sc["flags"])
        row[sysname] = tot
    wins[mapping[p["best"]]] += 1
    row["best"] = mapping[p["best"]]
    per_paper.append(row)

n = 50
print(f"{'system':<10} {'faith/3':>8} {'read/2':>7} {'hook/2':>7} {'total/7':>8} "
      f"{'wins':>5} {'halluc':>7} {'flags':>6}")
print("-" * 62)
for s in SYSTEMS:
    a = agg[s]
    print(f"{s:<10} {a['faithful']/n:>8.2f} {a['readable']/n:>7.2f} "
          f"{a['hook']/n:>7.2f} {a['total']/n:>8.2f} {wins[s]:>5} "
          f"{len(hallucinations[s]):>7} {flags_count[s]:>6}")

print("\nbest-pick wins (out of 50):", dict(wins))
print("hallucination papers (faithful==0):")
for s in SYSTEMS:
    print(f"  {s}: {hallucinations[s]}")

# head-to-head gemma425 vs granite per paper
g_better = gr_better = tie = 0
for r in per_paper:
    if r["gemma425"] > r["granite"]:
        g_better += 1
    elif r["granite"] > r["gemma425"]:
        gr_better += 1
    else:
        tie += 1
print(f"\ngemma425 vs granite (per-paper total score): "
      f"gemma425 higher {g_better}, granite higher {gr_better}, tie {tie}")

(JDIR / "aggregate_out.json").write_text(json.dumps({
    "per_system": {s: {k: agg[s][k] for k in agg[s]} for s in SYSTEMS},
    "wins": dict(wins),
    "hallucinations": hallucinations,
    "flags_count": dict(flags_count),
    "per_paper": per_paper,
}, indent=2))
