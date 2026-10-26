#!/usr/bin/env python3
"""Aggregate round-3 4-way: df50 / df100 / df25 / granite. Bootstrap CIs,
deltas (df50 vs each), hallucination tails, inter-rater. judge4way_df50/scores/*."""
import json, glob, random
from pathlib import Path
from collections import defaultdict

J = Path("quant_probe/judge4way_df50")
key = json.load(open(J/"key_SECRET.json"))
MODELS = ["df50", "df100", "df25", "granite"]
SLOTS = ["A", "B", "C", "D"]
rng = random.Random(20260616)

def collect(set_name):
    data = defaultdict(lambda: defaultdict(lambda: {"faithful": [], "readable": [], "hook": [], "flags": []}))
    perslot = defaultdict(lambda: defaultdict(dict)); agree = []
    for f in glob.glob(str(J/f"scores/{set_name}_shard*_p*.json")):
        d = json.load(open(f)); p = d["pass"]
        for aid, slots in d["scores"].items():
            if aid not in key[set_name]:
                continue
            smap = key[set_name][aid]
            for slot in SLOTS:
                if slot not in slots:
                    continue
                m = smap[slot]; v = slots[slot]
                data[aid][m]["faithful"].append(v["faithful"])
                data[aid][m]["readable"].append(v.get("readable", 0))
                data[aid][m]["hook"].append(v.get("hook", 0))
                data[aid][m]["flags"] += v.get("flags", [])
                perslot[aid][slot][p] = v["faithful"]
    for aid, slots in perslot.items():
        for slot, passes in slots.items():
            if 1 in passes and 2 in passes:
                agree.append((passes[1], passes[2]))
    return data, agree

def pmeans(data, metric):
    out = {}
    for aid, md in data.items():
        if not all(m in md and md[m][metric] for m in MODELS):
            continue
        out[aid] = {m: sum(md[m][metric])/len(md[m][metric]) for m in MODELS}
    return out

def ci(vals, nb=4000):
    n = len(vals); ms = sorted(sum(vals[rng.randrange(n)] for _ in range(n))/n for _ in range(nb))
    return ms[int(0.025*nb)], ms[int(0.975*nb)]

def report(set_name):
    data, agree = collect(set_name); pm = pmeans(data, "faithful"); n = len(pm)
    pr = pmeans(data, "readable"); ph = pmeans(data, "hook")
    print(f"\n{'='*60}\n{set_name.upper()}  (n={n} papers, all 4 scored)\n{'='*60}")
    print(f"{'model':<10}{'faith':>7}{'  95% CI':>16}{'read':>7}{'hook':>7}")
    for m in MODELS:
        fv = [pm[a][m] for a in pm]; lo, hi = ci(fv)
        rv = [pr[a][m] for a in pr]; hv = [ph[a][m] for a in ph]
        print(f"{m:<10}{sum(fv)/n:>7.3f}  [{lo:.3f}, {hi:.3f}]{sum(rv)/len(rv):>7.2f}{sum(hv)/len(hv):>7.2f}")
    print("\npaired faithfulness deltas (per-paper, bootstrap 95% CI):")
    for a_, b_ in [("df50","df100"),("df50","df25"),("df50","granite"),("df100","granite")]:
        diffs = [pm[a][a_]-pm[a][b_] for a in pm]; md = sum(diffs)/n; lo, hi = ci(diffs)
        print(f"  {a_:<7} - {b_:<7}: {md:+.3f}  [{lo:+.3f}, {hi:+.3f}]{'' if lo<=0<=hi else '  *sig*'}")
    print(f"\nhallucination tails (/{n}):")
    for m in MODELS:
        fv = [pm[a][m] for a in pm]; mis = sum(1 for x in fv if x<=1.0); hard = sum(1 for x in fv if x<=0.5)
        print(f"   {m:<10} <=1: {mis:>3} ({100*mis/n:>4.1f}%)   <=0.5: {hard:>3} ({100*hard/n:>4.1f}%)")
    if agree:
        ex = sum(1 for a,b in agree if a==b)/len(agree); w1 = sum(1 for a,b in agree if abs(a-b)<=1)/len(agree)
        print(f"\ninter-rater (n={len(agree)}): exact={ex*100:.0f}% within1={w1*100:.0f}% MAD={sum(abs(a-b) for a,b in agree)/len(agree):.2f}")

report("repr"); report("stress")
