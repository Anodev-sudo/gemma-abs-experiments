#!/usr/bin/env python3
"""Aggregate the blind 3-way cond_hook vs faith vs granite. Per-model means (faithful/
readable/hook) with bootstrap CIs, paired model-vs-model deltas (the spec's primary =
styleonly-faith, secondary = styleonly-granite), hallucination tails, inter-rater agreement.
repr (headline) and stress (worst-case) reported separately. Adapted from aggregate_threeway.py."""
import json, glob, random
from pathlib import Path
from collections import defaultdict

J = Path("quant_probe/judge3way_styleonly")
key = json.load(open(J / "key_SECRET.json"))
MODELS = ["styleonly", "faith", "granite"]
DIMS = ["faithful", "readable", "hook"]
rng = random.Random(20260617)


def collect(setn):
    data = defaultdict(lambda: defaultdict(lambda: {d: [] for d in DIMS}))
    perslot = defaultdict(lambda: defaultdict(dict)); agree = []
    for f in glob.glob(str(J / f"scores/{setn}_shard*_p*.json")):
        d = json.load(open(f)); p = d["pass"]
        for aid, slots in d["scores"].items():
            if aid not in key[setn]:
                continue
            sm = key[setn][aid]
            for slot in "ABC":
                if slot not in slots:
                    continue
                m = sm[slot]; v = slots[slot]
                for dim in DIMS:
                    if dim in v:
                        data[aid][m][dim].append(v[dim])
                if "faithful" in v:
                    perslot[aid][slot][p] = v["faithful"]
    for aid, slots in perslot.items():
        for slot, passes in slots.items():
            if 1 in passes and 2 in passes:
                agree.append((passes[1], passes[2]))
    return data, agree


def pm(data, dim):
    return {a: {m: sum(v[m][dim]) / len(v[m][dim]) for m in MODELS}
            for a, v in data.items() if all(v[m][dim] for m in MODELS)}


def ci(d, nb=4000):
    n = len(d)
    xs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(nb))
    return xs[int(.025 * nb)], xs[int(.975 * nb)]


def rep(setn):
    data, agree = collect(setn)
    print(f"\n{'='*60}\n{setn.upper()}  (styleonly=97MB cond_hook_q5 | faith=97MB faithful_only_q5 | granite=374MB q8)\n{'='*60}")
    for dim in DIMS:
        p = pm(data, dim); n = len(p)
        if not n:
            continue
        print(f"\n[{dim}] n={n}")
        for m in MODELS:
            fv = [p[a][m] for a in p]; lo, hi = ci(fv)
            print(f"  {m:<10}{sum(fv)/n:.3f}  [{lo:.3f},{hi:.3f}]")
        for x, y in (("styleonly", "faith"), ("styleonly", "granite")):
            d = [p[a][x] - p[a][y] for a in p]; md = sum(d) / n; lo, hi = ci(d)
            sig = "" if lo <= 0 <= hi else " *sig*"
            print(f"    {x}-{y:<8}: {md:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
        if dim == "faithful":
            print("  hallucination tails (per-paper mean faithful):")
            for m in MODELS:
                fv = [p[a][m] for a in p]
                mis = sum(1 for x in fv if x <= 1); hard = sum(1 for x in fv if x <= 0.5)
                print(f"    {m:<10} <=1 (misleading) {mis:3d} ({100*mis/n:4.1f}%) | <=0.5 (hard) {hard:3d} ({100*hard/n:4.1f}%)")
    if agree:
        ex = sum(1 for a, b in agree if a == b) / len(agree) * 100
        mad = sum(abs(a - b) for a, b in agree) / len(agree)
        print(f"\ninter-rater (faithful) n={len(agree)} exact={ex:.0f}% MAD={mad:.2f}")


rep("repr")
