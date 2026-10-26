#!/usr/bin/env python3
"""Aggregate the blind three-way: map A/B/C->model, per-model faithfulness with
bootstrap CIs, paired model-vs-model deltas, inter-rater agreement, hallucination tails.
Reports headline (repr) and worst-case (stress) separately."""
import json, glob, random
from pathlib import Path
from collections import defaultdict

J = Path("quant_probe/judge3way")
key = json.load(open(J/"key_SECRET.json"))   # {set:{arxiv_id:{A:model,...}}}
MODELS = ["q4qlora", "q5recover", "granite"]
rng = random.Random(20260614)

# collect: per set -> per arxiv_id -> per model -> list of faithful/readable/hook across passes
def collect(set_name):
    data = defaultdict(lambda: defaultdict(lambda: {"faithful": [], "readable": [], "hook": [], "flags": []}))
    agree = []   # (pass1_faithful, pass2_faithful) per (paper,slot) where both exist
    perslot = defaultdict(lambda: defaultdict(dict))  # arxiv->slot->pass->faithful
    for f in glob.glob(str(J/f"scores/{set_name}_shard*_p*.json")):
        d = json.load(open(f)); p = d["pass"]
        for aid, slots in d["scores"].items():
            if aid not in key[set_name]:
                continue
            smap = key[set_name][aid]
            for slot in ("A", "B", "C"):
                if slot not in slots:
                    continue
                m = smap[slot]
                v = slots[slot]
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

def model_paper_means(data, metric):
    # arxiv_id -> model -> mean over passes
    out = {}
    for aid, md in data.items():
        if not all(m in md and md[m][metric] for m in MODELS):
            continue   # need all 3 models present for paired analysis
        out[aid] = {m: sum(md[m][metric])/len(md[m][metric]) for m in MODELS}
    return out

def boot_ci(vals, nb=4000):
    n = len(vals)
    means = []
    for _ in range(nb):
        s = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s)/n)
    means.sort()
    return means[int(0.025*nb)], means[int(0.975*nb)]

def report(set_name):
    data, agree = collect(set_name)
    pm = model_paper_means(data, "faithful")
    n = len(pm)
    print(f"\n{'='*64}\n{set_name.upper()}  (papers with all 3 models scored: {n})\n{'='*64}")
    # per-model mean faithful + CI
    print(f"{'model':<12}{'faith':>7}{'  95% CI':>16}{'read':>7}{'hook':>7}")
    pm_read = model_paper_means(data, "readable"); pm_hook = model_paper_means(data, "hook")
    rows = {}
    for m in MODELS:
        fv = [pm[a][m] for a in pm]
        lo, hi = boot_ci(fv)
        rv = [pm_read[a][m] for a in pm_read]; hv = [pm_hook[a][m] for a in pm_hook]
        rows[m] = sum(fv)/n
        print(f"{m:<12}{sum(fv)/n:>7.3f}  [{lo:.3f}, {hi:.3f}]{sum(rv)/len(rv):>7.2f}{sum(hv)/len(hv):>7.2f}")
    # paired deltas vs granite and q4q vs q5
    print("\npaired faithfulness deltas (per-paper, bootstrap 95% CI):")
    for a_, b_ in [("q4qlora","granite"),("q5recover","granite"),("q4qlora","q5recover")]:
        diffs = [pm[a][a_]-pm[a][b_] for a in pm]
        md = sum(diffs)/n; lo,hi = boot_ci(diffs)
        sig = "" if lo<=0<=hi else "  *significant*"
        print(f"  {a_:<10} - {b_:<10}: {md:+.3f}  [{lo:+.3f}, {hi:+.3f}]{sig}")
    # hallucination tails (faithful averaged over passes, count per model)
    print("\nhallucination tails (per-paper mean faithful, /%d papers):" % n)
    print(f"{'model':<12}{'<=1 (misleading)':>18}{'<=0.5 (hard/inverted)':>24}")
    for m in MODELS:
        fv = [pm[a][m] for a in pm]
        mis = sum(1 for x in fv if x<=1.0); hard = sum(1 for x in fv if x<=0.5)
        print(f"{m:<12}{mis:>10} ({100*mis/n:>4.1f}%){hard:>14} ({100*hard/n:>4.1f}%)")
    # win rate (highest faithful per paper, ties split)
    print("\nbest-of-3 win share (ties split):")
    wins = defaultdict(float)
    for a in pm:
        mx = max(pm[a][m] for m in MODELS)
        top = [m for m in MODELS if pm[a][m]==mx]
        for m in top: wins[m]+=1/len(top)
    for m in MODELS: print(f"  {m:<12}{wins[m]/n*100:>6.1f}%")
    # inter-rater agreement
    if agree:
        exact = sum(1 for a,b in agree if a==b)/len(agree)
        within1 = sum(1 for a,b in agree if abs(a-b)<=1)/len(agree)
        mad = sum(abs(a-b) for a,b in agree)/len(agree)
        print(f"\ninter-rater (n={len(agree)} paper-slot pairs both passes): exact={exact*100:.0f}%  within1={within1*100:.0f}%  MAD={mad:.2f}")
    return rows

repr_rows = report("repr")
stress_rows = report("stress")
