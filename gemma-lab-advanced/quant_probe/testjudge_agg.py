import json, glob, random
from pathlib import Path
from collections import defaultdict
J=Path("quant_probe/judge_testfaith"); key=json.load(open(J/"key_SECRET.json"))
M=["faithonly","faithvoice","baseline"]; rng=random.Random(20260616); DIMS=["faithful","readable","hook"]
data=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
for f in glob.glob(str(J/"scores/shard*_p*.json")):
    d=json.load(open(f))
    for a,slots in d["scores"].items():
        if a not in key: continue
        sm=key[a]
        for s in "ABC":
            if s in slots:
                for dim in DIMS:
                    if dim in slots[s]: data[a][sm[s]][dim].append(slots[s][dim])
def pm(dim): return {a:{m:sum(v[m][dim])/len(v[m][dim]) for m in M} for a,v in data.items() if all(m in v and v[m].get(dim) for m in M)}
def ci(d,nb=4000):
    n=len(d); xs=sorted(sum(d[rng.randrange(n)] for _ in range(n))/n for _ in range(nb)); return xs[int(.025*nb)],xs[int(.975*nb)]
for dim in DIMS:
    p=pm(dim); n=len(p)
    if not n: continue
    print(f"\n[{dim}] n={n}")
    for m in M:
        fv=[p[a][m] for a in p]; lo,hi=ci(fv); print(f"  {m:<11}{sum(fv)/n:.3f}  [{lo:.3f},{hi:.3f}]")
    for a_,b_ in [("faithonly","baseline"),("faithvoice","baseline"),("faithonly","faithvoice")]:
        d=[p[a][a_]-p[a][b_] for a in p]; md=sum(d)/n; lo,hi=ci(d)
        print(f"    {a_}-{b_:<10}: {md:+.3f} [{lo:+.3f},{hi:+.3f}]{'' if lo<=0<=hi else ' *sig*'}")
    if dim=="faithful":
        for m in M:
            fv=[p[a][m] for a in p]; print(f"  halluc<=0.5 {m:<11}{sum(1 for x in fv if x<=0.5)} ({100*sum(1 for x in fv if x<=0.5)/n:.0f}%)")
