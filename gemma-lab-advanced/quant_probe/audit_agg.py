import json, glob, random
from pathlib import Path
from collections import defaultdict
J=Path("quant_probe/audit_faith"); key=json.load(open(J/"key_SECRET.json"))
M=["hooky","faithful"]; rng=random.Random(20260616); DIMS=["faithful","readable"]
data=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
for f in glob.glob(str(J/"scores/shard*_p*.json")):
    d=json.load(open(f))
    for a,slots in d["scores"].items():
        if a not in key: continue
        sm=key[a]
        for s in "XY":
            if s in slots:
                for dim in DIMS:
                    if dim in slots[s]: data[a][sm[s]][dim].append(slots[s][dim])
def pm(dim): return {a:{m:sum(v[m][dim])/len(v[m][dim]) for m in M} for a,v in data.items() if all(m in v and v[m].get(dim) for m in M)}
def ci(d,nb=4000):
    n=len(d); xs=sorted(sum(d[rng.randrange(n)] for _ in range(n))/n for _ in range(nb)); return xs[int(.025*nb)],xs[int(.975*nb)]
for dim in DIMS:
    p=pm(dim); n=len(p)
    print(f"\n[{dim}] n={n}")
    for m in M:
        fv=[p[a][m] for a in p]; lo,hi=ci(fv); print(f"  {m:<9}{sum(fv)/n:.3f}  [{lo:.3f},{hi:.3f}]")
    d=[p[a]["faithful" if False else "faithful"] for a in p] if False else None
    dl=[p[a]["faithful"] for a in p]  # noop guard
    delta=[p[a]["faithful"]-p[a]["hooky"] if dim=="faithful" else p[a]["faithful"]-p[a]["hooky"] for a in p]
    md=sum(delta)/n; lo,hi=ci(delta)
    print(f"  faithful-hooky: {md:+.3f} [{lo:+.3f},{hi:+.3f}]{'' if lo<=0<=hi else ' *sig*'}")
    if dim=="faithful":
        for m in M:
            fv=[p[a][m] for a in p]; print(f"  halluc<=0.5 {m:<9}{sum(1 for x in fv if x<=0.5)} ({100*sum(1 for x in fv if x<=0.5)/n:.0f}%)")
            print(f"  >=2.5 strong {m:<9}{sum(1 for x in fv if x>=2.5)} ({100*sum(1 for x in fv if x>=2.5)/n:.0f}%)")
