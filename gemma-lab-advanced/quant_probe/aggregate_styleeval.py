import json, glob, random
from pathlib import Path
from collections import defaultdict
J=Path("quant_probe/judge_styleeval"); key=json.load(open(J/"key_SECRET.json"))
M=["faith_only","faith_style","candidate"]; rng=random.Random(20260616); DIMS=["faithful","readable","hook"]
def collect(setn):
    data=defaultdict(lambda:defaultdict(lambda:defaultdict(list))); per=defaultdict(lambda:defaultdict(dict)); agree=[]
    for f in glob.glob(str(J/f"scores/{setn}_shard*_p*.json")):
        d=json.load(open(f)); p=d["pass"]
        for a,slots in d["scores"].items():
            if a not in key[setn]: continue
            sm=key[setn][a]
            for s in "ABC":
                if s in slots:
                    for dim in DIMS:
                        if dim in slots[s]: data[a][sm[s]][dim].append(slots[s][dim])
                    if "faithful" in slots[s]: per[a][s][p]=slots[s]["faithful"]
    for a,sl in per.items():
        for s,ps in sl.items():
            if 1 in ps and 2 in ps: agree.append((ps[1],ps[2]))
    return data,agree
def pm(data,dim): return {a:{m:sum(v[m][dim])/len(v[m][dim]) for m in M} for a,v in data.items() if all(m in v and v[m].get(dim) for m in M)}
def ci(d,nb=4000):
    n=len(d); xs=sorted(sum(d[rng.randrange(n)] for _ in range(n))/n for _ in range(nb)); return xs[int(.025*nb)],xs[int(.975*nb)]
def rep(setn):
    data,agree=collect(setn)
    print(f"\n{'='*56}\n{setn.upper()}\n{'='*56}")
    for dim in DIMS:
        p=pm(data,dim); n=len(p)
        if not n: continue
        print(f"\n[{dim}] n={n}")
        for m in M:
            fv=[p[a][m] for a in p]; lo,hi=ci(fv); print(f"  {m:<12}{sum(fv)/n:.3f}  [{lo:.3f},{hi:.3f}]")
        for a_,b_ in [("faith_style","faith_only"),("faith_style","candidate"),("faith_only","candidate")]:
            d=[p[a][a_]-p[a][b_] for a in p]; md=sum(d)/n; lo,hi=ci(d)
            print(f"    {a_}-{b_}: {md:+.3f} [{lo:+.3f},{hi:+.3f}]{'' if lo<=0<=hi else ' *sig*'}")
        if dim=="faithful":
            for m in M:
                fv=[p[a][m] for a in p]; print(f"  halluc<=0.5 {m:<12}{sum(1 for x in fv if x<=0.5)} ({100*sum(1 for x in fv if x<=0.5)/n:.1f}%)")
    if agree: print(f"\ninter-rater (faithful) n={len(agree)} exact={sum(1 for a,b in agree if a==b)/len(agree)*100:.0f}% MAD={sum(abs(a-b) for a,b in agree)/len(agree):.2f}")
rep("repr"); rep("stress")
