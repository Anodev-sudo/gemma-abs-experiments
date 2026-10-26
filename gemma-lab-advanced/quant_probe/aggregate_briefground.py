import json, glob, random
from pathlib import Path
from collections import defaultdict
J=Path("quant_probe/judge_briefground"); key=json.load(open(J/"key_SECRET.json"))
MODELS=["briefground","baseline","granite"]; rng=random.Random(20260618)
def collect(setn):
    data=defaultdict(lambda:defaultdict(list)); per=defaultdict(lambda:defaultdict(dict)); agree=[]
    for f in glob.glob(str(J/f"scores/{setn}_shard*_p*.json")):
        d=json.load(open(f)); p=d["pass"]
        for a,slots in d["scores"].items():
            if a not in key[setn]: continue
            sm=key[setn][a]
            for s in "ABC":
                if s in slots: data[a][sm[s]].append(slots[s]["faithful"]); per[a][s][p]=slots[s]["faithful"]
    for a,sl in per.items():
        for s,ps in sl.items():
            if 1 in ps and 2 in ps: agree.append((ps[1],ps[2]))
    return data,agree
def pm(data): return {a:{m:sum(v[m])/len(v[m]) for m in MODELS} for a,v in data.items() if all(m in v and v[m] for m in MODELS)}
def ci(d,nb=4000):
    n=len(d); ms=sorted(sum(d[rng.randrange(n)] for _ in range(n))/n for _ in range(nb)); return ms[int(.025*nb)],ms[int(.975*nb)]
def rep(setn):
    data,agree=collect(setn); p=pm(data); n=len(p)
    print(f"\n{'='*54}\n{setn.upper()} (n={n})\n{'='*54}")
    for m in MODELS:
        fv=[p[a][m] for a in p]; lo,hi=ci(fv); print(f"{m:<12}{sum(fv)/n:.3f}  [{lo:.3f},{hi:.3f}]")
    print("deltas:")
    for a_,b_ in [("briefground","baseline"),("briefground","granite"),("baseline","granite")]:
        d=[p[a][a_]-p[a][b_] for a in p]; md=sum(d)/n; lo,hi=ci(d)
        print(f"  {a_:<11}-{b_:<10}: {md:+.3f} [{lo:+.3f},{hi:+.3f}]{'' if lo<=0<=hi else ' *sig*'}")
    print("hard-halluc <=0.5:")
    for m in MODELS:
        fv=[p[a][m] for a in p]; print(f"  {m:<12}{sum(1 for x in fv if x<=0.5)} ({100*sum(1 for x in fv if x<=0.5)/n:.1f}%)")
    if agree:
        print(f"inter-rater n={len(agree)} exact={sum(1 for a,b in agree if a==b)/len(agree)*100:.0f}% MAD={sum(abs(a-b) for a,b in agree)/len(agree):.2f}")
rep("repr"); rep("stress")
