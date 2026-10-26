#!/usr/bin/env python3
"""Small held-out test set for fast student evaluation of the faithfulness-refinement experiment.
Recent, feed-representative VPS papers, deduped against eval (repr/stress), gold, AND the full 12k
training pool — so it overlaps neither training nor any prior eval.

  python build_testset.py --n 60 --out quant_probe/eval_testset_faith.json
"""
import json, argparse, re, subprocess
def norm(t): return re.sub(r"[^a-z0-9]", "", (t or "").lower())

def excluded_titles():
    bad=set()
    for f in ("quant_probe/eval_repr_240.json","quant_probe/eval_stress_150.json","quant_probe/distill/pool.json"):
        for p in json.load(open(f)): bad.add(norm(p.get("title","")))
    gold=("gold_pipeline/05_milestone_assembly/gold_set/mlx_lora/"
          "title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5/train.jsonl")
    for l in open(gold):
        u=next(m["content"] for m in json.loads(l)["messages"] if m["role"]=="user")
        m=re.search(r"Title:\n(.*?)\n\nAbstract:\n",u,re.S)
        if m: bad.add(norm(m.group(1)))
    return bad

def from_vps(n):
    q=(f"SELECT arxiv_id||'\t'||replace(title,char(10),' ')||'\t'||replace(abstract,char(10),' ') "
       f"FROM papers WHERE is_deleted=0 AND length(abstract)>250 "
       f"ORDER BY created_date DESC, arxiv_id DESC LIMIT {n};")
    cmd=["ssh","composer-vps",f"sqlite3 /srv/composer/data/runtime/current/arxiv.sqlite \"{q}\""]
    res=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    out=[]
    for line in res.stdout.splitlines():
        parts=line.split("\t")
        if len(parts)==3: out.append({"arxiv_id":parts[0],"title":parts[1].strip(),"abstract":parts[2].strip()})
    return out

ap=argparse.ArgumentParser()
ap.add_argument("--n",type=int,default=60)
ap.add_argument("--out",required=True)
a=ap.parse_args()
bad=excluded_titles()
# pull a big recent slice; the newest ~12k are the training pool, so survivors are the next-recent papers
raw=from_vps(40000)
pool,seen=[],set()
for p in raw:
    nt=norm(p["title"])
    if not nt or nt in bad or nt in seen: continue
    seen.add(nt); pool.append(p)
    if len(pool)>=a.n: break
json.dump(pool,open(a.out,"w"))
print(f"[testset] {len(pool)} papers (excluded {len(bad)} train+eval+gold) -> {a.out}")
print("sample:", [p["arxiv_id"] for p in pool[:5]])
