#!/usr/bin/env python3
"""Assemble a paper pool (arxiv_id/title/abstract) for distillation, deduped against the
eval sets (repr-240, stress-150) and the gold 894 so we never train on eval material.

Sources:
  --source local : the existing 10k grounding pool (title+abstract parsed from stage-1 data)
  --source vps   : pull fresh from the VPS arxiv DB (more real data); needs ssh composer-vps

  python gather_papers.py --n 8000 --out pool.json [--source local|vps]
"""
import sys, json, argparse, re, subprocess
from pathlib import Path

def norm(t): return re.sub(r"[^a-z0-9]", "", (t or "").lower())

def eval_gold_titles():
    bad = set()
    for f in ("quant_probe/eval_repr_240.json", "quant_probe/eval_stress_150.json"):
        for p in json.load(open(f)): bad.add(norm(p.get("title", "")))
    gold = ("gold_pipeline/05_milestone_assembly/gold_set/mlx_lora/"
            "title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5/train.jsonl")
    for l in open(gold):
        u = next(m["content"] for m in json.loads(l)["messages"] if m["role"] == "user")
        m = re.search(r"Title:\n(.*?)\n\nAbstract:\n", u, re.S)
        if m: bad.add(norm(m.group(1)))
    return bad

def from_local():
    src = "training/data/curriculum_stage1_summaries_briefprompt_v1/train.jsonl"
    out = []
    for i, l in enumerate(open(src)):
        u = next(m["content"] for m in json.loads(l)["messages"] if m["role"] == "user")
        m = re.search(r"Title:\n(.*?)\n\nAbstract:\n(.*)$", u, re.S)
        if m: out.append({"arxiv_id": f"loc{i:05d}", "title": m.group(1).strip(), "abstract": m.group(2).strip()})
    return out

def from_vps(n):
    # FirstPass DB: papers(arxiv_id,title,abstract,created_date,is_deleted). Read-only pull.
    # Order by created_date DESC (the app's feed order) so we get RECENT, feed-representative
    # papers — NOT arxiv_id DESC, which lexically floats 1990s prefixed IDs (hep-th/, supr-con/)
    # above modern YYMM.NNNNN ids and yields the oldest, math/physics-heaviest corner of arXiv.
    q = (f"SELECT arxiv_id||'\t'||replace(title,char(10),' ')||'\t'||replace(abstract,char(10),' ') "
         f"FROM papers WHERE is_deleted=0 AND length(abstract)>250 "
         f"ORDER BY created_date DESC, arxiv_id DESC LIMIT {n*2};")
    cmd = ["ssh", "composer-vps", f"sqlite3 /srv/composer/data/runtime/current/arxiv.sqlite \"{q}\""]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    out = []
    for line in res.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            out.append({"arxiv_id": parts[0], "title": parts[1].strip(), "abstract": parts[2].strip()})
    return out

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=8000)
ap.add_argument("--out", required=True)
ap.add_argument("--source", choices=["local", "vps"], default="local")
a = ap.parse_args()

bad = eval_gold_titles()
raw = from_vps(a.n) if a.source == "vps" else from_local()
pool, seen = [], set()
for p in raw:
    nt = norm(p["title"])
    if not nt or nt in bad or nt in seen: continue
    seen.add(nt); pool.append(p)
    if len(pool) >= a.n: break
json.dump(pool, open(a.out, "w"))
print(f"[gather:{a.source}] {len(pool)} papers (deduped vs {len(bad)} eval+gold) -> {a.out}")
