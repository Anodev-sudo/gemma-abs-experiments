#!/usr/bin/env python3
"""Write two readable Markdown files (Faithful, Granite) of feed blurbs for the same recent papers."""
import json
from pathlib import Path
papers=json.load(open("quant_probe/recent_papers.json"))
def load(f): return {json.loads(l)["arxiv_id"]:json.loads(l) for l in open(f"quant_probe/eval_out/{f}.jsonl")}
gens={"Faithful":("recent_faith","faithful_only_q5 — 97 MB gemma-3-270M q5"),
      "Granite":("recent_granite","Granite 4.0 350m q8 — 374 MB")}
DATE="2026-06-16"
for name,(f,desc) in gens.items():
    g=load(f)
    out=[f"# {name} — arXiv feed blurbs",
         f"_{desc}. {len(papers)} most-recent papers (created_date DESC), generated {DATE}. Same papers/order as the other file._",""]
    for i,p in enumerate(papers,1):
        a=p["arxiv_id"]; b=g.get(a,{}).get("brief","(missing)"); w=g.get(a,{}).get("words","?")
        out.append(f"**{i}. {p['title']}**  ")
        out.append(f"{b}  ")
        out.append(f"[{a}](https://arxiv.org/abs/{a}) · {w}w")
        out.append("")
    Path(f"recent_briefs_{name.lower()}.md").write_text("\n".join(out))
    print(f"wrote recent_briefs_{name.lower()}.md ({len(papers)} papers)")
