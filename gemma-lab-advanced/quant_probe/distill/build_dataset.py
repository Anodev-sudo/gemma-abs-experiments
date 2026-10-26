#!/usr/bin/env python3
"""Assemble verified briefs into a chat-format stage-1 grounding dataset (brief-objective).
Reuses the brief 50/50 valid/test so eval stays comparable across runs.

  python build_dataset.py --papers pool.json --verified verified.jsonl --name granitexl_briefground_v1
"""
import json, argparse, shutil
from pathlib import Path

BRIEF_SYSTEM = "You write short, faithful paper-feed blurbs. Output only one sentence."
BRIEF_USER = ('The title is already visible in the app. Write one readable feed blurb, '
              '14 words or less, that gives a broad reader a reason to open this paper. '
              'Never exceed 14 words. Prefer the main takeaway over implementation detail. '
              'Use exact numbers only when they are central and easy to bind. Avoid '
              'repeating the title. Avoid jargon unless necessary. Avoid dense numbers, '
              'equations, shorthand, author/date references, and filler like "This paper" '
              'or "We propose".\n\nTitle:\n{title}\n\nAbstract:\n{abstract}')

ap = argparse.ArgumentParser()
ap.add_argument("--papers", required=True)
ap.add_argument("--verified", required=True)
ap.add_argument("--name", required=True)
a = ap.parse_args()

papers = {p["arxiv_id"]: p for p in json.load(open(a.papers))}
out = Path("training/data") / ("curriculum_stage1_" + a.name)
out.mkdir(parents=True, exist_ok=True)
kept = 0
with (out / "train.jsonl").open("w") as fh:
    for l in open(a.verified):
        r = json.loads(l); p = papers.get(r["arxiv_id"])
        if not p: continue
        row = {"messages": [
            {"role": "system", "content": BRIEF_SYSTEM},
            {"role": "user", "content": BRIEF_USER.format(title=p["title"], abstract=p["abstract"])},
            {"role": "assistant", "content": r["brief"].strip()}]}
        fh.write(json.dumps(row) + "\n"); kept += 1
src = Path("training/data/curriculum_stage1_summaries_briefprompt_v1")
for f in ("valid.jsonl", "test.jsonl"):
    shutil.copy(src / f, out / f)
json.dump({"stage": 1, "purpose": "brief-OBJECTIVE grounding (bigger-Gemma teacher, prompt-only, faithfulness-verified)",
           "train_rows": kept}, open(out / "manifest.json", "w"), indent=2)
print(f"[build] {kept} verified brief-grounding rows -> {out}")
