#!/usr/bin/env python3
"""Join 3k grounding papers + Granite brief targets -> chat-format stage1 dataset
(brief-OBJECTIVE grounding). Light heuristic filter; reuse brief 50/50 valid/test."""
import json, shutil
from pathlib import Path
BRIEF_SYSTEM = "You write short, faithful paper-feed blurbs. Output only one sentence."
BRIEF_USER = ('The title is already visible in the app. Write one readable feed blurb, '
              '14 words or less, that gives a broad reader a reason to open this paper. '
              'Never exceed 14 words. Prefer the main takeaway over implementation detail. '
              'Use exact numbers only when they are central and easy to bind. Avoid '
              'repeating the title. Avoid jargon unless necessary. Avoid dense numbers, '
              'equations, shorthand, author/date references, and filler like "This paper" '
              'or "We propose".\n\nTitle:\n{title}\n\nAbstract:\n{abstract}')
papers={p["arxiv_id"]:p for p in json.load(open("quant_probe/briefground_3k.json"))}
briefs={json.loads(l)["arxiv_id"]:json.loads(l)["brief"] for l in open("quant_probe/eval_out/bg_granite.jsonl")}
out=Path("training/data/curriculum_stage1_granitebrief_v1"); out.mkdir(parents=True, exist_ok=True)
kept=dropped=0; seen=set()
with (out/"train.jsonl").open("w") as fh:
    for aid,b in briefs.items():
        b=b.strip(); wc=len(b.split())
        if not b or wc<4 or wc>16 or b.lower() in seen:  # drop empty/degenerate/over-long/dup
            dropped+=1; continue
        seen.add(b.lower()); p=papers[aid]
        row={"messages":[{"role":"system","content":BRIEF_SYSTEM},
             {"role":"user","content":BRIEF_USER.format(title=p["title"],abstract=p["abstract"])},
             {"role":"assistant","content":b}]}
        fh.write(json.dumps(row)+"\n"); kept+=1
# reuse the brief 50/50 valid+test (same as summary-grounding baseline used) for comparable eval
src=Path("training/data/curriculum_stage1_summaries_briefprompt_v1")
for f in ("valid.jsonl","test.jsonl"): shutil.copy(src/f, out/f)
json.dump({"stage":1,"purpose":"brief-OBJECTIVE grounding (Granite teacher)","train_rows":kept,"teacher":"granite_4_0_350m_v5_q8"}, open(out/"manifest.json","w"), indent=2)
print(f"kept {kept}, dropped {dropped} -> {out}")
