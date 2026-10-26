#!/usr/bin/env python3
"""Generate brief-mode on the enriched eval set (papers that re-segment at DF>=K)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab_common import build_prompt, clean, word_count
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

BRIEF_SYSTEM = "You write short, faithful paper-feed blurbs. Output only one sentence."
BRIEF_USER = ('The title is already visible in the app. Write one readable feed blurb, '
              '14 words or less, that gives a broad reader a reason to open this paper. '
              'Never exceed 14 words. Prefer the main takeaway over implementation detail. '
              'Use exact numbers only when they are central and easy to bind. Avoid '
              'repeating the title. Avoid jargon unless necessary. Avoid dense numbers, '
              'equations, shorthand, author/date references, and filler like "This paper" '
              'or "We propose".\n\nTitle:\n{title}\n\nAbstract:\n{abstract}')

model_path, label, mindf_cut, cap = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
papers = json.load(open("quant_probe/recent_eval_tagged.json"))
papers = [p for p in papers if p["_mindf"] < mindf_cut][:cap]   # affected at DF>=mindf_cut
model, tok = load(model_path)
sampler = make_sampler(temp=0.0)
out = Path("quant_probe/eval_out"); out.mkdir(exist_ok=True)
with (out/f"{label}.jsonl").open("w") as fh:
    for p in papers:
        msgs=[{"role":"system","content":BRIEF_SYSTEM},
              {"role":"user","content":BRIEF_USER.format(title=p.get("title",""),abstract=p.get("abstract",""))}]
        ids=build_prompt(tok,msgs)
        raw=generate(model,tok,prompt=ids,max_tokens=40,sampler=sampler,verbose=False)
        b=clean(raw)
        fh.write(json.dumps({"arxiv_id":p["arxiv_id"],"brief":b,"words":word_count(b),"mindf":p["_mindf"]})+"\n")
print(f"[{label}] generated {len(papers)} briefs")
