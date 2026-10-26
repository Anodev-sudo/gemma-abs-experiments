#!/usr/bin/env python3
"""Generate on the enriched eval with a QLoRA model: frozen q4 base + LoRA adapter."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab_common import build_prompt, clean, word_count
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
base, adapter, label, cap = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
BRIEF_SYSTEM="You write short, faithful paper-feed blurbs. Output only one sentence."
BRIEF_USER=('The title is already visible in the app. Write one readable feed blurb, '
 '14 words or less, that gives a broad reader a reason to open this paper. Never exceed '
 '14 words. Prefer the main takeaway over implementation detail. Use exact numbers only '
 'when they are central and easy to bind. Avoid repeating the title. Avoid jargon unless '
 'necessary. Avoid dense numbers, equations, shorthand, author/date references, and filler '
 'like "This paper" or "We propose".\n\nTitle:\n{title}\n\nAbstract:\n{abstract}')
papers=[p for p in json.load(open("quant_probe/recent_eval_tagged.json")) if p["_mindf"]<25][:cap]
model,tok=load(base, adapter_path=adapter)
sampler=make_sampler(temp=0.0)
out=Path("quant_probe/eval_out"); out.mkdir(exist_ok=True)
with (out/f"{label}.jsonl").open("w") as fh:
    for p in papers:
        msgs=[{"role":"system","content":BRIEF_SYSTEM},{"role":"user","content":BRIEF_USER.format(title=p.get("title",""),abstract=p.get("abstract",""))}]
        raw=generate(model,tok,prompt=build_prompt(tok,msgs),max_tokens=40,sampler=sampler,verbose=False)
        b=clean(raw); fh.write(json.dumps({"arxiv_id":p["arxiv_id"],"brief":b,"words":word_count(b)})+"\n")
print(f"[{label}] {len(papers)} briefs")
