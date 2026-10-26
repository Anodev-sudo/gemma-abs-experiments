#!/usr/bin/env python3
"""Generate brief-mode briefs for one model over an explicit paper-list json.

Unified across the three-way: standalone model OR base+adapter (QLoRA sidecar).
Identical BRIEF prompt, temp 0, max 40 tokens, so A/B/C are apples-to-apples.

  python gen_threeway.py <papers.json> <label> --model PATH [--adapter PATH]
"""
import sys, json, argparse
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

ap = argparse.ArgumentParser()
ap.add_argument("papers")
ap.add_argument("label")
ap.add_argument("--model", required=True)
ap.add_argument("--adapter", default=None)
ap.add_argument("--system", default=None, help="override BRIEF_SYSTEM (faithfulness-first prompt test)")
a = ap.parse_args()
if a.system:
    BRIEF_SYSTEM = a.system

papers = json.load(open(a.papers))
model, tok = load(a.model, adapter_path=a.adapter) if a.adapter else load(a.model)
sampler = make_sampler(temp=0.0)
out = Path("quant_probe/eval_out"); out.mkdir(exist_ok=True)
with (out/f"{a.label}.jsonl").open("w") as fh:
    for p in papers:
        msgs = [{"role": "system", "content": BRIEF_SYSTEM},
                {"role": "user", "content": BRIEF_USER.format(title=p.get("title", ""), abstract=p.get("abstract", ""))}]
        ids = build_prompt(tok, msgs)
        raw = generate(model, tok, prompt=ids, max_tokens=40, sampler=sampler, verbose=False)
        b = clean(raw)
        fh.write(json.dumps({"arxiv_id": p["arxiv_id"], "brief": b, "words": word_count(b), "mindf": p.get("_mindf", -1)}) + "\n")
print(f"[{a.label}] generated {len(papers)} briefs -> quant_probe/eval_out/{a.label}.jsonl")
