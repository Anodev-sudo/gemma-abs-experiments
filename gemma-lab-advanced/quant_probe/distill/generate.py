#!/usr/bin/env python3
"""Few-shot teacher generation: bigger Gemma writes brief-objective blurbs, prompt-only.
Voice comes from K gold exemplars + an explicit brief-vs-summary rules block. No finetune.

  python generate.py --papers pool.json --shots 6 --temp 0.3 --out cand.jsonl

Teacher is the LMStudio-served gemma-4-26b-a4b-qat (HTTP, OpenAI-compatible) — see client.py.
"""
import sys, json, argparse, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lab_common import clean, word_count
sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat, MODEL

GOLD = ("gold_pipeline/05_milestone_assembly/gold_set/mlx_lora/"
        "title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5/train.jsonl")

SYS = (
 "You are an expert science writer for an arXiv reading app. For each paper you write ONE "
 "feed blurb: a single sentence, 14 words or fewer, that gives a broad reader a compelling, "
 "faithful reason to open it.\n"
 "This is a BLURB, not a summary. A summary neutrally states what the paper is about; a blurb "
 "leads with the single most interesting, abstract-supported takeaway — the reason someone "
 "would click. Write the blurb.\n"
 "Rules: every claim must be directly supported by the abstract; never invert a result, "
 "overstate, or invent specifics; prefer the result/why-it-matters over the method; no jargon, "
 "equations, dense numbers, author/date references, or filler like \"This paper\" or \"We "
 "propose\". Output only the blurb sentence."
)

def load_exemplars(k, seed):
    import random
    rows = [json.loads(l) for l in open(GOLD)]
    random.Random(seed).shuffle(rows)
    ex = []
    for r in rows[:k]:
        u = next(m["content"] for m in r["messages"] if m["role"] == "user")
        a = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
        m = re.search(r"Title:\n(.*?)\n\nAbstract:\n(.*)$", u, re.S)
        t, ab = (m.group(1), m.group(2)) if m else ("", u)
        ex.append((t.strip(), ab.strip()[:500], a.strip()))   # truncate exemplar abstracts
    return ex

def build_msgs(ex, title, abstract):
    msgs = [{"role": "system", "content": SYS}]
    for t, ab, br in ex:
        msgs.append({"role": "user", "content": f"Title: {t}\nAbstract: {ab}"})
        msgs.append({"role": "assistant", "content": br})
    msgs.append({"role": "user", "content": f"Title: {title}\nAbstract: {abstract}"})
    return msgs

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=MODEL)
ap.add_argument("--papers", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--shots", type=int, default=6)
ap.add_argument("--temp", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=20260615)
ap.add_argument("--cap", type=int, default=0)
ap.add_argument("--resume", action="store_true", help="skip arxiv_ids already in --out")
a = ap.parse_args()

papers = json.load(open(a.papers))
if a.cap: papers = papers[:a.cap]
done = set()
mode = "w"
if a.resume and Path(a.out).exists():
    for l in open(a.out): done.add(json.loads(l)["arxiv_id"])
    mode = "a"
ex = load_exemplars(a.shots, a.seed)
n = 0
with open(a.out, mode) as fh:
    for p in papers:
        if p["arxiv_id"] in done: continue
        msgs = build_msgs(ex, p.get("title", ""), p.get("abstract", ""))
        b = clean(chat(msgs, temp=a.temp, max_tokens=48, model=a.model))
        fh.write(json.dumps({"arxiv_id": p["arxiv_id"], "brief": b, "words": word_count(b)}) + "\n")
        fh.flush()
        n += 1
        if n % 250 == 0: print(f"  generated {n}/{len(papers)}", flush=True)
print(f"[generate] {n} candidate briefs -> {a.out}")
