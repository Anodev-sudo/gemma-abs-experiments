#!/usr/bin/env python3
"""Tokenize every (non-deleted) arXiv abstract+title with the gemma tokenizer and
record, per vocab id: total occurrence count AND document frequency (# papers that
contain it). DF is what bounds vocab-strip risk: dropping a token re-segments only
the DF papers that use it.

Read-only on the served snapshot. Deliberate one-time full scan (not casual).

  /srv/composer/.venv/bin/python vocab_coverage.py \
      --db /srv/composer/data/runtime/current/arxiv.sqlite \
      --tokenizer ~/vocab_strip/ship_tokenizer --out ~/vocab_strip/token_stats.npz [--limit N]
"""
import argparse, json, sqlite3, time, sys
import numpy as np
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True)
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--batch", type=int, default=2000)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.tokenizer)
V = len(tok.get_vocab())
total = np.zeros(V, dtype=np.int64)   # occurrences
docf  = np.zeros(V, dtype=np.int64)   # document frequency
n_papers = 0
n_tokens = 0
t0 = time.time()

uri = f"file:{a.db}?mode=ro&immutable=1"
con = sqlite3.connect(uri, uri=True)
con.execute("PRAGMA query_only=1")
sql = "SELECT title, abstract FROM papers WHERE is_deleted=0"
if a.limit:
    sql += f" LIMIT {a.limit}"
cur = con.execute(sql)

while True:
    rows = cur.fetchmany(a.batch)
    if not rows:
        break
    texts = [f"{(t or '')}\n{(ab or '')}" for (t, ab) in rows]
    enc = tok(texts, add_special_tokens=False, truncation=False)["input_ids"]
    batch_all = []          # all ids (for total)
    batch_uniq = []         # per-doc unique ids (for DF)
    for ids in enc:
        arr = np.asarray(ids, dtype=np.int64)
        batch_all.append(arr)
        batch_uniq.append(np.unique(arr))
        n_tokens += arr.size
    if batch_all:
        total += np.bincount(np.concatenate(batch_all), minlength=V)
        docf  += np.bincount(np.concatenate(batch_uniq), minlength=V)
    n_papers += len(rows)
    if n_papers % 50000 < a.batch:
        el = time.time() - t0
        seen = int((total > 0).sum())
        print(f"  {n_papers:>9,} papers | {n_tokens/1e6:7.1f}M tok | "
              f"{seen:>7,} vocab seen ({100*seen/V:4.1f}%) | {el:6.0f}s | "
              f"{n_papers/max(el,1):,.0f} papers/s", flush=True)

con.close()
seen = int((total > 0).sum())
np.savez_compressed(a.out, total=total, docf=docf)
meta = {
    "n_papers": n_papers, "n_tokens": int(n_tokens), "vocab_size": V,
    "vocab_seen": seen, "vocab_seen_pct": round(100 * seen / V, 2),
    "vocab_never": V - seen, "elapsed_s": round(time.time() - t0, 1),
}
with open(a.out.replace(".npz", "_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2))
