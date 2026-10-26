#!/usr/bin/env python3
"""Turn the full-arXiv token stats into a vocab-strip decision.

Sweeps keep-thresholds by document frequency (DF = # papers containing the token —
the metric that bounds strip risk). For each threshold reports kept vocab,
occurrence-coverage, and projected model size. Always keeps the byte alphabet +
special/added tokens so any input stays representable (requires byte_fallback)."""
import argparse, json, re
import numpy as np
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--stats", required=True)      # token_stats.npz
ap.add_argument("--tokenizer", required=True)
ap.add_argument("--emit-keep", type=int, default=0, help="DF threshold to materialize a keep-list for")
ap.add_argument("--emit-out", default="quant_probe/keep_ids.json")
a = ap.parse_args()

z = np.load(a.stats)
total, docf = z["total"], z["docf"]
V = total.size
N_TOK = int(total.sum())

tok = AutoTokenizer.from_pretrained(a.tokenizer)
vocab = tok.get_vocab()                          # token_str -> id
id2tok = {i: s for s, i in vocab.items()}

# must-keep: byte tokens (<0xHH>) + all special/added ids
byte_ids = {i for s, i in vocab.items() if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", s)}
special_ids = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_encoder", {}).values())
must_keep = byte_ids | special_ids
print(f"vocab={V:,}  total_tokens={N_TOK:,}  byte_tokens={len(byte_ids)}  special/added={len(special_ids)}")

HIDDEN = 640
EMB = V * HIDDEN
NON_EMB = 268_098_000 - EMB                       # ~100.3M non-embedding params
def proj(kept):
    tot = NON_EMB + kept * HIDDEN
    return tot, tot*2/1e6, tot*1.06/1e6, tot*(5.5/8)/1e6   # params, bf16MB, q8MB, q5MB

print(f"\n{'DF>=':>6} {'kept':>8} {'%vocab':>7} {'occ_cov%':>9} {'params':>9} {'bf16MB':>8} {'q8MB':>7} {'q5MB':>7}")
print("-"*74)
for K in [1,2,3,5,10,25,50,100,500,1000,5000]:
    keep = set(np.where(docf >= K)[0].tolist()) | must_keep
    kept = len(keep)
    occ = total[list(keep)].sum() / N_TOK
    p,b,q8,q5 = proj(kept)
    print(f"{K:>6} {kept:>8,} {100*kept/V:>6.1f}% {100*occ:>8.4f}% {p/1e6:>8.1f}M {b:>7.0f} {q8:>6.0f} {q5:>6.0f}")

# context: granite shipped q8 ~374MB ; current gemma q5 fused 208MB / q8 304MB
print("\nref: shipped Granite q8 ~374MB | current (unstripped) gemma q5 208MB, q8 304MB")
# dropped-token risk profile at a few thresholds
for K in [2,5,10]:
    dropped = (docf < K) & (docf > 0)
    dropped_only = dropped & ~np.isin(np.arange(V), list(must_keep))
    print(f"DF>= {K}: drops {int(dropped_only.sum()):,} seen tokens, "
          f"each in <{K} papers; their total occ = {100*total[dropped_only].sum()/N_TOK:.4f}% of stream")

if a.emit_keep:
    keep = sorted(set(np.where(docf >= a.emit_keep)[0].tolist()) | must_keep)
    json.dump({"df_threshold": a.emit_keep, "kept_ids": keep, "kept": len(keep),
               "vocab": V, "must_keep": len(must_keep)}, open(a.emit_out, "w"))
    print(f"\nwrote keep-list (DF>={a.emit_keep}, {len(keep):,} ids) -> {a.emit_out}")
