#!/usr/bin/env python3
"""Vocabulary-strip a (tied-embedding) gemma model to the arXiv-reachable token set.

Keep-set = (tokens seen in arXiv, by DF threshold) ∪ byte tokens ∪ special/added,
CLOSED under BPE merge-parents (so every kept token is still reconstructible). Slices
the tied embed_tokens.weight, remaps ALL token ids, rebuilds the HF tokenizer.

  python strip_vocab.py --src quant_probe/fused_bf16 --tokenizer quant_probe/ship_tokenizer \
      --stats quant_probe/token_stats.npz --df 1 --out quant_probe/stripped_df1
"""
import argparse, json, re, struct, shutil
from collections import deque
from pathlib import Path
import numpy as np
import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)        # fused HF model dir
ap.add_argument("--tokenizer", required=True)  # gemma tokenizer dir
ap.add_argument("--stats", required=True)
ap.add_argument("--df", type=int, default=1)
ap.add_argument("--trim-specials", action="store_true",
                help="drop dead reserved/multimodal special tokens (keep only functional ones)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

# functional specials the brief chat-format actually needs (everything else added is dead)
FUNCTIONAL_SPECIALS = {"<bos>", "<eos>", "<pad>", "<unk>", "<mask>",
                       "<start_of_turn>", "<end_of_turn>"}

src, out = Path(a.src), Path(a.out)
out.mkdir(parents=True, exist_ok=True)

# embedding rows available (text vocab); ids >= EMB_ROWS have no embedding (image token)
hdr_n = struct.unpack("<Q", open(src/"model.safetensors","rb").read(8))[0]
hdr = json.loads(open(src/"model.safetensors","rb").read(8+hdr_n)[8:])
EMB_ROWS = hdr["model.embed_tokens.weight"]["shape"][0]
print(f"embedding rows (text vocab): {EMB_ROWS}")

tj = json.load(open(Path(a.tokenizer)/"tokenizer.json"))
mvocab = tj["model"]["vocab"]              # str -> id  (BPE vocab)
merges = tj["model"]["merges"]             # [A,B]
docf = np.load(a.stats)["docf"]

# --- keep-set ---
seen = set(np.where(docf >= a.df)[0].tolist())
byte_ids = {i for s, i in mvocab.items() if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", s)}
if a.trim_specials:
    # drop ONLY the reserved <unusedN> slots (dead everywhere). Keep all other added
    # tokens incl. newline-runs used in the prompt boilerplate (docf is abstract-only,
    # so they aren't in `seen` — must keep them explicitly to preserve prompt tokenization).
    dead = {at["id"] for at in tj["added_tokens"] if re.fullmatch(r"<unused\d+>", at["content"])}
    special_ids = {at["id"] for at in tj["added_tokens"]} - dead
    print(f"trim-specials: dropping {len(dead)} <unusedN> reserved tokens")
else:
    special_ids = {at["id"] for at in tj["added_tokens"]}
keep = (seen | byte_ids | special_ids)
keep = {i for i in keep if i < EMB_ROWS}   # drop image token (no embedding row)
base = len(keep)

# merge-parent map (id-level): a token can be produced by SEVERAL merge rules; BPE
# may build it via any of them depending on priority, so close over ALL decompositions.
parent = {}
for A, B in merges:
    ab = A + B
    if A in mvocab and B in mvocab and ab in mvocab:
        parent.setdefault(mvocab[ab], []).append((mvocab[A], mvocab[B]))
dq = deque(keep)
while dq:
    x = dq.popleft()
    for pa, pb in parent.get(x, ()):
        for p in (pa, pb):
            if p < EMB_ROWS and p not in keep:
                keep.add(p); dq.append(p)
added_by_closure = len(keep) - base
keep_sorted = sorted(keep)
K = len(keep_sorted)
old2new = {old: new for new, old in enumerate(keep_sorted)}
print(f"DF>={a.df}: seen={len(seen)} +bytes/special -> {base} +closure {added_by_closure} = KEPT {K} "
      f"({100*K/EMB_ROWS:.1f}% of {EMB_ROWS})")

# --- slice embedding, save model.safetensors ---
W = mx.load(str(src/"model.safetensors"))
W["model.embed_tokens.weight"] = W["model.embed_tokens.weight"][mx.array(keep_sorted)]
mx.save_safetensors(str(out/"model.safetensors"), W, metadata={"format": "pt"})

# --- rebuild tokenizer.json ---
ntj = json.loads(json.dumps(tj))  # deep copy
ntj["model"]["vocab"] = {t: old2new[i] for t, i in mvocab.items() if i in keep}
ntj["model"]["merges"] = [[A, B] for A, B in merges
                          if A in mvocab and B in mvocab and (A+B) in mvocab
                          and mvocab[A] in keep and mvocab[B] in keep and mvocab[A+B] in keep]
ntj["added_tokens"] = [{**at, "id": old2new[at["id"]]} for at in tj["added_tokens"] if at["id"] in keep]
pp = ntj.get("post_processor")
if pp and "special_tokens" in pp:
    for st in pp["special_tokens"].values():
        st["ids"] = [old2new[i] for i in st["ids"] if i in keep]
json.dump(ntj, open(out/"tokenizer.json", "w"))

# --- tokenizer_config.json: remap added_tokens_decoder keys, scrub image fields ---
tc = json.load(open(Path(a.tokenizer)/"tokenizer_config.json"))
tc["added_tokens_decoder"] = {str(old2new[int(k)]): v for k, v in tc.get("added_tokens_decoder", {}).items()
                              if int(k) in keep}
for f in ("image_token", "boi_token", "eoi_token", "image_seq_length"):
    tc.pop(f, None)
json.dump(tc, open(out/"tokenizer_config.json", "w"))
# scrub dangling multimodal refs (image tokens have no embedding row / are trimmed)
stm = json.load(open(Path(a.tokenizer)/"special_tokens_map.json"))
for f in ("boi_token", "eoi_token", "image_token"):
    stm.pop(f, None)
json.dump(stm, open(out/"special_tokens_map.json", "w"))
ct = Path(a.tokenizer)/"chat_template.jinja"
if ct.exists(): shutil.copy(ct, out/"chat_template.jinja")

# --- config.json + generation_config.json: vocab_size + special ids ---
def remap_id(v):
    if isinstance(v, list): return [old2new.get(x, x) for x in v]
    return old2new.get(v, v)
cfg = json.load(open(src/"config.json"))
cfg["vocab_size"] = K
for f in ("bos_token_id", "eos_token_id", "pad_token_id"):
    if f in cfg and cfg[f] is not None: cfg[f] = remap_id(cfg[f])
for f in ("image_token_index", "boi_token_index", "eoi_token_index"):
    cfg.pop(f, None)
json.dump(cfg, open(out/"config.json", "w"), indent=2)
gc_path = src/"generation_config.json"
if gc_path.exists():
    gc = json.load(open(gc_path))
    for f in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if f in gc and gc[f] is not None: gc[f] = remap_id(gc[f])
    json.dump(gc, open(out/"generation_config.json", "w"), indent=2)

json.dump({"df": a.df, "kept": K, "emb_rows": EMB_ROWS, "old2new": old2new},
          open(out/"strip_map.json", "w"))
print(f"wrote stripped model+tokenizer -> {out}  (vocab {EMB_ROWS} -> {K})")
