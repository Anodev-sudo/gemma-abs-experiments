#!/usr/bin/env python3
"""Identity check: on real arXiv text, stripped_tok.encode == remap(orig_tok.encode).
Proves the vocab/merge/id rebuild is lossless on the distribution."""
import json, sys
from pathlib import Path
from transformers import AutoTokenizer

LAB = Path(__file__).resolve().parent.parent
orig = AutoTokenizer.from_pretrained("quant_probe/ship_tokenizer")
strip = AutoTokenizer.from_pretrained("quant_probe/stripped_df1")
o2n = {int(k): v for k, v in json.load(open("quant_probe/stripped_df1/strip_map.json"))["old2new"].items()}

# corpus sample: 20k weighted corpus + the 50 held-out test papers
texts = []
for l in open("gold_pipeline/01_corpus/arxiv_abstract_corpus_weighted_20000.jsonl"):
    r = json.loads(l); texts.append((r.get("title","")+"\n"+r.get("abstract","")))
test = "gold_pipeline/05_milestone_assembly/gold_set/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5/test.jsonl"
for l in open(test):
    for m in json.loads(l)["messages"]:
        texts.append(m["content"])

keep = set(o2n)
mism = 0; oov = 0; checked = 0
n2o = {v: k for k, v in o2n.items()}
for t in texts:
    oids = orig.encode(t, add_special_tokens=False)
    if any(i not in keep for i in oids):
        oov += 1; continue          # text touches a genuinely-dropped token -> expected to re-segment
    checked += 1
    remap = [o2n[i] for i in oids]
    sids = strip.encode(t, add_special_tokens=False)
    if remap != sids:
        mism += 1
        if mism <= 3:
            for j, (x, y) in enumerate(zip(remap, sids)):
                if x != y:
                    print(f"  MISMATCH tok {j}: orig {orig.convert_ids_to_tokens(n2o[x])!r} "
                          f"vs strip {strip.convert_ids_to_tokens(y)!r}")
                    break
print(f"in-distribution texts checked: {checked} | mismatches: {mism} | "
      f"OOV texts (touch dropped token, expected re-seg): {oov}")
print("RESULT:", "IDENTICAL on in-distribution text ✓" if mism == 0 else "DIVERGENCE ✗")
