#!/usr/bin/env python3
"""Generate briefs on the 50-row held-out test split for one LoRA checkpoint.

Writes a jsonl compatible with the eval_v5_vs_granite harness so new runs can be
merged/judged alongside the round-1 results.

    python training/generate_for_run.py \
        --base google/gemma-3-270m-it \
        --checkpoint training/runs/<name>/adapters/0000XXX_adapters.safetensors \
        --label <name>_ckptXXX
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab_common import TEST_SPLIT, build_prompt, clean, word_count  # noqa: E402

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

LAB = Path(__file__).resolve().parent.parent
OUT = LAB / "eval_v5_vs_granite/out"


def adapter_dir_for(checkpoint: Path) -> tuple[str, tempfile.TemporaryDirectory | None]:
    """mlx_lm loads `adapters.safetensors` + `adapter_config.json` from a dir.
    If given a specific checkpoint file, stage a temp dir pointing at it."""
    checkpoint = Path(checkpoint)
    cfg = checkpoint.parent / "adapter_config.json"
    if checkpoint.name == "adapters.safetensors":
        return str(checkpoint.parent), None
    tmp = tempfile.TemporaryDirectory()
    d = Path(tmp.name)
    (d / "adapters.safetensors").symlink_to(checkpoint.resolve())
    shutil.copy(cfg, d / "adapter_config.json")
    return str(d), tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-3-270m-it")
    ap.add_argument("--checkpoint", required=True, help="path to a *_adapters.safetensors")
    ap.add_argument("--label", required=True)
    ap.add_argument("--max-tokens", type=int, default=40)
    args = ap.parse_args()

    adapter_path, tmp = adapter_dir_for(Path(args.checkpoint))
    print(f"[{args.label}] base={args.base} adapter={args.checkpoint}")
    model, tokenizer = load(args.base, adapter_path=adapter_path)
    sampler = make_sampler(temp=0.0)

    rows = [json.loads(l) for l in TEST_SPLIT.read_text().splitlines() if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{args.label}.jsonl"
    results = []
    with out_path.open("w") as fh:
        for i, row in enumerate(rows):
            msgs = row["messages"]
            gold = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            ids = build_prompt(tokenizer, msgs)
            raw = generate(model, tokenizer, prompt=ids, max_tokens=args.max_tokens,
                           sampler=sampler, verbose=False)
            brief = clean(raw)
            rec = {"idx": i, "paper_id": row.get("paper_id"),
                   "primary_category": row.get("primary_category"),
                   "gold": gold, "brief": brief, "words": word_count(brief), "raw": raw}
            results.append(rec)
            fh.write(json.dumps(rec) + "\n")
            print(f"  [{i:02d}] ({rec['words']:>2}w) {brief}")
    if tmp:
        tmp.cleanup()

    n = len(results)
    wc = [r["words"] for r in results]
    over = sum(w > 14 for w in wc)
    print(f"[{args.label}] {n} rows | avg {sum(wc)/n:.1f}w | <=14: {n-over}/{n} -> {out_path}")


if __name__ == "__main__":
    main()
