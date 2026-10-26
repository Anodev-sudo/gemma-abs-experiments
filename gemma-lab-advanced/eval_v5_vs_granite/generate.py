#!/usr/bin/env python3
"""Generate feed-blurb briefs for one model variant over the held-out test split.

Reads the bundled 50-row test split (already chat-formatted), strips the gold
assistant turn, applies each model's own chat template, and greedily generates a
brief. Writes one generations.jsonl per variant.

Run once per variant (keeps peak memory to a single model):
    python generate.py --variant gemma425
    python generate.py --variant gemma500
    python generate.py --variant granite
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

LAB = Path(__file__).resolve().parent.parent  # gemma-lab-advanced root
SRC = Path(__file__).resolve().parents[2]  # parent repo root
TEST = (
    LAB
    / "gold_pipeline/05_milestone_assembly/gold_set/mlx_lora"
    / "title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5/test.jsonl"
)

VARIANTS = {
    "gemma425": {
        "model": "google/gemma-3-270m-it",
        "adapter": str(LAB / "eval_v5_vs_granite/adapters/gemma_v5_ckpt425"),
    },
    "gemma500": {
        "model": "google/gemma-3-270m-it",
        "adapter": str(LAB / "eval_v5_vs_granite/adapters/gemma_v5_ckpt500"),
    },
    "granite": {
        "model": str(
            SRC
            / "hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_ckpt425_mlx_swift_q8_export_v1/fused_q8_g64"
        ),
        "adapter": None,
    },
}


def build_prompt(tokenizer, messages):
    """Apply chat template; fall back to folding system into user if the
    template rejects a standalone system turn (some gemma templates do)."""
    convo = [m for m in messages if m["role"] != "assistant"]
    try:
        return tokenizer.apply_chat_template(
            convo, add_generation_prompt=True, tokenize=True
        )
    except Exception:
        sys_txt = "\n\n".join(m["content"] for m in convo if m["role"] == "system")
        folded = []
        for m in convo:
            if m["role"] == "system":
                continue
            if m["role"] == "user" and sys_txt:
                folded.append({"role": "user", "content": f"{sys_txt}\n\n{m['content']}"})
                sys_txt = ""
            else:
                folded.append(m)
        return tokenizer.apply_chat_template(
            folded, add_generation_prompt=True, tokenize=True
        )


def clean(text: str) -> str:
    # First line / first sentence only; strip the model's stop tokens & quotes.
    text = text.strip()
    for stop in ("<end_of_turn>", "<eos>", "<|end_of_text|>", "</s>"):
        text = text.split(stop)[0]
    text = text.strip().strip('"').strip()
    # keep only the first line
    text = text.splitlines()[0].strip() if text else text
    return text


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--max-tokens", type=int, default=40)
    args = ap.parse_args()

    spec = VARIANTS[args.variant]
    print(f"[{args.variant}] loading model={spec['model']} adapter={spec['adapter']}")
    model, tokenizer = load(spec["model"], adapter_path=spec["adapter"])
    sampler = make_sampler(temp=0.0)  # greedy / deterministic

    rows = [json.loads(l) for l in TEST.read_text().splitlines() if l.strip()]
    out_path = LAB / "eval_v5_vs_granite/out" / f"{args.variant}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with out_path.open("w") as fh:
        for i, row in enumerate(rows):
            msgs = row["messages"]
            gold = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            prompt_ids = build_prompt(tokenizer, msgs)
            raw = generate(
                model,
                tokenizer,
                prompt=prompt_ids,
                max_tokens=args.max_tokens,
                sampler=sampler,
                verbose=False,
            )
            brief = clean(raw)
            rec = {
                "idx": i,
                "paper_id": row.get("paper_id"),
                "primary_category": row.get("primary_category"),
                "gold": gold,
                "brief": brief,
                "words": word_count(brief),
                "raw": raw,
            }
            results.append(rec)
            fh.write(json.dumps(rec) + "\n")
            print(f"  [{i:02d}] ({rec['words']:>2}w) {brief}")

    n = len(results)
    wc = [r["words"] for r in results]
    over14 = sum(w > 14 for w in wc)
    print(
        f"[{args.variant}] {n} rows | avg {sum(wc)/n:.1f}w | "
        f"<=14: {n-over14}/{n} | over14: {over14}/{n} -> {out_path}"
    )


if __name__ == "__main__":
    main()
