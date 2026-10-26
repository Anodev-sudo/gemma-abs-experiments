#!/usr/bin/env python3
"""Produce blind, position-shuffled judge shards from combined.jsonl.

Each paper shows abstract + title + three anonymized briefs (A/B/C) in a
per-paper randomized order. A key file records which system each letter maps to,
so the judges stay blind to model identity and position."""
from __future__ import annotations
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
JDIR = Path(__file__).resolve().parent / "judge"
SYSTEMS = ["gemma425", "gemma500", "granite"]
N_SHARDS = 3


def main():
    JDIR.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in (OUT / "combined.jsonl").read_text().splitlines() if l.strip()]
    key = {}
    shards = {s: [] for s in range(N_SHARDS)}
    for c in rows:
        idx = c["idx"]
        rng = random.Random(1000 + idx)  # deterministic per-paper shuffle
        order = SYSTEMS[:]
        rng.shuffle(order)
        letters = ["A", "B", "C"]
        mapping = {letters[k]: order[k] for k in range(3)}
        key[idx] = mapping
        block = [
            f"### PAPER {idx}",
            f"TITLE: {c['title']}",
            f"ABSTRACT: {c['abstract']}",
            "BRIEFS:",
        ]
        for L in letters:
            block.append(f"  {L}: {c[mapping[L]]}")
        block.append("")
        shards[idx % N_SHARDS].append("\n".join(block))

    (JDIR / "key.json").write_text(json.dumps(key, indent=2))
    for s, blocks in shards.items():
        (JDIR / f"shard_{s}.txt").write_text("\n".join(blocks))
        print(f"shard_{s}: {len(blocks)} papers")
    print(f"key -> {JDIR/'key.json'}")


if __name__ == "__main__":
    main()
