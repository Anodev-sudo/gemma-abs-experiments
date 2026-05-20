#!/usr/bin/env python3
"""Shard q5 heuristic repair hits for binary manual/subagent review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/"
    "eval/burnin150_style_probe_5000_v1/heuristic_screen_checkpoint150/repair_candidates.jsonl"
)
DEFAULT_OUT = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/"
    "eval/burnin150_style_probe_5000_v1/heuristic_hit_review_v1"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def compact_row(row: dict[str, Any], review_id: str) -> dict[str, Any]:
    heur = row.get("style_heuristics", {})
    return {
        "review_id": review_id,
        "paper_id": row.get("paper_id", ""),
        "split": row.get("split", ""),
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "generated_summary": row.get("generated_summary", ""),
        "heuristic_flags": heur.get("flags", []),
        "flag_categories": heur.get("flag_categories", {}),
        "repair_score": heur.get("repair_score", 0),
        "word_count": heur.get("word_count", None),
        "numbers": heur.get("numbers", []),
        "title_overlap_ratio": heur.get("title_overlap_ratio", None),
    }


def write_prompt(path: Path, shards: int) -> None:
    prompt = f"""# q5 Heuristic Hit Review Prompt

You are reviewing generated paper summaries that tripped loose heuristics.

Goal: decide whether each generated summary is an actual failure that should be repaired before it can be used as a training label or self-anchor.

Important:

- The heuristic flags are hints, not rules.
- Mark `actual_failure=false` when the summary is acceptable despite the flags.
- Mark `actual_failure=true` when the summary has a real issue: digit/formula residue, malformed wording, title rehash, unsupported claim, value/polarity mismatch, bad grammar, too vague, too long, or too thin.
- Notes must be very concise: ideally 3-10 words.
- Use the title and abstract when needed for accuracy/polarity.
- Do not propose new prompts, guards, DPO, or training changes.

Output JSONL schema, one row per input row:

```json
{{"review_id":"hit_000001","paper_id":"...","actual_failure":true,"notes":"repeated word and malformed term"}}
```

Allowed values:

- `actual_failure`: boolean
- `notes`: short string. Use `"acceptable"` for non-failures unless a tiny note is useful.

There are {shards} shards. Each reviewer owns one shard and one output file.
"""
    path.write_text(prompt, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shard heuristic repair candidates for binary review.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--shards", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shards < 1:
        raise SystemExit("--shards must be at least 1")
    rows = load_jsonl(args.input)
    args.out_root.mkdir(parents=True, exist_ok=True)

    compact = [compact_row(row, f"hit_{idx:06d}") for idx, row in enumerate(rows, start=1)]
    write_jsonl(args.out_root / "all_review_items.jsonl", compact)

    shard_paths: list[str] = []
    expected_output_paths: list[str] = []
    for shard_index in range(args.shards):
        shard_rows = [row for idx, row in enumerate(compact) if idx % args.shards == shard_index]
        shard_path = args.out_root / f"review_items_shard_{shard_index + 1:02d}.jsonl"
        output_path = args.out_root / f"review_results_shard_{shard_index + 1:02d}.jsonl"
        write_jsonl(shard_path, shard_rows)
        shard_paths.append(str(shard_path))
        expected_output_paths.append(str(output_path))

    write_prompt(args.out_root / "SUBAGENT_BINARY_REVIEW_PROMPT.md", args.shards)
    manifest = {
        "created_by": "scripts/build_q5_heuristic_hit_review_shards_v1.py",
        "input": str(args.input),
        "row_count": len(rows),
        "shards": args.shards,
        "shard_paths": shard_paths,
        "expected_output_paths": expected_output_paths,
        "review_question": "Is this generated summary an actual failure requiring repair before training/self-anchor use?",
        "output_schema": {
            "review_id": "string",
            "paper_id": "string",
            "actual_failure": "boolean",
            "notes": "short string",
        },
    }
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[heuristic-hit-shards] wrote {len(rows)} rows into {args.shards} shards at {args.out_root}")


if __name__ == "__main__":
    main()
