#!/usr/bin/env python3
"""Build a three-way review pack for the selected q5 style-normalization finalist."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin200_style_norm_v1_lr5e7")
REVIEW_INPUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/eval/manual_review_100/review_input_100.jsonl")
CHECKPOINT700_OUTPUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/eval/manual_review_100_outputs/checkpoint700_outputs_100.jsonl")
BURNIN_OUTPUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/eval/manual_review_100_outputs/checkpoint_200_outputs_100.jsonl")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def norm(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def outputs_by_pid(path: Path) -> dict[str, str]:
    return {
        str(row["paper_id"]): norm(row.get("normalized_output") or row.get("raw_output"))
        for row in load_jsonl(path)
        if row.get("paper_id")
    }


def build_pack(args: argparse.Namespace) -> list[dict[str, Any]]:
    sources = load_jsonl(args.review_input)
    checkpoint700 = outputs_by_pid(args.checkpoint700_output)
    burnin = outputs_by_pid(args.burnin_output)
    finalist = outputs_by_pid(args.finalist_output)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for index, source in enumerate(sources, start=1):
        pid = str(source["paper_id"])
        if pid not in checkpoint700 or pid not in burnin or pid not in finalist:
            missing.append(pid)
            continue
        rows.append(
            {
                **source,
                "review_id": f"finalist_review_{index:03d}",
                "checkpoint700_output": checkpoint700[pid],
                "burnin200_output": burnin[pid],
                "normalization_output": finalist[pid],
                "normalization_label": args.finalist_label,
            }
        )
    if missing:
        raise SystemExit(f"Missing outputs for {len(missing)} rows: {missing[:10]}")
    return rows


def write_markdown(path: Path, rows: list[dict[str, Any]], finalist_label: str) -> None:
    lines = [
        "# q5 style-normalization finalist review",
        "",
        "Compare checkpoint-700, hard burn-in checkpoint-200, and the selected normalization candidate.",
        f"Normalization candidate: `{finalist_label}`",
        "",
        "Rubric: judge whether normalization preserves the hard numeric/value fix while improving feed style.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['review_id']} `{row['paper_id']}` [{row.get('source_bucket', '')}]",
                f"Title: {row.get('title', '')}",
                f"Checkpoint-700: {row['checkpoint700_output']}",
                f"Burn-in-200: {row['burnin200_output']}",
                f"{finalist_label}: {row['normalization_output']}",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build q5 style-normalization finalist review pack.")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--review-input", type=Path, default=REVIEW_INPUT)
    parser.add_argument("--checkpoint700-output", type=Path, default=CHECKPOINT700_OUTPUT)
    parser.add_argument("--burnin-output", type=Path, default=BURNIN_OUTPUT)
    parser.add_argument("--finalist-output", type=Path, required=True)
    parser.add_argument("--finalist-label", default="style_norm_checkpoint_100")
    parser.add_argument("--output-name", default="finalist_review_checkpoint_100")
    parser.add_argument("--shard-size", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_pack(args)
    out = args.output_root / "eval" / args.output_name
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "finalist_review_100.jsonl", rows)
    for shard_index, offset in enumerate(range(0, len(rows), args.shard_size), start=1):
        write_jsonl(out / f"finalist_review_shard_{shard_index}.jsonl", rows[offset : offset + args.shard_size])
    write_markdown(out / "FINALIST_REVIEW_100.md", rows, args.finalist_label)
    manifest = {
        "created_by": "scripts/build_q5_style_norm_finalist_review_pack_v1.py",
        "rows": len(rows),
        "finalist_label": args.finalist_label,
        "review_input": str(args.review_input),
        "checkpoint700_output": str(args.checkpoint700_output),
        "burnin_output": str(args.burnin_output),
        "finalist_output": str(args.finalist_output),
        "shard_size": args.shard_size,
        "shards": [str(path) for path in sorted(out.glob("finalist_review_shard_*.jsonl"))],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[finalist-review-pack] wrote {out / 'FINALIST_REVIEW_100.md'}")
    print(f"[finalist-review-pack] wrote {out / 'finalist_review_100.jsonl'}")


if __name__ == "__main__":
    main()
