#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("hf_dataset/releases/recent_3y_balanced_10746_v1/training_rows.jsonl")
DEFAULT_OUTPUT = Path(
    "hf_bucket/experiments/mlx_lora/runtime_shift/"
    "llama_cpp_q5_clean_slate_broad_v1_20260508/"
    "preference_mining/q5_step2400_2000_v1/eval_inputs_2000.jsonl"
)
DEFAULT_MANIFEST = DEFAULT_OUTPUT.with_name("input_manifest.json")
DEFAULT_BUCKET = "q5_step2400_random_2000"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def umbrella(row: dict[str, Any]) -> str:
    primary = str(row.get("primary_category", "")).strip()
    return primary.split(".", 1)[0] if primary else "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic q5 step-2400 mining input set."
    )
    parser.add_argument("--source-jsonl", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument(
        "--exclude-jsonl",
        action="append",
        default=[],
        help="Optional JSONL path whose paper_id values should be excluded. May be repeated.",
    )
    parser.add_argument("--selection-bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source_jsonl)
    output = Path(args.output_jsonl)
    manifest_path = Path(args.manifest)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite")

    excluded_paper_ids: set[str] = set()
    for exclude_path in args.exclude_jsonl:
        for row in load_jsonl(Path(exclude_path)):
            paper_id = row.get("paper_id")
            if paper_id is not None:
                excluded_paper_ids.add(str(paper_id))

    rows = [
        row
        for row in load_jsonl(source)
        if str(row.get("paper_id")) not in excluded_paper_ids
    ]
    if args.limit > len(rows):
        raise ValueError(f"Requested {args.limit} rows from only {len(rows)} source rows.")

    rng = random.Random(args.seed)
    selected = rng.sample(rows, args.limit)
    selected.sort(key=lambda row: str(row.get("paper_id", "")))

    output_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        output_rows.append(
            {
                "eval_id": f"q5mine_{index:04d}",
                "paper_id": str(row["paper_id"]),
                "title": str(row.get("title", "")).strip(),
                "abstract": str(row.get("abstract", "")).strip(),
                "summary": str(row.get("summary", "")).strip(),
                "primary_category": row.get("primary_category"),
                "categories": row.get("categories", []),
                "created_date": row.get("created_date"),
                "selection_bucket": args.selection_bucket,
                "source_split": row.get("source_split", ""),
                "source_release_summary": str(row.get("summary", "")).strip(),
            }
        )

    write_jsonl(output, output_rows)

    manifest = {
        "purpose": f"Random {len(output_rows)}-row mining set for q5 step-2400 llama.cpp failure harvesting.",
        "source_jsonl": str(source),
        "output_jsonl": str(output),
        "rows": len(output_rows),
        "seed": args.seed,
        "excluded_rows": len(excluded_paper_ids),
        "exclude_jsonl": [str(path) for path in args.exclude_jsonl],
        "selection_bucket": args.selection_bucket,
        "selection": "random sample without replacement, sorted by paper_id for stable generation order",
        "umbrella_counts": dict(sorted(Counter(umbrella(row) for row in output_rows).items())),
        "primary_category_top20": dict(
            Counter(str(row.get("primary_category", "")) for row in output_rows).most_common(20)
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[q5-mining-inputs] wrote {len(output_rows)} rows to {output}")
    print(f"[q5-mining-inputs] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
