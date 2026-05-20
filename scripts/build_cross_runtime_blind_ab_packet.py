#!/usr/bin/env python3
"""Build a preference-only blind A/B packet from two generation JSONLs."""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def norm(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def generation_map(path: Path) -> dict[str, dict[str, str]]:
    mapped: dict[str, dict[str, str]] = {}
    for row in load_jsonl(path):
        paper_id = str(row.get("paper_id", ""))
        if not paper_id:
            continue
        summary = norm(row.get("summary") or row.get("normalized_output") or row.get("raw_output"))
        mapped[paper_id] = {
            "summary": summary,
            "summary_raw": str(row.get("summary_raw") or row.get("raw_output") or summary),
        }
    return mapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cross-runtime blind A/B packet.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--left-generations", type=Path, required=True)
    parser.add_argument("--right-generations", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260518)
    return parser.parse_args()


def build_rows(
    *,
    source_rows: list[dict[str, Any]],
    left_outputs: dict[str, dict[str, str]],
    right_outputs: dict[str, dict[str, str]],
    left_label: str,
    right_label: str,
    seed: int,
    count: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    usable = [
        row
        for row in source_rows
        if str(row.get("paper_id")) in left_outputs
        and str(row.get("paper_id")) in right_outputs
    ]
    if count is not None:
        if count > len(usable):
            raise ValueError(f"Requested {count} rows but only {len(usable)} have both outputs.")
        usable = usable[:count]

    review_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    unblinded_rows: list[dict[str, Any]] = []
    for index, source in enumerate(usable, start=1):
        paper_id = str(source["paper_id"])
        left = left_outputs[paper_id]
        right = right_outputs[paper_id]
        left_is_a = rng.random() < 0.5
        summary_a = left["summary"] if left_is_a else right["summary"]
        summary_b = right["summary"] if left_is_a else left["summary"]
        summary_a_model = left_label if left_is_a else right_label
        summary_b_model = right_label if left_is_a else left_label
        review_id = f"blind_ab_{index:03d}"
        key_rows.append(
            {
                "review_id": review_id,
                "paper_id": paper_id,
                "summary_a_model": summary_a_model,
                "summary_b_model": summary_b_model,
                "split": source.get("split", ""),
            }
        )
        review_rows.append(
            {
                "review_id": review_id,
                "paper_id": paper_id,
                "title": source.get("title", ""),
                "abstract": source.get("abstract", ""),
                "primary_category": source.get("primary_category", ""),
                "created_date": source.get("created_date", ""),
                "summary_a": summary_a,
                "summary_b": summary_b,
            }
        )
        unblinded_rows.append(
            {
                **review_rows[-1],
                "summary_a_model": summary_a_model,
                "summary_b_model": summary_b_model,
                "left_label": left_label,
                "right_label": right_label,
                "left_summary": left["summary"],
                "left_summary_raw": left["summary_raw"],
                "right_summary": right["summary"],
                "right_summary_raw": right["summary_raw"],
                "outputs_match": left["summary"] == right["summary"],
            }
        )
    return review_rows, key_rows, unblinded_rows


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Blind A/B Summary Review",
        "",
        "For each row, read the title and abstract, then choose the summary you would rather show in a broad mobile paper feed.",
        "",
        "Judge naturally: usefulness, readability, specificity, and faithfulness all matter.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['review_id']} `{row['paper_id']}`",
                f"Title: {row.get('title', '')}",
                "",
                f"Abstract: {row.get('abstract', '')}",
                "",
                f"Summary A: {row['summary_a']}",
                f"Summary B: {row['summary_b']}",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_subagent_prompt(path: Path, row_count: int, shard_count: int) -> None:
    prompt = f"""# Preference-Only Blind A/B Review

You are reviewing generated paper summaries without knowing which model produced which side.

For each row, read the title and abstract, then compare Summary A and Summary B.

Choose the summary you would rather show in a broad mobile research feed. Judge naturally: usefulness, readability, specificity, and faithfulness all matter. Do not apply mechanical heuristics.

For every input row, write exactly one JSONL object:

```json
{{"review_id":"blind_ab_001","paper_id":"...","preferred_summary":"summary_a","notes":"A is clearer and better grounded"}}
```

Allowed values for `preferred_summary`: `summary_a`, `summary_b`, `tie`, or `neither`.

Use `tie` when both are similarly good or similarly flawed. Use `neither` only when both are clearly bad. Keep `notes` very concise.

There are {row_count} rows across {shard_count} shards. Each reviewer owns one shard and one output file.
"""
    path.write_text(prompt, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_rows = load_jsonl(args.source)
    left_outputs = generation_map(args.left_generations)
    right_outputs = generation_map(args.right_generations)
    review_rows, key_rows, unblinded_rows = build_rows(
        source_rows=source_rows,
        left_outputs=left_outputs,
        right_outputs=right_outputs,
        left_label=args.left_label,
        right_label=args.right_label,
        seed=args.seed,
        count=args.count,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    row_count = len(review_rows)
    review_path = args.output_root / f"blind_review_{row_count}.jsonl"
    key_path = args.output_root / f"blind_key_{row_count}.jsonl"
    write_jsonl(review_path, review_rows)
    write_jsonl(key_path, key_rows)
    write_jsonl(args.output_root / "blind_review_unblinded_inputs.jsonl", unblinded_rows)

    shard_paths: list[str] = []
    expected_output_paths: list[str] = []
    for shard_index, offset in enumerate(range(0, row_count, args.shard_size), start=1):
        shard = review_rows[offset : offset + args.shard_size]
        shard_path = args.output_root / f"blind_review_shard_{shard_index:02d}.jsonl"
        result_path = args.output_root / f"blind_review_results_shard_{shard_index:02d}.jsonl"
        write_jsonl(shard_path, shard)
        write_markdown(args.output_root / f"BLIND_REVIEW_SHARD_{shard_index:02d}.md", shard)
        shard_paths.append(str(shard_path))
        expected_output_paths.append(str(result_path))
    write_subagent_prompt(args.output_root / "SUBAGENT_BLIND_AB_REVIEW_PROMPT.md", row_count, len(shard_paths))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "scripts/build_cross_runtime_blind_ab_packet.py",
        "source": str(args.source),
        "left_generations": str(args.left_generations),
        "right_generations": str(args.right_generations),
        "left_label": args.left_label,
        "right_label": args.right_label,
        "output_root": str(args.output_root),
        "review_path": str(review_path),
        "key_path": str(key_path),
        "shard_paths": shard_paths,
        "expected_output_paths": expected_output_paths,
        "row_count": row_count,
        "shard_size": args.shard_size,
        "seed": args.seed,
        "matching_output_count": sum(1 for row in unblinded_rows if row["outputs_match"]),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[cross-runtime-blind-ab] wrote {review_path}")
    print(f"[cross-runtime-blind-ab] wrote {key_path}")
    print(f"[cross-runtime-blind-ab] wrote {len(shard_paths)} shards")


if __name__ == "__main__":
    main()
