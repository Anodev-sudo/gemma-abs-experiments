#!/usr/bin/env python3
"""Summarize and unblind q5 blind A/B review results."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


CHOICES = {"summary_a", "summary_b", "tie", "neither"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def pct(part: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{part / total:.1%}"


def model_for_choice(key: dict[str, Any], choice: str) -> str:
    if choice == "summary_a":
        return str(key["summary_a_model"])
    if choice == "summary_b":
        return str(key["summary_b_model"])
    return choice


def count_model_choices(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter: collections.Counter[str] = collections.Counter()
    for row in rows:
        counter[str(row[field])] += 1
    return dict(sorted(counter.items()))


def render_model_table(title: str, counts: dict[str, int], total: int) -> list[str]:
    lines = [f"## {title}", "", "| choice | count | share |", "| --- | ---: | ---: |"]
    for choice, count in counts.items():
        lines.append(f"| {choice} | {count} | {pct(count, total)} |")
    lines.append("")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize q5 blind A/B reviews.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key_path = Path(manifest["key_path"])
    review_path = Path(manifest["review_path"])
    if not key_path.exists():
        raise SystemExit(f"Missing blind key: {key_path}")
    if not review_path.exists():
        raise SystemExit(f"Missing blind review packet: {review_path}")

    review_rows = load_jsonl(review_path)
    expected_by_id = {str(row["review_id"]): row for row in review_rows}
    key_by_id = {str(row["review_id"]): row for row in load_jsonl(key_path)}

    result_paths = sorted(args.root.glob("blind_review_results_shard_*.jsonl"))
    if not result_paths:
        raise SystemExit(f"No blind review result files found under {args.root}")

    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    duplicate_ids: list[str] = []
    schema_errors: list[str] = []
    per_file: dict[str, dict[str, int]] = {}

    for path in result_paths:
        rows = load_jsonl(path)
        file_counter: collections.Counter[str] = collections.Counter()
        for idx, row in enumerate(rows, start=1):
            review_id = row.get("review_id")
            if not isinstance(review_id, str):
                schema_errors.append(f"{path.name}:{idx}: missing string review_id")
                continue
            if review_id in seen:
                duplicate_ids.append(review_id)
                continue
            seen.add(review_id)
            if review_id not in expected_by_id:
                schema_errors.append(f"{path.name}:{idx}: unexpected review_id {review_id}")
                continue
            preferred = row.get("preferred_summary")
            accurate = row.get("more_accurate_summary")
            if preferred not in CHOICES:
                schema_errors.append(f"{path.name}:{idx}: bad preferred_summary {preferred!r}")
                continue
            if accurate not in CHOICES:
                schema_errors.append(f"{path.name}:{idx}: bad more_accurate_summary {accurate!r}")
                continue
            notes = row.get("notes")
            if not isinstance(notes, str) or not notes.strip():
                schema_errors.append(f"{path.name}:{idx}: missing notes")
                continue
            key = key_by_id[review_id]
            packet = expected_by_id[review_id]
            preferred_model = model_for_choice(key, preferred)
            accurate_model = model_for_choice(key, accurate)
            merged_row = {
                **packet,
                "preferred_summary": preferred,
                "more_accurate_summary": accurate,
                "preferred_model": preferred_model,
                "more_accurate_model": accurate_model,
                "review_notes": notes.strip(),
                "summary_a_model": key["summary_a_model"],
                "summary_b_model": key["summary_b_model"],
                "split": key.get("split", ""),
            }
            merged.append(merged_row)
            file_counter[preferred_model] += 1
        per_file[path.name] = {"total": len(rows), **dict(sorted(file_counter.items()))}

    missing_ids = sorted(set(expected_by_id) - seen)
    merged.sort(key=lambda row: row["review_id"])
    write_jsonl(args.root / "blind_review_merged_unblinded.jsonl", merged)

    total = len(merged)
    preferred_counts = count_model_choices(merged, "preferred_model")
    accurate_counts = count_model_choices(merged, "more_accurate_model")

    split_rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in merged:
        split_rows[str(row.get("split", ""))].append(row)
    by_split = {
        split: {
            "preferred": count_model_choices(rows, "preferred_model"),
            "more_accurate": count_model_choices(rows, "more_accurate_model"),
            "total": {"rows": len(rows)},
        }
        for split, rows in sorted(split_rows.items())
    }

    summary = {
        "root": str(args.root),
        "expected_rows": len(expected_by_id),
        "reviewed_rows": total,
        "missing_ids": missing_ids,
        "duplicate_ids": duplicate_ids,
        "schema_errors": schema_errors,
        "preferred_model_counts": preferred_counts,
        "more_accurate_model_counts": accurate_counts,
        "result_files": per_file,
        "by_split": by_split,
    }
    (args.root / "blind_review_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# q5 Blind A/B Review Summary",
        "",
        f"- Expected rows: {len(expected_by_id)}",
        f"- Reviewed rows: {total}",
        f"- Missing IDs: {len(missing_ids)}",
        f"- Duplicate IDs: {len(duplicate_ids)}",
        f"- Schema errors: {len(schema_errors)}",
        "",
    ]
    lines.extend(render_model_table("Preferred Summary", preferred_counts, total))
    lines.extend(render_model_table("More Accurate Summary", accurate_counts, total))
    lines.extend(["## Result Files", "", "| file | rows | preferred counts |", "| --- | ---: | --- |"])
    for name, counts in sorted(per_file.items()):
        row_total = counts.pop("total")
        rendered = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        lines.append(f"| {name} | {row_total} | {rendered} |")
        counts["total"] = row_total
    lines.append("")
    lines.extend(["## By Split", ""])
    for split, split_summary in by_split.items():
        rows = split_summary["total"]["rows"]
        lines.append(f"### {split} ({rows})")
        preferred = ", ".join(f"{k}: {v}" for k, v in split_summary["preferred"].items())
        accurate = ", ".join(f"{k}: {v}" for k, v in split_summary["more_accurate"].items())
        lines.append(f"- Preferred: {preferred}")
        lines.append(f"- More accurate: {accurate}")
        lines.append("")
    if missing_ids:
        lines.extend(["## Missing IDs", "", ", ".join(missing_ids[:50]), ""])
    if duplicate_ids:
        lines.extend(["## Duplicate IDs", "", ", ".join(duplicate_ids[:50]), ""])
    if schema_errors:
        lines.extend(["## Schema Errors", ""])
        lines.extend(f"- {error}" for error in schema_errors[:100])
        if len(schema_errors) > 100:
            lines.append(f"- ... and {len(schema_errors) - 100} more")
        lines.append("")
    (args.root / "BLIND_REVIEW_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        "[blind-ab-summary] "
        f"reviewed={total} missing={len(missing_ids)} duplicates={len(duplicate_ids)} "
        f"schema_errors={len(schema_errors)} preferred={preferred_counts} accurate={accurate_counts}"
    )


if __name__ == "__main__":
    main()
