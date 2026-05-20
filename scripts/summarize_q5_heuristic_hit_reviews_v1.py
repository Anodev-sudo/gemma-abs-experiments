#!/usr/bin/env python3
"""Summarize binary reviews for q5 heuristic-hit candidates."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/"
    "eval/burnin150_style_probe_5000_v1/heuristic_hit_review_v1"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def bool_label(value: bool) -> str:
    return "actual_failure" if value else "acceptable_false_positive"


def pct(part: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{part / total:.1%}"


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in rows:
        label = str(row.get(key, ""))
        counts[label][bool_label(bool(row["actual_failure"]))] += 1
        counts[label]["total"] += 1
    return {name: dict(counter) for name, counter in sorted(counts.items())}


def count_categories(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in rows:
        categories = row.get("flag_categories") or {}
        for category, enabled in categories.items():
            if not enabled:
                continue
            counts[category][bool_label(bool(row["actual_failure"]))] += 1
            counts[category]["total"] += 1
    return {name: dict(counter) for name, counter in sorted(counts.items())}


def render_count_table(counts: dict[str, dict[str, int]], title: str) -> list[str]:
    lines = [f"## {title}", "", "| group | total | actual failures | acceptable | failure rate |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, counter in counts.items():
        total = int(counter.get("total", 0))
        failures = int(counter.get("actual_failure", 0))
        acceptable = int(counter.get("acceptable_false_positive", 0))
        lines.append(f"| {name} | {total} | {failures} | {acceptable} | {pct(failures, total)} |")
    lines.append("")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize q5 heuristic-hit binary reviews.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_items_path = args.root / "all_review_items.jsonl"
    if not all_items_path.exists():
        raise SystemExit(f"Missing review item file: {all_items_path}")

    items = load_jsonl(all_items_path)
    expected_by_id = {row["review_id"]: row for row in items}

    result_paths = sorted(args.root.glob("review_results_shard_*.jsonl"))
    if not result_paths:
        raise SystemExit(f"No review result files found under {args.root}")

    merged: list[dict[str, Any]] = []
    duplicate_ids: list[str] = []
    seen: set[str] = set()
    schema_errors: list[str] = []
    per_file: dict[str, dict[str, int]] = {}

    for path in result_paths:
        rows = load_jsonl(path)
        failures = 0
        acceptable = 0
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
            actual_failure = row.get("actual_failure")
            if not isinstance(actual_failure, bool):
                schema_errors.append(f"{path.name}:{idx}: actual_failure is not boolean")
                continue
            notes = row.get("notes")
            if not isinstance(notes, str) or not notes.strip():
                schema_errors.append(f"{path.name}:{idx}: missing notes")
                continue
            source = expected_by_id[review_id]
            merged_row = {
                **source,
                "actual_failure": actual_failure,
                "review_notes": notes.strip(),
            }
            merged.append(merged_row)
            if actual_failure:
                failures += 1
            else:
                acceptable += 1
        per_file[path.name] = {"total": len(rows), "actual_failure": failures, "acceptable_false_positive": acceptable}

    missing_ids = sorted(set(expected_by_id) - seen)
    merged.sort(key=lambda row: row["review_id"])
    write_jsonl(args.root / "binary_review_merged.jsonl", merged)

    total = len(merged)
    total_failures = sum(1 for row in merged if row["actual_failure"])
    total_acceptable = total - total_failures
    summary = {
        "root": str(args.root),
        "expected_rows": len(items),
        "reviewed_rows": total,
        "actual_failures": total_failures,
        "acceptable_false_positives": total_acceptable,
        "failure_rate": total_failures / total if total else 0.0,
        "result_files": per_file,
        "missing_ids": missing_ids,
        "duplicate_ids": duplicate_ids,
        "schema_errors": schema_errors,
        "by_split": count_by(merged, "split"),
        "by_flag_category": count_categories(merged),
    }
    (args.root / "binary_review_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# q5 Heuristic Hit Binary Review Summary",
        "",
        f"- Expected rows: {len(items)}",
        f"- Reviewed rows: {total}",
        f"- Actual failures: {total_failures} ({pct(total_failures, total)})",
        f"- Acceptable false positives: {total_acceptable} ({pct(total_acceptable, total)})",
        f"- Missing IDs: {len(missing_ids)}",
        f"- Duplicate IDs: {len(duplicate_ids)}",
        f"- Schema errors: {len(schema_errors)}",
        "",
        "## Result Files",
        "",
        "| file | rows | actual failures | acceptable |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, counter in sorted(per_file.items()):
        lines.append(
            f"| {name} | {counter['total']} | {counter['actual_failure']} | {counter['acceptable_false_positive']} |"
        )
    lines.append("")
    lines.extend(render_count_table(summary["by_split"], "By Split"))
    lines.extend(render_count_table(summary["by_flag_category"], "By Heuristic Category"))
    if missing_ids:
        lines.extend(["## Missing IDs", "", ", ".join(missing_ids[:50])])
        if len(missing_ids) > 50:
            lines.append(f"... and {len(missing_ids) - 50} more")
        lines.append("")
    if duplicate_ids:
        lines.extend(["## Duplicate IDs", "", ", ".join(duplicate_ids[:50]), ""])
    if schema_errors:
        lines.extend(["## Schema Errors", ""])
        lines.extend(f"- {error}" for error in schema_errors[:100])
        if len(schema_errors) > 100:
            lines.append(f"- ... and {len(schema_errors) - 100} more")
        lines.append("")

    (args.root / "BINARY_REVIEW_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        "[heuristic-hit-review-summary] "
        f"reviewed={total} failures={total_failures} acceptable={total_acceptable} "
        f"missing={len(missing_ids)} duplicates={len(duplicate_ids)} schema_errors={len(schema_errors)}"
    )


if __name__ == "__main__":
    main()
