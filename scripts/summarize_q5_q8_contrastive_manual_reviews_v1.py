#!/usr/bin/env python3
"""Validate and summarize q8-vs-q5 contrastive manual review shards."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "hf_bucket/experiments/sft/"
    "q5_promoted218_vs_q8_contrastive_mine_10000_v1/"
    "manual_review_hp_and_fog_union_gpt54mini"
)
REQUIRED = {
    "review_id",
    "paper_id",
    "actual_failure",
    "failure_type",
    "severity",
    "preferred",
    "note",
}
FAILURE_TYPES = {
    "invented_or_wrong_entity",
    "lost_main_result",
    "value_or_polarity_error",
    "semantic_fog",
    "malformed_terms",
    "digit_or_metric_issue",
    "not_a_failure",
    "both_bad_or_unclear",
}
SEVERITIES = {"none", "minor", "major"}
PREFERENCES = {"q8", "q5", "neither", "tie"}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize manual review shard results.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    return parser.parse_args()


def validate_result(row: dict[str, Any], *, path: Path) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED - set(row)
    if missing:
        errors.append(f"{path}: missing {sorted(missing)}")
    if not isinstance(row.get("actual_failure"), bool):
        errors.append(f"{path}: actual_failure must be boolean for {row.get('review_id')}")
    if row.get("failure_type") not in FAILURE_TYPES:
        errors.append(f"{path}: invalid failure_type for {row.get('review_id')}: {row.get('failure_type')}")
    if row.get("severity") not in SEVERITIES:
        errors.append(f"{path}: invalid severity for {row.get('review_id')}: {row.get('severity')}")
    if row.get("preferred") not in PREFERENCES:
        errors.append(f"{path}: invalid preferred for {row.get('review_id')}: {row.get('preferred')}")
    return errors


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected_rows = load_jsonl(root / "review_rows_union.jsonl")
    expected_by_id = {row["review_id"]: row for row in expected_rows}

    result_paths = sorted((root / "results").glob("review_results_shard_*.jsonl"))
    merged: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in result_paths:
        rows = load_jsonl(path)
        expected_shard = next(
            (
                shard
                for shard in manifest["shards"]
                if Path(shard["result_jsonl"]).name == path.name
            ),
            None,
        )
        if expected_shard is not None and len(rows) != expected_shard["rows"]:
            errors.append(f"{path}: row count {len(rows)} != expected {expected_shard['rows']}")
        for row in rows:
            errors.extend(validate_result(row, path=path))
            review_id = row.get("review_id")
            expected = expected_by_id.get(review_id)
            if expected is None:
                errors.append(f"{path}: unexpected review_id {review_id}")
                continue
            if str(row.get("paper_id")) != str(expected.get("paper_id")):
                errors.append(
                    f"{path}: paper_id mismatch for {review_id}: {row.get('paper_id')} != {expected.get('paper_id')}"
                )
                continue
            merged.append({**expected, **row})

    seen = [row["review_id"] for row in merged]
    duplicate_ids = [review_id for review_id, count in Counter(seen).items() if count > 1]
    if duplicate_ids:
        errors.append(f"duplicate review_ids: {duplicate_ids[:20]}")
    missing_ids = sorted(set(expected_by_id) - set(seen))
    if missing_ids:
        errors.append(f"missing review_ids: {missing_ids[:20]} count={len(missing_ids)}")

    merged.sort(key=lambda row: row["review_id"])
    write_jsonl(root / "manual_review_merged.jsonl", merged)

    by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    for row in merged:
        bucket = str(row.get("source_bucket"))
        by_bucket[bucket]["rows"] += 1
        by_bucket[bucket]["actual_failure"] += int(bool(row.get("actual_failure")))
        by_bucket[bucket][f"preferred_{row.get('preferred')}"] += 1
        by_bucket[bucket][f"severity_{row.get('severity')}"] += 1
        by_bucket[bucket][f"type_{row.get('failure_type')}"] += 1

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "expected_rows": len(expected_rows),
        "result_files": [str(path) for path in result_paths],
        "merged_rows": len(merged),
        "errors": errors,
        "overall": dict(Counter(
            {
                "rows": len(merged),
                "actual_failure": sum(bool(row.get("actual_failure")) for row in merged),
            }
        )),
        "bucket_summary": {bucket: dict(counter) for bucket, counter in sorted(by_bucket.items())},
    }
    (root / "manual_review_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# q8/q5 Contrastive Manual Review Summary",
        "",
        f"- Generated at: `{summary['created_at']}`",
        f"- Expected rows: `{len(expected_rows)}`",
        f"- Merged rows: `{len(merged)}`",
        f"- Validation errors: `{len(errors)}`",
        "",
        "## Overall",
        "",
        f"- Actual failures: `{summary['overall']['actual_failure']}`",
        "",
        "## By Bucket",
        "",
    ]
    for bucket, counts in sorted(by_bucket.items()):
        rows = counts["rows"]
        failures = counts["actual_failure"]
        rate = failures / rows if rows else 0.0
        lines.extend(
            [
                f"### {bucket}",
                "",
                f"- Rows: `{rows}`",
                f"- Actual failures: `{failures}` (`{rate:.1%}`)",
                f"- Preferred q8/q5/neither/tie: `{counts['preferred_q8']}` / `{counts['preferred_q5']}` / `{counts['preferred_neither']}` / `{counts['preferred_tie']}`",
                f"- Major/minor/none: `{counts['severity_major']}` / `{counts['severity_minor']}` / `{counts['severity_none']}`",
                "",
            ]
        )
    if errors:
        lines.extend(["## Validation Errors", ""])
        lines.extend(f"- {error}" for error in errors[:100])
    (root / "MANUAL_REVIEW_SUMMARY.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[manual-review-summary] merged {len(merged)}/{len(expected_rows)} rows")
    print(f"[manual-review-summary] validation errors: {len(errors)}")
    print(f"[manual-review-summary] summary: {root / 'MANUAL_REVIEW_SUMMARY.md'}")


if __name__ == "__main__":
    main()
