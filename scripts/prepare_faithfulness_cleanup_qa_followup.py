#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from faithfulness_cleanup_common import (
    CONFIDENCE_VALUES,
    REVIEW_DECISIONS,
    load_jsonl,
    utc_now_iso,
    write_decision_tsv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a QA follow-up shard for low-confidence primary keeps/edits."
        )
    )
    parser.add_argument("--review-bundle-dir", required=True)
    parser.add_argument(
        "--output-name",
        default="followup_low_confidence",
        help="Subdirectory name under qa/ for the follow-up shard.",
    )
    return parser.parse_args()


def read_decision_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_primary_decisions(bundle_dir: Path) -> dict[str, dict[str, str]]:
    decisions: dict[str, dict[str, str]] = {}
    for path in sorted((bundle_dir / "primary").glob("agent_*/decisions.tsv")):
        for row in read_decision_tsv(path):
            review_id = str(row.get("review_id", "")).strip()
            if not review_id:
                continue
            copied = dict(row)
            copied["_decision_file"] = str(path)
            decisions[review_id] = copied
    return decisions


def normalize_decision(row: dict[str, str]) -> tuple[str, str, str]:
    decision = str(row.get("decision", "")).strip().lower()
    confidence = str(row.get("confidence", "")).strip().lower()
    final_summary = " ".join(str(row.get("final_summary", "")).split())
    if decision not in REVIEW_DECISIONS:
        decision = ""
    if confidence not in CONFIDENCE_VALUES:
        confidence = ""
    return decision, confidence, final_summary


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.review_bundle_dir)
    items = load_jsonl(bundle_dir / "review_items.jsonl")
    items_by_id = {item["review_id"]: item for item in items}
    primary_decisions = load_primary_decisions(bundle_dir)

    followup_items: list[dict[str, Any]] = []
    for review_id, decision_row in sorted(primary_decisions.items()):
        decision, confidence, final_summary = normalize_decision(decision_row)
        if decision not in {"keep", "edit"}:
            continue
        if confidence != "low":
            continue
        item = dict(items_by_id[review_id])
        item["primary_low_confidence_summary"] = final_summary
        item["primary_decision_file"] = decision_row.get("_decision_file", "")
        followup_items.append(item)

    output_dir = bundle_dir / "qa" / args.output_name
    write_jsonl(output_dir / "review_items.jsonl", followup_items)
    write_decision_tsv(output_dir / "decisions.tsv", followup_items)
    manifest = {
        "created_at": utc_now_iso(),
        "review_bundle_dir": str(bundle_dir),
        "output_dir": str(output_dir),
        "followup_policy": "primary keep/edit with low confidence",
        "followup_items": len(followup_items),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(f"[faithfulness-qa-followup] wrote {len(followup_items)} items to {output_dir}")


if __name__ == "__main__":
    main()
