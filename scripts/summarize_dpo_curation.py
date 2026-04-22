#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

DEFAULT_BUNDLE_DIR = (
    "hf_bucket/experiments/dpo/curation/"
    "title_abstract_preference_balanced200_pool_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize manual DPO curation decisions from a bundle directory."
    )
    parser.add_argument(
        "--bundle-dir",
        default=DEFAULT_BUNDLE_DIR,
        help="Bundle directory that contains manifest.json and decisions/*.tsv.",
    )
    return parser.parse_args()


def load_jsonl_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    decisions_dir = bundle_dir / "decisions"
    by_axis_dir = bundle_dir / "by_axis"

    summary: dict[str, dict[str, int]] = {}
    for axis in sorted(manifest["axis_counts"]):
        decision_path = decisions_dir / f"{axis}.tsv"
        reviewed = 0
        keep = 0
        drop = 0
        a_tier_keep = 0

        with decision_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                candidate_id = str(row.get("candidate_id", "")).strip()
                if not candidate_id:
                    continue
                reviewed += 1
                decision = str(row.get("decision", "")).strip().lower()
                if decision == "keep":
                    keep += 1
                    if str(row.get("priority_tier", "")).strip().upper() == "A":
                        a_tier_keep += 1
                elif decision == "drop":
                    drop += 1

        total_candidates = load_jsonl_count(by_axis_dir / f"{axis}.jsonl")
        summary[axis] = {
            "total_candidates": total_candidates,
            "reviewed": reviewed,
            "unreviewed": total_candidates - reviewed,
            "keep": keep,
            "drop": drop,
            "a_tier_keep": a_tier_keep,
            "target_keep_count": int(manifest["axis_targets"].get(axis, 0)),
        }

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
