#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from faithfulness_cleanup_common import load_jsonl, stats_for_rows, utc_now_iso, write_json, write_jsonl
from materialize_faithfulness_cleanup_overlay import (
    build_selected_candidate as build_cleanup_candidate,
    load_primary_decisions,
    load_qa_decisions,
    selection_sort_key as cleanup_sort_key,
)

DEFAULT_CLEANUP_BUNDLE = (
    "hf_bucket/experiments/faithfulness_cleanup/"
    "recent_3y_balanced_10746_faithfulness_overlay_800_v1/manual_review"
)
DEFAULT_SPECIFICITY_RELEASE = (
    "hf_dataset/releases/recent_3y_balanced_10746_specificity_preserving_400_v1"
)
DEFAULT_HOLDOUT_JSONL = (
    "hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_v1/"
    "review_samples/holdout_100_20260419-075505/generations.jsonl"
)
DEFAULT_OUTPUT_RELEASE = (
    "hf_dataset/releases/recent_3y_balanced_10746_faithfulness_overlay_1400_v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the 1,400-row faithfulness overlay v2 release."
    )
    parser.add_argument("--cleanup-review-bundle-dir", default=DEFAULT_CLEANUP_BUNDLE)
    parser.add_argument("--specificity-release-dir", default=DEFAULT_SPECIFICITY_RELEASE)
    parser.add_argument("--holdout-jsonl", default=DEFAULT_HOLDOUT_JSONL)
    parser.add_argument("--output-release-dir", default=DEFAULT_OUTPUT_RELEASE)
    parser.add_argument("--cleanup-target-rows", type=int, default=1000)
    parser.add_argument("--specificity-target-rows", type=int, default=400)
    return parser.parse_args()


def load_paper_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("paper_id", "")).strip() for row in load_jsonl(path) if row.get("paper_id")}


def cleanup_rows(bundle_dir: Path, holdout_ids: set[str], target: int) -> list[dict[str, Any]]:
    items = load_jsonl(bundle_dir / "review_items.jsonl")
    primary_decisions = load_primary_decisions(bundle_dir)
    qa_decisions = load_qa_decisions(bundle_dir)
    rows: list[dict[str, Any]] = []
    for item in items:
        selected, _ = build_cleanup_candidate(
            item,
            primary_decisions[item["review_id"]],
            qa_decisions.get(item["review_id"]),
            allow_low_confidence_without_qa=False,
        )
        if selected is None:
            continue
        if selected["paper_id"] in holdout_ids:
            continue
        selected["overlay_component"] = "faithfulness_cleanup"
        selected["source_split"] = "faithfulness_overlay_1400_v2"
        rows.append(selected)
    rows.sort(key=cleanup_sort_key)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if row["paper_id"] in seen:
            continue
        seen.add(row["paper_id"])
        deduped.append(row)
    if len(deduped) < target:
        raise ValueError(f"Only {len(deduped)} cleanup rows available; need {target}")
    return deduped[:target]


def specificity_rows(release_dir: Path, holdout_ids: set[str], target: int) -> list[dict[str, Any]]:
    rows = load_jsonl(release_dir / "training_rows.jsonl")
    overlap = sorted({row["paper_id"] for row in rows} & holdout_ids)
    if overlap:
        raise ValueError(f"Specificity release overlaps fixed holdout: {overlap[:10]}")
    if len(rows) != target:
        raise ValueError(f"Specificity release has {len(rows)} rows; expected {target}")
    output = []
    for row in rows:
        copied = dict(row)
        copied["overlay_component"] = "specificity_preserving"
        copied["source_split"] = "faithfulness_overlay_1400_v2"
        output.append(copied)
    return output


def main() -> None:
    args = parse_args()
    holdout_ids = load_paper_ids(Path(args.holdout_jsonl))
    cleanup = cleanup_rows(
        Path(args.cleanup_review_bundle_dir),
        holdout_ids,
        args.cleanup_target_rows,
    )
    specificity = specificity_rows(
        Path(args.specificity_release_dir),
        holdout_ids,
        args.specificity_target_rows,
    )
    rows = cleanup + specificity
    expected_total = args.cleanup_target_rows + args.specificity_target_rows
    if len(rows) != expected_total:
        raise ValueError(f"Final row count {len(rows)} != {expected_total}")
    duplicate_ids = [
        paper_id
        for paper_id, count in Counter(row["paper_id"] for row in rows).items()
        if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"Duplicate paper IDs in final overlay: {duplicate_ids[:10]}")
    holdout_overlap = sorted({row["paper_id"] for row in rows} & holdout_ids)
    if holdout_overlap:
        raise ValueError(f"Final overlay overlaps fixed holdout: {holdout_overlap[:10]}")

    rows = sorted(rows, key=lambda row: (str(row.get("overlay_component", "")), str(row["paper_id"])))
    output_dir = Path(args.output_release_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "training_rows.jsonl", rows)
    manifest = {
        "created_at": utc_now_iso(),
        "release_name": output_dir.name,
        "final_rows": len(rows),
        "cleanup_rows": len(cleanup),
        "specificity_rows": len(specificity),
        "cleanup_review_bundle_dir": args.cleanup_review_bundle_dir,
        "specificity_release_dir": args.specificity_release_dir,
        "holdout_jsonl": args.holdout_jsonl,
        "holdout_excluded_ids": len(holdout_ids),
        "duplicate_paper_ids": duplicate_ids,
        "holdout_overlap": holdout_overlap,
        "component_counts": dict(Counter(row.get("overlay_component", "") for row in rows)),
        "final_stats": stats_for_rows(rows),
        "ordering": "overlay_component_then_paper_id",
        "policy": {
            "cleanup_component": "top reviewed survivors excluding fixed holdout",
            "specificity_component": "reviewed specificity-preserving rows",
            "fixed_holdout_excluded": True,
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(f"[overlay-1400] wrote {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
