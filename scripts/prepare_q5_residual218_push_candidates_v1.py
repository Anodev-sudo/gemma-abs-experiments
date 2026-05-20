#!/usr/bin/env python3
"""Prepare reviewed q5-severe/q8-clean rows for a checkpoint-218 repair pass."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path(
    "hf_bucket/experiments/mlx_lora/q5_hardening_distill_6000_v1/"
    "q5_severe_q8_clean_review_rows.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "hf_bucket/experiments/sft/q5_residual218_push_285_v1/source_candidates"
)
DEFAULT_EXCLUDE_DATASET = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7/"
    "dataset_193repair_600selfanchor_77hard_v1/train.jsonl"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def base_paper_id(value: Any) -> str:
    return str(value or "").split("::", 1)[0]


def load_excluded_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            pid = base_paper_id(row.get("paper_id"))
            if pid:
                excluded.add(pid)
    return excluded


def normalize_candidate(row: dict[str, Any]) -> dict[str, Any]:
    bucket = row.get("selection_bucket", "")
    return {
        "paper_id": row.get("paper_id", ""),
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "primary_category": row.get("primary_category", ""),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date", ""),
        "q5_summary": row.get("q5_summary", ""),
        "q8_summary": row.get("q8_summary", ""),
        "q5_issue_tags": row.get("q5_issue_tags", []),
        "q5_hardening_axes": row.get("q5_hardening_axes", []),
        "q5_hardening_review_id": row.get("review_id", ""),
        "q5_hardening_selection_bucket": bucket,
        "selection_bucket": bucket,
        "q5_severity_source": "q5_hardening_6000_blind_review:q5_severe_q8_clean",
        "reviewer_note": row.get("review_note", ""),
        "q8_clean": True,
        "q8_severe": False,
        "q8_mangled": False,
        "source_split": "q5_residual218_push_candidate_v1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build source candidates for the q5 residual-218 push repair pipeline."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--exclude-dataset",
        type=Path,
        action="append",
        default=[DEFAULT_EXCLUDE_DATASET],
        help="JSONL dataset whose paper IDs should be excluded; repeatable.",
    )
    parser.add_argument(
        "--include-overlap",
        action="store_true",
        help="Keep rows whose paper IDs already appear in excluded datasets.",
    )
    args = parser.parse_args()

    source_rows = load_jsonl(args.source)
    excluded_ids = set() if args.include_overlap else load_excluded_ids(args.exclude_dataset)
    rows: list[dict[str, Any]] = []
    skipped_overlap: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate_ids: list[str] = []

    for source_row in source_rows:
        pid = base_paper_id(source_row.get("paper_id"))
        if not pid:
            continue
        if pid in seen:
            duplicate_ids.append(pid)
            continue
        seen.add(pid)
        candidate = normalize_candidate(source_row)
        if pid in excluded_ids:
            skipped_overlap.append(candidate)
            continue
        rows.append(candidate)

    output_dir = args.output_dir
    output_path = output_dir / "q5_residual218_push_candidates.jsonl"
    overlap_path = output_dir / "excluded_checkpoint218_training_overlap.jsonl"
    write_jsonl(output_path, rows)
    write_jsonl(overlap_path, skipped_overlap)

    manifest = {
        "source": str(args.source),
        "output_jsonl": str(output_path),
        "excluded_overlap_jsonl": str(overlap_path),
        "source_rows": len(source_rows),
        "candidate_rows": len(rows),
        "excluded_checkpoint218_training_overlap": len(skipped_overlap),
        "duplicate_source_ids": duplicate_ids,
        "bucket_counts": dict(Counter(row["q5_hardening_selection_bucket"] for row in rows)),
        "issue_tag_counts": dict(
            Counter(tag for row in rows for tag in row.get("q5_issue_tags", []))
        ),
        "axis_counts": dict(
            Counter(axis for row in rows for axis in row.get("q5_hardening_axes", []))
        ),
        "notes": [
            "Rows are reviewed q5-severe/q8-clean cases from the 6000-row hardening mine.",
            "The q8 summary is only a hint for the rewriter, not the final training target.",
            "Default behavior excludes paper IDs already used by the checkpoint-218 training dataset.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
