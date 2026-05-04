#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from faithfulness_cleanup_common import (
    DEFAULT_SEED,
    counter_dict,
    grouped_by_umbrella,
    label_mentions,
    largest_remainder_allocations,
    load_jsonl,
    local_filter_flags,
    numeric_mentions,
    resolve_training_rows,
    source_supports_token,
    stats_for_rows,
    summary_flags,
    umbrella,
    utc_now_iso,
    write_json,
    write_jsonl,
)

DEFAULT_SOURCE_RELEASE = "hf_dataset/releases/recent_3y_balanced_10746_v1"
DEFAULT_OUTPUT_ROOT = (
    "hf_bucket/experiments/faithfulness_cleanup/"
    "recent_3y_balanced_10746_faithfulness_overlay_1400_v2"
)
DEFAULT_EXCLUDE_REVIEW_ITEMS = (
    "hf_bucket/experiments/faithfulness_cleanup/"
    "recent_3y_balanced_10746_faithfulness_overlay_800_v1/manual_review/review_items.jsonl"
)
DEFAULT_HOLDOUT_JSONL = (
    "hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_v1/"
    "review_samples/holdout_100_20260419-075505/generations.jsonl"
)
DEFAULT_CANDIDATE_TARGET = 800
DEFAULT_FINAL_TARGET = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select supported-specificity candidates for overlay expansion."
    )
    parser.add_argument("--source-release", default=DEFAULT_SOURCE_RELEASE)
    parser.add_argument(
        "--output-dir",
        default=str(Path(DEFAULT_OUTPUT_ROOT) / "specificity_candidate_selection"),
    )
    parser.add_argument("--exclude-review-items", default=DEFAULT_EXCLUDE_REVIEW_ITEMS)
    parser.add_argument("--holdout-jsonl", default=DEFAULT_HOLDOUT_JSONL)
    parser.add_argument("--candidate-target", type=int, default=DEFAULT_CANDIDATE_TARGET)
    parser.add_argument("--final-target", type=int, default=DEFAULT_FINAL_TARGET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def load_paper_ids(path_value: str | None) -> set[str]:
    if not path_value:
        return set()
    path = Path(path_value)
    if not path.exists():
        return set()
    return {str(row.get("paper_id", "")).strip() for row in load_jsonl(path) if row.get("paper_id")}


def specificity_profile(row: dict[str, Any]) -> dict[str, Any]:
    summary = str(row.get("summary", "")).strip()
    source_text = " ".join([str(row.get("title", "")), str(row.get("abstract", ""))])
    supported_numbers: list[str] = []
    unsupported_numbers: list[str] = []
    for mention in numeric_mentions(summary):
        raw = mention["raw"]
        canonical = mention["canonical"]
        if source_supports_token(source_text, raw, canonical):
            supported_numbers.append(raw)
        else:
            unsupported_numbers.append(raw)

    supported_labels: list[str] = []
    unsupported_labels: list[str] = []
    for label in label_mentions(summary):
        if source_supports_token(source_text, label):
            supported_labels.append(label)
        else:
            unsupported_labels.append(label)

    flags: list[str] = []
    if supported_numbers:
        flags.append("supported_number")
    if supported_labels:
        flags.append("supported_label")
    if unsupported_numbers:
        flags.append("unsupported_number")
    if unsupported_labels:
        flags.append("unsupported_label")

    score = 0
    score += min(len(supported_numbers), 3) * 4
    score += min(len(supported_labels), 4) * 2
    score += 2 if supported_numbers and supported_labels else 0
    score += 1 if unsupported_numbers or unsupported_labels else 0
    score -= 2 * len(local_filter_flags(summary, str(row.get("title", ""))))

    return {
        "specificity_score": score,
        "specificity_flags": flags,
        "supported_numbers": supported_numbers,
        "supported_labels": supported_labels,
        "unsupported_numbers": unsupported_numbers,
        "unsupported_labels": unsupported_labels,
        "summary_flags": summary_flags(summary, str(row.get("title", ""))),
        "local_filter_flags": local_filter_flags(summary, str(row.get("title", ""))),
    }


def build_scored_row(row: dict[str, Any], *, source_release: str) -> dict[str, Any]:
    profile = specificity_profile(row)
    return {
        "paper_id": str(row.get("paper_id", "")),
        "title": str(row.get("title", "")),
        "abstract": str(row.get("abstract", "")),
        "original_summary": str(row.get("summary", "")).strip(),
        "summary": str(row.get("summary", "")).strip(),
        "primary_category": str(row.get("primary_category", "")),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date"),
        "source_split": row.get("source_split"),
        "teacher_model": row.get("teacher_model"),
        "generator_prompt_version": row.get("generator_prompt_version"),
        "review_prompt_version": row.get("review_prompt_version"),
        "source_release": source_release,
        **profile,
    }


def candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    flags = set(row.get("specificity_flags", []))
    return (
        -int(row.get("specificity_score", 0)),
        0 if "supported_number" in flags else 1,
        0 if "supported_label" in flags else 1,
        len(row.get("local_filter_flags", [])),
        str(row.get("primary_category", "")),
        str(row.get("paper_id", "")),
    )


def select_candidates(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row.get("specificity_score", 0) > 0]
    grouped = grouped_by_umbrella(eligible)
    allocations = largest_remainder_allocations(
        {key: len(value) for key, value in grouped.items()},
        target,
    )
    selected: list[dict[str, Any]] = []
    for group_name in sorted(grouped):
        selected.extend(
            sorted(grouped[group_name], key=candidate_sort_key)[
                : allocations.get(group_name, 0)
            ]
        )
    return sorted(selected, key=candidate_sort_key)


def assign_candidate_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        copied = dict(row)
        copied["candidate_id"] = f"specificity_preserve:{index:04d}"
        copied["selection_rank"] = index
        copied["selection_reason"] = "supported_specificity_stratified"
        output.append(copied)
    return output


def main() -> None:
    args = parse_args()
    source_path = resolve_training_rows(Path(args.source_release))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded_review_ids = load_paper_ids(args.exclude_review_items)
    holdout_ids = load_paper_ids(args.holdout_jsonl)
    excluded_ids = excluded_review_ids | holdout_ids

    source_rows = load_jsonl(source_path)
    scored_rows = []
    excluded_counts: Counter[str] = Counter()
    for row in source_rows:
        paper_id = str(row.get("paper_id", ""))
        if paper_id in holdout_ids:
            excluded_counts["fixed_holdout"] += 1
            continue
        if paper_id in excluded_review_ids:
            excluded_counts["existing_cleanup_review_pool"] += 1
            continue
        scored_rows.append(build_scored_row(row, source_release=args.source_release))

    selected_rows = assign_candidate_ids(select_candidates(scored_rows, args.candidate_target))
    if len(selected_rows) != args.candidate_target:
        raise ValueError(
            f"Selected {len(selected_rows)} specificity candidates; expected {args.candidate_target}"
        )

    duplicate_selected_ids = [
        paper_id
        for paper_id, count in Counter(row["paper_id"] for row in selected_rows).items()
        if count > 1
    ]
    if duplicate_selected_ids:
        raise ValueError(f"Duplicate selected paper IDs: {duplicate_selected_ids[:10]}")
    overlap = sorted({row["paper_id"] for row in selected_rows} & excluded_ids)
    if overlap:
        raise ValueError(f"Selected rows overlap excluded IDs: {overlap[:10]}")

    write_jsonl(output_dir / "all_scored_rows.jsonl", scored_rows)
    write_jsonl(output_dir / "selected_candidates.jsonl", selected_rows)

    flag_counts: Counter[str] = Counter()
    for row in selected_rows:
        flag_counts.update(row.get("specificity_flags", []))

    manifest = {
        "created_at": utc_now_iso(),
        "source_release": args.source_release,
        "source_training_rows_path": str(source_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "candidate_target": args.candidate_target,
        "final_target": args.final_target,
        "source_rows": len(source_rows),
        "scored_rows_after_exclusion": len(scored_rows),
        "selected_candidates": len(selected_rows),
        "excluded_counts": dict(sorted(excluded_counts.items())),
        "excluded_review_items_path": args.exclude_review_items,
        "holdout_jsonl": args.holdout_jsonl,
        "holdout_excluded_ids": len(holdout_ids),
        "selected_flag_counts": dict(sorted(flag_counts.items())),
        "selected_stats": stats_for_rows(selected_rows),
        "selected_umbrella_counts": counter_dict([umbrella(row) for row in selected_rows]),
        "selected_primary_category_top30": dict(
            Counter(row.get("primary_category", "") for row in selected_rows).most_common(30)
        ),
        "duplicate_selected_paper_ids": duplicate_selected_ids,
        "selection_policy": {
            "requires_supported_number_or_label": True,
            "exclude_existing_cleanup_review_pool": True,
            "exclude_fixed_holdout": True,
            "allocation": "largest_remainder_by_umbrella_then_specificity_score",
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "README.md").write_text(
        "# Specificity-Preserving Candidate Selection\n\n"
        "Selects rows whose existing summaries contain directly supported numbers "
        "or labels, excluding existing cleanup-reviewed papers and fixed-holdout IDs.\n",
        encoding="utf-8",
    )
    print(f"[specificity-candidates] source rows: {len(source_rows)}")
    print(f"[specificity-candidates] scored rows after exclusion: {len(scored_rows)}")
    print(f"[specificity-candidates] selected candidates: {len(selected_rows)}")
    print(f"[specificity-candidates] wrote {output_dir / 'selected_candidates.jsonl'}")


if __name__ == "__main__":
    main()
