#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from faithfulness_cleanup_common import (
    DEFAULT_CANDIDATE_TARGET,
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_FINAL_TARGET,
    DEFAULT_SEED,
    DEFAULT_SOURCE_RELEASE,
    counter_dict,
    grouped_by_umbrella,
    largest_remainder_allocations,
    load_jsonl,
    resolve_training_rows,
    risk_profile,
    risk_sort_key,
    stats_for_rows,
    umbrella,
    utc_now_iso,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic high-risk candidate pool for the "
            "faithfulness cleanup overlay pack."
        )
    )
    parser.add_argument(
        "--source-release",
        default=DEFAULT_SOURCE_RELEASE,
        help="Release directory or training_rows.jsonl to score.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(DEFAULT_EXPERIMENT_ROOT) / "candidate_selection"),
        help="Directory for candidate pool artifacts.",
    )
    parser.add_argument(
        "--candidate-target",
        type=int,
        default=DEFAULT_CANDIDATE_TARGET,
        help=(
            "Target selected candidate count. All high-risk rows are included even "
            "if this target is exceeded; medium-risk rows top up to the target."
        ),
    )
    parser.add_argument(
        "--final-target",
        type=int,
        default=DEFAULT_FINAL_TARGET,
        help="Final overlay size goal recorded in the manifest.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Selection seed recorded for reproducibility.",
    )
    return parser.parse_args()


def build_scored_row(
    row: dict[str, Any],
    *,
    source_release: str,
) -> dict[str, Any]:
    profile = risk_profile(row)
    return {
        "paper_id": str(row.get("paper_id", "")),
        "title": str(row.get("title", "")),
        "abstract": str(row.get("abstract", "")),
        "original_summary": str(row.get("summary", "")).strip(),
        "primary_category": str(row.get("primary_category", "")),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date"),
        "source_split": row.get("source_split"),
        "teacher_model": row.get("teacher_model"),
        "generator_prompt_version": row.get("generator_prompt_version"),
        "review_prompt_version": row.get("review_prompt_version"),
        "review_decision": row.get("review_decision"),
        "review_source": row.get("review_source"),
        "source_release": source_release,
        **profile,
    }


def select_medium_topup(
    medium_rows: list[dict[str, Any]],
    needed: int,
) -> list[dict[str, Any]]:
    if needed <= 0:
        return []
    grouped = grouped_by_umbrella(medium_rows)
    counts = {key: len(value) for key, value in grouped.items()}
    allocations = largest_remainder_allocations(counts, needed)
    selected: list[dict[str, Any]] = []
    for group_name in sorted(grouped):
        group_rows = sorted(grouped[group_name], key=risk_sort_key)
        selected.extend(group_rows[: allocations.get(group_name, 0)])
    return selected


def assign_candidate_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(sorted(rows, key=risk_sort_key), start=1):
        copied = dict(row)
        copied["candidate_id"] = f"faithful_cleanup:{index:04d}"
        copied["selection_rank"] = index
        copied["selection_reason"] = (
            "high_risk_all_in"
            if copied["risk_tier"] == "high"
            else "medium_risk_stratified_topup"
        )
        output.append(copied)
    return output


def build_readme(manifest: dict[str, Any]) -> str:
    return (
        "# Faithfulness Cleanup Candidate Selection\n\n"
        "Purpose:\n"
        "- Select high-risk summaries from the 10.7k release for a targeted "
        "faithfulness cleanup overlay.\n"
        "- Heuristics are recall-oriented only; final inclusion requires drafted "
        "cleanup plus manual agent review.\n\n"
        "Selection policy:\n"
        "- Include every high-risk row.\n"
        "- Top up with a deterministic stratified sample of medium-risk rows.\n"
        "- Sort selected candidates by risk tier, risk score, and `paper_id`.\n\n"
        "Key counts:\n"
        f"- Source rows: `{manifest['source_rows']}`\n"
        f"- Selected candidates: `{manifest['selected_candidates']}`\n"
        f"- Final overlay target: `{manifest['final_target']}`\n\n"
        "Files:\n"
        "- `selected_candidates.jsonl`: candidate pool for the cleanup rewriter\n"
        "- `all_scored_rows.jsonl`: scored source rows for audit\n"
        "- `by_tier/*.jsonl`: selected candidates grouped by risk tier\n"
        "- `manifest.json`: reproducibility metadata and counts\n"
    )


def main() -> None:
    args = parse_args()
    source_path = resolve_training_rows(Path(args.source_release))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = load_jsonl(source_path)
    scored_rows = [
        build_scored_row(row, source_release=str(args.source_release))
        for row in source_rows
    ]
    tier_counts = Counter(row["risk_tier"] for row in scored_rows)
    high_rows = [row for row in scored_rows if row["risk_tier"] == "high"]
    medium_rows = [row for row in scored_rows if row["risk_tier"] == "medium"]
    medium_needed = max(0, args.candidate_target - len(high_rows))
    medium_selected = select_medium_topup(medium_rows, medium_needed)
    selected_rows = assign_candidate_ids(high_rows + medium_selected)

    selected_ids = {row["paper_id"] for row in selected_rows}
    duplicate_selected_ids = [
        paper_id
        for paper_id, count in Counter(row["paper_id"] for row in selected_rows).items()
        if count > 1
    ]
    if duplicate_selected_ids:
        raise ValueError(
            "Duplicate selected paper_id values: "
            + ", ".join(sorted(duplicate_selected_ids)[:20])
        )

    write_jsonl(output_dir / "all_scored_rows.jsonl", scored_rows)
    write_jsonl(output_dir / "selected_candidates.jsonl", selected_rows)
    by_tier_dir = output_dir / "by_tier"
    for tier in ("high", "medium"):
        write_jsonl(
            by_tier_dir / f"{tier}.jsonl",
            [row for row in selected_rows if row["risk_tier"] == tier],
        )

    flag_counts: Counter[str] = Counter()
    for row in selected_rows:
        flag_counts.update(row.get("risk_flags", []))

    manifest: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "source_release": str(args.source_release),
        "source_training_rows_path": str(source_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "candidate_target": args.candidate_target,
        "final_target": args.final_target,
        "source_rows": len(source_rows),
        "tier_counts_all_scored": dict(sorted(tier_counts.items())),
        "selected_candidates": len(selected_rows),
        "selected_high_risk": sum(row["risk_tier"] == "high" for row in selected_rows),
        "selected_medium_risk": sum(row["risk_tier"] == "medium" for row in selected_rows),
        "selected_stats": stats_for_rows(selected_rows),
        "selected_risk_flag_counts": dict(sorted(flag_counts.items())),
        "selected_umbrella_counts": counter_dict([umbrella(row) for row in selected_rows]),
        "selected_primary_category_top30": dict(
            Counter(row.get("primary_category", "") for row in selected_rows).most_common(30)
        ),
        "selection_policy": {
            "include_all_high_risk": True,
            "medium_topup_policy": "largest_remainder_by_umbrella_then_risk_sort",
            "candidate_id_policy": "risk_sort_order",
        },
        "duplicate_selected_paper_ids": duplicate_selected_ids,
        "selected_paper_id_count": len(selected_ids),
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "README.md").write_text(build_readme(manifest), encoding="utf-8")

    print(f"[faithfulness-candidates] source rows: {len(source_rows)}")
    print(f"[faithfulness-candidates] tier counts: {json.dumps(dict(sorted(tier_counts.items())), sort_keys=True)}")
    print(f"[faithfulness-candidates] selected candidates: {len(selected_rows)}")
    print(f"[faithfulness-candidates] wrote {output_dir / 'selected_candidates.jsonl'}")


if __name__ == "__main__":
    main()
