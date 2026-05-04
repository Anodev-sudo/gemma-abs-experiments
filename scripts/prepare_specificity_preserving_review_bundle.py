#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

from faithfulness_cleanup_common import (
    DEFAULT_SEED,
    load_jsonl,
    stats_for_rows,
    utc_now_iso,
    write_decision_tsv,
    write_json,
    write_jsonl,
)

DEFAULT_ROOT = (
    "hf_bucket/experiments/faithfulness_cleanup/"
    "recent_3y_balanced_10746_faithfulness_overlay_1400_v2"
)
DEFAULT_QA_OVERLAP_RATIO = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare manual-agent review bundles for specificity rewrites."
    )
    parser.add_argument(
        "--rewrites-jsonl",
        default=str(Path(DEFAULT_ROOT) / "specificity_rewrite_run" / "specificity_rewrites_ordered.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(DEFAULT_ROOT) / "specificity_manual_review"),
    )
    parser.add_argument("--primary-reviewers", type=int, default=4)
    parser.add_argument("--qa-overlap-ratio", type=float, default=DEFAULT_QA_OVERLAP_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def build_review_item(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "review_id": f"specificity_review:{index:04d}",
        "candidate_id": row["candidate_id"],
        "paper_id": row["paper_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "primary_category": row.get("primary_category"),
        "created_date": row.get("created_date"),
        "specificity_score": row.get("specificity_score"),
        "specificity_flags": row.get("specificity_flags", []),
        "supported_numbers": row.get("supported_numbers", []),
        "supported_labels": row.get("supported_labels", []),
        "unsupported_numbers": row.get("unsupported_numbers", []),
        "unsupported_labels": row.get("unsupported_labels", []),
        "original_summary": row.get("original_summary", ""),
        "draft_decision": row.get("specificity_decision", ""),
        "draft_summary": str(row.get("specificity_summary", "")).strip(),
        "draft_risk_tags": row.get("specificity_risk_tags", []),
        "draft_evidence_span": row.get("specificity_evidence_span", ""),
        "draft_reason_span": row.get("specificity_reason_span", ""),
        "draft_local_filter_flags": row.get("specificity_local_filter_flags", []),
        "specificity_prompt_version": row.get("specificity_prompt_version", ""),
        "specificity_model": row.get("specificity_model", ""),
        "review_instruction": (
            "Choose keep, edit, drop, or uncertain. For keep/edit, final_summary "
            "must preserve useful supported specificity while staying faithful."
        ),
    }


def split_primary(items: list[dict[str, Any]], reviewer_count: int) -> list[list[dict[str, Any]]]:
    if reviewer_count <= 0:
        raise ValueError("--primary-reviewers must be positive")
    assignments = [[] for _ in range(reviewer_count)]
    for index, item in enumerate(items):
        assignments[index % reviewer_count].append(item)
    return assignments


def qa_overlap_items(items: list[dict[str, Any]], *, ratio: float, seed: int) -> list[dict[str, Any]]:
    if ratio < 0 or ratio > 1:
        raise ValueError("--qa-overlap-ratio must be between 0 and 1")
    count = round(len(items) * ratio)
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:count], key=lambda row: row["review_id"])


def build_readme(manifest: dict[str, Any]) -> str:
    return (
        "# Specificity-Preserving Manual Review Bundle\n\n"
        "Review objective:\n"
        "- Keep or edit summaries that preserve directly supported numbers, labels, "
        "comparators, or mechanisms without hallucinating details.\n"
        "- Drop rows that become generic, reverse direction, or cannot produce a "
        "useful supported one-sentence feed summary.\n\n"
        "Decision schema in `decisions.tsv`:\n"
        "- `decision`: `keep`, `edit`, `drop`, or `uncertain`\n"
        "- `confidence`: `high`, `medium`, or `low`\n"
        "- `final_summary`: required for `keep` and `edit`; leave blank for drops\n"
        "- `issue_tags`: comma-separated short tags\n"
        "- `note`: one short sentence when useful\n\n"
        f"- Review items: `{manifest['review_items']}`\n"
        f"- Primary reviewer shards: `{manifest['primary_reviewer_count']}`\n"
        f"- QA overlap items: `{manifest['qa_overlap_items']}`\n"
    )


def main() -> None:
    args = parse_args()
    rewrites = load_jsonl(Path(args.rewrites_jsonl))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = [build_review_item(row, index) for index, row in enumerate(rewrites, start=1)]
    write_jsonl(output_dir / "review_items.jsonl", items)

    primary_assignments = split_primary(items, args.primary_reviewers)
    for index, assigned in enumerate(primary_assignments):
        shard_dir = output_dir / "primary" / f"agent_{index:02d}"
        write_jsonl(shard_dir / "review_items.jsonl", assigned)
        write_decision_tsv(shard_dir / "decisions.tsv", assigned)

    qa_items = qa_overlap_items(items, ratio=args.qa_overlap_ratio, seed=args.seed)
    qa_dir = output_dir / "qa" / "overlap"
    write_jsonl(qa_dir / "review_items.jsonl", qa_items)
    write_decision_tsv(qa_dir / "decisions.tsv", qa_items)

    manifest = {
        "created_at": utc_now_iso(),
        "rewrites_jsonl": str(args.rewrites_jsonl),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "review_items": len(items),
        "primary_reviewer_count": args.primary_reviewers,
        "primary_assignment_counts": {
            f"agent_{index:02d}": len(assigned)
            for index, assigned in enumerate(primary_assignments)
        },
        "qa_overlap_ratio": args.qa_overlap_ratio,
        "qa_overlap_items": len(qa_items),
        "draft_decision_counts": dict(
            sorted(Counter(str(row.get("draft_decision", "")) for row in items).items())
        ),
        "review_item_stats": stats_for_rows(
            [
                {
                    **row,
                    "summary": row.get("draft_summary") or row.get("original_summary", ""),
                    "risk_tier": "specificity",
                }
                for row in items
            ]
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "README.md").write_text(build_readme(manifest), encoding="utf-8")

    print(f"[specificity-review-bundle] review items: {len(items)}")
    print(f"[specificity-review-bundle] primary shards: {args.primary_reviewers}")
    print(f"[specificity-review-bundle] qa overlap: {len(qa_items)}")
    print(f"[specificity-review-bundle] wrote {output_dir}")


if __name__ == "__main__":
    main()
