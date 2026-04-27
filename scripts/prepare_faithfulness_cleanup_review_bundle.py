#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from faithfulness_cleanup_common import (
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_QA_OVERLAP_RATIO,
    DEFAULT_SEED,
    load_jsonl,
    stats_for_rows,
    utc_now_iso,
    write_decision_tsv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare manual-agent review bundles for faithfulness cleanup rewrites."
    )
    parser.add_argument(
        "--rewrites-jsonl",
        default=str(Path(DEFAULT_EXPERIMENT_ROOT) / "rewrite_run" / "cleanup_rewrites_ordered.jsonl"),
        help="Ordered cleanup rewrite outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(DEFAULT_EXPERIMENT_ROOT) / "manual_review"),
        help="Manual review bundle output directory.",
    )
    parser.add_argument("--primary-reviewers", type=int, default=4)
    parser.add_argument("--qa-overlap-ratio", type=float, default=DEFAULT_QA_OVERLAP_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def build_review_item(row: dict[str, Any], index: int) -> dict[str, Any]:
    draft_summary = str(row.get("cleanup_summary", "")).strip()
    return {
        "review_id": f"faithfulness_review:{index:04d}",
        "candidate_id": row["candidate_id"],
        "paper_id": row["paper_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "primary_category": row.get("primary_category"),
        "created_date": row.get("created_date"),
        "risk_tier": row.get("risk_tier"),
        "risk_score": row.get("risk_score"),
        "heuristic_risk_flags": row.get("heuristic_risk_flags", []),
        "heuristic_unsupported_numbers": row.get("heuristic_unsupported_numbers", []),
        "heuristic_unsupported_labels": row.get("heuristic_unsupported_labels", []),
        "original_summary": row.get("original_summary", ""),
        "draft_decision": row.get("cleanup_decision", ""),
        "draft_summary": draft_summary,
        "draft_risk_tags": row.get("cleanup_risk_tags", []),
        "draft_evidence_span": row.get("cleanup_evidence_span", ""),
        "draft_reason_span": row.get("cleanup_reason_span", ""),
        "draft_local_filter_flags": row.get("cleanup_local_filter_flags", []),
        "cleanup_prompt_version": row.get("cleanup_prompt_version", ""),
        "cleanup_model": row.get("cleanup_model", ""),
        "review_instruction": (
            "Choose keep, edit, drop, or uncertain. For keep/edit, final_summary "
            "must be one faithful feed sentence directly supported by title/abstract."
        ),
    }


def split_primary(items: list[dict[str, Any]], reviewer_count: int) -> list[list[dict[str, Any]]]:
    if reviewer_count <= 0:
        raise ValueError("--primary-reviewers must be positive")
    assignments = [[] for _ in range(reviewer_count)]
    for index, item in enumerate(items):
        assignments[index % reviewer_count].append(item)
    return assignments


def qa_overlap_items(
    items: list[dict[str, Any]],
    *,
    ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    if ratio < 0 or ratio > 1:
        raise ValueError("--qa-overlap-ratio must be between 0 and 1")
    count = round(len(items) * ratio)
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:count], key=lambda row: row["review_id"])


def build_readme(manifest: dict[str, Any]) -> str:
    return (
        "# Faithfulness Cleanup Manual Review Bundle\n\n"
        "Review objective:\n"
        "- Keep or edit only summaries that are faithful, useful in a mobile feed, "
        "and directly supported by the title/abstract.\n"
        "- Drop summaries with unsupported numbers, mechanisms, entity labels, "
        "causal direction, overstrong claims, title rehashes, or bland rewrites.\n\n"
        "Decision schema in `decisions.tsv`:\n"
        "- `decision`: `keep`, `edit`, `drop`, or `uncertain`\n"
        "- `confidence`: `high`, `medium`, or `low`\n"
        "- `final_summary`: required for `keep` and `edit`; leave blank for drops\n"
        "- `issue_tags`: comma-separated short tags\n"
        "- `note`: one short sentence when useful\n\n"
        "Review process:\n"
        "- Primary reviewers cover every item exactly once.\n"
        "- QA overlap covers a deterministic sample of the full bundle.\n"
        "- Low-confidence keeps should be sent through QA before materialization.\n\n"
        "Key counts:\n"
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
                }
                for row in items
            ]
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "README.md").write_text(build_readme(manifest), encoding="utf-8")

    print(f"[faithfulness-review-bundle] review items: {len(items)}")
    print(f"[faithfulness-review-bundle] primary shards: {args.primary_reviewers}")
    print(f"[faithfulness-review-bundle] qa overlap: {len(qa_items)}")
    print(f"[faithfulness-review-bundle] wrote {output_dir}")


if __name__ == "__main__":
    main()
