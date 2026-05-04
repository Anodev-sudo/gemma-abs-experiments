#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from faithfulness_cleanup_common import (
    CONFIDENCE_VALUES,
    REVIEW_DECISIONS,
    load_jsonl,
    local_filter_flags,
    read_decision_tsv,
    stats_for_rows,
    utc_now_iso,
    write_json,
    write_jsonl,
)

DEFAULT_REVIEW_BUNDLE = (
    "hf_bucket/experiments/faithfulness_cleanup/"
    "recent_3y_balanced_10746_faithfulness_overlay_1400_v2/specificity_manual_review"
)
DEFAULT_OUTPUT_RELEASE = (
    "hf_dataset/releases/recent_3y_balanced_10746_specificity_preserving_400_v1"
)
DEFAULT_TARGET_ROWS = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize reviewed specificity-preserving overlay rows."
    )
    parser.add_argument("--review-bundle-dir", default=DEFAULT_REVIEW_BUNDLE)
    parser.add_argument("--output-release-dir", default=DEFAULT_OUTPUT_RELEASE)
    parser.add_argument("--target-rows", type=int, default=DEFAULT_TARGET_ROWS)
    parser.add_argument("--allow-underfilled", action="store_true")
    parser.add_argument("--allow-low-confidence-without-qa", action="store_true")
    return parser.parse_args()


def load_primary_decisions(bundle_dir: Path) -> dict[str, dict[str, str]]:
    decisions: dict[str, dict[str, str]] = {}
    for path in sorted((bundle_dir / "primary").glob("agent_*/decisions.tsv")):
        for row in read_decision_tsv(path):
            review_id = str(row.get("review_id", "")).strip()
            if not review_id:
                continue
            if review_id in decisions:
                raise ValueError(f"Duplicate primary decision for {review_id}")
            copied = dict(row)
            copied["_decision_file"] = str(path)
            decisions[review_id] = copied
    return decisions


def load_qa_decisions(bundle_dir: Path) -> dict[str, dict[str, str]]:
    decisions: dict[str, dict[str, str]] = {}
    qa_dir = bundle_dir / "qa"
    if not qa_dir.exists():
        return decisions
    for path in sorted(qa_dir.glob("*/decisions.tsv")):
        for row in read_decision_tsv(path):
            review_id = str(row.get("review_id", "")).strip()
            decision = str(row.get("decision", "")).strip().lower()
            if not review_id or not decision:
                continue
            if review_id in decisions:
                raise ValueError(f"Duplicate QA decision for {review_id}")
            copied = dict(row)
            copied["_decision_file"] = str(path)
            decisions[review_id] = copied
    return decisions


def normalize_decision(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    decision = str(row.get("decision", "")).strip().lower()
    confidence = str(row.get("confidence", "")).strip().lower()
    summary = " ".join(str(row.get("final_summary", "")).split())
    issue_tags = str(row.get("issue_tags", "")).strip()
    note = str(row.get("note", "")).strip()
    if decision not in REVIEW_DECISIONS:
        decision = ""
    if confidence not in CONFIDENCE_VALUES:
        confidence = ""
    return decision, confidence, summary, issue_tags, note


def accepted_decision(decision: str) -> bool:
    return decision in {"keep", "edit"}


def build_selected_candidate(
    item: dict[str, Any],
    primary: dict[str, str],
    qa: dict[str, str] | None,
    *,
    allow_low_confidence_without_qa: bool,
) -> tuple[dict[str, Any] | None, str]:
    primary_decision, primary_confidence, primary_summary, primary_tags, primary_note = normalize_decision(primary)
    if not primary_decision:
        return None, "missing_primary_decision"
    if primary_decision in {"drop", "uncertain"}:
        return None, f"primary_{primary_decision}"
    if not primary_confidence:
        return None, "missing_primary_confidence"
    if not primary_summary:
        return None, "missing_primary_summary"

    qa_payload: dict[str, Any] | None = None
    qa_agrees = False
    if qa is not None:
        qa_decision, qa_confidence, qa_summary, qa_tags, qa_note = normalize_decision(qa)
        qa_payload = {
            "decision": qa_decision,
            "confidence": qa_confidence,
            "final_summary": qa_summary,
            "issue_tags": qa_tags,
            "note": qa_note,
            "decision_file": qa.get("_decision_file", ""),
        }
        if qa_decision in {"drop", "uncertain"}:
            return None, f"qa_{qa_decision}"
        if accepted_decision(qa_decision) and qa_summary:
            qa_agrees = qa_summary == primary_summary
            if qa_confidence == "high":
                primary_summary = qa_summary
                primary_confidence = qa_confidence
    elif primary_confidence == "low" and not allow_low_confidence_without_qa:
        return None, "low_confidence_without_qa"

    local_flags = local_filter_flags(primary_summary, str(item.get("title", "")))
    if local_flags:
        return None, "local_filter_" + ",".join(local_flags)

    selected = {
        "paper_id": item["paper_id"],
        "title": item["title"],
        "abstract": item["abstract"],
        "summary": primary_summary,
        "primary_category": item.get("primary_category"),
        "categories": item.get("categories", []),
        "created_date": item.get("created_date"),
        "source_split": "specificity_preserving_overlay_400_v1",
        "teacher_model": item.get("specificity_model"),
        "generator_prompt_version": item.get("specificity_prompt_version"),
        "review_prompt_version": "manual_specificity_preserving_primary_plus_qa_v1",
        "overlay_component": "specificity_preserving",
        "original_summary": item.get("original_summary", ""),
        "specificity_candidate_id": item.get("candidate_id"),
        "specificity_review_id": item.get("review_id"),
        "specificity_score": item.get("specificity_score"),
        "specificity_flags": item.get("specificity_flags", []),
        "specificity_supported_numbers": item.get("supported_numbers", []),
        "specificity_supported_labels": item.get("supported_labels", []),
        "specificity_unsupported_numbers": item.get("unsupported_numbers", []),
        "specificity_unsupported_labels": item.get("unsupported_labels", []),
        "specificity_draft_decision": item.get("draft_decision"),
        "specificity_draft_summary": item.get("draft_summary"),
        "specificity_draft_risk_tags": item.get("draft_risk_tags", []),
        "specificity_draft_evidence_span": item.get("draft_evidence_span", ""),
        "specificity_draft_reason_span": item.get("draft_reason_span", ""),
        "manual_primary_decision": {
            "decision": primary_decision,
            "confidence": primary_confidence,
            "final_summary": primary_summary,
            "issue_tags": primary_tags,
            "note": primary_note,
            "decision_file": primary.get("_decision_file", ""),
        },
        "manual_qa_decision": qa_payload,
        "manual_qa_agrees": qa_agrees,
    }
    return selected, "selected"


def selection_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    primary = row.get("manual_primary_decision") or {}
    flags = set(row.get("specificity_flags", []))
    return (
        confidence_rank.get(str(primary.get("confidence", "")), 9),
        0 if row.get("manual_qa_agrees") else 1,
        0 if row.get("manual_qa_decision") else 1,
        -int(row.get("specificity_score") or 0),
        0 if "supported_number" in flags else 1,
        0 if "supported_label" in flags else 1,
        str(row.get("paper_id", "")),
    )


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.review_bundle_dir)
    output_dir = Path(args.output_release_dir)
    items = load_jsonl(bundle_dir / "review_items.jsonl")
    primary_decisions = load_primary_decisions(bundle_dir)
    qa_decisions = load_qa_decisions(bundle_dir)

    missing_primary = [
        item["review_id"] for item in items if item["review_id"] not in primary_decisions
    ]
    if missing_primary:
        raise ValueError(f"Missing primary decisions for {len(missing_primary)} review items")

    selected_candidates: list[dict[str, Any]] = []
    exclusion_reasons: Counter[str] = Counter()
    for item in items:
        selected, reason = build_selected_candidate(
            item,
            primary_decisions[item["review_id"]],
            qa_decisions.get(item["review_id"]),
            allow_low_confidence_without_qa=args.allow_low_confidence_without_qa,
        )
        if selected is None:
            exclusion_reasons[reason] += 1
            continue
        selected_candidates.append(selected)

    selected_candidates.sort(key=selection_sort_key)
    seen_paper_ids: set[str] = set()
    deduped: list[dict[str, Any]] = []
    duplicate_paper_ids: list[str] = []
    for row in selected_candidates:
        paper_id = str(row["paper_id"])
        if paper_id in seen_paper_ids:
            duplicate_paper_ids.append(paper_id)
            continue
        seen_paper_ids.add(paper_id)
        deduped.append(row)

    if len(deduped) < args.target_rows and not args.allow_underfilled:
        raise ValueError(
            f"Only {len(deduped)} rows survived; target is {args.target_rows}."
        )
    final_rows = deduped[: args.target_rows]
    if len(final_rows) != args.target_rows and not args.allow_underfilled:
        raise ValueError(f"Final specificity row count is {len(final_rows)}, expected {args.target_rows}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "training_rows.jsonl", final_rows)
    manifest = {
        "created_at": utc_now_iso(),
        "release_name": output_dir.name,
        "review_bundle_dir": str(bundle_dir),
        "target_rows": args.target_rows,
        "final_rows": len(final_rows),
        "review_items": len(items),
        "primary_decisions": len(primary_decisions),
        "qa_decisions": len(qa_decisions),
        "candidate_rows_survived_before_cap": len(deduped),
        "duplicate_paper_ids": sorted(duplicate_paper_ids),
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "final_stats": stats_for_rows(final_rows),
        "manual_policy": {
            "primary_decision_required": True,
            "qa_drop_or_uncertain_excludes": True,
            "low_confidence_keep_requires_qa": not args.allow_low_confidence_without_qa,
            "selection_priority": [
                "primary confidence",
                "qa agreement",
                "qa presence",
                "specificity score",
                "supported number",
                "supported label",
                "paper_id",
            ],
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(f"[specificity-materialize] wrote {len(final_rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
