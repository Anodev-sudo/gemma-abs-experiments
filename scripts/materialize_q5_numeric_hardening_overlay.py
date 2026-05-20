#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_Q5_REVIEW_ROWS = (
    "hf_bucket/experiments/mlx_lora/q5_hardening_distill_6000_v1/"
    "q5_severe_q8_clean_review_rows.jsonl"
)
DEFAULT_OUTPUT_RELEASE = (
    "hf_dataset/releases/"
    "recent_3y_balanced_10746_q5_numeric_hardening_overlay_q4mix_v1"
)
DEFAULT_Q4_RELEASES = (
    "hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_combined_v1",
    "hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_distill_6000_v1_primary",
    "hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_distill_v1",
)

NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:\s?[%xX])?")
WORD_RE = re.compile(r"[A-Za-z0-9]+")
BAD_SUMMARY_FRAGMENTS = (
    "this paper",
    "we propose",
    "we present",
    "we introduce",
    "we study",
)
NUMERIC_TAGS = {
    "invented_number",
    "wrong_number_binding",
    "unsupported_number",
    "numeric_error",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit q5 numeric hardening targets and merge accepted q5 rows with "
            "old q4 numeric hardening rows."
        )
    )
    parser.add_argument("--q5-review-rows", default=DEFAULT_Q5_REVIEW_ROWS)
    parser.add_argument("--output-release", default=DEFAULT_OUTPUT_RELEASE)
    parser.add_argument("--q4-release", action="append", default=list(DEFAULT_Q4_RELEASES))
    parser.add_argument("--min-words", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=20)
    parser.add_argument("--max-title-overlap", type=float, default=0.72)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_space(value: str) -> str:
    return " ".join(str(value).split())


def words(value: str) -> list[str]:
    return WORD_RE.findall(str(value).lower())


def word_count(value: str) -> int:
    return len(words(value))


def numbers(value: str) -> set[str]:
    return {match.group(0).replace(" ", "") for match in NUMBER_RE.finditer(str(value))}


def title_overlap(title: str, summary: str) -> float:
    title_words = set(words(title))
    summary_words = words(summary)
    if not summary_words:
        return 0.0
    stop = {
        "a",
        "an",
        "and",
        "are",
        "for",
        "from",
        "in",
        "of",
        "on",
        "the",
        "to",
        "with",
    }
    content = [word for word in summary_words if word not in stop]
    if not content:
        return 0.0
    return sum(1 for word in content if word in title_words) / len(content)


def audit_q5_target(
    row: dict[str, Any],
    *,
    min_words: int,
    max_words: int,
    max_title_overlap: float,
) -> list[str]:
    summary = normalize_space(row.get("q8_summary", ""))
    title = str(row.get("title", ""))
    source = f"{title} {row.get('abstract', '')}"
    reasons: list[str] = []

    if not summary:
        reasons.append("empty_summary")
    if "\ufffd" in summary or "����" in summary:
        reasons.append("replacement_character")
    count = word_count(summary)
    if count < min_words:
        reasons.append("too_short")
    if count > max_words:
        reasons.append("too_long")
    lowered = summary.lower()
    if any(fragment in lowered for fragment in BAD_SUMMARY_FRAGMENTS):
        reasons.append("boilerplate_opening")
    if len(set(words(summary))) <= max(2, count // 3):
        reasons.append("low_lexical_diversity")
    if title_overlap(title, summary) > max_title_overlap:
        reasons.append("high_title_overlap")
    summary_numbers = numbers(summary)
    source_numbers = numbers(source)
    if summary_numbers - source_numbers:
        reasons.append("target_number_not_in_source")
    return reasons


def audit_training_summary(
    row: dict[str, Any],
    *,
    min_words: int,
    max_words: int,
) -> list[str]:
    summary = normalize_space(row.get("summary", ""))
    reasons: list[str] = []
    if not summary:
        reasons.append("empty_summary")
    if "\ufffd" in summary or "����" in summary:
        reasons.append("replacement_character")
    count = word_count(summary)
    if count < min_words:
        reasons.append("too_short")
    if count > max_words:
        reasons.append("too_long")
    lowered = summary.lower()
    if any(fragment in lowered for fragment in BAD_SUMMARY_FRAGMENTS):
        reasons.append("boilerplate_opening")
    if len(set(words(summary))) <= max(2, count // 3):
        reasons.append("low_lexical_diversity")
    return reasons


def is_q5_numeric_candidate(row: dict[str, Any]) -> bool:
    return row.get("selection_bucket") == "numeric"


def make_q5_training_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": row["paper_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "summary": normalize_space(row["q8_summary"]),
        "primary_category": row.get("primary_category"),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date"),
        "source_split": "q5_numeric_hardening_overlay_q4mix_v1",
        "teacher_model": "q8_promoted",
        "generator_prompt_version": "q8_vs_q5_hardening_mine_6000_v1",
        "review_prompt_version": "manual_blind_q5_hardening_primary_5_4_mini_agents_v1",
        "q5_hardening_review_id": row.get("review_id"),
        "q5_hardening_selection_bucket": row.get("selection_bucket"),
        "q5_hardening_axes": row.get("q5_hardening_axes", []),
        "q5_summary": row.get("q5_summary", ""),
        "q5_issue_tags": row.get("q5_issue_tags", []),
        "q5_numeric_overlay_source": "q5_native_mined_numeric",
    }


def resolve_training_rows(release: str) -> tuple[Path, Path]:
    release_dir = Path(release)
    training_rows = release_dir / "training_rows.jsonl"
    if not training_rows.exists():
        raise FileNotFoundError(f"Missing training_rows.jsonl at {training_rows}")
    return release_dir, training_rows


def q4_row_is_numeric(row: dict[str, Any]) -> bool:
    bucket = row.get("q4_hardening_selection_bucket") or row.get("selection_bucket")
    if bucket == "numeric":
        return True
    tags = set(row.get("q4_issue_tags", [])) | set(row.get("correction_issue_tags", []))
    return bool(tags & NUMERIC_TAGS)


def make_q4_training_row(row: dict[str, Any], release_dir: Path) -> dict[str, Any]:
    copied = dict(row)
    copied["summary"] = normalize_space(str(copied["summary"]))
    copied["source_split"] = "q5_numeric_hardening_overlay_q4mix_v1"
    copied["q5_numeric_overlay_source"] = "q4_prior_numeric_hardening"
    copied["q5_numeric_overlay_source_release"] = str(release_dir)
    return copied


def row_priority(row: dict[str, Any]) -> tuple[int, int]:
    source = row.get("q5_numeric_overlay_source")
    if source == "q5_native_mined_numeric":
        return (0, 0)
    release = str(row.get("q5_numeric_overlay_source_release", ""))
    if "q4_hardening_combined_v1" in release:
        return (1, 0)
    if "q4_hardening_distill_6000_v1_primary" in release:
        return (2, 0)
    return (3, 0)


def main() -> None:
    args = parse_args()
    q5_review_path = Path(args.q5_review_rows)
    output_release = Path(args.output_release)

    q5_rows = [row for row in load_jsonl(q5_review_path) if is_q5_numeric_candidate(row)]
    accepted_q5: list[dict[str, Any]] = []
    rejected_q5: list[dict[str, Any]] = []
    for row in q5_rows:
        reasons = audit_q5_target(
            row,
            min_words=args.min_words,
            max_words=args.max_words,
            max_title_overlap=args.max_title_overlap,
        )
        if reasons:
            rejected_q5.append({**row, "q5_target_audit_reasons": reasons})
        else:
            accepted_q5.append(make_q5_training_row(row))

    q4_rows: list[dict[str, Any]] = []
    rejected_q4: list[dict[str, Any]] = []
    q4_source_counts: Counter[str] = Counter()
    for release in args.q4_release:
        release_dir, training_rows_path = resolve_training_rows(release)
        for row in load_jsonl(training_rows_path):
            if not q4_row_is_numeric(row):
                continue
            converted = make_q4_training_row(row, release_dir)
            reasons = audit_training_summary(
                converted,
                min_words=args.min_words,
                max_words=args.max_words,
            )
            if reasons:
                rejected_q4.append({**converted, "q4_target_audit_reasons": reasons})
                continue
            q4_rows.append(converted)
            q4_source_counts[str(release_dir)] += 1

    candidates = accepted_q5 + q4_rows
    by_paper: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for row in sorted(candidates, key=row_priority):
        paper_id = str(row.get("paper_id", ""))
        if not paper_id:
            continue
        if paper_id in by_paper:
            duplicates.append(
                {
                    "paper_id": paper_id,
                    "kept_source": by_paper[paper_id].get("q5_numeric_overlay_source"),
                    "dropped_source": row.get("q5_numeric_overlay_source"),
                    "dropped_source_release": row.get("q5_numeric_overlay_source_release"),
                }
            )
            continue
        by_paper[paper_id] = row

    training_rows = list(by_paper.values())
    output_release.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_release / "training_rows.jsonl", training_rows)
    write_jsonl(output_release / "q5_target_audit_rejected_rows.jsonl", rejected_q5)
    write_jsonl(output_release / "q4_target_audit_rejected_rows.jsonl", rejected_q4)
    write_jsonl(output_release / "dedupe_dropped_rows.jsonl", duplicates)

    manifest = {
        "created_at": utc_now_iso(),
        "mode": "q5_numeric_hardening_overlay_q4mix_v1",
        "q5_review_rows": str(q5_review_path),
        "q4_releases": list(args.q4_release),
        "output_release": str(output_release),
        "audit_thresholds": {
            "min_words": args.min_words,
            "max_words": args.max_words,
            "max_title_overlap": args.max_title_overlap,
        },
        "summary": {
            "q5_numeric_candidates": len(q5_rows),
            "q5_accepted": len(accepted_q5),
            "q5_rejected": len(rejected_q5),
            "q5_rejection_reason_counts": dict(
                Counter(reason for row in rejected_q5 for reason in row["q5_target_audit_reasons"])
            ),
            "q4_numeric_rows_before_dedupe": len(q4_rows),
            "q4_rejected": len(rejected_q4),
            "q4_rejection_reason_counts": dict(
                Counter(reason for row in rejected_q4 for reason in row["q4_target_audit_reasons"])
            ),
            "q4_source_counts": dict(q4_source_counts),
            "combined_rows_before_dedupe": len(candidates),
            "dedupe_dropped_rows": len(duplicates),
            "training_rows": len(training_rows),
            "source_counts": dict(Counter(row.get("q5_numeric_overlay_source") for row in training_rows)),
            "selection_bucket_counts": dict(
                Counter(
                    row.get("q5_hardening_selection_bucket")
                    or row.get("q4_hardening_selection_bucket")
                    or row.get("selection_bucket")
                    for row in training_rows
                )
            ),
            "summary_word_count": {
                "min": min(word_count(row["summary"]) for row in training_rows),
                "max": max(word_count(row["summary"]) for row in training_rows),
                "avg": round(
                    sum(word_count(row["summary"]) for row in training_rows)
                    / len(training_rows),
                    2,
                ),
            },
        },
        "outputs": {
            "training_rows": str(output_release / "training_rows.jsonl"),
            "q5_target_audit_rejected_rows": str(
                output_release / "q5_target_audit_rejected_rows.jsonl"
            ),
            "q4_target_audit_rejected_rows": str(
                output_release / "q4_target_audit_rejected_rows.jsonl"
            ),
            "dedupe_dropped_rows": str(output_release / "dedupe_dropped_rows.jsonl"),
        },
    }
    write_json(output_release / "run_manifest.json", manifest)
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
