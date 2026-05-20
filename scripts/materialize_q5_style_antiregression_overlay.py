#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
from typing import Any


DEFAULT_EVAL_ROOT = (
    "hf_bucket/experiments/mlx_lora/review_comparisons/"
    "q5_numeric_overlay_eval_20260507"
)
DEFAULT_OUTPUT_RELEASE = (
    "hf_dataset/releases/"
    "recent_3y_balanced_10746_q5_style_antiregression_overlay_v1"
)
DEFAULT_STYLE_RELEASE = "hf_dataset/releases/recent_3y_balanced_10746_v1"
DEFAULT_Q4_RELEASES = (
    "hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_combined_v1",
    "hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_distill_6000_v1_primary",
    "hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_distill_v1",
)

WORD_RE = re.compile(r"[A-Za-z0-9]+")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:\s?[%xX])?")
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
            "Build a small q5 style anti-regression overlay from the q5 numeric "
            "overlay eval, blind-review outcomes, style anchors, and numeric "
            "rehearsal rows."
        )
    )
    parser.add_argument("--eval-root", default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--style-release", default=DEFAULT_STYLE_RELEASE)
    parser.add_argument("--q4-release", action="append", default=list(DEFAULT_Q4_RELEASES))
    parser.add_argument("--output-release", default=DEFAULT_OUTPUT_RELEASE)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--max-words", type=int, default=16)
    parser.add_argument("--style-anchor-count", type=int, default=110)
    parser.add_argument("--style-drift-count", type=int, default=80)
    parser.add_argument("--numeric-rehearsal-count", type=int, default=60)
    parser.add_argument("--q4-rehearsal-count", type=int, default=30)
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


def normalize_space(value: Any) -> str:
    return " ".join(str(value).split())


def words(value: Any) -> list[str]:
    return WORD_RE.findall(str(value).lower())


def word_count(value: Any) -> int:
    return len(words(value))


def numbers(value: Any) -> set[str]:
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


def audit_target(row: dict[str, Any], summary: str, max_words: int) -> list[str]:
    reasons: list[str] = []
    summary = normalize_space(summary)
    count = word_count(summary)
    if not summary:
        reasons.append("empty_summary")
    if "\ufffd" in summary:
        reasons.append("replacement_character")
    if count < 5:
        reasons.append("too_short")
    if count > max_words:
        reasons.append("too_long")
    lowered = summary.lower()
    if any(fragment in lowered for fragment in BAD_SUMMARY_FRAGMENTS):
        reasons.append("boilerplate_opening")
    if len(set(words(summary))) <= max(2, count // 3):
        reasons.append("low_lexical_diversity")
    if title_overlap(str(row.get("title", "")), summary) > 0.72:
        reasons.append("high_title_overlap")
    source_numbers = numbers(f"{row.get('title', '')} {row.get('abstract', '')}")
    extra_numbers = numbers(summary) - source_numbers
    if extra_numbers:
        reasons.append("target_number_not_in_source")
    return reasons


def base_training_row(
    row: dict[str, Any],
    summary: str,
    *,
    source_split: str,
    source_kind: str,
    source_detail: str,
    source_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "paper_id": str(row["paper_id"]),
        "title": normalize_space(row["title"]),
        "abstract": normalize_space(row["abstract"]),
        "summary": normalize_space(summary),
        "primary_category": row.get("primary_category"),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date"),
        "source_split": source_split,
        "teacher_model": "mixed_q5_overlay_antiregression_v1",
        "generator_prompt_version": "q5_numeric_overlay_eval_20260507",
        "review_prompt_version": "targeted_blind_review_v1_plus_heuristic_audit",
        "q5_style_antiregression_source": source_kind,
        "q5_style_antiregression_detail": source_detail,
    }
    if source_id:
        out["q5_style_antiregression_source_id"] = source_id
    if extra:
        out.update(extra)
    return out


def parse_decisions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return {row["review_id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def load_blind_maps(eval_root: Path) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, str]],
]:
    blind_root = eval_root / "targeted_blind_review_v1"
    items = {row["review_id"]: row for row in load_jsonl(blind_root / "review_items.jsonl")}
    side_map = {row["review_id"]: row for row in load_jsonl(blind_root / "side_map.jsonl")}
    decisions = parse_decisions(blind_root / "decisions.tsv")
    return items, side_map, decisions


def side_for_source(side_map: dict[str, Any], source: str) -> str:
    if side_map["left_source"] == source:
        return "left"
    if side_map["right_source"] == source:
        return "right"
    raise ValueError(f"Source {source} not found in side map for {side_map.get('review_id')}")


def summary_for_source(item: dict[str, Any], side_map: dict[str, Any], source: str) -> str:
    side = side_for_source(side_map, source)
    return normalize_space(item[f"{side}_summary"])


def load_generation_rows(eval_root: Path, subdir: str) -> list[dict[str, Any]]:
    return load_jsonl(eval_root / subdir / "generations.jsonl")


def is_q4_numeric_row(row: dict[str, Any]) -> bool:
    bucket = row.get("q4_hardening_selection_bucket") or row.get("selection_bucket")
    if bucket == "numeric":
        return True
    tags = set(row.get("q4_issue_tags", [])) | set(row.get("correction_issue_tags", []))
    return bool(tags & NUMERIC_TAGS)


def add_candidate(
    candidates: list[dict[str, Any]],
    rejects: list[dict[str, Any]],
    source_row: dict[str, Any],
    summary: str,
    *,
    max_words: int,
    source_split: str,
    source_kind: str,
    source_detail: str,
    source_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    reasons = audit_target(source_row, summary, max_words=max_words)
    candidate = base_training_row(
        source_row,
        summary,
        source_split=source_split,
        source_kind=source_kind,
        source_detail=source_detail,
        source_id=source_id,
        extra=extra,
    )
    if reasons:
        rejected = dict(candidate)
        rejected["reject_reasons"] = reasons
        rejects.append(rejected)
        return
    candidates.append(candidate)


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    output_release = Path(args.output_release)
    source_split = output_release.name
    rng = random.Random(args.seed)

    candidates: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []

    items, side_maps, decisions = load_blind_maps(eval_root)

    for review_id, item in items.items():
        decision = decisions[review_id]
        side_map = side_maps[review_id]
        overlay_side = side_for_source(side_map, "overlay")
        pre_side = side_for_source(side_map, "pre_q5")
        overlay_severe = decision[f"{overlay_side}_severe"] == "yes"
        overlay_usable = decision[f"{overlay_side}_target_usable"] == "yes"
        pre_usable = decision[f"{pre_side}_target_usable"] == "yes"
        winner = decision["winner"]
        winning_source = None
        if winner in {"left", "right"}:
            winning_source = side_map[f"{winner}_source"]

        if (winning_source == "pre_q5" and pre_usable) or (overlay_severe and pre_usable):
            add_candidate(
                candidates,
                rejects,
                item,
                summary_for_source(item, side_map, "pre_q5"),
                max_words=args.max_words,
                source_split=source_split,
                source_kind="blind_review_overlay_regression",
                source_detail=item["selection_reason"],
                source_id=review_id,
                extra={"blind_review_note": decision["note"]},
            )

        if overlay_usable and winning_source == "overlay" and item["source"] == "numeric":
            add_candidate(
                candidates,
                rejects,
                item,
                summary_for_source(item, side_map, "overlay"),
                max_words=args.max_words,
                source_split=source_split,
                source_kind="blind_review_numeric_rehearsal",
                source_detail=item["selection_reason"],
                source_id=review_id,
                extra={"blind_review_note": decision["note"]},
            )

        q8_target = normalize_space(item.get("q8_target_summary", ""))
        if item["source"] == "numeric" and q8_target and decision["winner"] == "both_bad":
            add_candidate(
                candidates,
                rejects,
                item,
                q8_target,
                max_words=args.max_words,
                source_split=source_split,
                source_kind="q8_target_numeric_repair",
                source_detail=item["selection_reason"],
                source_id=review_id,
                extra={"blind_review_note": decision["note"]},
            )

    general_rows = load_generation_rows(eval_root, "pre_q5_vs_numeric_overlay_general_200")
    style_drift_candidates: list[dict[str, Any]] = []
    for row in general_rows:
        left = normalize_space(row["left_summary"])
        right = normalize_space(row["right_summary"])
        left_count = word_count(left)
        right_count = word_count(right)
        if right_count > 15 or right_count - left_count >= 3:
            if left_count <= 15 and left_count >= 5:
                style_drift_candidates.append(row)
    style_drift_candidates.sort(
        key=lambda row: (
            word_count(row["right_summary"]) - word_count(row["left_summary"]),
            word_count(row["right_summary"]),
        ),
        reverse=True,
    )
    for row in style_drift_candidates[: args.style_drift_count]:
        add_candidate(
            candidates,
            rejects,
            row,
            row["left_summary"],
            max_words=args.max_words,
            source_split=source_split,
            source_kind="general_style_drift_pre_q5_target",
            source_detail="right_overlay_longer_or_over15",
            source_id=str(row.get("eval_id") or row["paper_id"]),
            extra={
                "pre_q5_summary": row["left_summary"],
                "numeric_overlay_summary": row["right_summary"],
            },
        )

    numeric_rows = load_generation_rows(eval_root, "pre_q5_vs_numeric_overlay_numeric_severe")
    numeric_rehearsal_candidates: list[dict[str, Any]] = []
    for row in numeric_rows:
        left = normalize_space(row["left_summary"])
        right = normalize_space(row["right_summary"])
        source_nums = numbers(f"{row.get('title', '')} {row.get('abstract', '')}")
        left_extra = bool(numbers(left) - source_nums)
        right_extra = bool(numbers(right) - source_nums)
        if left_extra and not right_extra and 5 <= word_count(right) <= args.max_words:
            numeric_rehearsal_candidates.append(row)
    numeric_rehearsal_candidates.sort(key=lambda row: str(row["paper_id"]))
    for row in numeric_rehearsal_candidates[: args.numeric_rehearsal_count]:
        add_candidate(
            candidates,
            rejects,
            row,
            row["right_summary"],
            max_words=args.max_words,
            source_split=source_split,
            source_kind="mechanical_numeric_rehearsal",
            source_detail="pre_q5_extra_number_overlay_clean",
            source_id=str(row.get("eval_id") or row["paper_id"]),
            extra={
                "pre_q5_summary": row["left_summary"],
                "numeric_overlay_summary": row["right_summary"],
            },
        )

    style_rows = load_jsonl(Path(args.style_release) / "training_rows.jsonl")
    style_anchor_candidates = [
        row
        for row in style_rows
        if 8 <= word_count(row.get("summary", "")) <= 15
        and not numbers(row.get("summary", ""))
        and title_overlap(str(row.get("title", "")), str(row.get("summary", ""))) <= 0.55
    ]
    rng.shuffle(style_anchor_candidates)
    for row in style_anchor_candidates[: args.style_anchor_count]:
        add_candidate(
            candidates,
            rejects,
            row,
            row["summary"],
            max_words=args.max_words,
            source_split=source_split,
            source_kind="promoted_q8_style_anchor",
            source_detail="concise_numeric_free_original_training_row",
            source_id=str(row["paper_id"]),
        )

    q4_rehearsal_candidates: list[dict[str, Any]] = []
    for release in args.q4_release:
        path = Path(release) / "training_rows.jsonl"
        if not path.exists():
            continue
        for row in load_jsonl(path):
            if is_q4_numeric_row(row) and 5 <= word_count(row.get("summary", "")) <= args.max_words:
                copied = dict(row)
                copied["q4_source_release"] = release
                q4_rehearsal_candidates.append(copied)
    q4_rehearsal_candidates.sort(key=lambda row: str(row["paper_id"]))
    for row in q4_rehearsal_candidates[: args.q4_rehearsal_count]:
        add_candidate(
            candidates,
            rejects,
            row,
            row["summary"],
            max_words=args.max_words,
            source_split=source_split,
            source_kind="q4_numeric_rehearsal",
            source_detail=Path(str(row.get("q4_source_release", ""))).name,
            source_id=str(row["paper_id"]),
            extra={"q4_source_release": row.get("q4_source_release")},
        )

    by_paper: dict[str, dict[str, Any]] = {}
    duplicate_drops: list[dict[str, Any]] = []
    for row in candidates:
        paper_id = str(row["paper_id"])
        if paper_id in by_paper:
            duplicate_drops.append(row)
            continue
        by_paper[paper_id] = row

    final_rows = list(by_paper.values())
    final_rows.sort(
        key=lambda row: (
            str(row["q5_style_antiregression_source"]),
            str(row["paper_id"]),
        )
    )

    output_release.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_release / "training_rows.jsonl", final_rows)
    write_jsonl(output_release / "rejected_rows.jsonl", rejects)
    write_jsonl(output_release / "duplicate_dropped_rows.jsonl", duplicate_drops)

    source_counts = Counter(row["q5_style_antiregression_source"] for row in final_rows)
    detail_counts = Counter(
        (row["q5_style_antiregression_source"], row["q5_style_antiregression_detail"])
        for row in final_rows
    )
    reject_counts = Counter(reason for row in rejects for reason in row["reject_reasons"])
    word_counts = [word_count(row["summary"]) for row in final_rows]
    manifest = {
        "created_at": utc_now_iso(),
        "output_release": str(output_release),
        "eval_root": str(eval_root),
        "style_release": args.style_release,
        "q4_releases": args.q4_release,
        "seed": args.seed,
        "max_words": args.max_words,
        "rows": len(final_rows),
        "source_counts": dict(source_counts),
        "detail_counts": {f"{key[0]}::{key[1]}": value for key, value in detail_counts.items()},
        "rejected_rows": len(rejects),
        "reject_counts": dict(reject_counts),
        "duplicate_dropped_rows": len(duplicate_drops),
        "summary_word_count": {
            "min": min(word_counts) if word_counts else None,
            "max": max(word_counts) if word_counts else None,
            "avg": round(sum(word_counts) / len(word_counts), 2) if word_counts else None,
        },
        "outputs": {
            "training_rows": str(output_release / "training_rows.jsonl"),
            "rejected_rows": str(output_release / "rejected_rows.jsonl"),
            "duplicate_dropped_rows": str(output_release / "duplicate_dropped_rows.jsonl"),
            "manifest": str(output_release / "run_manifest.json"),
        },
    }
    write_json(output_release / "run_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
