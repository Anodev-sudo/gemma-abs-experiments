#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

try:
    from scripts.prepare_q4_hardening_review_packets import DECISION_HEADERS, ISSUE_TAGS
except ImportError:  # pragma: no cover - used when running as scripts/foo.py
    from prepare_q4_hardening_review_packets import DECISION_HEADERS, ISSUE_TAGS

DEFAULT_REVIEW_PACKET_DIR = "hf_bucket/experiments/mlx_lora/q4_hardening_distill_v1/review_packets"
DEFAULT_OUTPUT_ROOT = "hf_bucket/experiments/mlx_lora/q4_hardening_distill_v1"
DEFAULT_RELEASE_DIR = "hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_distill_v1"
DEFAULT_SEED = 20260501

WINNER_VALUES = {"left", "right", "tie", "both_bad"}
CONFIDENCE_VALUES = {"high", "medium", "low"}
RESERVED_EVAL_ONLY_PAPER_IDS = {
    "2510.10676",
    "2408.12758",
    "2510.04118",
    "2504.20147",
    "2306.16665",
    "2501.12757",
    "2305.19741",
    "2601.02875",
    "2306.11084",
    "2412.03821",
    "2309.14613",
    "2602.10701",
    "2406.00094",
    "2309.02934",
    "2304.08441",
    "2305.01768",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate q4 hardening reviews and materialize distill rows."
    )
    parser.add_argument("--review-packet-dir", default=DEFAULT_REVIEW_PACKET_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--release-dir", default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--qa-control-ratio", type=float, default=0.10)
    parser.add_argument("--qa-packet-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--write-qa-followup", action="store_true")
    parser.add_argument("--require-qa-for-distill", action="store_true")
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


def read_decision_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader if any(str(value).strip() for value in row.values())]


def write_blank_decisions(path: Path, review_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECISION_HEADERS, delimiter="\t")
        writer.writeheader()
        for review_id in review_ids:
            writer.writerow({"review_id": review_id})


def parse_bool(value: str) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in {"yes", "true", "1", "y"}:
        return True
    if normalized in {"no", "false", "0", "n"}:
        return False
    return None


def parse_tags(value: str) -> list[str]:
    aliases = {
        "no_answer": "non_answer",
        "unsupported_number": "numeric_error",
    }
    tags = [aliases.get(tag.strip(), tag.strip()) for tag in str(value).split(",") if tag.strip()]
    accepted_tags = set(ISSUE_TAGS) | {"numeric_error"}
    unknown = sorted(set(tags) - accepted_tags)
    if unknown:
        raise ValueError(f"Unknown issue tags: {', '.join(unknown)}")
    return tags


def parse_confidence(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in CONFIDENCE_VALUES:
        return normalized
    try:
        score = float(normalized)
    except ValueError:
        return ""
    if score >= 0.85:
        return "high"
    if score >= 0.65:
        return "medium"
    return "low"


def load_decisions(root: Path, glob_pattern: str) -> dict[str, dict[str, str]]:
    decisions: dict[str, dict[str, str]] = {}
    for path in sorted(root.glob(glob_pattern)):
        for row in read_decision_tsv(path):
            review_id = str(row.get("review_id", "")).strip()
            if not review_id:
                continue
            if not str(row.get("winner", "")).strip():
                continue
            if review_id in decisions:
                raise ValueError(f"Duplicate decision for {review_id}")
            copied = dict(row)
            copied["_decision_file"] = str(path)
            decisions[review_id] = copied
    return decisions


def normalize_decision(row: dict[str, str] | None) -> tuple[dict[str, Any] | None, str | None]:
    if row is None:
        return None, "missing_decision"
    winner = str(row.get("winner", "")).strip().lower()
    confidence = parse_confidence(row.get("confidence", ""))
    left_severe = parse_bool(row.get("left_severe", ""))
    right_severe = parse_bool(row.get("right_severe", ""))
    left_usable = parse_bool(row.get("left_target_usable", ""))
    right_usable = parse_bool(row.get("right_target_usable", ""))
    if winner not in WINNER_VALUES:
        return None, "invalid_winner"
    if confidence not in CONFIDENCE_VALUES:
        return None, "invalid_confidence"
    if left_severe is None or right_severe is None:
        return None, "missing_severe_flag"
    if left_usable is None or right_usable is None:
        return None, "missing_target_usable_flag"
    return (
        {
            "winner": winner,
            "left_severe": left_severe,
            "right_severe": right_severe,
            "left_issue_tags": parse_tags(row.get("left_issue_tags", "")),
            "right_issue_tags": parse_tags(row.get("right_issue_tags", "")),
            "left_target_usable": left_usable,
            "right_target_usable": right_usable,
            "confidence": confidence,
            "note": str(row.get("note", "")).strip(),
            "decision_file": row.get("_decision_file", ""),
        },
        None,
    )


def decision_disagrees(primary: dict[str, Any], qa: dict[str, Any]) -> bool:
    keys = (
        "winner",
        "left_severe",
        "right_severe",
        "left_target_usable",
        "right_target_usable",
    )
    return any(primary[key] != qa[key] for key in keys)


def source_value(side_map: dict[str, Any], side: str, field: str) -> Any:
    source = side_map[f"{side}_source"]
    return side_map[f"{source}_{field}"]


def map_decision_to_sources(
    item: dict[str, Any],
    side_map: dict[str, Any],
    primary_raw: dict[str, str] | None,
    qa_raw: dict[str, str] | None,
) -> dict[str, Any]:
    primary, primary_error = normalize_decision(primary_raw)
    qa, qa_error = normalize_decision(qa_raw) if qa_raw is not None else (None, None)
    effective = qa if qa is not None else primary
    errors = [error for error in (primary_error, qa_error) if error]
    if primary is not None and qa is not None and decision_disagrees(primary, qa):
        errors.append("primary_qa_conflict")

    if effective is None:
        q8_severe = q4_severe = q8_target_usable = q4_target_usable = False
        q8_issue_tags: list[str] = []
        q4_issue_tags: list[str] = []
        winner_source = ""
    else:
        q8_side = "left" if side_map["left_source"] == "q8" else "right"
        q4_side = "left" if side_map["left_source"] == "q4" else "right"
        q8_severe = bool(effective[f"{q8_side}_severe"])
        q4_severe = bool(effective[f"{q4_side}_severe"])
        q8_target_usable = bool(effective[f"{q8_side}_target_usable"])
        q4_target_usable = bool(effective[f"{q4_side}_target_usable"])
        q8_issue_tags = list(effective[f"{q8_side}_issue_tags"])
        q4_issue_tags = list(effective[f"{q4_side}_issue_tags"])
        winner_source = effective["winner"]
        if winner_source in {"left", "right"}:
            winner_source = side_map[f"{winner_source}_source"]

    return {
        "review_id": item["review_id"],
        "paper_id": item["paper_id"],
        "title": item["title"],
        "abstract": item["abstract"],
        "primary_category": item.get("primary_category"),
        "categories": item.get("categories", []),
        "created_date": item.get("created_date"),
        "selection_bucket": item.get("selection_bucket"),
        "q4_hardening_axes": item.get("q4_hardening_axes", []),
        "q8_summary": side_map["q8_summary"],
        "q4_summary": side_map["q4_summary"],
        "q8_severe": q8_severe,
        "q4_severe": q4_severe,
        "q8_issue_tags": q8_issue_tags,
        "q4_issue_tags": q4_issue_tags,
        "q8_target_usable": q8_target_usable,
        "q4_target_usable": q4_target_usable,
        "winner": winner_source,
        "primary_decision": primary,
        "qa_decision": qa,
        "review_errors": errors,
    }


def distill_exclusion_reason(row: dict[str, Any], *, require_qa: bool) -> str | None:
    if row["paper_id"] in RESERVED_EVAL_ONLY_PAPER_IDS:
        return "reserved_eval_only_prior_q4_failure"
    if row["review_errors"]:
        return ",".join(row["review_errors"])
    if require_qa and row.get("qa_decision") is None:
        return "missing_qa"
    if not row["q4_severe"]:
        return "q4_not_severe"
    if row["q8_severe"]:
        return "q8_severe"
    if not row["q8_target_usable"]:
        return "q8_target_not_usable"
    if not str(row.get("q8_summary", "")).strip():
        return "empty_q8_summary"
    return None


def build_training_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": row["paper_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "summary": row["q8_summary"],
        "primary_category": row.get("primary_category"),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date"),
        "source_split": "q4_hardening_distill_v1",
        "teacher_model": "faith_overlay_1400_v2_final_600_q8_distill",
        "generator_prompt_version": "q8_vs_q4_hardening_mine_1000_v1",
        "review_prompt_version": "manual_q4_hardening_primary_plus_qa_v1",
        "q4_hardening_review_id": row["review_id"],
        "q4_hardening_selection_bucket": row.get("selection_bucket"),
        "q4_hardening_axes": row.get("q4_hardening_axes", []),
        "q4_summary": row["q4_summary"],
        "q4_issue_tags": row["q4_issue_tags"],
    }


def build_qa_followup_rows(
    reviewed_rows: list[dict[str, Any]],
    review_items_by_id: dict[str, dict[str, Any]],
    *,
    ratio: float,
    seed: int,
) -> list[dict[str, Any]]:
    if ratio < 0 or ratio > 1:
        raise ValueError("--qa-control-ratio must be between 0 and 1")
    candidate_ids = {
        row["review_id"]
        for row in reviewed_rows
        if row["q4_severe"] and not row["q8_severe"] and row["q8_target_usable"] and not row["review_errors"]
    }
    clean_controls = [
        row["review_id"]
        for row in reviewed_rows
        if not row["q4_severe"] and not row["q8_severe"] and not row["review_errors"]
    ]
    rng = random.Random(seed)
    rng.shuffle(clean_controls)
    control_count = round(len(clean_controls) * ratio)
    selected_ids = candidate_ids | set(clean_controls[:control_count])
    return [review_items_by_id[review_id] for review_id in sorted(selected_ids)]


def packetize(rows: list[dict[str, Any]], packet_size: int) -> list[list[dict[str, Any]]]:
    if packet_size <= 0:
        raise ValueError("--qa-packet-size must be positive")
    return [rows[start : start + packet_size] for start in range(0, len(rows), packet_size)]


def summarize(rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "reviewed_rows": len(rows),
        "distill_rows": len(selected_rows),
        "q4_severe": sum(1 for row in rows if row["q4_severe"]),
        "q8_severe": sum(1 for row in rows if row["q8_severe"]),
        "q4_severe_q8_clean": sum(
            1 for row in rows if row["q4_severe"] and not row["q8_severe"]
        ),
        "winner_counts": dict(Counter(row["winner"] for row in rows)),
        "selection_bucket_counts": dict(Counter(row.get("selection_bucket") for row in rows)),
        "distill_bucket_counts": dict(Counter(row.get("selection_bucket") for row in selected_rows)),
        "review_error_counts": dict(Counter(error for row in rows for error in row["review_errors"])),
    }


def main() -> None:
    args = parse_args()
    review_dir = Path(args.review_packet_dir)
    output_root = Path(args.output_root)
    release_dir = Path(args.release_dir)

    review_items = load_jsonl(review_dir / "review_items.jsonl")
    side_maps = load_jsonl(review_dir / "side_map.jsonl")
    review_items_by_id = {str(row["review_id"]): row for row in review_items}
    side_maps_by_id = {str(row["review_id"]): row for row in side_maps}
    primary_decisions = load_decisions(review_dir, "primary/packet_*/decisions.tsv")
    qa_decisions = load_decisions(review_dir, "qa/*/decisions.tsv")

    reviewed_rows = [
        map_decision_to_sources(
            review_items_by_id[review_id],
            side_maps_by_id[review_id],
            primary_decisions.get(review_id),
            qa_decisions.get(review_id),
        )
        for review_id in sorted(review_items_by_id)
    ]

    if args.write_qa_followup:
        qa_rows = build_qa_followup_rows(
            reviewed_rows,
            review_items_by_id,
            ratio=args.qa_control_ratio,
            seed=args.seed,
        )
        qa_dir = review_dir / "qa" / "followup"
        write_jsonl(qa_dir / "review_items.jsonl", qa_rows)
        write_blank_decisions(qa_dir / "decisions.tsv", [str(row["review_id"]) for row in qa_rows])
        for packet_index, packet_rows in enumerate(packetize(qa_rows, args.qa_packet_size)):
            packet_dir = review_dir / "qa" / f"packet_{packet_index:02d}"
            write_jsonl(packet_dir / "review_items.jsonl", packet_rows)
            write_blank_decisions(
                packet_dir / "decisions.tsv",
                [str(row["review_id"]) for row in packet_rows],
            )
        print(f"[q4-hardening-aggregate] wrote QA follow-up rows: {len(qa_rows)}")

    selected_review_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for row in reviewed_rows:
        reason = distill_exclusion_reason(row, require_qa=args.require_qa_for_distill)
        if reason is None:
            selected_review_rows.append(row)
        else:
            excluded_rows.append({**row, "distill_exclusion_reason": reason})

    training_rows = [build_training_row(row) for row in selected_review_rows]
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "reviewed_rows.jsonl", reviewed_rows)
    write_jsonl(output_root / "distill_candidates.jsonl", selected_review_rows)
    write_jsonl(output_root / "excluded_review_rows.jsonl", excluded_rows)

    release_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(release_dir / "training_rows.jsonl", training_rows)
    manifest = {
        "created_at": utc_now_iso(),
        "review_packet_dir": str(review_dir),
        "output_root": str(output_root),
        "release_dir": str(release_dir),
        "require_qa_for_distill": args.require_qa_for_distill,
        "reserved_eval_only_paper_ids": sorted(RESERVED_EVAL_ONLY_PAPER_IDS),
        "primary_decisions": len(primary_decisions),
        "qa_decisions": len(qa_decisions),
        "summary": summarize(reviewed_rows, selected_review_rows),
        "outputs": {
            "reviewed_rows": str(output_root / "reviewed_rows.jsonl"),
            "distill_candidates": str(output_root / "distill_candidates.jsonl"),
            "excluded_review_rows": str(output_root / "excluded_review_rows.jsonl"),
            "training_rows": str(release_dir / "training_rows.jsonl"),
        },
    }
    write_json(output_root / "review_summary.json", manifest)
    write_json(release_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
