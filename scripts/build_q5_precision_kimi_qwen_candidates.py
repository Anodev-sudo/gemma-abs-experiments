#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OUTPUT_ROOT = Path("hf_bucket/experiments/mlx_lora/q5_precision_kimi_qwen_overlay_v1")
DEFAULT_OUTPUT = OUTPUT_ROOT / "source_candidates" / "q5_precision_candidates.jsonl"
DEFAULT_REPORT = OUTPUT_ROOT / "source_candidates" / "candidate_manifest.json"

Q5_SEVERE_Q8_CLEAN = Path(
    "hf_bucket/experiments/mlx_lora/q5_hardening_distill_v1/"
    "identified_rows/q5_severe_q8_clean.jsonl"
)
FRESH_50_ROOT = Path(
    "hf_bucket/experiments/mlx_lora/review_comparisons/"
    "q8_vs_q5_guardrail_replay_75_fresh_50_v1"
)
FRESH_200_ROOT = Path(
    "hf_bucket/experiments/mlx_lora/review_comparisons/"
    "q8_vs_q5_guardrail_replay_75_fresh_200_v1"
)

VALID_TAGS = {
    "invented_number",
    "wrong_number_binding",
    "numeric_error",
    "unsupported_number",
    "unsupported_entity",
    "unsupported_entity_label",
    "unsupported_mechanism",
    "unsupported_causal_claim",
    "overstrong_claim",
    "polarity_flip",
    "polarity_risk",
    "misses_point",
    "malformed_repetition",
    "non_answer",
    "title_rehash",
    "too_generic",
    "copies_q5_failure",
    "style_drift",
    "local_filter",
    "none",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
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


def normalize_tags(tags: Any) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for tag in tags or []:
        clean = str(tag).strip()
        if clean in {"truncation", "prompt_echo"}:
            clean = "non_answer"
        if not clean or clean not in VALID_TAGS or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
    return result


def compact_note(*parts: Any) -> str:
    text = " | ".join(str(part).strip() for part in parts if str(part or "").strip())
    return " ".join(text.split())


def generation_lookup(root: Path) -> dict[str, dict[str, Any]]:
    return {str(row["paper_id"]): row for row in load_jsonl(root / "generations.jsonl")}


def side_for_q5(review_row: dict[str, Any]) -> dict[str, Any] | None:
    for side in review_row.get("sides", {}).values():
        if side.get("source") == "q5_guardrail_replay_75":
            return side
    return None


def side_for_q8(review_row: dict[str, Any]) -> dict[str, Any] | None:
    for side in review_row.get("sides", {}).values():
        if side.get("source") == "q8_promoted":
            return side
    return None


def row_from_review(
    review_row: dict[str, Any],
    generation: dict[str, Any],
    *,
    bucket: str,
    source: str,
) -> dict[str, Any] | None:
    q5 = side_for_q5(review_row)
    q8 = side_for_q8(review_row)
    if not q5:
        return None
    q5_tags = normalize_tags(q5.get("issue_tags", []))
    if not q5.get("severe") and q5.get("usable", True) and not q5_tags:
        return None
    return {
        "paper_id": review_row["paper_id"],
        "title": review_row.get("title") or generation.get("title"),
        "abstract": generation.get("abstract", ""),
        "primary_category": generation.get("primary_category"),
        "categories": generation.get("categories", []),
        "created_date": generation.get("created_date"),
        "selection_bucket": bucket,
        "q5_hardening_axes": q5_tags or ["misses_point"],
        "review_id": review_row.get("review_id"),
        "q5_summary": q5.get("summary", ""),
        "q8_summary": q8.get("summary", "") if q8 else "",
        "q8_clean": bool(q8 and q8.get("usable") and not q8.get("severe")),
        "q8_severe": bool(q8 and q8.get("severe")),
        "q8_mangled": False,
        "q5_issue_tags": q5_tags or ["misses_point"],
        "q5_severity_source": source,
        "source_split": "q5_precision_kimi_qwen_overlay_v1",
        "note": compact_note(review_row.get("note"), f"winner={review_row.get('winner')}"),
    }


def tags_from_mechanical(q5: dict[str, Any], risk_score: int) -> list[str]:
    tags: list[str] = []
    if q5.get("numbers_not_in_source"):
        tags.extend(["unsupported_number", "wrong_number_binding"])
    if q5.get("malformed_flags"):
        tags.append("malformed_repetition")
    if q5.get("over20"):
        tags.append("style_drift")
    if risk_score >= 8:
        tags.append("misses_point")
    return normalize_tags(tags) or ["misses_point"]


def row_from_mechanical(record: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any] | None:
    q5 = record.get("q5", {})
    risk_score = int(record.get("risk_score", 0) or 0)
    if not (
        q5.get("numbers_not_in_source")
        or q5.get("malformed_flags")
        or q5.get("over20")
        or risk_score >= 8
    ):
        return None
    tags = tags_from_mechanical(q5, risk_score)
    return {
        "paper_id": record["paper_id"],
        "title": record.get("title") or generation.get("title"),
        "abstract": generation.get("abstract", ""),
        "primary_category": generation.get("primary_category") or record.get("primary_category"),
        "categories": generation.get("categories", []),
        "created_date": generation.get("created_date"),
        "selection_bucket": "fresh_replay75_mechanical_risk",
        "q5_hardening_axes": tags,
        "review_id": f"mechanical:{record['paper_id']}",
        "q5_summary": q5.get("summary", ""),
        "q8_summary": record.get("q8", {}).get("summary", ""),
        "q8_clean": None,
        "q8_severe": None,
        "q8_mangled": None,
        "q5_issue_tags": tags,
        "q5_severity_source": "mechanical_risk:q5_guardrail_replay_75",
        "source_split": "q5_precision_kimi_qwen_overlay_v1",
        "note": compact_note(
            f"risk_score={risk_score}",
            f"numbers_not_in_source={q5.get('numbers_not_in_source') or []}",
            f"malformed_flags={q5.get('malformed_flags') or []}",
        ),
    }


def row_from_hardening(row: dict[str, Any]) -> dict[str, Any]:
    tags = normalize_tags(row.get("q5_issue_tags", [])) or ["misses_point"]
    return {
        **row,
        "q5_issue_tags": tags,
        "q5_hardening_axes": normalize_tags(row.get("q5_hardening_axes", [])) or tags,
        "q5_severity_source": "mined_1000:q5_severe_q8_clean",
        "source_split": "q5_precision_kimi_qwen_overlay_v1",
    }


def add_unique(
    rows: list[dict[str, Any]],
    seen: set[str],
    row: dict[str, Any] | None,
    *,
    source_counts: Counter[str],
) -> None:
    if not row:
        return
    paper_id = str(row.get("paper_id", ""))
    if not paper_id or paper_id in seen:
        return
    if not row.get("abstract") or not row.get("q5_summary"):
        return
    seen.add(paper_id)
    source_counts[str(row.get("q5_severity_source"))] += 1
    rows.append(row)


def build_candidates() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_counts: Counter[str] = Counter()

    for root, review_rel, bucket, source in [
        (
            FRESH_200_ROOT,
            Path("risk_blind_review_v1/unblinded_review_summary.json"),
            "fresh_replay75_risk_review",
            "fresh_200_risk_review:q5_guardrail_replay_75",
        ),
        (
            FRESH_50_ROOT,
            Path("blind_review_v1/unblinded_review_summary.json"),
            "fresh_replay75_blind_review",
            "fresh_50_blind_review:q5_guardrail_replay_75",
        ),
    ]:
        generations = generation_lookup(root)
        review_path = root / review_rel
        if not review_path.exists():
            continue
        review_rows = json.loads(review_path.read_text(encoding="utf-8")).get("rows", [])
        for review_row in review_rows:
            generation = generations.get(str(review_row.get("paper_id")), {})
            add_unique(
                rows,
                seen,
                row_from_review(review_row, generation, bucket=bucket, source=source),
                source_counts=source_counts,
            )

    generations_200 = generation_lookup(FRESH_200_ROOT)
    mechanical_path = FRESH_200_ROOT / "mechanical_risk_screen.json"
    if mechanical_path.exists():
        records = json.loads(mechanical_path.read_text(encoding="utf-8")).get("records", [])
        for record in sorted(records, key=lambda item: int(item.get("risk_score", 0) or 0), reverse=True):
            generation = generations_200.get(str(record.get("paper_id")), {})
            add_unique(
                rows,
                seen,
                row_from_mechanical(record, generation),
                source_counts=source_counts,
            )

    for row in load_jsonl(Q5_SEVERE_Q8_CLEAN):
        add_unique(rows, seen, row_from_hardening(row), source_counts=source_counts)

    manifest = {
        "created_at": utc_now_iso(),
        "candidate_rows": len(rows),
        "source_counts": dict(source_counts),
        "selection_bucket_counts": dict(Counter(row.get("selection_bucket") for row in rows)),
        "q5_issue_tag_counts": dict(
            Counter(tag for row in rows for tag in row.get("q5_issue_tags", []))
        ),
        "policy": [
            "Prioritize current replay_75 blind-review failures.",
            "Add fresh mechanical-risk replay_75 rows for malformed text, unsupported numbers, overlong outputs, and high risk score.",
            "Fill with mined q5 severe rows where q8 was clean, deduplicated by paper_id.",
        ],
    }
    return rows, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build focused source rows for the q5 precision Kimi/Qwen overlay."
    )
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest-json", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, manifest = build_candidates()
    write_jsonl(Path(args.output_jsonl), rows)
    manifest["outputs"] = {
        "candidates": args.output_jsonl,
        "manifest": args.manifest_json,
    }
    write_json(Path(args.manifest_json), manifest)
    print(
        f"[q5-precision-candidates] wrote {len(rows)} rows to {args.output_jsonl}"
    )


if __name__ == "__main__":
    main()
