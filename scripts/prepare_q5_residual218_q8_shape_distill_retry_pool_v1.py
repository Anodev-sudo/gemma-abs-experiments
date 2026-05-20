#!/usr/bin/env python3
"""Prepare a retry pool from q8-shape rows rejected by local/Qwen screens."""
from __future__ import annotations

import argparse
import collections
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PRIOR_ROOT = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_residual218_q8_shape_distill_v1/"
    "kimi_qwen_repair_v1"
)
DEFAULT_OUT = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_residual218_q8_shape_distill_v1/"
    "deepseek_v4_retry_v1/candidate_pool/retry_candidates.jsonl"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def candidate_from_rejected(row: dict[str, Any], *, rejection_stage: str, index: int) -> dict[str, Any]:
    prior_flags = row.get("writer_hard_filter_flags", [])
    verifier_tags = row.get("verifier_issue_tags", [])
    note_parts = [
        str(row.get("category_note", "")),
        f"prior_rejection_stage: {rejection_stage}",
        f"prior_writer_summary: {norm(row.get('repair_summary', ''))}",
    ]
    if prior_flags:
        note_parts.append(f"prior_local_flags: {', '.join(str(flag) for flag in prior_flags)}")
    if verifier_tags:
        note_parts.append(f"prior_qwen_tags: {', '.join(str(tag) for tag in verifier_tags)}")
    if row.get("verifier_reason_span"):
        note_parts.append(f"prior_qwen_reason: {norm(row.get('verifier_reason_span', ''))}")
    return {
        "paper_id": str(row["paper_id"]),
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "primary_category": row.get("primary_category", ""),
        "q8_summary": row.get("q8_summary", ""),
        "checkpoint_218_summary": row.get("bad_summary", ""),
        "category": row.get("failure_family", ""),
        "category_note": " | ".join(part for part in note_parts if part),
        "reviewer_note": row.get("reviewer_note", ""),
        "more_accurate_model": row.get("more_accurate_model", ""),
        "preferred_model": row.get("preferred_model", ""),
        "source_split": row.get("source_split", ""),
        "source_review_id": row.get("source_review_id", ""),
        "retry_source_candidate_id": row.get("candidate_id", ""),
        "retry_candidate_id": f"q8_shape_deepseek_retry:{index:04d}",
        "prior_rejection_stage": rejection_stage,
        "prior_writer_model": row.get("writer_model", ""),
        "prior_writer_prompt_version": row.get("writer_prompt_version", ""),
        "prior_writer_hard_filter_flags": prior_flags,
        "prior_qwen_issue_tags": verifier_tags,
        "rewrite_source": "q8_shape_distill_kimi_qwen_rejected_retry_v1",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare q8-shape DeepSeek retry pool.")
    parser.add_argument("--prior-root", type=Path, default=DEFAULT_PRIOR_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rejected = load_jsonl(args.prior_root / "local_screen/rejected_before_qwen.jsonl")
    qwen_rejected = load_jsonl(args.prior_root / "materialized/rejected_after_qwen.jsonl")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    index = 1
    for rejection_stage, source_rows in (
        ("local_screen", local_rejected),
        ("qwen_verification", qwen_rejected),
    ):
        for row in source_rows:
            key = str(row.get("candidate_id") or row.get("paper_id"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(candidate_from_rejected(row, rejection_stage=rejection_stage, index=index))
            index += 1

    write_jsonl(args.out, rows)
    write_json(
        args.out.with_name("manifest.json"),
        {
            "created_at": utc_now(),
            "created_by": "scripts/prepare_q5_residual218_q8_shape_distill_retry_pool_v1.py",
            "prior_root": str(args.prior_root),
            "retry_rows": len(rows),
            "local_screen_rejected": len(local_rejected),
            "qwen_rejected": len(qwen_rejected),
            "prior_rejection_stage_counts": dict(collections.Counter(row["prior_rejection_stage"] for row in rows)),
            "category_counts": dict(collections.Counter(row.get("category", "") for row in rows)),
        },
    )
    print(f"[q8-shape-retry-pool] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
