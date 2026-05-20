#!/usr/bin/env python3
"""Materialize accepted clean2250 runtime-drift repairs into training rows."""
from __future__ import annotations

import argparse
import collections
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_llama_stream import SYSTEM_PROMPT, USER_TEMPLATE


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def norm(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def word_count(value: str) -> int:
    return len(WORD_RE.findall(value))


def training_row(row: dict[str, Any]) -> dict[str, Any]:
    summary = norm(row["repair_summary"])
    summary_words = word_count(summary)
    drift_flags = list(row.get("issue_flags", []))
    return {
        "paper_id": str(row["paper_id"]),
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "summary": summary,
        "summary_word_count": summary_words,
        "primary_category": row.get("primary_category", ""),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date", ""),
        "teacher_model": row.get("writer_model", ""),
        "generator_prompt_version": row.get("writer_prompt_version", ""),
        "review_prompt_version": row.get("reviewer_prompt_version", ""),
        "source_split": "clean2250_runtime_drift_repair_v1",
        "dataset_component": "clean2250_runtime_drift_repair_v1",
        "runtime_drift_candidate_split": row.get("split", ""),
        "runtime_drift_issue_flags": drift_flags,
        "runtime_drift_score": row.get("runtime_drift_score", 0),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(title=row.get("title", ""), abstract=row.get("abstract", ""))},
            {"role": "assistant", "content": summary},
        ],
        "loss_weight_hints": {
            "enabled": True,
            "base_weight": 2.2,
            "spans": [],
            "reason": "Qwen-verified repair for GGUF checkpoint-800 runtime drift from native PEFT behavior.",
        },
        "runtime_drift_repair_v1": {
            "candidate_id": row.get("candidate_id", ""),
            "eval_id": row.get("eval_id", ""),
            "bad_gguf_checkpoint_800_summary": row.get("bad_summary", ""),
            "native_peft_checkpoint_800_reference_summary": row.get("q8_summary", ""),
            "baseline_2250_summary": row.get("baseline_2250_summary", ""),
            "runtime_drift_flags": drift_flags,
            "runtime_drift_score": row.get("runtime_drift_score", 0),
            "writer_evidence_span": row.get("writer_evidence_span", ""),
            "writer_reason_span": row.get("writer_reason_span", ""),
            "verifier_evidence_span": row.get("verifier_evidence_span", ""),
            "verifier_reason_span": row.get("verifier_reason_span", ""),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_path = args.output_root / "qwen_verification_run" / "verifications_ordered.jsonl"
    local_reject_path = args.output_root / "local_screen" / "rejected_before_qwen.jsonl"
    if not verify_path.exists():
        raise SystemExit(f"Missing verification file: {verify_path}")
    verified = load_jsonl(verify_path)
    local_rejected = load_jsonl(local_reject_path) if local_reject_path.exists() else []
    accepted = [
        row
        for row in verified
        if row.get("verifier_decision") == "accept"
        and row.get("repair_summary")
        and not row.get("post_qwen_hard_filter_flags")
        and word_count(row.get("repair_summary", "")) <= 14
    ]
    rejected_after_qwen = [row for row in verified if row not in accepted]
    training_rows = [training_row(row) for row in accepted]

    out_dir = args.output_root / "materialized"
    write_jsonl(out_dir / "accepted_repairs.jsonl", accepted)
    write_jsonl(out_dir / "rejected_after_qwen.jsonl", rejected_after_qwen)
    write_jsonl(out_dir / "rejected_before_qwen.jsonl", local_rejected)
    write_jsonl(out_dir / "training_rows.jsonl", training_rows)
    write_json(
        out_dir / "run_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "verified_rows": len(verified),
            "accepted_rows": len(accepted),
            "rejected_after_qwen": len(rejected_after_qwen),
            "rejected_before_qwen": len(local_rejected),
            "accepted_word_count_max": max((word_count(row["summary"]) for row in training_rows), default=0),
            "accepted_word_count_avg": round(
                sum(word_count(row["summary"]) for row in training_rows) / max(len(training_rows), 1),
                2,
            ),
            "accepted_split_counts": dict(collections.Counter(row.get("split", "") for row in accepted)),
            "accepted_issue_flag_counts": dict(collections.Counter(flag for row in accepted for flag in row.get("issue_flags", []))),
            "source": "GGUF checkpoint-800 runtime drift from native PEFT checkpoint-800 on the 454-row heuristic set.",
        },
    )
    summary_lines = [
        "# clean2250 runtime drift repair v1",
        "",
        f"- Verified rows: {len(verified)}",
        f"- Accepted repairs: {len(accepted)}",
        f"- Rejected after Qwen: {len(rejected_after_qwen)}",
        f"- Rejected before Qwen: {len(local_rejected)}",
        f"- Training rows: {len(training_rows)}",
        "",
        "This shard targets rows where GGUF checkpoint 800 drifted longer, repetitive, or more numeric than native PEFT checkpoint 800.",
    ]
    (out_dir / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[materialize-runtime-drift] accepted={len(accepted)} training_rows={len(training_rows)} out={out_dir}")


if __name__ == "__main__":
    main()
