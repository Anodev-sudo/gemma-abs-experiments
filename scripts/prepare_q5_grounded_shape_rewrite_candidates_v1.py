#!/usr/bin/env python3
"""Prepare quarantined grounded-shape rows for a fresh Kimi/Qwen rewrite pass."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_QUARANTINED = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_residual218_grounded_shape_scaffold_v1/"
    "rewrite_candidate_pack_v1/quarantined_grounded_rewrite_candidates.jsonl"
)
DEFAULT_FAILURE_SHAPE = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_hard77_burnin150_residual218_grounded_abstraction_"
    "overfit14_diag_v1_lr1e5/eval/repair14_failure_shape_diagnostic.jsonl"
)
DEFAULT_OUT = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_residual218_grounded_shape_scaffold_v1/"
    "rewrite_candidate_pack_v1/grounded_shape_rewrite_review_input.jsonl"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def norm(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def trajectory_note(diag: dict[str, Any]) -> str:
    wanted = {"checkpoint_70", "checkpoint_120", "checkpoint_150"}
    bits: list[str] = []
    for item in diag.get("trajectory", []):
        label = str(item.get("label", ""))
        if label not in wanted:
            continue
        bits.append(f"{label}: {norm(item.get('output', ''))}")
    return " | ".join(bits)


def build_row(row: dict[str, Any], diag_by_pid: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pid = str(row["paper_id"])
    diag = diag_by_pid.get(pid, {})
    prior_target = norm(row.get("summary", ""))
    note_parts = [
        f"diagnostic bucket: {row.get('bucket', '')}",
        f"old target to reconsider: {prior_target}",
        f"manual verdict: {row.get('manual_verdict', '')}",
        f"likely issue: {row.get('likely_issue', '')}",
        f"rewrite guidance: {row.get('next_label_guidance', '')}",
        f"overfit trajectory: {trajectory_note(diag)}",
    ]
    return {
        "paper_id": pid,
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "primary_category": row.get("primary_category", ""),
        "q8_summary": row.get("q8_summary", ""),
        "checkpoint_218_summary": row.get("checkpoint218_summary", ""),
        "category": row.get("bucket", ""),
        "category_note": " ".join(part for part in note_parts if part.strip()),
        "reviewer_note": row.get("manual_verdict", ""),
        "more_accurate_model": "fresh_kimi_rewrite_required",
        "prior_target_summary": prior_target,
        "target_source_support": row.get("target_source_support", 0.0),
        "target_terms_not_in_source": row.get("target_terms_not_in_source", []),
        "rewrite_source": "q5_grounded_shape_quarantine_v1",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare grounded-shape rewrite candidates.")
    parser.add_argument("--quarantined", type=Path, default=DEFAULT_QUARANTINED)
    parser.add_argument("--failure-shape", type=Path, default=DEFAULT_FAILURE_SHAPE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quarantined = load_jsonl(args.quarantined)
    diag_by_pid = {str(row["paper_id"]): row for row in load_jsonl(args.failure_shape)}
    rows = [build_row(row, diag_by_pid) for row in quarantined]
    write_jsonl(args.out, rows)
    manifest = {
        "created_by": "scripts/prepare_q5_grounded_shape_rewrite_candidates_v1.py",
        "quarantined_source": str(args.quarantined),
        "failure_shape_source": str(args.failure_shape),
        "review_input": str(args.out),
        "rows": len(rows),
        "paper_ids": [row["paper_id"] for row in rows],
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[grounded-shape-rewrite-candidates] wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
