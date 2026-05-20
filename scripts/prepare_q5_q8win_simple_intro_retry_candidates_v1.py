#!/usr/bin/env python3
"""Prepare retry candidates from failed q8-win simple-intro Kimi/Qwen labels."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


DEFAULT_RUN_ROOT = Path(
    "hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/"
    "cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/eval/"
    "q8win_simple_intro_after_ckpt50/kimi_qwen_simple_intro_v2_balanced"
)
DEFAULT_OUT_DIR = Path(
    "hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/"
    "cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/eval/"
    "q8win_simple_intro_after_ckpt50/retry_after_v2/source_candidates"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def norm(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def original_sort_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("candidate_id", "")), str(row.get("paper_id", "")))


def failure_note(row: dict[str, Any], stage: str) -> str:
    prior = norm(row.get("generated_summary"))
    if stage == "local_filter":
        flags = ", ".join(str(flag) for flag in row.get("writer_local_filter_flags", []))
        return norm(
            f"Previous Kimi summary failed local gates: {flags}. "
            f"Previous summary: {prior}"
        )
    tags = ", ".join(str(tag) for tag in row.get("reviewer_issue_tags", []))
    reason = norm(row.get("reviewer_reason_span"))
    return norm(
        f"Qwen rejected previous Kimi summary. Tags: {tags}. "
        f"Reason: {reason} Previous summary: {prior}"
    )


def retry_row(row: dict[str, Any], *, new_id: str, stage: str) -> dict[str, Any]:
    out = dict(row)
    out["candidate_id"] = new_id
    out["prior_candidate_id"] = row.get("candidate_id", "")
    out["prior_kimi_summary"] = norm(row.get("generated_summary"))
    out["prior_failure_stage"] = stage
    out["prior_failure_note"] = failure_note(row, stage)
    out["prior_writer_prompt_version"] = row.get("writer_prompt_version", "")
    out["prior_reviewer_prompt_version"] = row.get("reviewer_prompt_version", "")
    out["retry_source"] = "q5_q8win50_simple_intro_retry_after_v2"
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--candidate-prefix", default="q5_q8win50_simple_intro_retry")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_excluded = load_jsonl(args.run_root / "local_eligible_for_review/excluded.jsonl")
    reviewed = load_jsonl(args.run_root / "reviewer_run/reviewer_outputs_ordered.jsonl")

    candidates: list[dict[str, Any]] = []
    seen_prior_ids: set[str] = set()

    source_rows: list[tuple[str, dict[str, Any]]] = []
    for row in local_excluded:
        if not norm(row.get("generated_summary")):
            continue
        source_rows.append(("local_filter", row))
    for row in reviewed:
        if row.get("reviewer_decision") != "reject":
            continue
        if not norm(row.get("generated_summary")):
            continue
        source_rows.append(("qwen_verifier", row))

    for stage, row in sorted(source_rows, key=lambda item: original_sort_key(item[1])):
        prior_id = str(row.get("candidate_id", ""))
        if prior_id in seen_prior_ids:
            continue
        seen_prior_ids.add(prior_id)
        new_id = f"{args.candidate_prefix}:{len(candidates) + 1:04d}"
        candidates.append(retry_row(row, new_id=new_id, stage=stage))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "retry_candidates.jsonl"
    write_jsonl(out_path, candidates)
    write_json(
        args.output_dir / "manifest.json",
        {
            "created_by": "scripts/prepare_q5_q8win_simple_intro_retry_candidates_v1.py",
            "source_run_root": str(args.run_root),
            "output": str(out_path),
            "candidate_rows": len(candidates),
            "candidate_prefix": args.candidate_prefix,
            "prior_failure_stage_counts": dict(
                collections.Counter(row["prior_failure_stage"] for row in candidates)
            ),
            "local_filter_counts": dict(
                collections.Counter(
                    flag
                    for row in candidates
                    for flag in row.get("writer_local_filter_flags", [])
                )
            ),
            "qwen_tag_counts": dict(
                collections.Counter(
                    tag
                    for row in candidates
                    for tag in row.get("reviewer_issue_tags", [])
                )
            ),
        },
    )
    print(f"[simple-intro-retry-candidates] wrote {len(candidates)} rows to {out_path}")


if __name__ == "__main__":
    main()
