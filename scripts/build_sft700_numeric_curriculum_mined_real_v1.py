#!/usr/bin/env python3
"""Materialize real mined checkpoint-700 numeric repairs for the numeric curriculum."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

NUMERIC_TAGS = {
    "unsupported_number",
    "decimal_corruption",
    "metric_role_flip",
    "number_role_binding",
    "numeric_error",
    "wrong_number_binding",
}
NUMERIC_POLICIES = {"removed_risky_number", "kept_supported_number"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def word_count(text: str) -> int:
    return len(text.strip().split())


def is_numeric_repair(row: dict[str, Any]) -> bool:
    tags = set(row.get("failure_tags", []))
    policy = row.get("number_policy")
    return bool(tags & NUMERIC_TAGS) or policy in NUMERIC_POLICIES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--accepted-repairs",
        default="hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/failure_mining/checkpoint700_10000_v1/repairs_targeted_v1/materialized/training_rows.jsonl",
    )
    ap.add_argument(
        "--output-dir",
        default="hf_bucket/experiments/sft/q5_step2400_sft700_numeric_role_curriculum_v1/real_mined_numeric_v1",
    )
    args = ap.parse_args()

    source = Path(args.accepted_repairs)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    accepted = load_jsonl(source)
    rows = []
    dropped = []
    seen = set()
    for row in accepted:
        pid = row.get("paper_id")
        if not is_numeric_repair(row):
            continue
        wc = word_count(row.get("summary", ""))
        if not 10 <= wc <= 15:
            dropped.append({**row, "drop_reason": f"target_word_count_{wc}"})
            continue
        if pid in seen:
            dropped.append({**row, "drop_reason": "duplicate_paper_id"})
            continue
        seen.add(pid)
        copied = dict(row)
        copied["source_split"] = "real_mined_sft700_numeric_repair_v1"
        copied["curriculum_component"] = "real_mined_numeric_repair"
        copied["source_training_row_path"] = str(source)
        rows.append(copied)

    write_jsonl(out / "training_rows.jsonl", rows)
    write_jsonl(out / "dropped_rows.jsonl", dropped)

    manifest = {
        "dataset_id": "real_mined_sft700_numeric_repair_v1",
        "purpose": "Real checkpoint-700 numeric failures repaired by Kimi and accepted by Qwen verifier, filtered for numeric curriculum use.",
        "source": str(source),
        "accepted_repair_rows": len(accepted),
        "numeric_curriculum_rows": len(rows),
        "dropped_rows": len(dropped),
        "selection_rule": "failure_tags intersect numeric tags OR number_policy is removed_risky_number/kept_supported_number; target summary 10-15 words; dedupe by paper_id.",
        "number_policy_counts": dict(Counter(row.get("number_policy", "") for row in rows)),
        "failure_tag_counts": dict(Counter(tag for row in rows for tag in row.get("failure_tags", []))),
        "word_count_counts": dict(sorted(Counter(word_count(row.get("summary", "")) for row in rows).items())),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
