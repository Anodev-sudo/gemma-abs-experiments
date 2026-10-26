#!/usr/bin/env python3
"""Combine clean rows, under-14 support, and numeric guard rows."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CLEAN = Path(
    "hf_bucket/experiments/sft/q5_checkpoint50_natural_language_clean_digitfree_overlay_v1/"
    "dataset_clean_digitfree_10to14_no_supdigit_v1/train.jsonl"
)
DEFAULT_UNDER14_SUPPORT = Path(
    "hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_under14_highsupport_v1/"
    "dataset_clean1648_under14_highsupport_v1/train.jsonl"
)
DEFAULT_NUMERIC_GUARD = Path(
    "hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_numeric_guard_v1/"
    "dataset_clean1648_numericguard200_v1/train.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_guarded_support_v1/"
    "dataset_clean1648_under14support482_numericguard165_v1"
)

DIGIT_OR_SUP_RE = re.compile(r"[0-9\u00b2\u00b3\u00b9\u2070-\u2079\u2080-\u2089]")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dataset", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--under14-support-dataset", type=Path, default=DEFAULT_UNDER14_SUPPORT)
    parser.add_argument("--numeric-guard-dataset", type=Path, default=DEFAULT_NUMERIC_GUARD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def component(row: dict[str, Any]) -> str:
    return str(row.get("dataset_component", ""))


def main() -> None:
    args = parse_args()
    clean_rows = load_jsonl(args.clean_dataset)
    clean_ids = {str(row["paper_id"]) for row in clean_rows}

    raw_under14_rows = [
        row
        for row in load_jsonl(args.under14_support_dataset)
        if component(row).startswith("under14_highsupport::")
    ]
    numeric_rows = [
        row
        for row in load_jsonl(args.numeric_guard_dataset)
        if component(row).startswith("numeric_guard::")
    ]

    used_ids = set(clean_ids)
    combined = list(clean_rows)
    numeric_duplicate_ids: list[str] = []
    for row in numeric_rows:
        paper_id = str(row["paper_id"])
        if paper_id in used_ids:
            numeric_duplicate_ids.append(paper_id)
            continue
        combined.append(row)
        used_ids.add(paper_id)

    under14_rows: list[dict[str, Any]] = []
    under14_duplicate_ids: list[str] = []
    for row in raw_under14_rows:
        paper_id = str(row["paper_id"])
        if paper_id in used_ids:
            under14_duplicate_ids.append(paper_id)
            continue
        combined.append(row)
        under14_rows.append(row)
        used_ids.add(paper_id)

    ids = [str(row["paper_id"]) for row in combined]
    if len(ids) != len(set(ids)):
        counts = Counter(ids)
        dupes = sorted(pid for pid, count in counts.items() if count > 1)
        raise ValueError(f"combined dataset contains duplicate paper_id values: {dupes[:20]}")

    digit_rows = [row["paper_id"] for row in combined if DIGIT_OR_SUP_RE.search(str(row.get("summary", "")))]
    if digit_rows:
        raise ValueError(f"combined dataset has digit-bearing summaries: {digit_rows[:20]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    write_jsonl(train_path, combined)

    component_counts = Counter(component(row) for row in combined)
    weights = Counter(str((row.get("loss_weight_hints") or {}).get("base_weight")) for row in combined)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "scripts/build_q5_natural_language_guarded_support_v1_dataset.py",
        "purpose": "Existing-data-only clean stabilizer plus under-14 support plus numeric guard rows.",
        "train_path": str(train_path),
        "row_count": len(combined),
        "clean_rows": len(clean_rows),
        "under14_support_rows": len(under14_rows),
        "under14_support_source_rows": len(raw_under14_rows),
        "under14_duplicate_ids_dropped": len(under14_duplicate_ids),
        "numeric_guard_rows": len(numeric_rows),
        "numeric_duplicate_ids_dropped": len(numeric_duplicate_ids),
        "clean_id_count": len(clean_ids),
        "component_counts": dict(sorted(component_counts.items())),
        "base_weight_counts": dict(sorted(weights.items())),
        "word_count_distribution": dict(sorted(Counter(word_count(str(row.get("summary", ""))) for row in combined).items())),
        "sources": {
            "clean_dataset": str(args.clean_dataset),
            "under14_support_dataset": str(args.under14_support_dataset),
            "numeric_guard_dataset": str(args.numeric_guard_dataset),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[guarded-support] wrote {train_path}")
    print(json.dumps({k: manifest[k] for k in ("row_count", "clean_rows", "under14_support_rows", "numeric_guard_rows")}, indent=2))


if __name__ == "__main__":
    main()
