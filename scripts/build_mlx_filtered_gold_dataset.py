#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_GOLD_DIR = Path(
    "hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/"
    "mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5"
)
DEFAULT_OUTPUT_DIR = Path(
    "hf_bucket/experiments/mlx_lora/"
    "gemma_3_270m_it_natural_language_10746_heuristic3000_goldfinish_v1/"
    "dataset_gold_digitformula_free_feed_blurb_v5"
)
DIGIT_RE = re.compile(r"\d")
FORMULAISH_RE = re.compile(r"[$\\{}_=<>≤≥∑√∞]|\\b[A-Za-z]\^[0-9]|\\bO\(")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter strict gold MLX chat rows by target shape.")
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--drop-digit-targets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-formula-targets", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def assistant_text(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return " ".join(str(message.get("content", "")).split())
    return ""


def exclusion_reasons(summary: str, *, drop_digit_targets: bool, drop_formula_targets: bool) -> list[str]:
    reasons: list[str] = []
    if drop_digit_targets and DIGIT_RE.search(summary):
        reasons.append("digit_target")
    if drop_formula_targets and FORMULAISH_RE.search(summary):
        reasons.append("formulaish_target")
    return reasons


def filter_split(
    rows: list[dict[str, Any]],
    *,
    drop_digit_targets: bool,
    drop_formula_targets: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        summary = assistant_text(row)
        reasons = exclusion_reasons(
            summary,
            drop_digit_targets=drop_digit_targets,
            drop_formula_targets=drop_formula_targets,
        )
        if reasons:
            dropped.append(
                {
                    "paper_id": row.get("paper_id"),
                    "primary_category": row.get("primary_category"),
                    "summary": summary,
                    "exclusion_reasons": reasons,
                }
            )
            continue
        kept_row = dict(row)
        kept_row["dataset_component"] = "strict_gold_digitformula_free"
        kept.append(kept_row)
    return kept, dropped


def main() -> None:
    args = parse_args()
    train_rows = load_jsonl(args.gold_dir / "train.jsonl")
    valid_rows = load_jsonl(args.gold_dir / "valid.jsonl")
    test_rows = load_jsonl(args.gold_dir / "test.jsonl")

    train_kept, train_dropped = filter_split(
        train_rows,
        drop_digit_targets=args.drop_digit_targets,
        drop_formula_targets=args.drop_formula_targets,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train_kept)
    write_jsonl(args.output_dir / "valid.jsonl", valid_rows)
    write_jsonl(args.output_dir / "test.jsonl", test_rows)
    write_jsonl(args.output_dir / "dropped_train_rows.jsonl", train_dropped)

    reason_counts = Counter(reason for row in train_dropped for reason in row["exclusion_reasons"])
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "scripts/build_mlx_filtered_gold_dataset.py",
        "source_gold_dir": str(args.gold_dir),
        "output_dir": str(args.output_dir),
        "drop_digit_targets": args.drop_digit_targets,
        "drop_formula_targets": args.drop_formula_targets,
        "train_source_rows": len(train_rows),
        "train_rows": len(train_kept),
        "train_dropped_rows": len(train_dropped),
        "valid_rows": len(valid_rows),
        "test_rows": len(test_rows),
        "drop_reason_counts": dict(reason_counts.most_common()),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = [
        "# Filtered strict-gold feed-blurb dataset",
        "",
        f"- Source train rows: `{len(train_rows)}`",
        f"- Kept train rows: `{len(train_kept)}`",
        f"- Dropped train rows: `{len(train_dropped)}`",
        f"- Drop reasons: `{dict(reason_counts.most_common())}`",
        "",
        "The validation and test splits are copied from the full strict-gold release.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"[filtered-gold] wrote {len(train_kept)} train rows to {args.output_dir}")
    print(f"[filtered-gold] dropped={dict(reason_counts.most_common())}")


if __name__ == "__main__":
    main()
