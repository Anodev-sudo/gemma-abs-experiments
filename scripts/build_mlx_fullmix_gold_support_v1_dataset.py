#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_GOLD_DIR = Path(
    "hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/"
    "mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5"
)
DEFAULT_SUPPORT_PATHS = [
    Path(
        "hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_specificity_support_v1/"
        "dataset_clean1648_broadsupport_v1/train.jsonl"
    ),
    Path(
        "hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_guarded_support_v1/"
        "dataset_clean1648_under14support482_numericguard165_v1/train.jsonl"
    ),
]
DEFAULT_OUTPUT_DIR = Path(
    "hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_fullmix_gold4_support_v1/"
    "dataset_title_abstract_feed_blurb_v5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a frequency-weighted MLX full-finetune dataset."
    )
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--support-path", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gold-repeat", type=int, default=4)
    parser.add_argument("--exclude-component", action="append", default=[])
    parser.add_argument("--exclude-source-split", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260527)
    return parser.parse_args()


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
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def assistant_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Row {row.get('paper_id')} is missing messages.")
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    raise ValueError(f"Row {row.get('paper_id')} has no assistant message.")


def validate_chat_row(row: dict[str, Any], *, expected_prompt: str) -> None:
    if not row.get("paper_id"):
        raise ValueError("Found row without paper_id.")
    if row.get("prompt_version") != expected_prompt:
        raise ValueError(
            f"Unexpected prompt_version for {row.get('paper_id')}: {row.get('prompt_version')}"
        )
    if row.get("input_mode") != "title_abstract":
        raise ValueError(f"Unexpected input_mode for {row.get('paper_id')}: {row.get('input_mode')}")
    if not assistant_text(row):
        raise ValueError(f"Empty assistant text for {row.get('paper_id')}.")


def clone_for_training(row: dict[str, Any], *, tier: str, repeat_index: int, source_path: Path) -> dict[str, Any]:
    cloned = dict(row)
    cloned["fullmix_tier"] = tier
    cloned["fullmix_repeat_index"] = repeat_index
    cloned["fullmix_source_path"] = str(source_path)
    return cloned


def main() -> None:
    args = parse_args()
    if args.gold_repeat < 1:
        raise ValueError("--gold-repeat must be at least 1.")

    support_paths = args.support_path if args.support_path is not None else DEFAULT_SUPPORT_PATHS
    expected_prompt = "gemma_270m_it_abstract_to_feed_blurb_v5"

    gold_train_path = args.gold_dir / "train.jsonl"
    gold_valid_path = args.gold_dir / "valid.jsonl"
    gold_test_path = args.gold_dir / "test.jsonl"
    gold_train_rows = load_jsonl(gold_train_path)
    gold_valid_rows = load_jsonl(gold_valid_path)
    gold_test_rows = load_jsonl(gold_test_path)
    for row in gold_train_rows + gold_valid_rows + gold_test_rows:
        validate_chat_row(row, expected_prompt=expected_prompt)

    all_gold_ids = {str(row["paper_id"]) for row in gold_train_rows + gold_valid_rows + gold_test_rows}
    seen_support_ids = set(all_gold_ids)
    support_rows: list[dict[str, Any]] = []
    support_source_counts: Counter[str] = Counter()
    support_component_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    excluded_components = set(args.exclude_component)
    excluded_source_splits = set(args.exclude_source_split)

    for support_path in support_paths:
        for row in load_jsonl(support_path):
            validate_chat_row(row, expected_prompt=expected_prompt)
            paper_id = str(row["paper_id"])
            component = str(row.get("dataset_component") or "unknown")
            source_split = str(row.get("source_split") or "unknown")
            if paper_id in all_gold_ids:
                skipped_counts[f"{support_path}::gold_overlap"] += 1
                continue
            if paper_id in seen_support_ids:
                skipped_counts[f"{support_path}::duplicate_after_priority"] += 1
                continue
            if component in excluded_components:
                skipped_counts[f"{support_path}::excluded_component::{component}"] += 1
                continue
            if source_split in excluded_source_splits:
                skipped_counts[f"{support_path}::excluded_source_split::{source_split}"] += 1
                continue
            seen_support_ids.add(paper_id)
            support_rows.append(
                clone_for_training(
                    row,
                    tier="support_once",
                    repeat_index=0,
                    source_path=support_path,
                )
            )
            support_source_counts[source_split] += 1
            support_component_counts[component] += 1

    train_rows: list[dict[str, Any]] = []
    for repeat_index in range(args.gold_repeat):
        for row in gold_train_rows:
            train_rows.append(
                clone_for_training(
                    row,
                    tier="gold_train_repeat",
                    repeat_index=repeat_index,
                    source_path=gold_train_path,
                )
            )
    train_rows.extend(support_rows)
    random.Random(args.seed).shuffle(train_rows)

    valid_rows = [
        clone_for_training(row, tier="gold_valid_holdout", repeat_index=0, source_path=gold_valid_path)
        for row in gold_valid_rows
    ]
    test_rows = [
        clone_for_training(row, tier="gold_test_holdout", repeat_index=0, source_path=gold_test_path)
        for row in gold_test_rows
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "valid.jsonl", valid_rows)
    write_jsonl(args.output_dir / "test.jsonl", test_rows)

    tier_counts = Counter(str(row["fullmix_tier"]) for row in train_rows)
    train_unique_paper_ids = {str(row["paper_id"]) for row in train_rows}
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "scripts/build_mlx_fullmix_gold_support_v1_dataset.py",
        "purpose": "Frequency-weighted full-finetune dataset: strict gold repeated, lower-quality support once.",
        "prompt_version": expected_prompt,
        "input_mode": "title_abstract",
        "gold_dir": str(args.gold_dir),
        "gold_repeat": args.gold_repeat,
        "support_paths": [str(path) for path in support_paths],
        "excluded_components": sorted(excluded_components),
        "excluded_source_splits": sorted(excluded_source_splits),
        "seed": args.seed,
        "split_counts": {
            "train_examples": len(train_rows),
            "valid_examples": len(valid_rows),
            "test_examples": len(test_rows),
            "train_unique_paper_ids": len(train_unique_paper_ids),
        },
        "train_tier_counts": dict(sorted(tier_counts.items())),
        "gold_counts": {
            "train_unique": len(gold_train_rows),
            "valid_unique": len(gold_valid_rows),
            "test_unique": len(gold_test_rows),
            "all_gold_unique": len(all_gold_ids),
            "gold_train_exposures": len(gold_train_rows) * args.gold_repeat,
        },
        "support_counts": {
            "selected_unique": len(support_rows),
            "source_split_counts": dict(support_source_counts.most_common()),
            "component_counts": dict(support_component_counts.most_common()),
            "skipped_counts": dict(sorted(skipped_counts.items())),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote MLX fullmix dataset to {args.output_dir}")
    print(json.dumps(manifest["split_counts"], indent=2, sort_keys=True))
    print(json.dumps(manifest["train_tier_counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
