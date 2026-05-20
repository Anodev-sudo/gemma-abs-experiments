#!/usr/bin/env python3
"""Build a repair-only overfit diagnostic dataset for checkpoint-218.

This is a diagnostic dataset, not a promotion candidate. It repeats the
Kimi/Qwen-accepted grounded-abstraction repairs and hydrates the original
title-plus-abstract inputs from the blind review source packet so the training
prompt matches runtime eval.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any

from generate_llama_stream import SYSTEM_PROMPT, USER_TEMPLATE

ROOT = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_hard77_burnin150_residual218_grounded_abstraction_"
    "overfit14_diag_v1_lr1e5"
)
REPAIR_SRCS = (
    Path(
        "hf_bucket/experiments/sft/"
        "q5_step2400_sft700_hard77_burnin150_residual218_grounded_abstraction_v1/"
        "kimi_qwen_repair_v1/materialized/training_rows.jsonl"
    ),
    Path(
        "hf_bucket/experiments/sft/"
        "q5_step2400_sft700_hard77_burnin150_residual218_grounded_abstraction_v1/"
        "kimi_qwen_repair_retry_v1/materialized/training_rows.jsonl"
    ),
)
SOURCE_PACKET = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7/"
    "eval/blind_ab_q8_mlx_python_vs_residual218_100/source_100.jsonl"
)

DIGIT_RE = re.compile(r"\d")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")
SENTENCE_END_RE = re.compile(r"""[.!?]+(?:['")\]]+)?(?=\s|$)""")


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


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def sentence_count(text: str) -> int:
    text = norm(text)
    if not text:
        return 0
    endings = SENTENCE_END_RE.findall(text)
    return len(endings) if endings else 1


def messages(title: str, abstract: str, summary: str) -> list[dict[str, str]]:
    user = USER_TEMPLATE.format(title=title, abstract=abstract)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": summary},
    ]


def load_repairs(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in load_jsonl(path):
            pid = str(row["paper_id"])
            if pid in seen:
                raise SystemExit(f"Duplicate repair paper_id across repair sources: {pid}")
            seen.add(pid)
            summary = norm(row["summary"])
            reject_reasons = []
            if DIGIT_RE.search(summary):
                reject_reasons.append("digit_label")
            if sentence_count(summary) != 1:
                reject_reasons.append("not_one_sentence")
            if not (8 <= word_count(summary) <= 18):
                reject_reasons.append(f"word_count_{word_count(summary)}")
            if reject_reasons:
                raise SystemExit(f"Repair label failed diagnostic screen: {pid} {reject_reasons}")
            repairs.append({**row, "summary": summary})
    if len(repairs) != 14:
        raise SystemExit(f"Expected exactly 14 accepted repair rows, got {len(repairs)}")
    return repairs


def hydrate_repair(row: dict[str, Any], source_by_pid: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pid = str(row["paper_id"])
    source = source_by_pid.get(pid)
    if not source:
        raise SystemExit(f"Missing source packet row for repair paper: {pid}")
    title = norm(source.get("title") or row.get("title"))
    abstract = norm(source.get("abstract") or row.get("abstract"))
    if not title or not abstract:
        raise SystemExit(f"Missing hydrated title or abstract for repair paper: {pid}")
    return {
        **row,
        "paper_id": pid,
        "title": title,
        "abstract": abstract,
        "primary_category": source.get("primary_category", row.get("primary_category", "")),
        "source_eval_id": source.get("eval_id", ""),
        "source_split": source.get("split", row.get("source_split", "")),
        "source_selection_bucket": source.get("selection_bucket", ""),
    }


def train_row(row: dict[str, Any], repeat_index: int, base_weight: float) -> dict[str, Any]:
    pid = str(row["paper_id"])
    summary = norm(row["summary"])
    out = {
        **row,
        "paper_id": f"{pid}::overfit14_repeat_{repeat_index:03d}",
        "source_paper_id": pid,
        "summary": summary,
        "diagnostic_only": True,
        "diagnostic_component": "grounded_abstraction_overfit14_repair",
        "weighted_numeric_patch_row_type": "grounded_abstraction_overfit14_repair",
        "messages": messages(row["title"], row["abstract"], summary),
        "loss_weight_hints": {
            "enabled": True,
            "base_weight": base_weight,
            "spans": [],
            "policy": f"repair-only overfit diagnostic; full assistant target {base_weight:g}x",
        },
    }
    return out


def eval_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_id": row.get("source_eval_id") or f"overfit14_{row['paper_id']}",
        "paper_id": row["paper_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "primary_category": row.get("primary_category", ""),
        "split": row.get("source_split", ""),
        "target_summary": row["summary"],
        "checkpoint218_summary": row.get("checkpoint218_summary", ""),
        "q8_summary": row.get("q8_summary", ""),
        "failure_family": row.get("failure_family", ""),
        "category_note": row.get("category_note", ""),
        "reviewer_note": row.get("reviewer_note", ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build repair-only q5 overfit diagnostic rows.")
    parser.add_argument("--out", type=Path, default=ROOT / "dataset_14repair_repeats100_fulltarget8_v1")
    parser.add_argument("--source-packet", type=Path, default=SOURCE_PACKET)
    parser.add_argument("--repair-src", type=Path, action="append", default=None)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--base-weight", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.base_weight <= 0:
        raise SystemExit("--base-weight must be positive")

    repair_paths = tuple(args.repair_src or REPAIR_SRCS)
    repairs = load_repairs(repair_paths)
    source_by_pid = {str(row["paper_id"]): row for row in load_jsonl(args.source_packet)}
    hydrated = [hydrate_repair(row, source_by_pid) for row in repairs]

    train_rows: list[dict[str, Any]] = []
    for repeat_index in range(args.repeats):
        for row in hydrated:
            train_rows.append(train_row(row, repeat_index, args.base_weight))

    eval_rows = [eval_row(row) for row in hydrated]
    unique_rows = [train_row(row, 0, args.base_weight) for row in hydrated]
    duplicate_train_ids = [
        pid for pid, count in collections.Counter(row["paper_id"] for row in train_rows).items() if count > 1
    ]
    if duplicate_train_ids:
        raise SystemExit(f"Duplicate training IDs: {duplicate_train_ids[:20]}")
    digit_labels = [row["paper_id"] for row in train_rows if DIGIT_RE.search(row["summary"])]
    if digit_labels:
        raise SystemExit(f"Digit labels in diagnostic dataset: {digit_labels[:20]}")
    missing_abstracts = [row["paper_id"] for row in eval_rows if not norm(row["abstract"])]
    if missing_abstracts:
        raise SystemExit(f"Missing eval abstracts after hydration: {missing_abstracts}")

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train_rows)
    write_jsonl(args.out / "eval_repair14.jsonl", eval_rows)
    write_jsonl(args.out / "unique_training_rows.jsonl", unique_rows)
    manifest = {
        "created_by": "scripts/build_q5_grounded_abstraction_overfit14_diag_v1_dataset.py",
        "diagnostic_only": True,
        "repair_sources": [str(path) for path in repair_paths],
        "source_packet": str(args.source_packet),
        "train_path": str(args.out / "train.jsonl"),
        "eval_input_path": str(args.out / "eval_repair14.jsonl"),
        "unique_rows_path": str(args.out / "unique_training_rows.jsonl"),
        "unique_repair_rows": len(hydrated),
        "repeats_per_repair": args.repeats,
        "train_rows": len(train_rows),
        "digit_label_rows": 0,
        "empty_eval_abstracts": 0,
        "loss_weight_policy": f"assistant-only full target {args.base_weight:g}x; no anchors; no hard77",
        "runtime_prompt_source": "scripts/generate_llama_stream.py SYSTEM_PROMPT and USER_TEMPLATE",
        "paper_ids": [row["paper_id"] for row in hydrated],
        "failure_family_counts": dict(collections.Counter(row.get("failure_family", "") for row in hydrated)),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[overfit14-dataset] wrote {len(train_rows)} rows to {args.out / 'train.jsonl'}")
    print(f"[overfit14-dataset] wrote eval rows to {args.out / 'eval_repair14.jsonl'}")


if __name__ == "__main__":
    main()
