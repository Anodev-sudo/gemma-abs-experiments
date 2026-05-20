#!/usr/bin/env python3
"""Build the clean2250 runtime-drift repair v1 SFT dataset."""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
SENTENCE_END_RE = re.compile(r"""[.!?]+(?:['")\]]+)?(?=\s|$)""")


DEFAULT_REPAIRS = Path(
    "hf_bucket/experiments/sft/gemma270m_clean_slate_under14_v1/lr1_5e5/"
    "semantic_binding_repair_v1/lr6e7/runtime_drift_repair_v1/materialized/training_rows.jsonl"
)
DEFAULT_ANCHORS = Path(
    "hf_bucket/experiments/sft/gemma270m_clean_slate_under14_v1/lr1_5e5/"
    "semantic_binding_repair_v1/dataset_semantic_binding_repair_v1/train.jsonl"
)
DEFAULT_OUT = Path(
    "hf_bucket/experiments/sft/gemma270m_clean_slate_under14_v1/lr1_5e5/"
    "semantic_binding_repair_v1/lr6e7/runtime_drift_repair_v1/"
    "dataset_runtime_drift_repair_v1"
)


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


def sentence_count(value: str) -> int:
    text = norm(value)
    if not text:
        return 0
    endings = SENTENCE_END_RE.findall(text)
    return len(endings) if endings else 1


def summary_from_row(row: dict[str, Any]) -> str:
    if row.get("summary"):
        return norm(row["summary"])
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return norm(messages[-1].get("content", ""))
    return ""


def with_weight(row: dict[str, Any], *, weight: float, component: str, policy: str) -> dict[str, Any]:
    out = dict(row)
    summary = summary_from_row(out)
    out["summary"] = summary
    out["summary_word_count"] = word_count(summary)
    out["dataset_component"] = component
    out["source_split"] = component
    out["loss_weight_hints"] = {
        "enabled": True,
        "base_weight": weight,
        "spans": [],
        "policy": policy,
    }
    return out


def anchor_ok(row: dict[str, Any], excluded_ids: set[str]) -> bool:
    if str(row.get("paper_id", "")) in excluded_ids:
        return False
    summary = summary_from_row(row)
    if not summary:
        return False
    if word_count(summary) > 14:
        return False
    if sentence_count(summary) != 1:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repairs", type=Path, default=DEFAULT_REPAIRS)
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--anchor-count", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--repair-weight", type=float, default=2.4)
    parser.add_argument("--anchor-weight", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repairs_raw = load_jsonl(args.repairs)
    excluded_ids = {str(row.get("paper_id", "")) for row in repairs_raw}
    anchor_pool = [row for row in load_jsonl(args.anchors) if anchor_ok(row, excluded_ids)]
    rng = random.Random(args.seed)
    anchors = rng.sample(anchor_pool, min(args.anchor_count, len(anchor_pool)))

    repairs = [
        with_weight(
            row,
            weight=args.repair_weight,
            component="runtime_drift_repair_qwen_accepted",
            policy="runtime_drift_repair_v1 accepted repair high weight",
        )
        for row in repairs_raw
    ]
    anchor_rows = [
        with_weight(
            row,
            weight=args.anchor_weight,
            component="semantic_binding_anchor_low_weight",
            policy="runtime_drift_repair_v1 low-weight preservation anchor",
        )
        for row in anchors
    ]
    final_rows = [*repairs, *anchor_rows]
    rng.shuffle(final_rows)

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", final_rows)
    write_jsonl(args.out / "repairs.jsonl", repairs)
    write_jsonl(args.out / "anchors.jsonl", anchor_rows)

    summary_word_counts = [word_count(summary_from_row(row)) for row in final_rows]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repairs_path": str(args.repairs),
        "anchors_path": str(args.anchors),
        "train_rows": len(final_rows),
        "repair_rows": len(repairs),
        "anchor_rows": len(anchor_rows),
        "anchor_pool_rows": len(anchor_pool),
        "seed": args.seed,
        "repair_weight": args.repair_weight,
        "anchor_weight": args.anchor_weight,
        "component_counts": dict(collections.Counter(row.get("dataset_component", "") for row in final_rows)),
        "max_summary_words": max(summary_word_counts, default=0),
        "avg_summary_words": round(sum(summary_word_counts) / max(len(summary_word_counts), 1), 2),
        "over14_rows": sum(1 for count in summary_word_counts if count > 14),
        "purpose": (
            "Train a small continuation from checkpoint 800 against GGUF sidecar drift, "
            "with low-weight existing semantic-binding anchors to preserve prior behavior."
        ),
    }
    write_json(args.out / "manifest.json", manifest)
    (args.out / "README.md").write_text(
        "\n".join(
            [
                "# clean2250 runtime drift repair v1 dataset",
                "",
                f"- Train rows: {len(final_rows)}",
                f"- Qwen-accepted runtime repairs: {len(repairs)} at weight {args.repair_weight}",
                f"- Low-weight semantic-binding anchors: {len(anchor_rows)} at weight {args.anchor_weight}",
                f"- Max summary words: {manifest['max_summary_words']}",
                "",
                "Repairs target cases where GGUF checkpoint 800 drifted longer, repetitive, or more numeric than native PEFT checkpoint 800.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[runtime-drift-dataset] train_rows={len(final_rows)} repairs={len(repairs)} anchors={len(anchor_rows)} out={args.out}")


if __name__ == "__main__":
    main()
