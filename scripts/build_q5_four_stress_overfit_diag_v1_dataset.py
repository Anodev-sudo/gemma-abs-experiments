#!/usr/bin/env python3
"""Build a four-paper q5 overfit diagnostic dataset.

This is a diagnostic dataset, not a promotion candidate. It repeats four
Kimi/Qwen-accepted stress repairs to test whether LoRA-only SFT can overwrite
sticky q5 llama.cpp completions under the exact title-plus-abstract runtime
prompt.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from generate_llama_stream import SYSTEM_PROMPT, USER_TEMPLATE

SOURCE = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_numeric_hard_attractor_booster_v1/"
    "kimi_qwen_label_run/materialized/accepted_repairs.jsonl"
)
OUT = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_four_stress_overfit_diag_v1_lr2e6/"
    "dataset_repeats50_fulltarget_weighted_v1"
)

STRESS_IDS = ("1901.06548", "2111.10635", "2303.09722", "2406.04373")
DIGIT_RE = re.compile(r"\d")

CRUCIAL_SPANS = {
    "1901.06548": (
        "Tangle height minimization",
        "NP-hard",
        "restricted swap lists",
    ),
    "2111.10635": (
        "Reinforcement learning scheduling",
        "assigns",
        "mixed computing resources",
    ),
    "2303.09722": (
        "Nanoprinted microfibers",
        "low-loss broadband coupling",
        "optical fibers",
        "photonic waveguides",
    ),
    "2406.04373": (
        "LLM-guided code reading",
        "hardware validation stimuli",
        "uncovered branches",
    ),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def messages(title: str, abstract: str, summary: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(title=title, abstract=abstract)},
        {"role": "assistant", "content": summary},
    ]


def find_span(summary: str, phrase: str) -> dict[str, Any]:
    start = summary.find(phrase)
    if start < 0:
        raise ValueError(f"Missing crucial span {phrase!r} in summary {summary!r}")
    return {
        "start": start,
        "end": start + len(phrase),
        "text": phrase,
        "weight": 12.0,
        "kind": "four_stress_crucial_replacement_span",
    }


def training_row(source: dict[str, Any], repeat_index: int) -> dict[str, Any]:
    paper_id = str(source["paper_id"])
    summary = str(source["repair_summary"]).strip()
    if DIGIT_RE.search(summary):
        raise ValueError(f"Stress label contains a digit: {paper_id} {summary!r}")
    spans = [find_span(summary, phrase) for phrase in CRUCIAL_SPANS[paper_id]]
    return {
        "paper_id": paper_id,
        "diagnostic_row_id": f"{paper_id}:repeat:{repeat_index:03d}",
        "title": source.get("title", ""),
        "abstract": source.get("abstract", ""),
        "summary": summary,
        "primary_category": source.get("primary_category", ""),
        "categories": [],
        "created_date": "",
        "teacher_model": source.get("writer_model", ""),
        "generator_prompt_version": source.get("writer_prompt_version", ""),
        "review_prompt_version": source.get("reviewer_prompt_version", ""),
        "source_split": "q5_four_stress_overfit_diag_v1",
        "diagnostic_only": True,
        "diagnostic_policy": (
            "Four stress rows repeated to test whether LoRA-only SFT can overwrite "
            "sticky q5 llama.cpp completions. Not for promotion."
        ),
        "bad_summary_current_q5": source.get("bad_summary", ""),
        "failure_tags": source.get("failure_tags", []),
        "messages": messages(source.get("title", ""), source.get("abstract", ""), summary),
        "loss_weight_hints": {
            "enabled": True,
            "base_weight": 8.0,
            "spans": spans,
            "policy": "full assistant target 8x; crucial replacement spans 12x",
        },
    }


def eval_row(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_id": f"four_stress_{source['paper_id']}",
        "paper_id": source["paper_id"],
        "title": source.get("title", ""),
        "abstract": source.get("abstract", ""),
        "primary_category": source.get("primary_category", ""),
        "target_summary": source.get("repair_summary", ""),
        "bad_summary_current_q5": source.get("bad_summary", ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build q5 four-stress overfit diagnostic rows.")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    accepted = {str(row.get("paper_id")): row for row in load_jsonl(args.source)}
    missing = [paper_id for paper_id in STRESS_IDS if paper_id not in accepted]
    if missing:
        raise SystemExit(f"Missing stress rows in {args.source}: {missing}")

    ordered_sources = [accepted[paper_id] for paper_id in STRESS_IDS]
    train_rows: list[dict[str, Any]] = []
    for repeat_index in range(args.repeats):
        for source in ordered_sources:
            train_rows.append(training_row(source, repeat_index))

    eval_rows = [eval_row(source) for source in ordered_sources]
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train_rows)
    write_jsonl(args.out / "eval_input_4.jsonl", eval_rows)
    write_jsonl(args.out / "unique_training_rows.jsonl", [training_row(source, 0) for source in ordered_sources])

    counts = Counter(row["paper_id"] for row in train_rows)
    manifest = {
        "created_by": "scripts/build_q5_four_stress_overfit_diag_v1_dataset.py",
        "source": str(args.source),
        "train_path": str(args.out / "train.jsonl"),
        "eval_input_path": str(args.out / "eval_input_4.jsonl"),
        "unique_rows_path": str(args.out / "unique_training_rows.jsonl"),
        "repeats_per_paper": args.repeats,
        "train_rows": len(train_rows),
        "paper_ids": list(STRESS_IDS),
        "row_counts": dict(counts),
        "digit_summary_rows": sum(1 for row in train_rows if DIGIT_RE.search(row["summary"])),
        "loss_weight_policy": "assistant target 8x; crucial replacement spans 12x",
        "runtime_prompt_source": "scripts/generate_llama_stream.py SYSTEM_PROMPT and USER_TEMPLATE",
        "diagnostic_only": True,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[four-stress] wrote {len(train_rows)} train rows to {args.out / 'train.jsonl'}")
    print(f"[four-stress] wrote eval rows to {args.out / 'eval_input_4.jsonl'}")


if __name__ == "__main__":
    main()
