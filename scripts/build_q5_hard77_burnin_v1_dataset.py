#!/usr/bin/env python3
"""Build the q5 hard77 burn-in dataset.

This dataset is a first-stage curriculum burn-in, not a promotion dataset. It
uses the clean Kimi/Qwen hard-attractor repairs and repeats them under the
exact title-plus-abstract runtime prompt so the q5 LoRA can overwrite sticky
numeric/value completions before any style-normalization pass.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from generate_llama_stream import build_prompt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_numeric_hard_attractor_booster_v1_dataset import (  # noqa: E402
    BOOSTER_ACCEPTED,
    FORMULA_RE,
    HIDDEN_RE,
    messages,
    surgical_hints,
)
from run_sft_gemma import GEMMA_TRAINING_CHAT_TEMPLATE, resolve_model_path  # noqa: E402

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Run this with the project training runtime; transformers is required.") from exc


OUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/dataset_hard77_repeats20_v1")
DEFAULT_HF_HOME = Path("hf_bucket/experiments/mlx_lora/.hf-home")
DEFAULT_MODEL = "google/gemma-3-270m-it"
STRESS_IDS = {"1901.06548", "2111.10635", "2303.09722", "2406.04373"}
DIGIT_RE = re.compile(r"\d")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def clean_reason(summary: str) -> list[str]:
    reasons: list[str] = []
    words = summary.split()
    if not (10 <= len(words) <= 15):
        reasons.append(f"wc={len(words)}")
    if DIGIT_RE.search(summary):
        reasons.append("digit")
    if FORMULA_RE.search(summary):
        reasons.append("formula")
    if HIDDEN_RE.search(summary):
        reasons.append("hidden_quant")
    if not summary.endswith("."):
        reasons.append("period")
    return reasons


def repair_to_base_row(row: dict[str, Any]) -> dict[str, Any]:
    summary = str(row.get("repair_summary", "")).strip()
    reasons = clean_reason(summary)
    if reasons:
        raise ValueError(f"Unclean hard row {row.get('paper_id')}: {reasons} {summary!r}")
    base = {
        "paper_id": str(row["paper_id"]),
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "summary": summary,
        "primary_category": row.get("primary_category", ""),
        "categories": [],
        "created_date": "",
        "teacher_model": row.get("writer_model", ""),
        "generator_prompt_version": row.get("writer_prompt_version", ""),
        "review_prompt_version": row.get("reviewer_prompt_version", ""),
        "source_split": "q5_hard77_burnin_v1",
        "bad_summary_current_q5": row.get("bad_summary", ""),
        "failure_tags": row.get("failure_tags", []),
        "number_policy": row.get("number_policy", ""),
        "repair_candidate_id": row.get("candidate_id", ""),
        "diagnostic_only": False,
        "curriculum_stage": "hard_burnin",
    }
    base["messages"] = messages(base["title"], base["abstract"], base["summary"])
    hints = surgical_hints(base, "hard77_burnin")
    boosted_spans = []
    for span in hints.get("spans", []):
        boosted = dict(span)
        if boosted.get("kind") == "source_specific_anchor":
            boosted["weight"] = 2.0
        else:
            boosted["weight"] = 12.0
        boosted_spans.append(boosted)
    base["loss_weight_hints"] = {
        "enabled": True,
        "base_weight": 8.0,
        "spans": boosted_spans,
        "policy": "hard77 burn-in: full assistant target 8x; non-source high-value spans 12x; source anchors unboosted",
    }
    return base


def eval_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval_id": f"hard77_{row['paper_id']}",
        "paper_id": row["paper_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "primary_category": row.get("primary_category", ""),
        "split": "hard77_burnin",
        "target_summary": row["summary"],
        "bad_summary_current_q5": row.get("bad_summary_current_q5", ""),
    }


def verify_prompt_render(rows: list[dict[str, Any]], hf_home: Path, model: str) -> None:
    model_path = resolve_model_path(None, hf_home, model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.chat_template = GEMMA_TRAINING_CHAT_TEMPLATE
    for row in rows:
        rendered = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
        runtime_prompt = build_prompt(row["title"], row["abstract"])
        if rendered != runtime_prompt:
            raise ValueError(f"Training/runtime prompt mismatch for {row['paper_id']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build q5 hard77 burn-in dataset.")
    parser.add_argument("--source", type=Path, default=BOOSTER_ACCEPTED)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--expected-rows", type=int, default=77)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    raw_rows = load_jsonl(args.source)
    base_rows = [repair_to_base_row(row) for row in raw_rows if not clean_reason(str(row.get("repair_summary", "")).strip())]
    by_pid = {row["paper_id"]: row for row in base_rows}
    if len(by_pid) != len(base_rows):
        duplicates = [pid for pid, count in Counter(row["paper_id"] for row in base_rows).items() if count > 1]
        raise SystemExit(f"Duplicate hard77 paper ids: {duplicates}")
    if len(base_rows) != args.expected_rows:
        raise SystemExit(f"Expected {args.expected_rows} clean hard rows, got {len(base_rows)}")
    missing_stress = sorted(STRESS_IDS - set(by_pid))
    if missing_stress:
        raise SystemExit(f"Missing remembered stress rows: {missing_stress}")
    if any(DIGIT_RE.search(row["summary"]) for row in base_rows):
        raise SystemExit("At least one hard77 label contains digits")
    verify_prompt_render(base_rows, args.hf_home, args.model)

    train_rows: list[dict[str, Any]] = []
    for repeat_index in range(args.repeats):
        for row in base_rows:
            repeated = dict(row)
            repeated["burnin_repeat_index"] = repeat_index
            repeated["diagnostic_row_id"] = f"{row['paper_id']}:repeat:{repeat_index:03d}"
            train_rows.append(repeated)

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train_rows)
    write_jsonl(args.out / "unique_training_rows.jsonl", base_rows)
    write_jsonl(args.out / "eval_hard77.jsonl", [eval_row(row) for row in base_rows])
    manifest = {
        "created_by": "scripts/build_q5_hard77_burnin_v1_dataset.py",
        "source": str(args.source),
        "train_path": str(args.out / "train.jsonl"),
        "unique_training_rows_path": str(args.out / "unique_training_rows.jsonl"),
        "eval_hard77_path": str(args.out / "eval_hard77.jsonl"),
        "unique_rows": len(base_rows),
        "train_rows": len(train_rows),
        "repeats_per_paper": args.repeats,
        "digit_summary_rows": sum(1 for row in base_rows if DIGIT_RE.search(row["summary"])),
        "remembered_stress_rows": sorted(STRESS_IDS),
        "loss_weight_policy": "assistant target 8x; non-source high-value spans 12x; source anchors unboosted",
        "runtime_prompt_check": "passed against scripts/generate_llama_stream.py",
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[hard77-builder] wrote {len(train_rows)} rows to {args.out / 'train.jsonl'}")
    print(f"[hard77-builder] wrote {len(base_rows)} eval rows to {args.out / 'eval_hard77.jsonl'}")


if __name__ == "__main__":
    main()
