#!/usr/bin/env python3
"""Summarize q5 checkpoint-200 style-normalization runtime evals."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin200_style_norm_v1_lr5e7")
BURNIN_ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5")
MANUAL_INPUT = BURNIN_ROOT / "eval/manual_review_100/review_input_100.jsonl"
HARD77_INPUT = BURNIN_ROOT / "dataset_hard77_repeats20_v1/eval_hard77.jsonl"
COMBINED_INPUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_hard_attractor_booster_v1_lr1e6/eval/heuristic_inputs/combined_454.jsonl")
STRESS_IDS = ("1901.06548", "2111.10635", "2303.09722", "2406.04373")
DIGIT_RE = re.compile(r"\d")
NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?(?:\s?[%x×]|e[-+]?\d+)?|\b\d+[A-Za-z]+|\b[A-Za-z]+\d+[A-Za-z0-9-]*")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def output_by_pid(path: Path) -> dict[str, str]:
    return {
        str(row["paper_id"]): norm(row.get("normalized_output") or row.get("raw_output") or row.get("summary"))
        for row in load_jsonl(path)
        if row.get("paper_id")
    }


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def metrics(rows: list[dict[str, Any]], outputs: dict[str, str]) -> dict[str, Any]:
    flags = Counter()
    for row in rows:
        out = outputs.get(str(row["paper_id"]), "")
        wc = word_count(out)
        if wc < 10:
            flags["under10"] += 1
        if wc > 15:
            flags["over15"] += 1
        if DIGIT_RE.search(out):
            flags["digit"] += 1
        if len(NUM_RE.findall(out)) >= 2:
            flags["dense"] += 1
    return {
        "rows": len(rows),
        "digit_rows": flags["digit"],
        "dense_number_rows": flags["dense"],
        "under10": flags["under10"],
        "over15": flags["over15"],
    }


def exact_targets(rows: list[dict[str, Any]], outputs: dict[str, str]) -> int:
    return sum(
        1
        for row in rows
        if row.get("target_summary") and outputs.get(str(row["paper_id"]), "") == norm(row["target_summary"])
    )


def checkpoint_outputs(root: Path, eval_name: str, step: int, rows: int) -> Path:
    return root / f"eval/{eval_name}_outputs/checkpoint_{step}_outputs_{rows}.jsonl"


def summarize_step(root: Path, step: int, manual_rows: list[dict[str, Any]], hard77_rows: list[dict[str, Any]], combined_rows: list[dict[str, Any]]) -> dict[str, Any]:
    manual_out = output_by_pid(checkpoint_outputs(root, "manual_review_100", step, 100))
    hard_out = output_by_pid(checkpoint_outputs(root, "hard77", step, 77))
    combined_path = checkpoint_outputs(root, "combined_454", step, 454)
    combined_out = output_by_pid(combined_path) if combined_path.exists() else {}
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in combined_rows:
        by_split[str(row.get("split", ""))].append(row)
    return {
        "manual_100": metrics(manual_rows, manual_out),
        "hard77": {**metrics(hard77_rows, hard_out), "exact_target_rows": exact_targets(hard77_rows, hard_out)},
        "combined_454": metrics(combined_rows, combined_out) if combined_out else {},
        "combined_by_split": {split: metrics(rows, combined_out) for split, rows in by_split.items()} if combined_out else {},
        "stress_outputs": {pid: manual_out.get(pid, hard_out.get(pid, "")) for pid in STRESS_IDS},
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize q5 style-normalization eval outputs.")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--burnin-root", type=Path, default=BURNIN_ROOT)
    ap.add_argument("--manual-input", type=Path, default=MANUAL_INPUT)
    ap.add_argument("--hard77-input", type=Path, default=HARD77_INPUT)
    ap.add_argument("--combined-input", type=Path, default=COMBINED_INPUT)
    ap.add_argument("--steps", nargs="+", type=int, required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    manual_rows = load_jsonl(args.manual_input)
    hard77_rows = load_jsonl(args.hard77_input)
    combined_rows = load_jsonl(args.combined_input)
    summary = {str(step): summarize_step(args.root, step, manual_rows, hard77_rows, combined_rows) for step in args.steps}
    out_dir = args.root / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "style_norm_eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# q5 style-normalization eval summary", ""]
    lines.append("| step | manual digit | manual >15 | hard77 digit | hard77 exact | combined digit | combined >15 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for step in args.steps:
        s = summary[str(step)]
        combined = s.get("combined_454", {})
        lines.append(
            f"| {step} | {s['manual_100']['digit_rows']} | {s['manual_100']['over15']} | "
            f"{s['hard77']['digit_rows']} | {s['hard77']['exact_target_rows']} | "
            f"{combined.get('digit_rows', '')} | {combined.get('over15', '')} |"
        )
    lines.extend(["", "## Stress Outputs", ""])
    for step in args.steps:
        lines.append(f"### checkpoint-{step}")
        for pid, text in summary[str(step)]["stress_outputs"].items():
            lines.append(f"- `{pid}`: {text}")
        lines.append("")
    (out_dir / "STYLE_NORM_EVAL_SUMMARY.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(out_dir / "STYLE_NORM_EVAL_SUMMARY.md")


if __name__ == "__main__":
    main()
