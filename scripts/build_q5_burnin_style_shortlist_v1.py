#!/usr/bin/env python3
"""Build a human style-failure shortlist after q5 hard77 burn-in."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_q5_heuristic_sweep import heur, norm  # noqa: E402


ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5")
COMBINED_INPUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_hard_attractor_booster_v1_lr1e6/eval/heuristic_inputs/combined_454.jsonl")
HARD77_INPUT = ROOT / "dataset_hard77_repeats20_v1/eval_hard77.jsonl"
STRESS_IDS = {"1901.06548", "2111.10635", "2303.09722", "2406.04373"}
DIGIT_RE = re.compile(r"\d")
BENCHMARK_RE = re.compile(r"\b(outperform|outperforms|best|state-of-the-art|benchmark|random testing|superior)\b", re.I)
GENERIC_RE = re.compile(r"\b(new|novel|various|different|multiple|better|improved performance|high performance|method|approach)\b", re.I)
AWKWARD_RE = re.compile(r"\b(couplings|bypass random|train heterogeneous models|for discrete data|test designs|code that bypass)\b", re.I)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def output_by_pid(path: Path) -> dict[str, str]:
    rows = load_jsonl(path)
    return {
        str(row["paper_id"]): norm(row.get("normalized_output") or row.get("raw_output"))
        for row in rows
        if row.get("paper_id")
    }


def style_reasons(source: dict[str, Any], baseline: str, burnin: str) -> list[str]:
    base_h = heur(baseline, source)
    burn_h = heur(burnin, source)
    reasons: list[str] = []
    if burnin != baseline:
        reasons.append("changed")
    if burn_h["word_count"] < 10:
        reasons.append("under10")
    if burn_h["word_count"] > 15:
        reasons.append("over15")
    if burn_h["title_overlap"] >= max(5, base_h["title_overlap"] + 2):
        reasons.append("title_overlap_regression")
    if BENCHMARK_RE.search(burnin):
        reasons.append("benchmark_or_winner_wording")
    if GENERIC_RE.search(burnin) and burn_h["title_overlap"] >= base_h["title_overlap"]:
        reasons.append("generic_or_title_rehash_risk")
    if AWKWARD_RE.search(burnin):
        reasons.append("awkward_phrase")
    if DIGIT_RE.search(burnin) and not DIGIT_RE.search(baseline):
        reasons.append("new_digit")
    return reasons


def record(section: str, source: dict[str, Any], baseline: str, burnin: str, reasons: list[str]) -> dict[str, Any]:
    return {
        "section": section,
        "paper_id": source["paper_id"],
        "eval_id": source.get("eval_id", ""),
        "split": source.get("split", ""),
        "title": source.get("title", ""),
        "primary_category": source.get("primary_category", ""),
        "baseline_output": baseline,
        "burnin_output": burnin,
        "baseline_flags": heur(baseline, source)["flags"],
        "burnin_flags": heur(burnin, source)["flags"],
        "baseline_word_count": heur(baseline, source)["word_count"],
        "burnin_word_count": heur(burnin, source)["word_count"],
        "reasons": reasons,
    }


def priority(row: dict[str, Any]) -> tuple[int, int, int]:
    weights = {
        "new_digit": 100,
        "benchmark_or_winner_wording": 70,
        "awkward_phrase": 60,
        "over15": 50,
        "under10": 45,
        "title_overlap_regression": 40,
        "generic_or_title_rehash_risk": 25,
        "changed": 1,
    }
    score = sum(weights.get(reason, 0) for reason in row["reasons"])
    return (-score, -abs(int(row["burnin_word_count"]) - int(row["baseline_word_count"])), row["paper_id"])


def split_metrics(rows: list[dict[str, Any]], outputs: dict[str, str]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        text = outputs.get(str(row["paper_id"]), "")
        h = heur(text, row)
        by_split[str(row.get("split", ""))].append({"heuristics": h})
    metrics = {}
    for split, items in by_split.items():
        flags = Counter(flag for item in items for flag in item["heuristics"]["flags"])
        metrics[split] = {
            "rows": len(items),
            "digit_rows": flags.get("has_digit", 0),
            "under10": flags.get("under10", 0),
            "over15": flags.get("over15", 0),
            "unsupported_number_rows": flags.get("unsupported_number", 0),
            "number_dense_rows": flags.get("number_dense", 0),
        }
    return metrics


def write_markdown(path: Path, records: list[dict[str, Any]], metrics: dict[str, Any], selected_label: str) -> None:
    sections = [
        ("stress_rows", "Remembered Stress Rows"),
        ("hard77_failures", "Hard77 Remaining Failures"),
        ("fresh_normal_changed", "Fresh Normal Changed Shortlist"),
        ("fresh_normal_sanity", "Fresh Normal Sanity Slice"),
    ]
    lines = [
        "# q5 hard77 burn-in style failure shortlist",
        "",
        f"Selected burn-in checkpoint: `{selected_label}`",
        "",
        "This is a human review packet only. No normalization dataset was compiled.",
        "",
        "## Split Metrics",
        "",
        "| model | split | rows | digit | dense nums | unsupported nums | <10 | >15 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name, split_map in metrics.items():
        for split, row in sorted(split_map.items()):
            lines.append(
                f"| {model_name} | {split} | {row['rows']} | {row['digit_rows']} | {row['number_dense_rows']} | "
                f"{row['unsupported_number_rows']} | {row['under10']} | {row['over15']} |"
            )
    for section_key, section_title in sections:
        section_rows = [row for row in records if row["section"] == section_key]
        lines.extend(["", f"## {section_title}", ""])
        if not section_rows:
            lines.append("_None._")
            continue
        for row in section_rows:
            lines.append(f"### `{row['paper_id']}` {row['title']}")
            lines.append(f"- Reasons: {', '.join(row['reasons']) if row['reasons'] else 'review'}")
            lines.append(f"- Baseline ({row['baseline_word_count']}w; flags={row['baseline_flags']}): {row['baseline_output']}")
            lines.append(f"- Burn-in ({row['burnin_word_count']}w; flags={row['burnin_flags']}): {row['burnin_output']}")
            lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a human style review shortlist after hard77 burn-in.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--selected-label", required=True)
    parser.add_argument("--combined-input", type=Path, default=COMBINED_INPUT)
    parser.add_argument("--hard77-input", type=Path, default=HARD77_INPUT)
    parser.add_argument("--baseline-combined-output", type=Path, required=True)
    parser.add_argument("--burnin-combined-output", type=Path, required=True)
    parser.add_argument("--hard77-output", type=Path, required=True)
    parser.add_argument("--fresh-normal-limit", type=int, default=40)
    parser.add_argument("--sanity-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260515)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined_rows = load_jsonl(args.combined_input)
    hard77_rows = load_jsonl(args.hard77_input)
    source_by_pid = {str(row["paper_id"]): row for row in combined_rows}
    hard77_by_pid = {str(row["paper_id"]): row for row in hard77_rows}
    baseline = output_by_pid(args.baseline_combined_output)
    burnin = output_by_pid(args.burnin_combined_output)
    hard77_out = output_by_pid(args.hard77_output)

    records: list[dict[str, Any]] = []
    for pid in sorted(STRESS_IDS):
        source = source_by_pid[pid]
        records.append(record("stress_rows", source, baseline.get(pid, ""), burnin.get(pid, ""), style_reasons(source, baseline.get(pid, ""), burnin.get(pid, ""))))

    for pid, source in sorted(hard77_by_pid.items()):
        out = hard77_out.get(pid, "")
        target = norm(source.get("target_summary", ""))
        reasons = []
        if DIGIT_RE.search(out):
            reasons.append("has_digit")
        if target and out.casefold() != target.casefold():
            reasons.append("not_exact_target")
        if reasons:
            pseudo_source = {
                **source,
                "title": source.get("title", ""),
                "split": "hard77_burnin",
            }
            records.append(record("hard77_failures", pseudo_source, target, out, reasons))

    fresh_rows = [row for row in combined_rows if row.get("split") == "fresh_recent_normal_100"]
    changed_records = []
    for source in fresh_rows:
        pid = str(source["paper_id"])
        base = baseline.get(pid, "")
        burn = burnin.get(pid, "")
        reasons = style_reasons(source, base, burn)
        if burn != base:
            changed_records.append(record("fresh_normal_changed", source, base, burn, reasons))
    records.extend(sorted(changed_records, key=priority)[: args.fresh_normal_limit])

    rng = random.Random(args.seed)
    sanity_rows = rng.sample(fresh_rows, min(args.sanity_size, len(fresh_rows)))
    for source in sanity_rows:
        pid = str(source["paper_id"])
        records.append(record("fresh_normal_sanity", source, baseline.get(pid, ""), burnin.get(pid, ""), style_reasons(source, baseline.get(pid, ""), burnin.get(pid, ""))))

    metrics = {
        "checkpoint700": split_metrics(combined_rows, baseline),
        args.selected_label: split_metrics(combined_rows, burnin),
    }
    out_dir = args.root / "eval/style_failure_review"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "style_failure_shortlist.jsonl", records)
    write_markdown(out_dir / "STYLE_FAILURE_SHORTLIST.md", records, metrics, args.selected_label)
    print(f"[style-shortlist] wrote {out_dir / 'STYLE_FAILURE_SHORTLIST.md'}")
    print(f"[style-shortlist] wrote {out_dir / 'style_failure_shortlist.jsonl'}")


if __name__ == "__main__":
    main()
