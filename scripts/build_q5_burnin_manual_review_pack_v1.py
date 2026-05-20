#!/usr/bin/env python3
"""Build a 100-row manual review pack for q5 hard77 burn-in."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5")
OUT = ROOT / "eval/manual_review_100"
HARD77_INPUT = ROOT / "dataset_hard77_repeats20_v1/eval_hard77.jsonl"
COMBINED_INPUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_hard_attractor_booster_v1_lr1e6/eval/heuristic_inputs/combined_454.jsonl")
BASELINE_BAD = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_hard_attractor_booster_v1_lr1e6/eval/baseline_ckpt700_bad_summaries_pool350.jsonl")
STRESS_IDS = ("1901.06548", "2111.10635", "2303.09722", "2406.04373")
DIGIT_RE = re.compile(r"\d")
NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?(?:\s?[%x×]|e[-+]?\d+)?|\b\d+[A-Za-z]+|\b[A-Za-z]+\d+[A-Za-z0-9-]*")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def norm(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def output_by_pid(path: Path) -> dict[str, str]:
    return {
        str(row["paper_id"]): norm(row.get("normalized_output") or row.get("raw_output"))
        for row in load_jsonl(path)
        if row.get("paper_id")
    }


def hard_priority(row: dict[str, Any]) -> tuple[int, str]:
    bad = str(row.get("bad_summary_current_q5", ""))
    tags = set(row.get("failure_tags") or [])
    score = 0
    if row["paper_id"] in STRESS_IDS:
        score += 10_000
    if DIGIT_RE.search(bad):
        score += 1_000
    if "number_dense" in tags:
        score += 300
    if "unit_or_metric_trap" in tags:
        score += 200
    if "benchmark_headline_trap" in tags:
        score += 100
    return (-score, str(row["paper_id"]))


def numeric_priority(row: dict[str, Any], baseline_bad: dict[str, str]) -> tuple[int, str]:
    bad = baseline_bad.get(str(row["paper_id"]), "")
    score = 0
    nums = NUM_RE.findall(bad)
    if DIGIT_RE.search(bad):
        score += 1_000
    score += 200 * min(len(nums), 4)
    if any(token in bad.lower() for token in ("outperform", "state-of-the-art", "best", "higher", "lower", "cost", "accuracy", "efficiency")):
        score += 200
    return (-score, str(row["paper_id"]))


def review_row(row: dict[str, Any], source_bucket: str, index: int) -> dict[str, Any]:
    return {
        "review_id": f"review_{index:03d}",
        "source_bucket": source_bucket,
        "eval_id": row.get("eval_id", ""),
        "paper_id": str(row["paper_id"]),
        "split": row.get("split", source_bucket),
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "primary_category": row.get("primary_category", ""),
        "target_summary": row.get("target_summary", ""),
        "bad_summary_current_q5": row.get("bad_summary_current_q5", ""),
    }


def build_input(args: argparse.Namespace) -> list[dict[str, Any]]:
    hard_rows = load_jsonl(args.hard77_input)
    combined = load_jsonl(args.combined_input)
    baseline_bad = {
        str(row["paper_id"]): norm(row.get("summary", ""))
        for row in load_jsonl(args.baseline_bad)
        if row.get("paper_id")
    }

    hard_sorted = sorted(hard_rows, key=hard_priority)
    hard_selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pid in STRESS_IDS:
        row = next(row for row in hard_rows if str(row["paper_id"]) == pid)
        hard_selected.append(row)
        seen.add(pid)
    for row in hard_sorted:
        pid = str(row["paper_id"])
        if pid not in seen:
            hard_selected.append(row)
            seen.add(pid)
        if len(hard_selected) == args.hard_count:
            break

    numeric_rows = [row for row in combined if row.get("split") == "numeric_ckpt700_risk_pool350" and str(row["paper_id"]) not in seen]
    numeric_selected = sorted(numeric_rows, key=lambda row: numeric_priority(row, baseline_bad))[: args.numeric_count]
    seen.update(str(row["paper_id"]) for row in numeric_selected)

    fresh_rows = [row for row in combined if row.get("split") == "fresh_recent_normal_100" and str(row["paper_id"]) not in seen]
    rng = random.Random(args.seed)
    fresh_selected = rng.sample(fresh_rows, args.fresh_count)

    selected: list[dict[str, Any]] = []
    for row in hard_selected:
        selected.append(review_row(row, "hard77", len(selected) + 1))
    for row in numeric_selected:
        enriched = dict(row)
        enriched["bad_summary_current_q5"] = baseline_bad.get(str(row["paper_id"]), "")
        selected.append(review_row(enriched, "numeric_risk", len(selected) + 1))
    for row in fresh_selected:
        selected.append(review_row(row, "fresh_normal", len(selected) + 1))

    if len(selected) != args.total_count:
        raise SystemExit(f"Expected {args.total_count} selected rows, got {len(selected)}")
    if len({row["paper_id"] for row in selected}) != len(selected):
        raise SystemExit("Selected rows contain duplicate paper_ids")
    return selected


def materialize_pairs(args: argparse.Namespace) -> None:
    rows = load_jsonl(args.input_path)
    baseline = output_by_pid(args.baseline_output)
    burnin = output_by_pid(args.burnin_output)
    paired = []
    missing = []
    for row in rows:
        pid = str(row["paper_id"])
        if pid not in baseline or pid not in burnin:
            missing.append(pid)
            continue
        paired.append(
            {
                **row,
                "baseline_label": args.baseline_label,
                "burnin_label": args.burnin_label,
                "baseline_output": baseline[pid],
                "burnin_output": burnin[pid],
            }
        )
    if missing:
        raise SystemExit(f"Missing outputs for {len(missing)} rows: {missing[:10]}")
    write_jsonl(args.out / "paired_review_100.jsonl", paired)
    for shard_index in range(5):
        shard = paired[shard_index * 20 : (shard_index + 1) * 20]
        write_jsonl(args.out / f"paired_review_shard_{shard_index + 1}.jsonl", shard)
    args.out.mkdir(parents=True, exist_ok=True)
    lines = [
        "# q5 hard77 burn-in manual review pack",
        "",
        f"Baseline: `{args.baseline_label}`",
        f"Burn-in: `{args.burnin_label}`",
        "",
        "Rubric: verdict is better, same, worse, or mixed. Tag concrete failure modes.",
        "",
    ]
    for row in paired:
        lines.extend(
            [
                f"## {row['review_id']} `{row['paper_id']}` [{row['source_bucket']}]",
                f"Title: {row['title']}",
                f"Baseline: {row['baseline_output']}",
                f"Burn-in: {row['burnin_output']}",
                "",
            ]
        )
    (args.out / "PAIRED_REVIEW_100.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "created_by": "scripts/build_q5_burnin_manual_review_pack_v1.py",
        "rows": len(paired),
        "baseline_label": args.baseline_label,
        "burnin_label": args.burnin_label,
        "source_counts": {
            "hard77": sum(1 for row in paired if row["source_bucket"] == "hard77"),
            "numeric_risk": sum(1 for row in paired if row["source_bucket"] == "numeric_risk"),
            "fresh_normal": sum(1 for row in paired if row["source_bucket"] == "fresh_normal"),
        },
        "shards": [str(args.out / f"paired_review_shard_{idx}.jsonl") for idx in range(1, 6)],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[manual-review-pack] wrote paired review pack to {args.out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or pair q5 burn-in manual review rows.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build-input")
    build.add_argument("--out", type=Path, default=OUT)
    build.add_argument("--hard77-input", type=Path, default=HARD77_INPUT)
    build.add_argument("--combined-input", type=Path, default=COMBINED_INPUT)
    build.add_argument("--baseline-bad", type=Path, default=BASELINE_BAD)
    build.add_argument("--hard-count", type=int, default=30)
    build.add_argument("--numeric-count", type=int, default=40)
    build.add_argument("--fresh-count", type=int, default=30)
    build.add_argument("--total-count", type=int, default=100)
    build.add_argument("--seed", type=int, default=20260515)

    pair = sub.add_parser("pair-outputs")
    pair.add_argument("--out", type=Path, default=OUT)
    pair.add_argument("--input-path", type=Path, default=OUT / "review_input_100.jsonl")
    pair.add_argument("--baseline-output", type=Path, required=True)
    pair.add_argument("--burnin-output", type=Path, required=True)
    pair.add_argument("--baseline-label", default="checkpoint700")
    pair.add_argument("--burnin-label", default="hard77_burnin_checkpoint_200")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "build-input":
        selected = build_input(args)
        args.out.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.out / "review_input_100.jsonl", selected)
        print(f"[manual-review-pack] wrote {len(selected)} rows to {args.out / 'review_input_100.jsonl'}")
    elif args.cmd == "pair-outputs":
        materialize_pairs(args)


if __name__ == "__main__":
    main()
