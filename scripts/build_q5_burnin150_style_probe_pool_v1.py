#!/usr/bin/env python3
"""Build a broad source packet for q5 burn-in checkpoint-150 style probing."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


OUT_ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/eval/burnin150_style_probe_5000_v1")
RELEASE_ROWS = Path("hf_dataset/releases/recent_3y_balanced_10746_v1/training_rows.jsonl")
COMBINED_454 = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_hard_attractor_booster_v1_lr1e6/eval/heuristic_inputs/combined_454.jsonl")
HARD77 = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/dataset_hard77_repeats20_v1/eval_hard77.jsonl")
MANUAL_100 = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/eval/manual_review_100/review_input_100.jsonl")

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")
DIGIT_RE = re.compile(r"\d")
NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?(?:\s?[%x×]|e[-+]?\d+)?|\b\d+[A-Za-z]+|\b[A-Za-z]+\d+[A-Za-z0-9-]*")
FORMULA_RE = re.compile(r"[$<>≤≥≈]|\\[A-Za-z]+|\bO\(|\^|_|𝔽|[α-ωΑ-Ω]")
UNIT_RE = re.compile(
    r"\b(percent|accuracy|efficiency|latency|throughput|speed|cost|energy|power|carbon|"
    r"GHz|MHz|kHz|nm|mm|cm|km|kg|mA|mAh|eV|MeV|GeV|Tesla|Kelvin|FLOP|token|benchmark)\b",
    re.I,
)
POLARITY_RE = re.compile(
    r"\b(but|however|although|despite|except|fail|fails|failed|failure|limited|limitation|"
    r"cannot|can not|does not|do not|without|lack|lacks|tradeoff|trade-off|risk|uncertain)\b",
    re.I,
)
BENCHMARK_RE = re.compile(r"\b(outperform|outperforms|state-of-the-art|benchmark|leaderboard|best|superior|beats?|achieves?)\b", re.I)


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


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def risk_tags(row: dict[str, Any]) -> list[str]:
    text = f"{row.get('title', '')}\n{row.get('abstract', '')}"
    tags: list[str] = []
    nums = NUM_RE.findall(text)
    if DIGIT_RE.search(text):
        tags.append("has_digit_source")
    if len(nums) >= 4:
        tags.append("dense_numeric_source")
    if FORMULA_RE.search(text):
        tags.append("formula_or_symbol_source")
    if UNIT_RE.search(text):
        tags.append("metric_or_unit_source")
    if POLARITY_RE.search(text):
        tags.append("caveat_or_polarity_source")
    if BENCHMARK_RE.search(text):
        tags.append("benchmark_or_winner_source")
    if len([w for w in words(text) if w.isupper() and len(w) >= 3]) >= 4:
        tags.append("acronym_dense_source")
    return tags


def generation_row(row: dict[str, Any], *, eval_id: str, split: str, pool_source: str) -> dict[str, Any]:
    out = {
        "eval_id": eval_id,
        "paper_id": str(row["paper_id"]),
        "title": norm(row.get("title", "")),
        "abstract": norm(row.get("abstract", "")),
        "primary_category": row.get("primary_category", ""),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date", ""),
        "split": split,
        "pool_source": pool_source,
        "source_risk_tags": risk_tags(row),
    }
    return out


def add_required(
    selected: list[dict[str, Any]],
    seen: set[str],
    rows: list[dict[str, Any]],
    *,
    split_prefix: str,
    pool_source: str,
) -> None:
    for row in rows:
        pid = str(row.get("paper_id", ""))
        if not pid or pid in seen:
            continue
        split = str(row.get("split") or split_prefix)
        eval_id = str(row.get("eval_id") or f"{split_prefix}_{len(selected) + 1:04d}")
        selected.append(generation_row(row, eval_id=eval_id, split=split, pool_source=pool_source))
        seen.add(pid)


def category_balanced_select(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("primary_category") or "unknown")].append(row)
    rng = random.Random(seed)
    for bucket in by_category.values():
        rng.shuffle(bucket)
    selected: list[dict[str, Any]] = []
    cycle = sorted(by_category)
    while cycle and len(selected) < count:
        next_cycle: list[str] = []
        for category in cycle:
            bucket = by_category[category]
            if not bucket:
                continue
            selected.append(bucket.pop())
            if bucket:
                next_cycle.append(category)
            if len(selected) >= count:
                break
        cycle = next_cycle
    return selected


def source_candidate(row: dict[str, Any], seen: set[str], min_abstract_chars: int) -> bool:
    pid = str(row.get("paper_id", ""))
    if not pid or pid in seen:
        return False
    if len(norm(row.get("abstract", ""))) < min_abstract_chars:
        return False
    if not norm(row.get("title", "")):
        return False
    return True


def build_pool(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    combined_rows = load_jsonl(args.combined_454)
    hard77_rows = load_jsonl(args.hard77)
    manual_rows = load_jsonl(args.manual_100)
    release_rows = load_jsonl(args.release_rows)

    add_required(selected, seen, combined_rows, split_prefix="known_combined_454", pool_source=str(args.combined_454))
    add_required(selected, seen, hard77_rows, split_prefix="known_hard77", pool_source=str(args.hard77))
    add_required(selected, seen, manual_rows, split_prefix="known_manual_100", pool_source=str(args.manual_100))

    release_candidates = [row for row in release_rows if source_candidate(row, seen, args.min_abstract_chars)]
    risky: list[dict[str, Any]] = []
    normal: list[dict[str, Any]] = []
    for row in release_candidates:
        tags = risk_tags(row)
        if tags:
            risky.append(row)
        else:
            normal.append(row)

    remaining = args.count - len(selected)
    if remaining < 0:
        raise SystemExit(f"Required rows exceed requested count: {len(selected)} > {args.count}")
    risk_target = min(args.risk_fill_count, remaining, len(risky))
    risky_selected = category_balanced_select(risky, risk_target, args.seed)
    for row in risky_selected:
        pid = str(row["paper_id"])
        if pid in seen:
            continue
        selected.append(
            generation_row(
                row,
                eval_id=f"release_risk_{len(selected) + 1:04d}",
                split="release_source_risk",
                pool_source=str(args.release_rows),
            )
        )
        seen.add(pid)

    remaining = args.count - len(selected)
    normal_pool = [row for row in normal if str(row.get("paper_id", "")) not in seen]
    if len(normal_pool) < remaining:
        extra_risky = [row for row in risky if str(row.get("paper_id", "")) not in seen]
        normal_pool.extend(extra_risky)
    normal_selected = category_balanced_select(normal_pool, remaining, args.seed + 17)
    for row in normal_selected:
        pid = str(row["paper_id"])
        if pid in seen:
            continue
        split = "release_source_normal" if not risk_tags(row) else "release_source_risk_backfill"
        selected.append(
            generation_row(
                row,
                eval_id=f"release_fill_{len(selected) + 1:04d}",
                split=split,
                pool_source=str(args.release_rows),
            )
        )
        seen.add(pid)

    if len(selected) != args.count:
        raise SystemExit(f"Expected {args.count} rows, got {len(selected)}")
    if len({row["paper_id"] for row in selected}) != len(selected):
        raise SystemExit("Output contains duplicate paper_id values")

    source_counts = Counter(row["split"] for row in selected)
    risk_counts = Counter(tag for row in selected for tag in row.get("source_risk_tags", []))
    category_counts = Counter(row.get("primary_category", "") for row in selected)
    manifest = {
        "created_by": "scripts/build_q5_burnin150_style_probe_pool_v1.py",
        "row_count": len(selected),
        "intended_checkpoint": "q5 hard77 burn-in checkpoint-150",
        "label_policy": "Generation input only; existing summary/target fields are intentionally stripped.",
        "prompt_contract": "scripts/generate_llama_stream.py title+abstract runtime prompt",
        "inputs": {
            "combined_454": str(args.combined_454),
            "hard77": str(args.hard77),
            "manual_100": str(args.manual_100),
            "release_rows": str(args.release_rows),
        },
        "selection_policy": {
            "required_rows_first": ["combined_454", "hard77", "manual_100"],
            "release_fill": "category-balanced source-risk rows first, then category-balanced normal rows",
            "seed": args.seed,
            "min_abstract_chars": args.min_abstract_chars,
            "requested_risk_fill_count": args.risk_fill_count,
        },
        "split_counts": dict(source_counts),
        "risk_tag_counts": dict(risk_counts),
        "top_primary_categories": dict(category_counts.most_common(30)),
    }
    return selected, manifest


def write_preview(path: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# q5 burn-in checkpoint-150 style probe pool",
        "",
        f"Rows: `{manifest['row_count']}`",
        "",
        "This file is a source packet for llama.cpp generation, not a training dataset.",
        "Existing label fields were stripped to avoid accidentally reusing old labels.",
        "",
        "## Split Counts",
        "",
        "| split | rows |",
        "|---|---:|",
    ]
    for split, count in sorted(manifest["split_counts"].items()):
        lines.append(f"| {split} | {count} |")
    lines.extend(["", "## Risk Tag Counts", "", "| tag | rows |", "|---|---:|"])
    for tag, count in sorted(manifest["risk_tag_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {tag} | {count} |")
    lines.extend(["", "## First Rows", ""])
    for row in rows[:20]:
        lines.append(f"- `{row['paper_id']}` `{row['split']}` tags={row['source_risk_tags']}: {row['title']}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 5k source rows for q5 burn-in checkpoint-150 generation.")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--release-rows", type=Path, default=RELEASE_ROWS)
    parser.add_argument("--combined-454", type=Path, default=COMBINED_454)
    parser.add_argument("--hard77", type=Path, default=HARD77)
    parser.add_argument("--manual-100", type=Path, default=MANUAL_100)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--risk-fill-count", type=int, default=2500)
    parser.add_argument("--min-abstract-chars", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260516)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, manifest = build_pool(args)
    args.out_root.mkdir(parents=True, exist_ok=True)
    input_path = args.out_root / "generation_input_5000.jsonl"
    write_jsonl(input_path, rows)
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_preview(args.out_root / "README.md", rows, manifest)
    print(f"[burnin150-style-probe-pool] wrote {len(rows)} rows to {input_path}")


if __name__ == "__main__":
    main()
