#!/usr/bin/env python3
"""Build a gold-only burn-in dataset for the promoted q5/q8 sidecar."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_q5_natural_language_token_weighted_gold_v1_dataset as q5_gold  # noqa: E402


DEFAULT_GOLD_DATASET = q5_gold.DEFAULT_GOLD_DATASET
DEFAULT_PROMPT_SPEC = q5_gold.DEFAULT_PROMPT_SPEC
DEFAULT_OUTPUT_DIR = Path(
    "hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin_v1/"
    "dataset_994gold_burnin_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dataset", type=Path, default=DEFAULT_GOLD_DATASET)
    parser.add_argument("--prompt-spec", type=Path, default=DEFAULT_PROMPT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-gold-rows", type=int, default=994)
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--title-risk-weight", type=float, default=0.90)
    parser.add_argument("--under10-weight", type=float, default=0.85)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260527)
    return parser.parse_args()


def word_count(text: str) -> int:
    return len([part for part in text.strip().split() if part])


def umbrella_category(category: str) -> str:
    if category.startswith("cs."):
        return "cs"
    if category.startswith("math."):
        return "math"
    if category.startswith(("astro-ph", "cond-mat", "gr-qc", "hep-", "nucl-", "physics.")) or category == "quant-ph":
        return "physics"
    return category.split(".")[0] if category else "unknown"


def row_base_weight(row: dict[str, Any], *, base_weight: float, title_risk_weight: float, under10_weight: float) -> float:
    summary = str(row["summary"])
    stats = q5_gold.title_overlap_stats(str(row["title"]), summary)
    weight = title_risk_weight if stats["title_overlap_risk"] else base_weight
    if word_count(summary) < 10:
        weight = min(weight, under10_weight)
    return weight


def convert_gold_row(
    row: dict[str, Any],
    *,
    prompt_spec: dict[str, Any],
    source_path: Path,
    base_weight: float,
    title_risk_weight: float,
    under10_weight: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = q5_gold.converted_base_row(
        row,
        component="strict_gold_burnin",
        prompt_spec=prompt_spec,
        source_path=source_path,
    )
    stats = q5_gold.title_overlap_stats(out["title"], out["summary"])
    wc = word_count(out["summary"])
    base = row_base_weight(
        out,
        base_weight=base_weight,
        title_risk_weight=title_risk_weight,
        under10_weight=under10_weight,
    )
    flags: list[str] = []
    if stats["title_overlap_risk"]:
        flags.append("title_overlap_risk_row_weight")
    if wc < 10:
        flags.append("under10_row_weight_cap")
    out["loss_weight_hints"] = {
        "enabled": True,
        "base_weight": base,
        "spans": [],
        "policy": "q5_natural_language_994_gold_burnin_v1 row-level-only weights",
    }
    out["burnin_policy"] = {
        "curriculum_stage": "gold_only_burnin",
        "word_count": wc,
        "title_overlap_risk": bool(stats["title_overlap_risk"]),
        "risk_flags": flags,
    }
    audit = {
        "paper_id": out["paper_id"],
        "title": out["title"],
        "summary": out["summary"],
        "primary_category": out["primary_category"],
        "umbrella_category": umbrella_category(str(out["primary_category"])),
        "word_count": wc,
        "title_overlap_stats": stats,
        "base_weight": base,
        "risk_flags": flags,
    }
    return out, audit


def main() -> None:
    args = parse_args()
    if args.repeat_count <= 0:
        raise ValueError("--repeat-count must be positive")
    for name in ("base_weight", "title_risk_weight", "under10_weight"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    prompt_spec = q5_gold.load_prompt_spec(args.prompt_spec)
    gold_rows = q5_gold.load_jsonl(args.gold_dataset)
    if len(gold_rows) != args.expected_gold_rows:
        raise ValueError(f"Expected {args.expected_gold_rows} gold rows, found {len(gold_rows)}")
    gold_ids = [str(row.get("paper_id")) for row in gold_rows]
    duplicates = [pid for pid, count in Counter(gold_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Gold dataset contains duplicate paper_id values: {duplicates[:10]}")

    unique_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for row in gold_rows:
        converted, audit = convert_gold_row(
            row,
            prompt_spec=prompt_spec,
            source_path=args.gold_dataset,
            base_weight=args.base_weight,
            title_risk_weight=args.title_risk_weight,
            under10_weight=args.under10_weight,
        )
        unique_rows.append(converted)
        audits.append(audit)

    rng = random.Random(args.seed)
    train_rows: list[dict[str, Any]] = []
    for repeat_index in range(args.repeat_count):
        shuffled = list(unique_rows)
        rng.shuffle(shuffled)
        for row in shuffled:
            repeated = dict(row)
            repeated["burnin_repeat_index"] = repeat_index
            repeated["diagnostic_row_id"] = f"{row['paper_id']}:burnin:{repeat_index:03d}"
            train_rows.append(repeated)

    output_path = args.output_dir / "train.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    q5_gold.write_jsonl(output_path, train_rows)
    q5_gold.write_jsonl(args.output_dir / "unique_training_rows.jsonl", unique_rows)
    q5_gold.write_jsonl(args.output_dir / "gold_burnin_audit.jsonl", audits)

    component_counts = Counter(row["dataset_component"] for row in train_rows)
    base_weight_counts = Counter(str(row["loss_weight_hints"]["base_weight"]) for row in train_rows)
    word_counts = Counter(str(audit["word_count"]) for audit in audits)
    title_risk_count = sum(1 for audit in audits if audit["title_overlap_stats"]["title_overlap_risk"])
    under10_count = sum(1 for audit in audits if audit["word_count"] < 10)
    umbrella_counts = Counter(audit["umbrella_category"] for audit in audits)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "scripts/build_q5_natural_language_gold_burnin_v1_dataset.py",
        "purpose": "Gold-only burn-in stage for the promoted q5/q8 sidecar before anchor or mixed-quality overlay training.",
        "output_path": str(output_path),
        "unique_training_rows_path": str(args.output_dir / "unique_training_rows.jsonl"),
        "audit_path": str(args.output_dir / "gold_burnin_audit.jsonl"),
        "prompt_spec": str(args.prompt_spec),
        "prompt_version": prompt_spec["prompt_version"],
        "input_mode": "title_abstract",
        "row_count": len(train_rows),
        "unique_gold_rows": len(unique_rows),
        "repeat_count": args.repeat_count,
        "component_counts": dict(sorted(component_counts.items())),
        "base_weight_counts": dict(sorted(base_weight_counts.items())),
        "word_count_distribution": dict(sorted(word_counts.items(), key=lambda item: int(item[0]))),
        "title_overlap_risk_rows": title_risk_count,
        "under10_rows": under10_count,
        "umbrella_counts": dict(sorted(umbrella_counts.items())),
        "weights": {
            "base_weight": args.base_weight,
            "title_risk_weight": args.title_risk_weight,
            "under10_weight": args.under10_weight,
        },
        "sources": {"gold": str(args.gold_dataset)},
        "validation": {
            "expected_unique_gold_rows": args.expected_gold_rows,
            "actual_unique_gold_rows": len(unique_rows),
            "duplicate_paper_ids": 0,
            "span_policy": "none",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = [
        "# q5 994-Gold Burn-In Dataset v1",
        "",
        f"- Rows: {len(train_rows)}",
        f"- Unique strict gold rows: {len(unique_rows)}",
        f"- Repeat count: {args.repeat_count}",
        f"- Prompt: `{prompt_spec['prompt_version']}` / `title_abstract`",
        f"- Title-overlap-risk rows: {title_risk_count}",
        f"- Under-10 rows: {under10_count}",
        "",
        "This is a curriculum burn-in dataset. It has no anchors, no silver rows, and no per-token spans.",
        "Only mild row-level dampers are used for title-overlap-risk and under-10 summaries.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(f"[gold-burnin] wrote {len(train_rows)} rows to {output_path}")
    print(f"[gold-burnin] unique gold rows={len(unique_rows)} repeat_count={args.repeat_count}")
    print(f"[gold-burnin] title_risk={title_risk_count} under10={under10_count}")
    print(f"[gold-burnin] base weights={dict(sorted(base_weight_counts.items()))}")


if __name__ == "__main__":
    main()
