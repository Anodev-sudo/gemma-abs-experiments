#!/usr/bin/env python3
"""Build a stabilizer dataset for q5 natural-language burn-in checkpoint 32."""
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
DEFAULT_ANCHORS = q5_gold.DEFAULT_ANCHORS
DEFAULT_FRESH20 = q5_gold.DEFAULT_EVAL
DEFAULT_HELD100 = Path(
    "hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_token_weighted_gold_v1/"
    "lr4e6_steps160/eval/blind_ab_ckpt96_vs_baseline_feed_preference_100_20260527/"
    "source_100_random_recent3y_excluding_train.jsonl"
)
DEFAULT_PROMPT_SPEC = q5_gold.DEFAULT_PROMPT_SPEC
DEFAULT_OUTPUT_DIR = Path(
    "hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v1/"
    "dataset_994gold_1000faithanchors_stabilizer_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dataset", type=Path, default=DEFAULT_GOLD_DATASET)
    parser.add_argument("--anchor-source", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--eval-jsonl", type=Path, action="append", default=[DEFAULT_FRESH20, DEFAULT_HELD100])
    parser.add_argument("--prompt-spec", type=Path, default=DEFAULT_PROMPT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anchor-count", type=int, default=1000)
    parser.add_argument("--gold-weight", type=float, default=0.65)
    parser.add_argument("--title-risk-gold-weight", type=float, default=0.50)
    parser.add_argument("--under10-gold-weight", type=float, default=0.40)
    parser.add_argument("--anchor-weight", type=float, default=0.35)
    parser.add_argument("--expected-gold-rows", type=int, default=994)
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


def load_eval_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        for row in q5_gold.load_jsonl(path):
            paper_id = row.get("paper_id")
            if paper_id:
                ids.add(str(paper_id))
    return ids


def base_weight_for_gold(
    row: dict[str, Any],
    *,
    gold_weight: float,
    title_risk_gold_weight: float,
    under10_gold_weight: float,
) -> tuple[float, list[str], int, dict[str, Any]]:
    summary = str(row["summary"])
    wc = word_count(summary)
    stats = q5_gold.title_overlap_stats(str(row["title"]), summary)
    weight = title_risk_gold_weight if stats["title_overlap_risk"] else gold_weight
    flags: list[str] = []
    if stats["title_overlap_risk"]:
        flags.append("title_overlap_risk_damped")
    if wc < 10:
        weight = min(weight, under10_gold_weight)
        flags.append("under10_damped")
    return weight, flags, wc, stats


def convert_row(
    row: dict[str, Any],
    *,
    component: str,
    weight: float,
    prompt_spec: dict[str, Any],
    source_path: Path,
    policy: str,
) -> dict[str, Any]:
    out = q5_gold.converted_base_row(
        row,
        component=component,
        prompt_spec=prompt_spec,
        source_path=source_path,
    )
    out["loss_weight_hints"] = {
        "enabled": True,
        "base_weight": weight,
        "spans": [],
        "policy": policy,
    }
    return out


def convert_gold_row(
    row: dict[str, Any],
    *,
    prompt_spec: dict[str, Any],
    source_path: Path,
    gold_weight: float,
    title_risk_gold_weight: float,
    under10_gold_weight: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_row = q5_gold.converted_base_row(
        row,
        component="strict_gold_stabilizer",
        prompt_spec=prompt_spec,
        source_path=source_path,
    )
    weight, flags, wc, stats = base_weight_for_gold(
        base_row,
        gold_weight=gold_weight,
        title_risk_gold_weight=title_risk_gold_weight,
        under10_gold_weight=under10_gold_weight,
    )
    base_row["loss_weight_hints"] = {
        "enabled": True,
        "base_weight": weight,
        "spans": [],
        "policy": "q5_natural_language_burnin32_stabilizer_v1 strict gold row-level damping",
    }
    base_row["stabilizer_policy"] = {
        "curriculum_stage": "burnin32_stabilizer",
        "word_count": wc,
        "title_overlap_risk": bool(stats["title_overlap_risk"]),
        "risk_flags": flags,
    }
    audit = {
        "paper_id": base_row["paper_id"],
        "dataset_component": base_row["dataset_component"],
        "title": base_row["title"],
        "summary": base_row["summary"],
        "primary_category": base_row["primary_category"],
        "umbrella_category": umbrella_category(str(base_row["primary_category"])),
        "word_count": wc,
        "title_overlap_stats": stats,
        "base_weight": weight,
        "risk_flags": flags,
    }
    return base_row, audit


def convert_anchor_row(
    row: dict[str, Any],
    *,
    prompt_spec: dict[str, Any],
    source_path: Path,
    anchor_weight: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = convert_row(
        row,
        component="faithfulness_anchor",
        weight=anchor_weight,
        prompt_spec=prompt_spec,
        source_path=source_path,
        policy="q5_natural_language_burnin32_stabilizer_v1 moderate-weight faithfulness anchor",
    )
    stats = q5_gold.title_overlap_stats(out["title"], out["summary"])
    audit = {
        "paper_id": out["paper_id"],
        "dataset_component": out["dataset_component"],
        "title": out["title"],
        "summary": out["summary"],
        "primary_category": out["primary_category"],
        "umbrella_category": umbrella_category(str(out["primary_category"])),
        "word_count": word_count(out["summary"]),
        "title_overlap_stats": stats,
        "base_weight": anchor_weight,
        "risk_flags": ["faithfulness_anchor_no_spans"],
    }
    return out, audit


def main() -> None:
    args = parse_args()
    if args.anchor_count <= 0:
        raise ValueError("--anchor-count must be positive")
    for name in ("gold_weight", "title_risk_gold_weight", "under10_gold_weight", "anchor_weight"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    prompt_spec = q5_gold.load_prompt_spec(args.prompt_spec)
    gold_rows = q5_gold.load_jsonl(args.gold_dataset)
    if len(gold_rows) != args.expected_gold_rows:
        raise ValueError(f"Expected {args.expected_gold_rows} gold rows, found {len(gold_rows)}")
    gold_ids = {str(row["paper_id"]) for row in gold_rows}
    if len(gold_ids) != len(gold_rows):
        raise ValueError("Gold dataset contains duplicate paper_id values")

    eval_ids = load_eval_ids(args.eval_jsonl)
    anchor_candidates = [
        row
        for row in q5_gold.load_jsonl(args.anchor_source)
        if str(row.get("paper_id")) not in gold_ids and str(row.get("paper_id")) not in eval_ids
    ]
    if len(anchor_candidates) < args.anchor_count:
        raise ValueError(f"Need {args.anchor_count} anchors, found {len(anchor_candidates)}")

    rng = random.Random(args.seed)
    rng.shuffle(anchor_candidates)
    anchor_rows = anchor_candidates[: args.anchor_count]

    converted: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for row in gold_rows:
        out, audit = convert_gold_row(
            row,
            prompt_spec=prompt_spec,
            source_path=args.gold_dataset,
            gold_weight=args.gold_weight,
            title_risk_gold_weight=args.title_risk_gold_weight,
            under10_gold_weight=args.under10_gold_weight,
        )
        converted.append(out)
        audits.append(audit)
    for row in anchor_rows:
        out, audit = convert_anchor_row(
            row,
            prompt_spec=prompt_spec,
            source_path=args.anchor_source,
            anchor_weight=args.anchor_weight,
        )
        converted.append(out)
        audits.append(audit)

    converted_ids = [str(row["paper_id"]) for row in converted]
    if len(set(converted_ids)) != len(converted_ids):
        raise ValueError("Converted dataset contains duplicate paper_id values")

    rng.shuffle(converted)
    output_path = args.output_dir / "train.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    q5_gold.write_jsonl(output_path, converted)
    q5_gold.write_jsonl(args.output_dir / "stabilizer_audit.jsonl", audits)

    component_counts = Counter(row["dataset_component"] for row in converted)
    base_weight_counts = Counter(str(row["loss_weight_hints"]["base_weight"]) for row in converted)
    gold_audits = [row for row in audits if row["dataset_component"] == "strict_gold_stabilizer"]
    anchor_audits = [row for row in audits if row["dataset_component"] == "faithfulness_anchor"]
    umbrella_counts = Counter(row["umbrella_category"] for row in audits)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "scripts/build_q5_natural_language_burnin32_stabilizer_v1_dataset.py",
        "purpose": "Low-LR stabilizer dataset for q5 natural-language burn-in checkpoint 32.",
        "output_path": str(output_path),
        "audit_path": str(args.output_dir / "stabilizer_audit.jsonl"),
        "prompt_spec": str(args.prompt_spec),
        "prompt_version": prompt_spec["prompt_version"],
        "input_mode": "title_abstract",
        "row_count": len(converted),
        "component_counts": dict(sorted(component_counts.items())),
        "base_weight_counts": dict(sorted(base_weight_counts.items())),
        "weights": {
            "strict_gold_default": args.gold_weight,
            "strict_gold_title_overlap_risk": args.title_risk_gold_weight,
            "strict_gold_under10_cap": args.under10_gold_weight,
            "faithfulness_anchor": args.anchor_weight,
        },
        "sources": {
            "gold": str(args.gold_dataset),
            "anchor": str(args.anchor_source),
            "eval_excluded": [str(path) for path in args.eval_jsonl],
        },
        "anchor_selection": {
            "seed": args.seed,
            "requested_count": args.anchor_count,
            "candidate_count_after_exclusions": len(anchor_candidates),
            "excluded_gold_ids": len(gold_ids),
            "excluded_eval_ids": len(eval_ids),
        },
        "gold_audit": {
            "title_overlap_risk_rows": sum(1 for row in gold_audits if row["title_overlap_stats"]["title_overlap_risk"]),
            "under10_rows": sum(1 for row in gold_audits if row["word_count"] < 10),
        },
        "anchor_audit": {
            "rows": len(anchor_audits),
            "under10_rows": sum(1 for row in anchor_audits if row["word_count"] < 10),
        },
        "umbrella_counts": dict(sorted(umbrella_counts.items())),
        "validation": {
            "expected_row_count": args.expected_gold_rows + args.anchor_count,
            "actual_row_count": len(converted),
            "duplicate_paper_ids": 0,
            "span_policy": "none",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = [
        "# q5 Burn-In 32 Stabilizer Dataset v1",
        "",
        f"- Rows: {len(converted)}",
        f"- Strict gold rows: {component_counts['strict_gold_stabilizer']}",
        f"- Faithfulness anchors: {component_counts['faithfulness_anchor']}",
        f"- Prompt: `{prompt_spec['prompt_version']}` / `title_abstract`",
        "",
        "This is a second-stage stabilizer dataset for burn-in checkpoint 32.",
        "It uses row-level weights only: no token spans and no lower-quality mined rows.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"[stabilizer] wrote {len(converted)} rows to {output_path}")
    print(f"[stabilizer] components={dict(sorted(component_counts.items()))}")
    print(f"[stabilizer] base weights={dict(sorted(base_weight_counts.items()))}")
    print(f"[stabilizer] anchor candidates after exclusions={len(anchor_candidates)}")


if __name__ == "__main__":
    main()
