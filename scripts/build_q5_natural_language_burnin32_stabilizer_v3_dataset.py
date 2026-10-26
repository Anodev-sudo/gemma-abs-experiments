#!/usr/bin/env python3
"""Build a digit-damped stabilizer dataset for q5 burn-in checkpoint 32."""
from __future__ import annotations

import argparse
import json
import re
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_q5_natural_language_burnin32_stabilizer_v1_dataset as stab_v1  # noqa: E402
import build_q5_natural_language_token_weighted_gold_v1_dataset as q5_gold  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(
    "hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v3/"
    "dataset_994gold_1000faithanchors_digitdamped_v3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dataset", type=Path, default=stab_v1.DEFAULT_GOLD_DATASET)
    parser.add_argument("--anchor-source", type=Path, default=stab_v1.DEFAULT_ANCHORS)
    parser.add_argument("--eval-jsonl", type=Path, action="append", default=[stab_v1.DEFAULT_FRESH20, stab_v1.DEFAULT_HELD100])
    parser.add_argument("--prompt-spec", type=Path, default=stab_v1.DEFAULT_PROMPT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anchor-count", type=int, default=1000)
    parser.add_argument("--gold-weight", type=float, default=0.55)
    parser.add_argument("--title-risk-gold-weight", type=float, default=0.35)
    parser.add_argument("--under10-gold-weight", type=float, default=0.15)
    parser.add_argument("--digit-gold-weight", type=float, default=0.30)
    parser.add_argument("--clean-anchor-weight", type=float, default=0.90)
    parser.add_argument("--title-risk-anchor-weight", type=float, default=0.70)
    parser.add_argument("--digit-anchor-weight", type=float, default=0.10)
    parser.add_argument("--under10-anchor-weight", type=float, default=0.25)
    parser.add_argument("--over14-anchor-weight", type=float, default=0.30)
    parser.add_argument("--expected-gold-rows", type=int, default=994)
    parser.add_argument("--seed", type=int, default=20260529)
    return parser.parse_args()


def has_digit(text: str) -> bool:
    return bool(re.search(r"\d", text))


def gold_weight(row: dict[str, Any], args: argparse.Namespace) -> tuple[float, list[str], dict[str, Any], int]:
    summary = str(row["summary"])
    wc = stab_v1.word_count(summary)
    stats = q5_gold.title_overlap_stats(str(row["title"]), summary)
    weight = args.title_risk_gold_weight if stats["title_overlap_risk"] else args.gold_weight
    flags: list[str] = []
    if stats["title_overlap_risk"]:
        flags.append("title_overlap_risk_damped")
    if has_digit(summary):
        weight = min(weight, args.digit_gold_weight)
        flags.append("digit_damped")
    if wc < 10:
        weight = min(weight, args.under10_gold_weight)
        flags.append("under10_damped")
    return weight, flags, stats, wc


def anchor_weight(row: dict[str, Any], args: argparse.Namespace) -> tuple[float, list[str], dict[str, Any], int]:
    summary = str(row["summary"])
    wc = stab_v1.word_count(summary)
    stats = q5_gold.title_overlap_stats(str(row["title"]), summary)
    flags: list[str] = []
    if has_digit(summary):
        flags.append("digit_damped")
        return args.digit_anchor_weight, flags, stats, wc
    if wc < 10:
        flags.append("under10_damped")
        return args.under10_anchor_weight, flags, stats, wc
    if wc > 14:
        flags.append("over14_damped")
        return args.over14_anchor_weight, flags, stats, wc
    if stats["title_overlap_risk"]:
        flags.append("title_overlap_risk_moderated")
        return args.title_risk_anchor_weight, flags, stats, wc
    flags.append("clean_digitfree_anchor")
    return args.clean_anchor_weight, flags, stats, wc


def convert_gold_row(
    row: dict[str, Any],
    *,
    prompt_spec: dict[str, Any],
    source_path: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = q5_gold.converted_base_row(
        row,
        component="strict_gold_stabilizer",
        prompt_spec=prompt_spec,
        source_path=source_path,
    )
    weight, flags, stats, wc = gold_weight(out, args)
    out["loss_weight_hints"] = {
        "enabled": True,
        "base_weight": weight,
        "spans": [],
        "policy": "q5_natural_language_burnin32_stabilizer_v3 digit-damped strict gold row weight",
    }
    out["stabilizer_policy"] = {
        "curriculum_stage": "burnin32_stabilizer_digit_damped",
        "word_count": wc,
        "title_overlap_risk": bool(stats["title_overlap_risk"]),
        "has_digit": has_digit(out["summary"]),
        "risk_flags": flags,
    }
    audit = {
        "paper_id": out["paper_id"],
        "dataset_component": out["dataset_component"],
        "title": out["title"],
        "summary": out["summary"],
        "primary_category": out["primary_category"],
        "umbrella_category": stab_v1.umbrella_category(str(out["primary_category"])),
        "word_count": wc,
        "has_digit": has_digit(out["summary"]),
        "title_overlap_stats": stats,
        "base_weight": weight,
        "risk_flags": flags,
    }
    return out, audit


def convert_anchor_row(
    row: dict[str, Any],
    *,
    prompt_spec: dict[str, Any],
    source_path: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = q5_gold.converted_base_row(
        row,
        component="faithfulness_anchor",
        prompt_spec=prompt_spec,
        source_path=source_path,
    )
    weight, flags, stats, wc = anchor_weight(out, args)
    out["loss_weight_hints"] = {
        "enabled": True,
        "base_weight": weight,
        "spans": [],
        "policy": "q5_natural_language_burnin32_stabilizer_v3 digit-damped faithfulness anchor row weight",
    }
    out["stabilizer_policy"] = {
        "curriculum_stage": "burnin32_stabilizer_digit_damped",
        "word_count": wc,
        "title_overlap_risk": bool(stats["title_overlap_risk"]),
        "has_digit": has_digit(out["summary"]),
        "risk_flags": flags,
    }
    audit = {
        "paper_id": out["paper_id"],
        "dataset_component": out["dataset_component"],
        "title": out["title"],
        "summary": out["summary"],
        "primary_category": out["primary_category"],
        "umbrella_category": stab_v1.umbrella_category(str(out["primary_category"])),
        "word_count": wc,
        "has_digit": has_digit(out["summary"]),
        "title_overlap_stats": stats,
        "base_weight": weight,
        "risk_flags": flags,
    }
    return out, audit


def main() -> None:
    args = parse_args()
    if args.anchor_count <= 0:
        raise ValueError("--anchor-count must be positive")
    weight_args = [
        "gold_weight",
        "title_risk_gold_weight",
        "under10_gold_weight",
        "digit_gold_weight",
        "clean_anchor_weight",
        "title_risk_anchor_weight",
        "digit_anchor_weight",
        "under10_anchor_weight",
        "over14_anchor_weight",
    ]
    for name in weight_args:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    prompt_spec = q5_gold.load_prompt_spec(args.prompt_spec)
    gold_rows = q5_gold.load_jsonl(args.gold_dataset)
    if len(gold_rows) != args.expected_gold_rows:
        raise ValueError(f"Expected {args.expected_gold_rows} gold rows, found {len(gold_rows)}")
    gold_ids = {str(row["paper_id"]) for row in gold_rows}
    if len(gold_ids) != len(gold_rows):
        raise ValueError("Gold dataset contains duplicate paper_id values")

    eval_ids = stab_v1.load_eval_ids(args.eval_jsonl)
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
            args=args,
        )
        converted.append(out)
        audits.append(audit)
    for row in anchor_rows:
        out, audit = convert_anchor_row(
            row,
            prompt_spec=prompt_spec,
            source_path=args.anchor_source,
            args=args,
        )
        converted.append(out)
        audits.append(audit)

    converted_ids = [str(row["paper_id"]) for row in converted]
    if len(set(converted_ids)) != len(converted_ids):
        raise ValueError("Converted dataset contains duplicate paper_id values")

    rng.shuffle(converted)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "train.jsonl"
    q5_gold.write_jsonl(output_path, converted)
    q5_gold.write_jsonl(args.output_dir / "stabilizer_audit.jsonl", audits)

    component_counts = Counter(row["dataset_component"] for row in converted)
    base_weight_counts = Counter(str(row["loss_weight_hints"]["base_weight"]) for row in converted)
    flag_counts = Counter(flag for row in audits for flag in row["risk_flags"])
    umbrella_counts = Counter(row["umbrella_category"] for row in audits)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "scripts/build_q5_natural_language_burnin32_stabilizer_v3_dataset.py",
        "purpose": "Digit-damped stabilizer dataset for q5 natural-language burn-in checkpoint 32.",
        "output_path": str(output_path),
        "audit_path": str(args.output_dir / "stabilizer_audit.jsonl"),
        "prompt_spec": str(args.prompt_spec),
        "prompt_version": prompt_spec["prompt_version"],
        "input_mode": "title_abstract",
        "row_count": len(converted),
        "component_counts": dict(sorted(component_counts.items())),
        "base_weight_counts": dict(sorted(base_weight_counts.items())),
        "risk_flag_counts": dict(sorted(flag_counts.items())),
        "weights": {name: getattr(args, name) for name in weight_args},
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
        "audit": {
            "digit_rows": sum(1 for row in audits if row["has_digit"]),
            "under10_rows": sum(1 for row in audits if row["word_count"] < 10),
            "over14_rows": sum(1 for row in audits if row["word_count"] > 14),
            "title_overlap_risk_rows": sum(1 for row in audits if row["title_overlap_stats"]["title_overlap_risk"]),
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
        "# q5 Burn-In 32 Stabilizer Dataset v3",
        "",
        f"- Rows: {len(converted)}",
        f"- Strict gold rows: {component_counts['strict_gold_stabilizer']}",
        f"- Faithfulness anchors: {component_counts['faithfulness_anchor']}",
        f"- Prompt: `{prompt_spec['prompt_version']}` / `title_abstract`",
        "",
        "This dataset uses existing labels only.",
        "It keeps clean digit-free anchors strong while damping digit-bearing,",
        "very short, and over-length anchor rows.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"[stabilizer-v3] wrote {len(converted)} rows to {output_path}")
    print(f"[stabilizer-v3] components={dict(sorted(component_counts.items()))}")
    print(f"[stabilizer-v3] base weights={dict(sorted(base_weight_counts.items()))}")
    print(f"[stabilizer-v3] risk flags={dict(sorted(flag_counts.items()))}")
    print(f"[stabilizer-v3] anchor candidates after exclusions={len(anchor_candidates)}")


if __name__ == "__main__":
    main()
