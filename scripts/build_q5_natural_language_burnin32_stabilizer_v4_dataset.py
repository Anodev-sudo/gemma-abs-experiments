#!/usr/bin/env python3
"""Build a clean-only stabilizer dataset for q5 burn-in checkpoint 32."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_q5_natural_language_burnin32_stabilizer_v1_dataset as stab_v1  # noqa: E402
import build_q5_natural_language_token_weighted_gold_v1_dataset as q5_gold  # noqa: E402


DEFAULT_OUTPUT_DIR = Path(
    "hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_gold_burnin32_stabilizer_v4/"
    "dataset_clean_digitfree_10to14_v4"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dataset", type=Path, default=stab_v1.DEFAULT_GOLD_DATASET)
    parser.add_argument("--anchor-source", type=Path, default=stab_v1.DEFAULT_ANCHORS)
    parser.add_argument("--eval-jsonl", type=Path, action="append", default=[stab_v1.DEFAULT_FRESH20, stab_v1.DEFAULT_HELD100])
    parser.add_argument("--prompt-spec", type=Path, default=stab_v1.DEFAULT_PROMPT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gold-weight", type=float, default=0.70)
    parser.add_argument("--title-risk-gold-weight", type=float, default=0.50)
    parser.add_argument("--anchor-weight", type=float, default=0.95)
    parser.add_argument("--title-risk-anchor-weight", type=float, default=0.75)
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--max-words", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20260530)
    return parser.parse_args()


def has_digit(text: str) -> bool:
    return bool(re.search(r"\d", text))


def clean_length_digitfree(row: dict[str, Any], *, min_words: int, max_words: int) -> bool:
    summary = str(row.get("summary", ""))
    words = stab_v1.word_count(summary)
    return min_words <= words <= max_words and not has_digit(summary)


def convert_row(
    row: dict[str, Any],
    *,
    component: str,
    prompt_spec: dict[str, Any],
    source_path: Path,
    base_weight: float,
    title_risk_weight: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = q5_gold.converted_base_row(
        row,
        component=component,
        prompt_spec=prompt_spec,
        source_path=source_path,
    )
    summary = str(out["summary"])
    wc = stab_v1.word_count(summary)
    stats = q5_gold.title_overlap_stats(str(out["title"]), summary)
    weight = title_risk_weight if stats["title_overlap_risk"] else base_weight
    flags = ["clean_digitfree_10to14"]
    if stats["title_overlap_risk"]:
        flags.append("title_overlap_risk_moderated")
    out["loss_weight_hints"] = {
        "enabled": True,
        "base_weight": weight,
        "spans": [],
        "policy": "q5_natural_language_burnin32_stabilizer_v4 clean-only row weight",
    }
    out["stabilizer_policy"] = {
        "curriculum_stage": "burnin32_stabilizer_clean_only",
        "word_count": wc,
        "title_overlap_risk": bool(stats["title_overlap_risk"]),
        "has_digit": False,
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
        "has_digit": False,
        "title_overlap_stats": stats,
        "base_weight": weight,
        "risk_flags": flags,
    }
    return out, audit


def main() -> None:
    args = parse_args()
    if args.min_words <= 0 or args.max_words < args.min_words:
        raise ValueError("Invalid word-count bounds")
    for name in ("gold_weight", "title_risk_gold_weight", "anchor_weight", "title_risk_anchor_weight"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    prompt_spec = q5_gold.load_prompt_spec(args.prompt_spec)
    all_gold_rows = q5_gold.load_jsonl(args.gold_dataset)
    gold_rows = [
        row
        for row in all_gold_rows
        if clean_length_digitfree(row, min_words=args.min_words, max_words=args.max_words)
    ]
    gold_ids = {str(row["paper_id"]) for row in all_gold_rows}

    eval_ids = stab_v1.load_eval_ids(args.eval_jsonl)
    anchor_candidates = [
        row
        for row in q5_gold.load_jsonl(args.anchor_source)
        if str(row.get("paper_id")) not in gold_ids and str(row.get("paper_id")) not in eval_ids
    ]
    anchor_rows = [
        row
        for row in anchor_candidates
        if clean_length_digitfree(row, min_words=args.min_words, max_words=args.max_words)
    ]

    rng = random.Random(args.seed)
    rng.shuffle(gold_rows)
    rng.shuffle(anchor_rows)

    converted: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for row in gold_rows:
        out, audit = convert_row(
            row,
            component="strict_gold_clean",
            prompt_spec=prompt_spec,
            source_path=args.gold_dataset,
            base_weight=args.gold_weight,
            title_risk_weight=args.title_risk_gold_weight,
        )
        converted.append(out)
        audits.append(audit)
    for row in anchor_rows:
        out, audit = convert_row(
            row,
            component="faithfulness_anchor_clean",
            prompt_spec=prompt_spec,
            source_path=args.anchor_source,
            base_weight=args.anchor_weight,
            title_risk_weight=args.title_risk_anchor_weight,
        )
        converted.append(out)
        audits.append(audit)

    converted_ids = [str(row["paper_id"]) for row in converted]
    if len(set(converted_ids)) != len(converted_ids):
        raise ValueError("Converted dataset contains duplicate paper_id values")
    if any(has_digit(str(row["summary"])) for row in converted):
        raise ValueError("Clean-only dataset unexpectedly contains a digit-bearing summary")

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
        "created_by": "scripts/build_q5_natural_language_burnin32_stabilizer_v4_dataset.py",
        "purpose": "Clean-only digit-free 10-14 word stabilizer dataset for q5 burn-in checkpoint 32.",
        "output_path": str(output_path),
        "audit_path": str(args.output_dir / "stabilizer_audit.jsonl"),
        "prompt_spec": str(args.prompt_spec),
        "prompt_version": prompt_spec["prompt_version"],
        "input_mode": "title_abstract",
        "row_count": len(converted),
        "component_counts": dict(sorted(component_counts.items())),
        "base_weight_counts": dict(sorted(base_weight_counts.items())),
        "risk_flag_counts": dict(sorted(flag_counts.items())),
        "weights": {
            "gold_weight": args.gold_weight,
            "title_risk_gold_weight": args.title_risk_gold_weight,
            "anchor_weight": args.anchor_weight,
            "title_risk_anchor_weight": args.title_risk_anchor_weight,
        },
        "selection": {
            "min_words": args.min_words,
            "max_words": args.max_words,
            "excluded_digit_summaries": True,
            "all_gold_rows": len(all_gold_rows),
            "selected_gold_rows": len(gold_rows),
            "anchor_candidates_after_exclusions": len(anchor_candidates),
            "selected_anchor_rows": len(anchor_rows),
            "excluded_gold_ids": len(gold_ids),
            "excluded_eval_ids": len(eval_ids),
            "seed": args.seed,
        },
        "sources": {
            "gold": str(args.gold_dataset),
            "anchor": str(args.anchor_source),
            "eval_excluded": [str(path) for path in args.eval_jsonl],
        },
        "audit": {
            "digit_rows": sum(1 for row in audits if row["has_digit"]),
            "under10_rows": sum(1 for row in audits if row["word_count"] < 10),
            "over14_rows": sum(1 for row in audits if row["word_count"] > 14),
            "title_overlap_risk_rows": sum(1 for row in audits if row["title_overlap_stats"]["title_overlap_risk"]),
        },
        "umbrella_counts": dict(sorted(umbrella_counts.items())),
        "validation": {
            "duplicate_paper_ids": 0,
            "span_policy": "none",
            "digit_rows": 0,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = [
        "# q5 Burn-In 32 Stabilizer Dataset v4",
        "",
        f"- Rows: {len(converted)}",
        f"- Strict gold clean rows: {component_counts['strict_gold_clean']}",
        f"- Faithfulness anchor clean rows: {component_counts['faithfulness_anchor_clean']}",
        f"- Prompt: `{prompt_spec['prompt_version']}` / `title_abstract`",
        "",
        "This dataset uses existing labels only.",
        "It excludes digit-bearing, under-10-word, and over-14-word labels instead",
        "of retaining them at low weight.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"[stabilizer-v4] wrote {len(converted)} rows to {output_path}")
    print(f"[stabilizer-v4] components={dict(sorted(component_counts.items()))}")
    print(f"[stabilizer-v4] base weights={dict(sorted(base_weight_counts.items()))}")
    print(f"[stabilizer-v4] risk flags={dict(sorted(flag_counts.items()))}")
    print(f"[stabilizer-v4] selected gold={len(gold_rows)} anchors={len(anchor_rows)}")


if __name__ == "__main__":
    main()
