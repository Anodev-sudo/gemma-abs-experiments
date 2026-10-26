#!/usr/bin/env python3
"""Build a q5 natural-language dataset with filtered under-14 support rows."""
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
import build_q5_natural_language_token_weighted_gold_v1_dataset as q5_gold  # noqa: E402
import build_q5_natural_language_burnin32_stabilizer_v1_dataset as stab_v1  # noqa: E402


DEFAULT_CLEAN = Path(
    "hf_bucket/experiments/sft/q5_checkpoint50_natural_language_clean_digitfree_overlay_v1/"
    "dataset_clean_digitfree_10to14_no_supdigit_v1/train.jsonl"
)
DEFAULT_UNDER14 = Path(
    "hf_bucket/experiments/sft/q8_promoted_broad_reweighted_under14_v1/"
    "dataset_under14_kimi_qwen_v1/train.jsonl"
)
DEFAULT_HELD100 = stab_v1.DEFAULT_HELD100
DEFAULT_OUTPUT_DIR = Path(
    "hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_under14_highsupport_v1/"
    "dataset_clean1648_under14_highsupport_v1"
)

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")
DIGIT_OR_SUP_RE = re.compile(r"[0-9\u00b2\u00b3\u00b9\u2070-\u2079\u2080-\u2089]")
FORMULAISH_RE = re.compile(r"[$\\{}^_*<>≤≥≈=%]|<sup>|</sup>")
BOILERPLATE_RE = re.compile(r"^(this paper|we propose|we present|we introduce)\b", re.I)

EXCLUDED_COMPONENTS = {
    "base_10746_low_weight",
}

SUPPORT_WEIGHTS = {
    "faithfulness_overlay_1400": 0.10,
    "recenter_broad_overlay": 0.12,
    "specificity_preserving_400": 0.20,
    "step2400_mined_kimi_qwen_10000": 0.22,
    "hard77_preserve": 0.25,
    "q5_correction_overlay_repair_kimi_qwen": 0.24,
    "q5_precision_plus_q4_source_kimi_qwen": 0.22,
    "q5_precision_kimi_qwen_combined": 0.22,
    "q5_residual218_push_kimi_qwen": 0.22,
    "q5_q8shape150_q8win_value_aware": 0.24,
    "q5_secondpass50_q8win_specificity": 0.24,
    "q5_value125_q8win_value_aware": 0.24,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dataset", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--under14-dataset", type=Path, default=DEFAULT_UNDER14)
    parser.add_argument("--prompt-spec", type=Path, default=q5_gold.DEFAULT_PROMPT_SPEC)
    parser.add_argument("--eval-jsonl", type=Path, action="append", default=[q5_gold.DEFAULT_EVAL, DEFAULT_HELD100])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260604)
    return parser.parse_args()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def has_digit(text: str) -> bool:
    return bool(DIGIT_OR_SUP_RE.search(text))


def component_name(row: dict[str, Any]) -> str:
    return str(row.get("dataset_component") or row.get("component") or row.get("source_component") or "")


def support_weight(component: str) -> float:
    return SUPPORT_WEIGHTS.get(component, 0.12)


def umbrella(row: dict[str, Any]) -> str:
    return stab_v1.umbrella_category(str(row.get("primary_category", "")))


def is_support_candidate(row: dict[str, Any]) -> tuple[bool, list[str]]:
    summary = q5_gold.row_summary(row)
    reasons: list[str] = []
    component = component_name(row)
    wc = word_count(summary)

    if component in EXCLUDED_COMPONENTS:
        reasons.append("excluded_component")
    if wc < 10:
        reasons.append("under10")
    if wc > 13:
        reasons.append("over13")
    if has_digit(summary):
        reasons.append("digit_or_supdigit")
    if FORMULAISH_RE.search(summary):
        reasons.append("formulaish")
    if BOILERPLATE_RE.search(summary):
        reasons.append("boilerplate")

    stats = q5_gold.title_overlap_stats(str(row.get("title", "")), summary)
    if stats["title_overlap_risk"]:
        reasons.append("title_overlap_risk")
    return not reasons, reasons


def main() -> None:
    args = parse_args()
    prompt_spec = q5_gold.load_prompt_spec(args.prompt_spec)
    clean_rows = q5_gold.load_jsonl(args.clean_dataset)
    clean_ids = {str(row["paper_id"]) for row in clean_rows}
    eval_ids = stab_v1.load_eval_ids(args.eval_jsonl)

    source_rows = q5_gold.load_jsonl(args.under14_dataset)
    support_rows: list[dict[str, Any]] = []
    reject_reasons: Counter[str] = Counter()
    excluded_duplicate_or_eval = 0
    for row in source_rows:
        paper_id = str(row.get("paper_id", ""))
        if not paper_id or paper_id in clean_ids or paper_id in eval_ids:
            excluded_duplicate_or_eval += 1
            continue
        ok, reasons = is_support_candidate(row)
        if not ok:
            reject_reasons.update(reasons)
            continue
        support_rows.append(row)

    rng = random.Random(args.seed)
    rng.shuffle(support_rows)

    converted_support: list[dict[str, Any]] = []
    support_audit: list[dict[str, Any]] = []
    for row in support_rows:
        original_component = component_name(row)
        out = q5_gold.converted_base_row(
            row,
            component=f"under14_highsupport::{original_component}",
            prompt_spec=prompt_spec,
            source_path=args.under14_dataset,
        )
        weight = support_weight(original_component)
        out["loss_weight_hints"] = {
            "enabled": True,
            "base_weight": weight,
            "spans": [],
            "policy": "q5_natural_language_under14_highsupport_v1 low-weight existing support",
        }
        out["under14_highsupport_policy"] = {
            "original_component": original_component,
            "word_count": word_count(out["summary"]),
            "support_weight": weight,
            "title_overlap_stats": q5_gold.title_overlap_stats(out["title"], out["summary"]),
        }
        converted_support.append(out)
        support_audit.append(
            {
                "paper_id": out["paper_id"],
                "summary": out["summary"],
                "primary_category": out["primary_category"],
                "umbrella_category": umbrella(out),
                "original_component": original_component,
                "base_weight": weight,
                "word_count": word_count(out["summary"]),
            }
        )

    combined = clean_rows + converted_support
    ids = [str(row["paper_id"]) for row in combined]
    if len(ids) != len(set(ids)):
        raise ValueError("combined dataset contains duplicate paper_id values")

    rng.shuffle(combined)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    audit_path = args.output_dir / "support_audit.jsonl"
    q5_gold.write_jsonl(train_path, combined)
    q5_gold.write_jsonl(audit_path, support_audit)

    component_counts = Counter(str(row.get("dataset_component", "")) for row in combined)
    support_component_counts = Counter(row["original_component"] for row in support_audit)
    support_umbrella_counts = Counter(row["umbrella_category"] for row in support_audit)
    weights = Counter(str((row.get("loss_weight_hints") or {}).get("base_weight")) for row in combined)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "scripts/build_q5_natural_language_under14_highsupport_v1_dataset.py",
        "purpose": "Existing-data-only clean natural-language rows plus low-weight filtered q8 under-14 repair support.",
        "train_path": str(train_path),
        "support_audit_path": str(audit_path),
        "row_count": len(combined),
        "clean_rows": len(clean_rows),
        "support_rows": len(converted_support),
        "under14_source_rows": len(source_rows),
        "excluded_duplicate_or_eval_ids": excluded_duplicate_or_eval,
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "component_counts": dict(sorted(component_counts.items())),
        "support_component_counts": dict(sorted(support_component_counts.items())),
        "support_umbrella_counts": dict(sorted(support_umbrella_counts.items())),
        "base_weight_counts": dict(sorted(weights.items())),
        "selection": {
            "seed": args.seed,
            "filters": [
                "exclude_base_10746_low_weight",
                "10_to_13_words",
                "no_ascii_or_superscript_digits",
                "no_formulaish_text",
                "no_boilerplate_opening",
                "no_title_overlap_risk",
                "exclude_clean_dataset_ids",
                "exclude_eval_ids",
            ],
        },
        "sources": {
            "clean_dataset": str(args.clean_dataset),
            "under14_dataset": str(args.under14_dataset),
            "prompt_spec": str(args.prompt_spec),
            "eval_excluded": [str(path) for path in args.eval_jsonl],
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[under14-highsupport] wrote {train_path}")
    print(json.dumps({k: manifest[k] for k in ("row_count", "clean_rows", "support_rows")}, indent=2))


if __name__ == "__main__":
    main()
