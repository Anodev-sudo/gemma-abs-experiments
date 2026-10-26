#!/usr/bin/env python3
"""Build a clean natural-language set with scaled existing numeric guard rows."""
from __future__ import annotations

import argparse
import json
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
DEFAULT_NUMERIC = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_targeted_booster_clean_v1_lr4e7/"
    "dataset_600normal_160numeric_40booster_clean_v1/train.jsonl"
)
DEFAULT_HELD100 = stab_v1.DEFAULT_HELD100
DEFAULT_OUTPUT_DIR = Path(
    "hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_numeric_guard_v1/"
    "dataset_clean1648_numericguard200_v1"
)

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")
DIGIT_OR_SUP_RE = re.compile(r"[0-9\u00b2\u00b3\u00b9\u2070-\u2079\u2080-\u2089]")
FORMULAISH_RE = re.compile(r"[$\\{}^_*<>≤≥≈=%]|<sup>|</sup>")
PATCH_TYPES = {
    "numeric_antagonist_clean_digit_free_targeted_weight",
    "numeric_antagonist_clean_targeted_booster",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dataset", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--numeric-dataset", type=Path, default=DEFAULT_NUMERIC)
    parser.add_argument("--prompt-spec", type=Path, default=q5_gold.DEFAULT_PROMPT_SPEC)
    parser.add_argument("--eval-jsonl", type=Path, action="append", default=[q5_gold.DEFAULT_EVAL, DEFAULT_HELD100])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--span-scale", type=float, default=0.35)
    parser.add_argument("--base-weight", type=float, default=0.15)
    return parser.parse_args()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def patch_type(row: dict[str, Any]) -> str:
    return str(row.get("weighted_numeric_patch_row_type") or row.get("patch_component") or "")


def validate_source_span(summary: str, span: dict[str, Any], paper_id: str) -> None:
    start = int(span["start"])
    end = int(span["end"])
    text = str(span["text"])
    if summary[start:end] != text:
        raise ValueError(f"Span mismatch for {paper_id}: {text!r} != {summary[start:end]!r}")


def scaled_hints(row: dict[str, Any], base_weight: float, span_scale: float) -> dict[str, Any]:
    summary = q5_gold.row_summary(row)
    spans: list[dict[str, Any]] = []
    for span in (row.get("loss_weight_hints") or {}).get("spans", []):
        validate_source_span(summary, span, str(row.get("paper_id", "")))
        weight = max(base_weight, round(float(span.get("weight", 1.0)) * span_scale, 3))
        spans.append(
            {
                "start": int(span["start"]),
                "end": int(span["end"]),
                "text": str(span["text"]),
                "kind": f"numeric_guard::{span.get('kind', 'span')}",
                "weight": weight,
            }
        )
    return {
        "enabled": bool(spans),
        "base_weight": base_weight,
        "spans": spans,
        "policy": f"q5_numeric_guard_v1 scaled existing spans by {span_scale}",
    }


def main() -> None:
    args = parse_args()
    if args.span_scale <= 0:
        raise ValueError("--span-scale must be positive")
    if args.base_weight <= 0:
        raise ValueError("--base-weight must be positive")

    prompt_spec = q5_gold.load_prompt_spec(args.prompt_spec)
    clean_rows = q5_gold.load_jsonl(args.clean_dataset)
    clean_ids = {str(row["paper_id"]) for row in clean_rows}
    eval_ids = stab_v1.load_eval_ids(args.eval_jsonl)

    source_rows = q5_gold.load_jsonl(args.numeric_dataset)
    converted_numeric: list[dict[str, Any]] = []
    numeric_audit: list[dict[str, Any]] = []
    reject_reasons: Counter[str] = Counter()
    used_patch_ids: set[str] = set()

    for row in source_rows:
        row_type = patch_type(row)
        paper_id = str(row.get("paper_id", ""))
        summary = q5_gold.row_summary(row)
        if row_type not in PATCH_TYPES:
            reject_reasons["non_patch_row_type"] += 1
            continue
        if not paper_id or paper_id in clean_ids or paper_id in eval_ids:
            reject_reasons["duplicate_or_eval_id"] += 1
            continue
        if paper_id in used_patch_ids:
            reject_reasons["duplicate_patch_id"] += 1
            continue
        if DIGIT_OR_SUP_RE.search(summary):
            reject_reasons["digit_or_supdigit"] += 1
            continue
        if FORMULAISH_RE.search(summary):
            reject_reasons["formulaish"] += 1
            continue
        wc = word_count(summary)
        if wc < 10:
            reject_reasons["under10"] += 1
            continue
        if wc > 15:
            reject_reasons["over15"] += 1
            continue

        out = q5_gold.converted_base_row(
            row,
            component=f"numeric_guard::{row_type}",
            prompt_spec=prompt_spec,
            source_path=args.numeric_dataset,
        )
        out["loss_weight_hints"] = scaled_hints(row, args.base_weight, args.span_scale)
        out["numeric_guard_policy"] = {
            "source_row_type": row_type,
            "word_count": wc,
            "span_scale": args.span_scale,
            "base_weight": args.base_weight,
            "source_span_count": len((row.get("loss_weight_hints") or {}).get("spans", [])),
            "scaled_span_count": len(out["loss_weight_hints"]["spans"]),
        }
        converted_numeric.append(out)
        used_patch_ids.add(paper_id)
        numeric_audit.append(
            {
                "paper_id": paper_id,
                "summary": summary,
                "primary_category": row.get("primary_category", ""),
                "row_type": row_type,
                "word_count": wc,
                "base_weight": args.base_weight,
                "span_count": len(out["loss_weight_hints"]["spans"]),
                "max_span_weight": max((span["weight"] for span in out["loss_weight_hints"]["spans"]), default=args.base_weight),
            }
        )

    combined = clean_rows + converted_numeric
    ids = [str(row["paper_id"]) for row in combined]
    if len(ids) != len(set(ids)):
        raise ValueError("combined dataset contains duplicate paper_id values")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    audit_path = args.output_dir / "numeric_guard_audit.jsonl"
    q5_gold.write_jsonl(train_path, combined)
    q5_gold.write_jsonl(audit_path, numeric_audit)

    component_counts = Counter(str(row.get("dataset_component", "")) for row in combined)
    row_type_counts = Counter(row["row_type"] for row in numeric_audit)
    weights = Counter(str((row.get("loss_weight_hints") or {}).get("base_weight")) for row in combined)
    max_span_weights = Counter(str(row["max_span_weight"]) for row in numeric_audit)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "scripts/build_q5_natural_language_numeric_guard_v1_dataset.py",
        "purpose": "Existing-data-only clean natural-language rows plus scaled q5 numeric guard rows.",
        "train_path": str(train_path),
        "numeric_guard_audit_path": str(audit_path),
        "row_count": len(combined),
        "clean_rows": len(clean_rows),
        "numeric_guard_rows": len(converted_numeric),
        "numeric_source_rows": len(source_rows),
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "component_counts": dict(sorted(component_counts.items())),
        "numeric_row_type_counts": dict(sorted(row_type_counts.items())),
        "base_weight_counts": dict(sorted(weights.items())),
        "numeric_max_span_weight_counts": dict(sorted(max_span_weights.items())),
        "selection": {
            "patch_types": sorted(PATCH_TYPES),
            "filters": [
                "10_to_15_words",
                "no_ascii_or_superscript_digits",
                "no_formulaish_text",
                "exclude_clean_dataset_ids",
                "exclude_eval_ids",
            ],
            "base_weight": args.base_weight,
            "span_scale": args.span_scale,
        },
        "sources": {
            "clean_dataset": str(args.clean_dataset),
            "numeric_dataset": str(args.numeric_dataset),
            "prompt_spec": str(args.prompt_spec),
            "eval_excluded": [str(path) for path in args.eval_jsonl],
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[numeric-guard] wrote {train_path}")
    print(json.dumps({k: manifest[k] for k in ("row_count", "clean_rows", "numeric_guard_rows")}, indent=2))


if __name__ == "__main__":
    main()
