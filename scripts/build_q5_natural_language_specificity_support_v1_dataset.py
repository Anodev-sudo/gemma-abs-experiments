#!/usr/bin/env python3
"""Build a natural-language q5 overlay with low-weight specificity support."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict, deque
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
DEFAULT_BROAD = Path(
    "hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v2/"
    "dataset_broad_base_low_repair_high_v2/train.jsonl"
)
DEFAULT_HELD100 = stab_v1.DEFAULT_HELD100
DEFAULT_OUTPUT_DIR = Path(
    "hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_specificity_support_v1/"
    "dataset_clean1648_broadsupport_v1"
)

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")
DIGIT_OR_SUP_RE = re.compile(r"[0-9\u00b2\u00b3\u00b9\u2070-\u2079\u2080-\u2089]")
FORMULAISH_RE = re.compile(r"[$\\{}^_*<>≤≥≈=%]|<sup>|</sup>")
BOILERPLATE_RE = re.compile(r"^(this paper|we propose|we present|we introduce)\b", re.I)

COMPONENT_WEIGHTS = {
    "base_10746_low_weight": 0.08,
    "faithfulness_overlay_1400": 0.12,
    "recenter_broad_overlay": 0.14,
    "specificity_preserving_400": 0.16,
    "step2400_mined_kimi_qwen_10000": 0.14,
    "hard77_preserve": 0.20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dataset", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--broad-dataset", type=Path, default=DEFAULT_BROAD)
    parser.add_argument("--prompt-spec", type=Path, default=q5_gold.DEFAULT_PROMPT_SPEC)
    parser.add_argument("--eval-jsonl", type=Path, action="append", default=[q5_gold.DEFAULT_EVAL, DEFAULT_HELD100])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--support-count", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=20260603)
    return parser.parse_args()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def has_digit(text: str) -> bool:
    return bool(DIGIT_OR_SUP_RE.search(text))


def component_name(row: dict[str, Any]) -> str:
    return str(row.get("dataset_component") or row.get("component") or row.get("source_component") or "")


def support_weight(component: str) -> float:
    if component.startswith("q5_"):
        return 0.18
    return COMPONENT_WEIGHTS.get(component, 0.12)


def umbrella(row: dict[str, Any]) -> str:
    return stab_v1.umbrella_category(str(row.get("primary_category", "")))


def is_support_candidate(row: dict[str, Any]) -> tuple[bool, list[str]]:
    summary = q5_gold.row_summary(row)
    reasons: list[str] = []
    wc = word_count(summary)
    if wc < 10:
        reasons.append("under10")
    if wc > 14:
        reasons.append("over14")
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


def select_round_robin(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(component_name(row), umbrella(row))].append(row)
    queues: list[deque[dict[str, Any]]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        rng.shuffle(bucket)
        queues.append(deque(bucket))
    rng.shuffle(queues)

    selected: list[dict[str, Any]] = []
    while queues and len(selected) < count:
        next_queues: list[deque[dict[str, Any]]] = []
        for queue in queues:
            if not queue:
                continue
            selected.append(queue.popleft())
            if len(selected) >= count:
                break
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return selected


def main() -> None:
    args = parse_args()
    if args.support_count < 0:
        raise ValueError("--support-count must be non-negative")

    prompt_spec = q5_gold.load_prompt_spec(args.prompt_spec)
    clean_rows = q5_gold.load_jsonl(args.clean_dataset)
    clean_ids = {str(row["paper_id"]) for row in clean_rows}
    eval_ids = stab_v1.load_eval_ids(args.eval_jsonl)

    broad_rows = q5_gold.load_jsonl(args.broad_dataset)
    candidate_rows: list[dict[str, Any]] = []
    reject_reasons: Counter[str] = Counter()
    excluded_duplicate_or_eval = 0
    for row in broad_rows:
        paper_id = str(row.get("paper_id", ""))
        if not paper_id or paper_id in clean_ids or paper_id in eval_ids:
            excluded_duplicate_or_eval += 1
            continue
        ok, reasons = is_support_candidate(row)
        if not ok:
            reject_reasons.update(reasons)
            continue
        candidate_rows.append(row)

    selected_support = select_round_robin(candidate_rows, args.support_count, args.seed)
    converted_support: list[dict[str, Any]] = []
    support_audit: list[dict[str, Any]] = []
    for row in selected_support:
        original_component = component_name(row)
        out = q5_gold.converted_base_row(
            row,
            component=f"broad_support::{original_component}",
            prompt_spec=prompt_spec,
            source_path=args.broad_dataset,
        )
        weight = support_weight(original_component)
        out["loss_weight_hints"] = {
            "enabled": True,
            "base_weight": weight,
            "spans": [],
            "policy": "q5_natural_language_specificity_support_v1 low-weight deterministic support",
        }
        out["specificity_support_policy"] = {
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

    rng = random.Random(args.seed + 1)
    rng.shuffle(combined)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.jsonl"
    audit_path = args.output_dir / "support_audit.jsonl"
    q5_gold.write_jsonl(train_path, combined)
    q5_gold.write_jsonl(audit_path, support_audit)

    component_counts = Counter(str(row.get("dataset_component", "")) for row in combined)
    support_component_counts = Counter(row["original_component"] for row in support_audit)
    umbrella_counts = Counter(row["umbrella_category"] for row in support_audit)
    weights = Counter(str((row.get("loss_weight_hints") or {}).get("base_weight")) for row in combined)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "scripts/build_q5_natural_language_specificity_support_v1_dataset.py",
        "purpose": "Existing-data-only clean natural-language rows plus low-weight broad specificity support.",
        "train_path": str(train_path),
        "support_audit_path": str(audit_path),
        "row_count": len(combined),
        "clean_rows": len(clean_rows),
        "support_rows": len(converted_support),
        "support_candidates_after_filter": len(candidate_rows),
        "broad_source_rows": len(broad_rows),
        "excluded_duplicate_or_eval_ids": excluded_duplicate_or_eval,
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "component_counts": dict(sorted(component_counts.items())),
        "support_component_counts": dict(sorted(support_component_counts.items())),
        "support_umbrella_counts": dict(sorted(umbrella_counts.items())),
        "base_weight_counts": dict(sorted(weights.items())),
        "selection": {
            "support_count": args.support_count,
            "seed": args.seed,
            "filters": [
                "10_to_14_words",
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
            "broad_dataset": str(args.broad_dataset),
            "prompt_spec": str(args.prompt_spec),
            "eval_excluded": [str(path) for path in args.eval_jsonl],
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[specificity-support] wrote {train_path}")
    print(json.dumps({k: manifest[k] for k in ("row_count", "clean_rows", "support_rows", "support_candidates_after_filter")}, indent=2))


if __name__ == "__main__":
    main()
