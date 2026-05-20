#!/usr/bin/env python3
"""Materialize the final 600 normal + 200 clean numeric-antagonist SFT dataset."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You write one-sentence paper summaries for a mobile research feed. "
    "Keep them concrete, readable, and useful. Output only the summary sentence."
)
USER_TEMPLATE = (
    "The app already shows the title, author list, and date. "
    "Write one sentence that tells a reader why they might open this paper. "
    "Aim for 10 to 15 words. Avoid repeating the title. "
    'Avoid filler like "This paper" or "We propose". '
    "Do not mention authors or dates. "
    "Keep the wording natural and specific.\n\n"
    "Title:\n{title}\n\n"
    "Abstract:\n{abstract}"
)
DIGIT_RE = re.compile(r"\d")
FORMULA_RE = re.compile(r"[$<>≤≥≈]|\\[A-Za-z]+|\bO\(|\^")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def wc(text: str) -> int:
    return len(str(text).split())


def messages(title: str, abstract: str, summary: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(title=title, abstract=abstract)},
        {"role": "assistant", "content": summary},
    ]


def normal_weight_hints() -> dict[str, Any]:
    return {
        "enabled": False,
        "base_weight": 1.0,
        "spans": [],
        "policy": "normal rows are unweighted style anchors",
    }


def numeric_weight_hints(summary: str, weight: float) -> dict[str, Any]:
    return {
        "enabled": True,
        "base_weight": 1.0,
        "spans": [
            {
                "start": 0,
                "end": len(summary),
                "text": summary,
                "weight": weight,
                "kind": "clean_digit_free_numeric_antagonist_full_target",
            }
        ],
        "policy": "upweight full assistant target; prompt/source tokens stay masked out",
    }


def final_row(base: dict[str, Any], summary: str, row_type: str, component: str, weight_hints: dict[str, Any], mix_index: int) -> dict[str, Any]:
    title = str(base.get("title") or "")
    abstract = str(base.get("abstract") or "")
    out = {
        "paper_id": str(base.get("paper_id") or ""),
        "title": title,
        "abstract": abstract,
        "summary": summary,
        "primary_category": str(base.get("primary_category") or ""),
        "categories": base.get("categories") if isinstance(base.get("categories"), list) else [],
        "created_date": str(base.get("created_date") or ""),
        "teacher_model": str(base.get("teacher_model") or base.get("writer_model") or ""),
        "generator_prompt_version": str(base.get("generator_prompt_version") or base.get("writer_prompt_version") or ""),
        "review_prompt_version": str(base.get("review_prompt_version") or base.get("reviewer_prompt_version") or ""),
        "source_split": "q5_numeric_antagonist_clean_v1_final_600normal_200numeric",
        "patch_component": component,
        "weighted_numeric_patch_row_type": row_type,
        "loss_weight_hints": weight_hints,
        "mix_index": mix_index,
        "messages": messages(title, abstract, summary),
    }
    for key in ["bad_summary", "pre_dpo_summary", "failure_tags", "number_policy", "repair_candidate_id"]:
        if key in base:
            out[key] = base[key]
    if "candidate_id" in base:
        out["repair_candidate_id"] = base["candidate_id"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="hf_bucket/experiments/sft/q5_step2400_sft700_numeric_antagonist_clean_v1_pool350")
    ap.add_argument("--output-dir", default="hf_bucket/experiments/sft/q5_step2400_sft700_numeric_antagonist_clean_v1_pool350/dataset_600normal_200numeric_clean_v1")
    ap.add_argument("--normal-count", type=int, default=600)
    ap.add_argument("--numeric-count", type=int, default=200)
    ap.add_argument("--numeric-weight", type=float, default=4.0)
    args = ap.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    normal_rows = load_jsonl(run_root / "normal_anchors" / "normal_anchor_rows.jsonl")
    accepted = load_jsonl(run_root / "kimi_qwen_label_run" / "materialized" / "accepted_repairs.jsonl")

    normal_selected = normal_rows[: args.normal_count]
    if len(normal_selected) != args.normal_count:
        raise SystemExit(f"Expected {args.normal_count} normal rows, got {len(normal_selected)}")

    numeric_clean = []
    numeric_excluded = []
    for row in accepted:
        summary = str(row.get("repair_summary") or row.get("summary") or "").strip()
        reasons = []
        if not (10 <= wc(summary) <= 15):
            reasons.append("outside_10_15_words")
        if DIGIT_RE.search(summary):
            reasons.append("digit_in_summary")
        if FORMULA_RE.search(summary):
            reasons.append("formula_symbol_in_summary")
        if reasons:
            bad = dict(row)
            bad["post_materialize_exclusion_reasons"] = reasons
            numeric_excluded.append(bad)
            continue
        numeric_clean.append(row)

    if len(numeric_clean) < args.numeric_count:
        raise SystemExit(f"Need {args.numeric_count} clean numeric rows, got {len(numeric_clean)}")
    numeric_selected = numeric_clean[: args.numeric_count]
    numeric_surplus = numeric_clean[args.numeric_count:]

    final = []
    for i, row in enumerate(normal_selected, start=1):
        final.append(final_row(
            row,
            str(row["summary"]).strip(),
            "normal_style_anchor",
            "checkpoint700_train_digit_free_normal_anchor",
            normal_weight_hints(),
            i,
        ))
    for j, row in enumerate(numeric_selected, start=1):
        summary = str(row.get("repair_summary") or "").strip()
        final.append(final_row(
            row,
            summary,
            "numeric_antagonist_clean_digit_free",
            "kimi_writer_qwen_verified_checkpoint700_numeric_risk",
            numeric_weight_hints(summary, args.numeric_weight),
            args.normal_count + j,
        ))

    write_jsonl(output_dir / "train.jsonl", final)
    write_jsonl(output_dir / "normal_selected_600.jsonl", normal_selected)
    write_jsonl(output_dir / "numeric_selected_200.jsonl", numeric_selected)
    write_jsonl(output_dir / "numeric_clean_surplus.jsonl", numeric_surplus)
    write_jsonl(output_dir / "numeric_post_materialize_excluded.jsonl", numeric_excluded)

    summary_counts = Counter(r["weighted_numeric_patch_row_type"] for r in final)
    wc_counts = Counter(wc(r["summary"]) for r in final)
    digit_rows = [r for r in final if DIGIT_RE.search(r["summary"])]
    formula_rows = [r for r in final if FORMULA_RE.search(r["summary"])]
    manifest = {
        "created_by": "scripts/materialize_ckpt700_numeric_antagonist_clean_dataset.py",
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "rows": len(final),
        "row_type_counts": dict(summary_counts),
        "word_count_distribution": dict(sorted(wc_counts.items())),
        "digit_summary_rows": len(digit_rows),
        "formula_symbol_summary_rows": len(formula_rows),
        "normal_rows": len(normal_selected),
        "numeric_accepted_before_hard_filter": len(accepted),
        "numeric_clean_after_hard_filter": len(numeric_clean),
        "numeric_selected": len(numeric_selected),
        "numeric_surplus_clean": len(numeric_surplus),
        "numeric_post_materialize_excluded": len(numeric_excluded),
        "numeric_full_target_weight": args.numeric_weight,
        "prompt_runtime_alignment": "messages use SYSTEM_PROMPT and USER_TEMPLATE from scripts/generate_llama_stream.py",
        "label_provenance": {
            "normal": "checkpoint-700 parent SFT training rows, digit-free style anchors",
            "numeric": "fresh Kimi writer labels verified by Qwen, then hard local digit/formula/length filtered",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    audit = [
        "# Dataset audit",
        "",
        f"Rows: {len(final)}",
        f"Normal anchors: {len(normal_selected)}",
        f"Numeric selected: {len(numeric_selected)}",
        f"Numeric clean surplus: {len(numeric_surplus)}",
        f"Numeric excluded after hard filter: {len(numeric_excluded)}",
        f"Digit-bearing final summaries: {len(digit_rows)}",
        f"Formula/symbol final summaries: {len(formula_rows)}",
        f"Word counts: {dict(sorted(wc_counts.items()))}",
        "",
        "Numeric labels are Kimi-generated, Qwen-verified, and hard-filtered for no digits/formula symbols.",
    ]
    (output_dir / "DATASET_AUDIT.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
