#!/usr/bin/env python3
"""Select exactly 160 accepted Kimi/Qwen style-normalization rows."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin200_style_norm_v1_lr5e7")
CATEGORIES = (
    "generic_title_drift",
    "value_attachment_or_polarity",
    "awkward_malformed_technical",
    "result_preserving_no_digit_numeric",
)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")
DIGIT_RE = re.compile(r"\d")
FORMULA_RE = re.compile(r"[$<>≤≥≈]|\\[A-Za-z]+|\bO\(|\^|_|𝔽|[α-ωΑ-Ω]")
BANNED_OPENINGS = ("this paper", "the paper", "the authors", "we propose", "in this work", "in this paper")
HIDDEN_QUANT_RE = re.compile(
    r"\b(most|often|widespread|best|highest|largest|substantial|dramatic|major|state-of-the-art|"
    r"outperforms|outperform|superior|strong|highly|significant)\b",
    re.I,
)


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


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(re.findall(r"[.!?]+(?:['\")\]]+)?(?=\s|$)", stripped))


def title_overlap_ratio(summary: str, title: str) -> float:
    toks = lambda s: {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z-]{2,}", s)}
    st = toks(summary)
    tt = toks(title)
    if not st:
        return 0.0
    return len(st & tt) / len(st)


def hard_filter_reasons(row: dict[str, Any]) -> list[str]:
    summary = norm(row.get("repair_summary", ""))
    low = summary.lower()
    reasons: list[str] = []
    wc = word_count(summary)
    if not summary:
        reasons.append("empty")
    if wc < 10:
        reasons.append("under_10_words")
    if wc > 15:
        reasons.append("over_15_words")
    if sentence_count(summary) != 1:
        reasons.append("not_single_sentence")
    if DIGIT_RE.search(summary):
        reasons.append("digit")
    if FORMULA_RE.search(summary):
        reasons.append("formula_or_symbol")
    if any(low.startswith(prefix) for prefix in BANNED_OPENINGS):
        reasons.append("banned_opening")
    if HIDDEN_QUANT_RE.search(summary):
        reasons.append("hidden_quantifier_or_superlative")
    if title_overlap_ratio(summary, row.get("title", "")) >= 0.75:
        reasons.append("high_title_overlap")
    if row.get("writer_local_filter_flags"):
        reasons.extend(f"writer_local_{flag}" for flag in row.get("writer_local_filter_flags", []))
    if row.get("verifier_decision") != "accept":
        reasons.append("verifier_not_accept")
    return reasons


def training_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": row["paper_id"],
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "summary": norm(row.get("repair_summary", "")),
        "primary_category": row.get("primary_category", ""),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date", ""),
        "teacher_model": row.get("writer_model", ""),
        "generator_prompt_version": row.get("writer_prompt_version", ""),
        "review_prompt_version": row.get("reviewer_prompt_version", ""),
        "source_split": "q5_burnin200_style_norm_targeted_v1",
        "normalization_category": row.get("normalization_category", ""),
        "bad_summary_checkpoint200": row.get("bad_summary", ""),
        "bad_summary_checkpoint700": row.get("pre_dpo_summary", ""),
        "failure_tags": row.get("failure_tags", []),
        "number_policy": row.get("number_policy", ""),
        "repair_candidate_id": row.get("candidate_id", ""),
    }


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("paper_id", "")),
            str(row.get("normalization_category", "")),
            norm(row.get("repair_summary", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def selection_order(by_category: dict[str, list[dict[str, Any]]]) -> list[str]:
    difficulty_bias = {
        "awkward_malformed_technical": 0,
        "value_attachment_or_polarity": 1,
        "result_preserving_no_digit_numeric": 2,
        "generic_title_drift": 3,
    }

    def available_unique(category: str) -> int:
        return len({str(row.get("paper_id", "")) for row in by_category.get(category, [])})

    return sorted(CATEGORIES, key=lambda category: (available_unique(category), difficulty_bias[category], category))


def select_rows(
    rows: list[dict[str, Any]],
    per_category: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hard_rejected: list[dict[str, Any]] = []
    for row in dedupe_rows(rows):
        reasons = hard_filter_reasons(row)
        if reasons:
            hard_rejected.append({**row, "style_norm_hard_filter_reasons": reasons})
            continue
        category = str(row.get("normalization_category", ""))
        if category in CATEGORIES:
            by_category[category].append(row)

    selected: list[dict[str, Any]] = []
    extras: list[dict[str, Any]] = []
    used_paper_ids: set[str] = set()
    order = selection_order(by_category)
    for category in order:
        ranked = sorted(
            by_category.get(category, []),
            key=lambda r: (-int(r.get("style_norm_candidate_score", 0)), str(r.get("candidate_id", ""))),
        )
        picked = []
        for row in ranked:
            pid = str(row.get("paper_id", ""))
            if pid in used_paper_ids:
                extras.append({**row, "style_norm_extra_reason": "duplicate_paper_id"})
                continue
            if len(picked) < per_category:
                picked.append(row)
                used_paper_ids.add(pid)
            else:
                extras.append({**row, "style_norm_extra_reason": "category_overflow"})
        if len(picked) != per_category:
            available_before_unique = len({str(row.get("paper_id", "")) for row in by_category.get(category, [])})
            raise SystemExit(
                f"Category {category} has {len(picked)} accepted hard-clean unique rows after earlier category picks; "
                f"expected {per_category}. It had {available_before_unique} unique hard-clean rows before cross-category dedupe."
            )
        selected.extend(picked)
    return selected, extras, hard_rejected, order


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Materialize exactly 160 q5 style-normalization target rows.")
    ap.add_argument("--output-root", type=Path, default=ROOT)
    ap.add_argument("--reviewer-outputs", type=Path, nargs="+", default=None)
    ap.add_argument("--per-category", type=int, default=40)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    reviewer_outputs = args.reviewer_outputs or [args.output_root / "reviewer_run/outputs.jsonl"]
    rows: list[dict[str, Any]] = []
    for path in reviewer_outputs:
        rows.extend(load_jsonl(path))
    selected, extras, hard_rejected, order = select_rows(rows, args.per_category)
    train = [training_row(row) for row in selected]
    out = args.output_root / "materialized"
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "targeted_training_rows_160.jsonl", train)
    write_jsonl(out / "selected_accepted_repairs_160.jsonl", selected)
    write_jsonl(out / "accepted_extras.jsonl", extras)
    write_jsonl(out / "hard_filter_rejected.jsonl", hard_rejected)
    manifest = {
        "created_by": "scripts/materialize_q5_burnin200_style_norm_targets_v1.py",
        "reviewer_outputs": [str(path) for path in reviewer_outputs],
        "selected_rows": len(train),
        "per_category": args.per_category,
        "selection_order": order,
        "category_counts": dict(Counter(row["normalization_category"] for row in train)),
        "hard_filter_rejected_rows": len(hard_rejected),
        "accepted_extra_rows": len(extras),
        "digit_summary_rows": sum(1 for row in train if DIGIT_RE.search(row.get("summary", ""))),
    }
    (out / "style_norm_target_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[style-norm-targets] selected {len(train)} rows")


if __name__ == "__main__":
    main()
