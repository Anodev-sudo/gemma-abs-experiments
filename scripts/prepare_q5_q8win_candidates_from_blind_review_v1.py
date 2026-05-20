#!/usr/bin/env python3
"""Prepare q8-winning blind-review rows for Kimi/Qwen q5 repair."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def issue_tags(split: str, note: str) -> list[str]:
    text = f"{split} {note}".casefold()
    tags: list[str] = []
    if "numeric" in text or "value" in text or any(ch.isdigit() for ch in note):
        tags.append("wrong_number_binding")
    if "polarity" in text or "decline" in text or "increase" in text or "decrease" in text:
        tags.append("polarity_flip")
    if "entity" in text or "muddled" in text or "off-target" in text:
        tags.append("unsupported_entity")
    if "notation" in text or "garbled" in text or "malformed" in text:
        tags.append("malformed_repetition")
    if "abstract" in text or "generic" in text or "loses" in text:
        tags.append("misses_point")
    if not tags:
        tags.append("too_generic")
    return sorted(set(tags))


def summary_for(row: dict[str, Any], model: str) -> str:
    if row.get("summary_a_model") == model:
        return str(row.get("summary_a", "")).strip()
    if row.get("summary_b_model") == model:
        return str(row.get("summary_b", "")).strip()
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--q8-label", default="q8_mlx_python")
    parser.add_argument("--q5-label", required=True)
    parser.add_argument("--candidate-prefix", required=True)
    parser.add_argument(
        "--exclude-training-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Optional training-row JSONL whose paper_id values should be skipped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.review_jsonl)
    excluded_paper_ids: set[str] = set()
    for path in args.exclude_training_jsonl:
        for row in load_jsonl(path):
            paper_id = str(row.get("paper_id", "")).strip()
            if paper_id:
                excluded_paper_ids.add(paper_id)

    candidates: list[dict[str, Any]] = []
    skipped_labeled = 0
    for row in rows:
        if row.get("preferred_model") != args.q8_label:
            continue
        if row.get("more_accurate_model") != args.q8_label:
            continue
        paper_id = str(row.get("paper_id", "")).strip()
        if paper_id in excluded_paper_ids:
            skipped_labeled += 1
            continue
        q8_summary = summary_for(row, args.q8_label)
        q5_summary = summary_for(row, args.q5_label)
        if not q8_summary or not q5_summary:
            raise ValueError(f"Missing summaries for {row.get('review_id')}")
        split = str(row.get("split", ""))
        note = str(row.get("review_notes", ""))
        candidates.append(
            {
                "candidate_id": f"{args.candidate_prefix}:{len(candidates) + 1:04d}",
                "paper_id": paper_id,
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
                "primary_category": row.get("primary_category", ""),
                "categories": row.get("categories", []),
                "created_date": row.get("created_date", ""),
                "q8_summary": q8_summary,
                "q5_summary": q5_summary,
                "q5_issue_tags": issue_tags(split, note),
                "q5_severity_source": f"blind_review:{args.q8_label}_preferred_and_more_accurate_vs_{args.q5_label}",
                "reviewer_note": note,
                "blind_review_id": row.get("review_id", ""),
                "blind_review_split": split,
                "summary_a_model": row.get("summary_a_model"),
                "summary_b_model": row.get("summary_b_model"),
                "preferred_model": row.get("preferred_model"),
                "more_accurate_model": row.get("more_accurate_model"),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "q8win_repair_candidates.jsonl"
    write_jsonl(out_path, candidates)
    manifest = {
        "created_by": "scripts/prepare_q5_q8win_candidates_from_blind_review_v1.py",
        "source_review_rows": str(args.review_jsonl),
        "output": str(out_path),
        "candidate_rows": len(candidates),
        "excluded_training_jsonl": [str(path) for path in args.exclude_training_jsonl],
        "excluded_paper_ids": len(excluded_paper_ids),
        "skipped_already_labeled": skipped_labeled,
        "q8_label": args.q8_label,
        "q5_label": args.q5_label,
        "candidate_prefix": args.candidate_prefix,
        "split_counts": dict(collections.Counter(row["blind_review_split"] for row in candidates)),
        "issue_tag_counts": dict(collections.Counter(tag for row in candidates for tag in row["q5_issue_tags"])),
        "selection": "preferred_model == q8_label and more_accurate_model == q8_label",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[q8win-candidates] wrote {len(candidates)} rows to {out_path}")


if __name__ == "__main__":
    main()
