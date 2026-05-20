#!/usr/bin/env python3
"""Prepare broad-base cleanup candidates from a blind basin diagnostic."""
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


def norm(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def summary_for(row: dict[str, Any], model: str) -> str:
    if row.get("summary_a_model") == model:
        return norm(row.get("summary_a", ""))
    if row.get("summary_b_model") == model:
        return norm(row.get("summary_b", ""))
    return ""


def output_by_pid(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rows = load_jsonl(path)
    outputs: dict[str, str] = {}
    for row in rows:
        paper_id = str(row.get("paper_id", "")).strip()
        if not paper_id:
            continue
        outputs[paper_id] = norm(row.get("normalized_output") or row.get("raw_output") or row.get("summary"))
    return outputs


def issue_tags(split: str, note: str, current_summary: str, old_summary: str) -> list[str]:
    text = f"{split} {note} {current_summary} {old_summary}".casefold()
    tags: list[str] = []
    if any(ch.isdigit() for ch in current_summary) or "numeric" in text or "value" in text:
        tags.append("wrong_number_binding")
    if "polarity" in text or "increase" in text or "decrease" in text or "decline" in text:
        tags.append("polarity_flip")
    if "entity" in text or "off-target" in text or "wrong" in text:
        tags.append("unsupported_entity")
    if "garbled" in text or "mangled" in text or "notation" in text:
        tags.append("malformed_repetition")
    if "generic" in text or "miss" in text or "closer" in text:
        tags.append("misses_point")
    if "faithful" in text or "unsupported" in text:
        tags.append("unsupported_causal_claim")
    if not tags:
        tags.append("too_generic")
    return sorted(set(tags))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-jsonl", type=Path, required=True)
    parser.add_argument("--q8-output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--current-label", required=True)
    parser.add_argument("--old-base-label", required=True)
    parser.add_argument("--candidate-prefix", required=True)
    parser.add_argument(
        "--repair-model",
        action="append",
        default=[],
        help="Model choice that should be included for cleanup, for example old_base_step2400 or neither.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.review_jsonl)
    q8_outputs = output_by_pid(args.q8_output)
    repair_models = set(args.repair_model or [args.old_base_label, "neither"])

    candidates: list[dict[str, Any]] = []
    skipped_missing = 0
    for row in rows:
        preferred = str(row.get("preferred_model", ""))
        accurate = str(row.get("more_accurate_model", ""))
        if preferred not in repair_models or accurate not in repair_models:
            continue
        paper_id = str(row.get("paper_id", "")).strip()
        current_summary = summary_for(row, args.current_label)
        old_summary = summary_for(row, args.old_base_label)
        if not paper_id or not current_summary:
            skipped_missing += 1
            continue
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
                "current_broad_summary": current_summary,
                "q5_summary": current_summary,
                "old_base_summary": old_summary,
                "q8_summary": q8_outputs.get(paper_id, ""),
                "q5_issue_tags": issue_tags(split, note, current_summary, old_summary),
                "q5_severity_source": (
                    f"blind_review:{preferred}_preferred_and_{accurate}_more_accurate_vs_{args.current_label}"
                ),
                "reviewer_note": note,
                "blind_review_id": row.get("review_id", ""),
                "blind_review_split": split,
                "preferred_model": preferred,
                "more_accurate_model": accurate,
                "summary_a_model": row.get("summary_a_model"),
                "summary_b_model": row.get("summary_b_model"),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "broad_basin_cleanup_candidates.jsonl"
    write_jsonl(out_path, candidates)
    manifest = {
        "created_by": "scripts/prepare_q5_broad_basin_cleanup_candidates_v1.py",
        "source_review_rows": str(args.review_jsonl),
        "q8_output": str(args.q8_output) if args.q8_output else "",
        "output": str(out_path),
        "candidate_rows": len(candidates),
        "skipped_missing": skipped_missing,
        "current_label": args.current_label,
        "old_base_label": args.old_base_label,
        "candidate_prefix": args.candidate_prefix,
        "repair_models": sorted(repair_models),
        "selection": "preferred_model and more_accurate_model are both in repair_models",
        "preferred_model_counts": dict(collections.Counter(row["preferred_model"] for row in candidates)),
        "more_accurate_model_counts": dict(collections.Counter(row["more_accurate_model"] for row in candidates)),
        "split_counts": dict(collections.Counter(row["blind_review_split"] for row in candidates)),
        "issue_tag_counts": dict(collections.Counter(tag for row in candidates for tag in row["q5_issue_tags"])),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[broad-cleanup-candidates] wrote {len(candidates)} rows to {out_path}")


if __name__ == "__main__":
    main()
