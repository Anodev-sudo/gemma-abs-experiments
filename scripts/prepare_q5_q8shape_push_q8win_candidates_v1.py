#!/usr/bin/env python3
"""Prepare q8-winning blind-review rows for value-aware Kimi/Qwen repair."""
from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any


REVIEW_ROWS = Path(
    "hf_bucket/experiments/sft/q5_q8shape150_residual_push_v1/lr1e7/eval/"
    "blind_ab_q8_mlx_python_vs_q8shape_push150_500/blind_review_merged_unblinded.jsonl"
)
OUT_DIR = Path("hf_bucket/experiments/sft/q5_q8shape150_q8win_repair_v1/source_candidates")


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


def main() -> None:
    rows = load_jsonl(REVIEW_ROWS)
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if row.get("preferred_model") != "q8_mlx_python":
            continue
        if row.get("more_accurate_model") != "q8_mlx_python":
            continue
        q8_summary = summary_for(row, "q8_mlx_python")
        q5_summary = summary_for(row, "q5_q8shape_push150")
        if not q8_summary or not q5_summary:
            raise ValueError(f"Missing summaries for {row.get('review_id')}")
        split = str(row.get("split", ""))
        note = str(row.get("review_notes", ""))
        candidates.append(
            {
                "candidate_id": f"q8win_repair:{len(candidates) + 1:04d}",
                "paper_id": str(row.get("paper_id", "")),
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
                "primary_category": row.get("primary_category", ""),
                "categories": row.get("categories", []),
                "created_date": row.get("created_date", ""),
                "q8_summary": q8_summary,
                "q5_summary": q5_summary,
                "q5_issue_tags": issue_tags(split, note),
                "q5_severity_source": "blind_review:q8_preferred_and_more_accurate_vs_q8shape_push150",
                "reviewer_note": note,
                "blind_review_id": row.get("review_id", ""),
                "blind_review_split": split,
                "summary_a_model": row.get("summary_a_model"),
                "summary_b_model": row.get("summary_b_model"),
                "preferred_model": row.get("preferred_model"),
                "more_accurate_model": row.get("more_accurate_model"),
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "q8win_repair_candidates.jsonl"
    write_jsonl(out_path, candidates)
    manifest = {
        "created_by": "scripts/prepare_q5_q8shape_push_q8win_candidates_v1.py",
        "source_review_rows": str(REVIEW_ROWS),
        "output": str(out_path),
        "candidate_rows": len(candidates),
        "split_counts": dict(collections.Counter(row["blind_review_split"] for row in candidates)),
        "issue_tag_counts": dict(
            collections.Counter(tag for row in candidates for tag in row["q5_issue_tags"])
        ),
        "selection": "preferred_model == q8_mlx_python and more_accurate_model == q8_mlx_python",
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[q8win-candidates] wrote {len(candidates)} rows to {out_path}")


if __name__ == "__main__":
    main()
