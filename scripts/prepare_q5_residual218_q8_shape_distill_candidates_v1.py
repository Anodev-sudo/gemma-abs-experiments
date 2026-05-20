#!/usr/bin/env python3
"""Mine q8-shape distillation candidates from q8-vs-q5 checkpoint-218 review."""
from __future__ import annotations

import argparse
import collections
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_FINAL500 = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7/"
    "eval/blind_ab_q8_mlx_python_vs_residual218_500/"
    "blind_review_merged_unblinded.jsonl"
)
DEFAULT_OUT_ROOT = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_residual218_q8_shape_distill_v1/"
    "candidate_pool_v1"
)

Q5_MODEL = "candidate_residual_checkpoint_218"
Q8_MODEL = "q8_mlx_python"
DIGIT_RE = re.compile(r"\d")
FOG_RE = re.compile(
    r"\b(?:enables?|supports?|characteri[sz]es?|maps?|mapping|frameworks?|"
    r"structures?|patterns?|constraints?|resource allocation|optimization|"
    r"validation|domain-specific|through|modular|pipeline|deployment|"
    r"modeling|classification|parameter|inference|dynamics?|mechanisms?)\b",
    re.IGNORECASE,
)
MALFORMED_RE = re.compile(
    r"\b(?:Bayron|Curravariance|Ather|Sterochastic|Parametriding|Hermetic|"
    r"Coordined|pierded|weakary|Arenormalizable|Schr[oe]der|"
    r"Thermoregulatory|Noyehoteles|chimerical|refutationicity|smallgain|"
    r"indoped|ansatzs|metastable metastable|supportive supportive|tired|"
    r"inheritance maps|hardware-agnostic)\b",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def summaries_by_model(row: dict[str, Any]) -> tuple[str, str]:
    q5_summary = ""
    q8_summary = ""
    for side in ("a", "b"):
        summary = norm(row.get(f"summary_{side}", ""))
        model = str(row.get(f"summary_{side}_model", ""))
        if model == Q5_MODEL:
            q5_summary = summary
        elif model == Q8_MODEL:
            q8_summary = summary
    if not q5_summary or not q8_summary:
        raise ValueError(f"Missing q5/q8 summary for {row.get('paper_id')}")
    return q5_summary, q8_summary


def q8_wins(row: dict[str, Any]) -> bool:
    return row.get("preferred_model") == Q8_MODEL or row.get("more_accurate_model") == Q8_MODEL


def q8_clean_win(row: dict[str, Any]) -> bool:
    return row.get("preferred_model") == Q8_MODEL and row.get("more_accurate_model") == Q8_MODEL


def failure_tags(q5_summary: str, q8_summary: str, split: str) -> list[str]:
    tags: list[str] = []
    if MALFORMED_RE.search(q5_summary):
        tags.append("malformed_or_invented_terms")
    if FOG_RE.search(q5_summary):
        tags.append("semantic_fog_abstraction")
    if DIGIT_RE.search(q8_summary):
        tags.append("q8_metric_to_qualitative_hook")
    if "entity" in split:
        tags.append("object_or_entity_binding")
    if "polarity" in split:
        tags.append("value_or_polarity_binding")
    if "long_notation" in split:
        tags.append("notation_to_plain_language")
    if not tags:
        tags.append("lost_concrete_result")
    return list(dict.fromkeys(tags))


def primary_category(tags: list[str]) -> str:
    priority = [
        "malformed_or_invented_terms",
        "semantic_fog_abstraction",
        "object_or_entity_binding",
        "value_or_polarity_binding",
        "q8_metric_to_qualitative_hook",
        "notation_to_plain_language",
        "lost_concrete_result",
    ]
    for tag in priority:
        if tag in tags:
            return tag
    return tags[0]


def score_candidate(row: dict[str, Any], q5_summary: str, q8_summary: str, tags: list[str]) -> int:
    score = 0
    if q8_clean_win(row):
        score += 100
    elif row.get("more_accurate_model") == Q8_MODEL:
        score += 70
    elif row.get("preferred_model") == Q8_MODEL:
        score += 50
    if "malformed_or_invented_terms" in tags:
        score += 80
    if "semantic_fog_abstraction" in tags:
        score += 45
    if "object_or_entity_binding" in tags:
        score += 25
    if "value_or_polarity_binding" in tags:
        score += 25
    if "q8_metric_to_qualitative_hook" in tags:
        score += 20
    if not DIGIT_RE.search(q8_summary):
        score += 15
    if DIGIT_RE.search(q5_summary):
        score += 10
    return score


def build_candidate(index: int, row: dict[str, Any]) -> dict[str, Any]:
    q5_summary, q8_summary = summaries_by_model(row)
    split = str(row.get("split", ""))
    tags = failure_tags(q5_summary, q8_summary, split)
    category = primary_category(tags)
    score = score_candidate(row, q5_summary, q8_summary, tags)
    note_bits = [
        f"blind_review_note: {norm(row.get('review_notes', ''))}",
        f"q5_failure_tags: {', '.join(tags)}",
        f"q8_preferred: {row.get('preferred_model') == Q8_MODEL}",
        f"q8_more_accurate: {row.get('more_accurate_model') == Q8_MODEL}",
        f"q8_has_digits: {bool(DIGIT_RE.search(q8_summary))}",
        "rewrite goal: keep q8's concrete semantic hook, remove fragile numbers, avoid q5 semantic fog",
    ]
    return {
        "paper_id": str(row["paper_id"]),
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "primary_category": row.get("primary_category", ""),
        "q8_summary": q8_summary,
        "checkpoint_218_summary": q5_summary,
        "category": category,
        "category_note": " | ".join(note_bits),
        "reviewer_note": row.get("review_notes", ""),
        "more_accurate_model": row.get("more_accurate_model", ""),
        "preferred_model": row.get("preferred_model", ""),
        "source_split": split,
        "source_review_id": row.get("review_id", ""),
        "q5_failure_tags": tags,
        "candidate_score": score,
        "candidate_rank": index,
        "rewrite_source": "q8_vs_residual218_final500",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare q8-shape distillation candidates.")
    parser.add_argument("--final500", type=Path, default=DEFAULT_FINAL500)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--only-clean-q8-wins",
        action="store_true",
        help="Require q8 to win both preference and accuracy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = load_jsonl(args.final500)
    candidates = []
    for row in source_rows:
        if not q8_wins(row):
            continue
        if args.only_clean_q8_wins and not q8_clean_win(row):
            continue
        candidates.append(build_candidate(0, row))
    candidates.sort(
        key=lambda row: (-int(row["candidate_score"]), str(row["source_split"]), str(row["paper_id"]))
    )
    if args.limit is not None:
        candidates = candidates[: args.limit]
    for index, row in enumerate(candidates, start=1):
        row["candidate_rank"] = index

    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "q8_shape_distill_candidates.jsonl", candidates)
    write_json(
        out / "manifest.json",
        {
            "created_at": utc_now(),
            "created_by": "scripts/prepare_q5_residual218_q8_shape_distill_candidates_v1.py",
            "source_final500": str(args.final500),
            "candidate_rows": len(candidates),
            "only_clean_q8_wins": bool(args.only_clean_q8_wins),
            "limit": args.limit,
            "category_counts": dict(collections.Counter(row["category"] for row in candidates)),
            "tag_counts": dict(collections.Counter(tag for row in candidates for tag in row["q5_failure_tags"])),
            "source_split_counts": dict(collections.Counter(row["source_split"] for row in candidates)),
        },
    )
    lines = [
        "# q8-shape distillation candidate pool",
        "",
        f"- Source rows: {len(source_rows)}",
        f"- Candidate rows: {len(candidates)}",
        "",
        "## By Category",
        "",
        "| category | rows |",
        "| --- | ---: |",
    ]
    for category, count in sorted(collections.Counter(row["category"] for row in candidates).items()):
        lines.append(f"| {category} | {count} |")
    lines.extend(["", "## By Source Split", "", "| split | rows |", "| --- | ---: |"])
    for split, count in sorted(collections.Counter(row["source_split"] for row in candidates).items()):
        lines.append(f"| {split} | {count} |")
    (out / "CANDIDATE_POOL_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[q8-shape-candidates] wrote {len(candidates)} rows to {out / 'q8_shape_distill_candidates.jsonl'}")


if __name__ == "__main__":
    main()
