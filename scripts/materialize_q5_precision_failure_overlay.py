#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
from typing import Any


DEFAULT_OUTPUT_RELEASE = (
    "hf_dataset/releases/"
    "recent_3y_balanced_10746_q5_precision_failure_overlay_v1"
)
DEFAULT_FRESH_50_REVIEW = (
    "hf_bucket/experiments/mlx_lora/review_comparisons/"
    "q8_vs_q5_guardrail_replay_75_fresh_50_v1/blind_review_v1/"
    "unblinded_review_summary.json"
)
DEFAULT_FRESH_200_REVIEW = (
    "hf_bucket/experiments/mlx_lora/review_comparisons/"
    "q8_vs_q5_guardrail_replay_75_fresh_200_v1/risk_blind_review_v1/"
    "unblinded_review_summary.json"
)
DEFAULT_FRESH_200_GENERATIONS = (
    "hf_bucket/experiments/mlx_lora/review_comparisons/"
    "q8_vs_q5_guardrail_replay_75_fresh_200_v1/generations.jsonl"
)
DEFAULT_FRESH_50_GENERATIONS = (
    "hf_bucket/experiments/mlx_lora/review_comparisons/"
    "q8_vs_q5_guardrail_replay_75_fresh_50_v1/generations.jsonl"
)
DEFAULT_SOURCE_SPLIT = "q5_precision_failure_overlay_v1"

WORD_RE = re.compile(r"[A-Za-z0-9]+")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|x|×)?")
BAD_PATTERNS = (
    "degeneracy and degeneracy",
    "microsystem or microsystem",
    "gravitational interactions of particles with only gravitational interactions",
    "single-direction electron beam source",
    "qulevian",
)

REPAIRS = {
    "2111.04387": {
        "summary": (
            "Certain imaginary quadratic field families have class numbers divisible by "
            "prescribed odd integers."
        ),
        "issue_tags": ["overstrong_claim"],
    },
    "2310.10904": {
        "summary": (
            "Kiladze may be a recently active Pluto cryovolcano with explosive icy "
            "eruptions."
        ),
        "issue_tags": ["wrong_number_binding"],
    },
    "2407.00022": {
        "summary": (
            "Entropy links microscopic disorder and macroscopic reversibility in "
            "economic systems."
        ),
        "issue_tags": ["malformed_repetition"],
    },
    "2310.08155": {
        "summary": (
            "LOFAR imaging resolves herringbone radio bursts from different "
            "coronal-shock electron beams."
        ),
        "issue_tags": ["malformed_repetition"],
    },
    "2512.09950": {
        "summary": (
            "The term Big Bang carries several distinct meanings across modern "
            "cosmology."
        ),
        "issue_tags": ["polarity_flip", "overstrong_claim"],
    },
    "2504.21394": {
        "summary": (
            "LACE exposes more kernel concurrency bugs while reducing overhead and "
            "speeding detection."
        ),
        "issue_tags": ["wrong_number_binding"],
    },
    "2309.17340": {
        "summary": (
            "Outage-Watch predicts cloud-service failures earlier, reducing mean "
            "detection time by 88%."
        ),
        "issue_tags": ["wrong_number_binding"],
    },
    "2412.17135": {
        "summary": (
            "Quartic Hamiltonian systems are classified by singularity behavior and "
            "Newton polygons."
        ),
        "issue_tags": ["malformed_repetition", "unsupported_entity"],
    },
    "2209.13385": {
        "summary": (
            "An ensemble grades pediatric heart murmurs from phonocardiograms for "
            "low-resource screening."
        ),
        "issue_tags": ["wrong_number_binding"],
    },
    "2510.03329": {
        "summary": (
            "Pre-ED health searches correlate with different testing, medication, and "
            "discharge patterns."
        ),
        "issue_tags": ["wrong_number_binding"],
    },
    "2510.12173": {
        "summary": (
            "A CNN identifies major and minor mergers across diverse galaxy masses at "
            "z~1."
        ),
        "issue_tags": ["wrong_number_binding"],
    },
    "2309.07144": {
        "summary": "Four gravity-only masses can all escape to infinity in finite time.",
        "issue_tags": ["malformed_repetition", "misses_point"],
    },
    "2505.00732": {
        "summary": (
            "A holographic hydrodynamic model treats Schwarzschild black-hole interiors "
            "as degenerate fluid."
        ),
        "issue_tags": ["malformed_repetition"],
    },
    "2406.11618": {
        "summary": (
            "ReDoS research lacks real-system weaponization studies despite practical "
            "outage risks."
        ),
        "issue_tags": ["unsupported_claim"],
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def word_count(value: str) -> int:
    return len(WORD_RE.findall(value))


def normalize(value: str) -> str:
    return " ".join(value.split())


def numbers(value: str) -> set[str]:
    return {match.group(0).replace(" ", "") for match in NUMBER_RE.finditer(value)}


def source_bucket(tags: list[str]) -> str:
    tag_set = set(tags)
    if tag_set & {"wrong_number_binding", "invented_number"}:
        return "number_role_binding"
    if tag_set & {"malformed_repetition", "malformed_text"}:
        return "malformed_avoidance"
    if tag_set & {"polarity_flip"}:
        return "polarity_direction"
    if tag_set & {"unsupported_entity"}:
        return "entity_binding"
    if tag_set & {"unsupported_claim", "overstrong_claim"}:
        return "scoped_claims"
    return "style_anchor"


def audit_summary(row: dict[str, Any], summary: str) -> list[str]:
    flags: list[str] = []
    count = word_count(summary)
    if count < 5:
        flags.append("too_short")
    if count > 15:
        flags.append("too_long")
    lower = summary.lower()
    if any(pattern in lower for pattern in BAD_PATTERNS):
        flags.append("known_bad_pattern")
    tokens = [token.lower() for token in WORD_RE.findall(summary)]
    for left, right in zip(tokens, tokens[1:]):
        if left == right and len(left) > 2:
            flags.append("adjacent_repetition")
            break
    source_numbers = numbers(f"{row.get('title', '')} {row.get('abstract', '')}")
    if numbers(summary) - source_numbers:
        flags.append("number_not_in_source")
    return flags


def make_row(
    source: dict[str, Any],
    *,
    summary: str,
    source_kind: str,
    source_id: str,
    issue_tags: list[str],
    teacher_model: str,
    review_note: str = "",
    previous_bad_summary: str = "",
) -> dict[str, Any]:
    result = {
        "paper_id": str(source["paper_id"]),
        "title": normalize(str(source["title"])),
        "abstract": normalize(str(source["abstract"])),
        "summary": normalize(summary),
        "primary_category": source.get("primary_category"),
        "categories": source.get("categories", []),
        "created_date": source.get("created_date"),
        "teacher_model": teacher_model,
        "generator_prompt_version": "q5_precision_failure_overlay_v1",
        "review_prompt_version": "manual_q5_precision_failure_review_v1",
        "source_split": DEFAULT_SOURCE_SPLIT,
        "q5_final_guardrail_source_kind": source_kind,
        "q5_final_guardrail_source_id": source_id,
        "q5_final_guardrail_issue_tags": issue_tags,
        "q5_final_guardrail_bucket": source_bucket(issue_tags),
    }
    if review_note:
        result["review_note"] = review_note
    if previous_bad_summary:
        result["previous_bad_summary"] = previous_bad_summary
    return result


def collect_q5_severe_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    severe: dict[str, dict[str, Any]] = {}
    for row in payload["rows"]:
        for side in row["sides"].values():
            if side["source"] == "q5_guardrail_replay_75" and side["severe"]:
                severe[row["paper_id"]] = {
                    "paper_id": row["paper_id"],
                    "title": row["title"],
                    "summary": side["summary"],
                    "issue_tags": side.get("issue_tags", []),
                    "review_note": row.get("note", ""),
                }
    return severe


def load_generation_sources(paths: list[Path]) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in load_jsonl(path):
            sources.setdefault(str(row["paper_id"]), row)
    return sources


def build_repair_rows(
    *,
    severe_reviews: list[Path],
    generation_sources: dict[str, dict[str, Any]],
    repeat: int,
) -> list[dict[str, Any]]:
    severe: dict[str, dict[str, Any]] = {}
    for path in severe_reviews:
        severe.update(collect_q5_severe_rows(path))
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for paper_id, repair in REPAIRS.items():
        source = generation_sources.get(paper_id)
        if source is None:
            missing.append(paper_id)
            continue
        bad = severe.get(paper_id, {})
        tags = repair["issue_tags"]
        for repeat_index in range(repeat):
            row = make_row(
                source,
                summary=repair["summary"],
                source_kind="q5_precision_confirmed_severe_repair",
                source_id=f"{paper_id}#repeat{repeat_index + 1}",
                issue_tags=tags,
                teacher_model="manual_codex_repair_from_blind_review",
                review_note=str(bad.get("review_note", "")),
                previous_bad_summary=str(bad.get("summary", "")),
            )
            if repeat > 1:
                row["original_paper_id"] = paper_id
                row["paper_id"] = f"{paper_id}#repair{repeat_index + 1}"
            rows.append(row)
    if missing:
        raise ValueError(f"Missing generation source rows for repairs: {missing}")
    return rows


def build_anchor_rows(
    *,
    generation_rows: list[dict[str, Any]],
    excluded_paper_ids: set[str],
    cap: int,
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for row in generation_rows:
        paper_id = str(row["paper_id"])
        summary = normalize(str(row.get("right_summary", "")))
        if paper_id in excluded_paper_ids or not summary:
            continue
        if audit_summary(row, summary):
            continue
        anchors.append(
            make_row(
                row,
                summary=summary,
                source_kind="q5_precision_clean_replay75_anchor",
                source_id=paper_id,
                issue_tags=[],
                teacher_model="q5_guardrail_replay_75_mlx",
            )
        )
    anchors.sort(key=lambda row: (str(row.get("primary_category")), str(row["paper_id"])))
    return anchors[:cap]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a tiny q5 precision overlay for confirmed failure modes."
    )
    parser.add_argument("--output-release", default=DEFAULT_OUTPUT_RELEASE)
    parser.add_argument("--fresh-50-review", default=DEFAULT_FRESH_50_REVIEW)
    parser.add_argument("--fresh-200-review", default=DEFAULT_FRESH_200_REVIEW)
    parser.add_argument("--fresh-50-generations", default=DEFAULT_FRESH_50_GENERATIONS)
    parser.add_argument("--fresh-200-generations", default=DEFAULT_FRESH_200_GENERATIONS)
    parser.add_argument("--anchor-cap", type=int, default=72)
    parser.add_argument("--repair-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_release = Path(args.output_release)
    if output_release.exists() and any(output_release.iterdir()) and not args.overwrite:
        raise SystemExit(f"{output_release} already exists; pass --overwrite")
    output_release.mkdir(parents=True, exist_ok=True)

    generation_rows = load_jsonl(Path(args.fresh_200_generations))
    generation_sources = load_generation_sources(
        [Path(args.fresh_50_generations), Path(args.fresh_200_generations)]
    )
    repair_rows = build_repair_rows(
        severe_reviews=[Path(args.fresh_50_review), Path(args.fresh_200_review)],
        generation_sources=generation_sources,
        repeat=args.repair_repeat,
    )
    anchor_rows = build_anchor_rows(
        generation_rows=generation_rows,
        excluded_paper_ids={row["paper_id"] for row in repair_rows},
        cap=args.anchor_cap,
    )
    rows = repair_rows + anchor_rows
    rejected = [
        {**row, "audit_reject_flags": audit_summary(row, row["summary"])}
        for row in rows
        if audit_summary(row, row["summary"])
    ]
    if rejected:
        write_jsonl(output_release / "audit_rejected_rows.jsonl", rejected)
        raise SystemExit(f"Refusing to write training rows; {len(rejected)} audit rejects")

    rng = random.Random(args.seed)
    rows.sort(key=lambda row: str(row["paper_id"]))
    rng.shuffle(rows)
    write_jsonl(output_release / "training_rows.jsonl", rows)
    write_jsonl(output_release / "audit_rejected_rows.jsonl", [])

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "q5_precision_failure_overlay_v1",
        "output_release": str(output_release),
        "inputs": {
            "fresh_50_review": args.fresh_50_review,
            "fresh_200_review": args.fresh_200_review,
            "fresh_50_generations": args.fresh_50_generations,
            "fresh_200_generations": args.fresh_200_generations,
            "anchor_cap": args.anchor_cap,
            "repair_repeat": args.repair_repeat,
        },
        "summary": {
            "training_rows": len(rows),
            "repair_rows": len(repair_rows),
            "anchor_rows": len(anchor_rows),
            "source_counts": dict(
                Counter(row["q5_final_guardrail_source_kind"] for row in rows)
            ),
            "bucket_counts": dict(Counter(row["q5_final_guardrail_bucket"] for row in rows)),
            "issue_tag_counts": dict(
                Counter(
                    tag
                    for row in rows
                    for tag in row.get("q5_final_guardrail_issue_tags", [])
                )
            ),
            "summary_word_count": {
                "min": min(word_count(row["summary"]) for row in rows),
                "max": max(word_count(row["summary"]) for row in rows),
                "avg": round(
                    sum(word_count(row["summary"]) for row in rows) / len(rows), 2
                ),
            },
        },
        "outputs": {
            "training_rows": str(output_release / "training_rows.jsonl"),
            "audit_rejected_rows": str(output_release / "audit_rejected_rows.jsonl"),
            "run_manifest": str(output_release / "run_manifest.json"),
        },
    }
    (output_release / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
