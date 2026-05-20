#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import random
import re
from typing import Any


DEFAULT_OUTPUT_RELEASE = (
    "hf_dataset/releases/recent_3y_balanced_10746_q5_final_guardrail_overlay_v1"
)
DEFAULT_SOURCE_SPLIT = "q5_final_guardrail_overlay_v1"
DEFAULT_KIMI_QWEN_ROWS = (
    "hf_bucket/experiments/mlx_lora/q5_kimi_qwen_correction_overlay_v1/"
    "rejection_repair_v1/materialized_release/combined_selected_review_rows.jsonl"
)
DEFAULT_REORIENT_ROOT = (
    "hf_bucket/experiments/mlx_lora/review_comparisons/"
    "q8_vs_q5_len15_step125_reorient_100_v1"
)
DEFAULT_LEN15_TARGETED_ROOT = (
    "hf_bucket/experiments/mlx_lora/review_comparisons/"
    "q5_numeric_len15_v2_eval_20260507/targeted_blind_review_v1"
)
DEFAULT_Q5_REVIEWED_ROWS = (
    "hf_bucket/experiments/mlx_lora/q5_hardening_distill_v1/"
    "identified_rows/all_reviewed_unblinded.jsonl"
)
DEFAULT_REPLAY_JSONL = (
    "hf_bucket/experiments/mlx_lora/review_comparisons/"
    "q5_numeric_len15_v2_eval_20260507/pre_q5_vs_len15_step125_general_200/"
    "review_rows_blinded.jsonl"
)

WORD_RE = re.compile(r"[A-Za-z0-9]+")
NUMBER_RE = re.compile(
    r"(?<![A-Za-z])(?:[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[xX%])?|"
    r"\d+(?:\.\d+)?\s*(?:e[+-]?\d+|×\s*10\^?[+-]?\d+))(?![A-Za-z])"
)
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "using",
    "with",
}
MALFORMED_PATTERNS = (
    "approximate approximations",
    "zero-sings",
    "2-1 color",
    "fcolon",
    "meg a",
    "schr-schr",
    "schrorsche",
    "pals are",
    "magenetic",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_space(value: str) -> str:
    return " ".join(str(value).split())


def words(value: str) -> list[str]:
    return WORD_RE.findall(str(value).lower())


def word_count(value: str) -> int:
    return len(words(value))


def numbers(value: str) -> set[str]:
    return {match.group(0).replace(" ", "") for match in NUMBER_RE.finditer(str(value))}


def title_overlap(title: str, summary: str) -> float:
    title_words = [word for word in words(title) if word not in STOP_WORDS]
    summary_words = [word for word in words(summary) if word not in STOP_WORDS]
    if not title_words or not summary_words:
        return 0.0
    title_set = set(title_words)
    summary_set = set(summary_words)
    return len(title_set & summary_set) / max(1, len(summary_set))


def audit_summary(row: dict[str, Any], summary: str, *, max_words: int) -> list[str]:
    flags: list[str] = []
    normalized = normalize_space(summary)
    count = word_count(normalized)
    if not normalized:
        flags.append("empty_summary")
    if count < 5:
        flags.append("too_short")
    if count > max_words:
        flags.append("too_long")
    if any(pattern in normalized.lower() for pattern in MALFORMED_PATTERNS):
        flags.append("known_malformed_pattern")
    summary_words = words(normalized)
    for left, right in zip(summary_words, summary_words[1:]):
        if left == right and left not in {"ii", "iii"}:
            flags.append("adjacent_repetition")
            break
    if len(set(summary_words)) <= max(2, len(summary_words) // 3):
        flags.append("low_word_diversity")
    source_text = f"{row.get('title', '')} {row.get('abstract', '')}"
    extra_numbers = numbers(normalized) - numbers(source_text)
    if extra_numbers:
        flags.append("target_number_not_in_source")
    if title_overlap(str(row.get("title", "")), normalized) > 0.72:
        flags.append("high_title_overlap")
    return flags


def split_tags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(tag) for tag in value if str(tag)]
    return [tag for tag in str(value or "").split(",") if tag]


def source_bucket_from_tags(tags: list[str]) -> str:
    tag_set = set(tags)
    if tag_set & {"wrong_number_binding", "invented_number", "numeric_error", "unsupported_number"}:
        return "number_role_binding"
    if tag_set & {"polarity_flip", "polarity_risk"}:
        return "polarity_direction"
    if tag_set & {"overstrong_claim", "unsupported_mechanism", "unsupported_causal_claim"}:
        return "scoped_claims"
    if tag_set & {"malformed_repetition", "non_answer"}:
        return "malformed_avoidance"
    if tag_set & {"unsupported_entity", "unsupported_entity_label"}:
        return "entity_binding"
    return "style_anchor"


def make_training_row(
    *,
    source: dict[str, Any],
    summary: str,
    source_kind: str,
    source_id: str,
    issue_tags: list[str],
    teacher_model: str,
    review_prompt_version: str,
    source_split: str = DEFAULT_SOURCE_SPLIT,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "paper_id": str(source["paper_id"]),
        "title": normalize_space(str(source["title"])),
        "abstract": normalize_space(str(source["abstract"])),
        "summary": normalize_space(summary),
        "primary_category": source.get("primary_category"),
        "categories": source.get("categories", []),
        "created_date": source.get("created_date"),
        "teacher_model": teacher_model,
        "generator_prompt_version": "q5_final_guardrail_harvest_v1",
        "review_prompt_version": review_prompt_version,
        "source_split": source_split,
        "q5_final_guardrail_source_kind": source_kind,
        "q5_final_guardrail_source_id": source_id,
        "q5_final_guardrail_issue_tags": issue_tags,
        "q5_final_guardrail_bucket": source_bucket_from_tags(issue_tags),
    }
    if extra:
        result.update(extra)
    return result


def harvest_kimi_qwen(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for row in load_jsonl(path):
        summary = row.get("correction_summary", "")
        if str(row.get("verifier_decision", "accept")).lower() not in {"accept", "accepted"}:
            continue
        tags = split_tags(row.get("correction_issue_tags") or row.get("q5_issue_tags"))
        rows.append(
            make_training_row(
                source=row,
                summary=summary,
                source_kind="kimi_qwen_verified_rewrite",
                source_id=str(row.get("candidate_id") or row.get("q5_hardening_review_id")),
                issue_tags=tags,
                teacher_model=str(row.get("correction_model", "moonshotai/kimi-k2.6")),
                review_prompt_version=str(
                    row.get("verifier_prompt_version")
                    or "qwen3_235b_a22b_2507_q5_correction_verifier_v1"
                ),
                extra={
                    "previous_bad_summary": row.get("q5_summary", ""),
                    "reference_q8_summary": row.get("q8_summary", ""),
                },
            )
        )
    return rows


def harvest_len15_replay(
    paths: list[Path],
    *,
    cap: int,
    max_words: int,
) -> list[dict[str, Any]]:
    if cap <= 0:
        return []
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            summary = str(row.get("right_summary", ""))
            if not summary or str(row.get("paper_id")) in seen:
                continue
            candidate = make_training_row(
                source=row,
                summary=summary,
                source_kind="len15_step125_replay_anchor",
                source_id=f"{path}:{row.get('paper_id')}",
                issue_tags=[],
                teacher_model="q5_len15_step125_mlx",
                review_prompt_version="local_replay_anchor_audit_v1",
                extra={
                    "q5_replay_source_jsonl": str(path),
                },
            )
            if audit_summary(candidate, summary, max_words=max_words):
                continue
            candidates.append(candidate)
            seen.add(str(row.get("paper_id")))
    candidates.sort(key=lambda row: (str(row.get("primary_category", "")), str(row["paper_id"])))
    return candidates[:cap]


def load_reorient_sources(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    review_items = {
        row["review_id"]: row
        for row in load_jsonl(root / "blind_review_packets" / "review_items.jsonl")
    }
    summary = json.loads(
        (root / "blind_review_packets" / "unblinded_review_summary.json").read_text(
            encoding="utf-8"
        )
    )
    return review_items, summary


def harvest_reorient(root: Path, *, max_clean_q5_anchors: int) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    review_items, summary = load_reorient_sources(root)
    rows: list[dict[str, Any]] = []
    q5_anchor_candidates: list[dict[str, Any]] = []
    for contrast in summary.get("contrast_rows", []):
        source = review_items[contrast["review_id"]]
        q5_tags = split_tags(contrast.get("q5_len15_tags"))
        q8_tags = split_tags(contrast.get("q8_tags"))
        if contrast.get("q5_len15_severe") and not contrast.get("q8_severe"):
            rows.append(
                make_training_row(
                    source=source,
                    summary=str(contrast.get("q8_summary", "")),
                    source_kind="reorient_q8_repair_for_q5_severe",
                    source_id=str(contrast["review_id"]),
                    issue_tags=q5_tags,
                    teacher_model="promoted_q8_mlx_standalone",
                    review_prompt_version="manual_blind_reorient_review_v1",
                    extra={
                        "previous_bad_summary": contrast.get("q5_len15_summary", ""),
                        "review_note": contrast.get("note", ""),
                    },
                )
            )
        elif contrast.get("q8_severe") and not contrast.get("q5_len15_severe"):
            q5_anchor_candidates.append(
                make_training_row(
                    source=source,
                    summary=str(contrast.get("q5_len15_summary", "")),
                    source_kind="reorient_q5_clean_anchor",
                    source_id=str(contrast["review_id"]),
                    issue_tags=q8_tags,
                    teacher_model="q5_len15_step125_mlx",
                    review_prompt_version="manual_blind_reorient_review_v1",
                    extra={
                        "reference_q8_summary": contrast.get("q8_summary", ""),
                        "review_note": contrast.get("note", ""),
                    },
                )
            )
        elif contrast.get("winner_source") == "q5_len15":
            q5_anchor_candidates.append(
                make_training_row(
                    source=source,
                    summary=str(contrast.get("q5_len15_summary", "")),
                    source_kind="reorient_q5_clean_win_anchor",
                    source_id=str(contrast["review_id"]),
                    issue_tags=[],
                    teacher_model="q5_len15_step125_mlx",
                    review_prompt_version="manual_blind_reorient_review_v1",
                    extra={
                        "reference_q8_summary": contrast.get("q8_summary", ""),
                        "review_note": contrast.get("note", ""),
                    },
                )
            )
    rows.extend(q5_anchor_candidates[:max_clean_q5_anchors])
    return rows


def load_len15_targeted(root: Path) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, str]],
]:
    items = {row["review_id"]: row for row in load_jsonl(root / "review_items.jsonl")}
    side_maps = {row["review_id"]: row for row in load_jsonl(root / "side_map.jsonl")}
    with (root / "decisions.tsv").open(encoding="utf-8") as handle:
        decisions = {row["review_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    return items, side_maps, decisions


def side_for_source(side_map: dict[str, Any], source: str) -> str | None:
    for side in ("A", "B", "C"):
        if side_map.get(f"{side}_source") == source:
            return side
    return None


def summary_for_side(item: dict[str, Any], side: str) -> str:
    return str(item.get(f"{side}_summary", ""))


def harvest_len15_targeted(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    items, side_maps, decisions = load_len15_targeted(root)
    rows: list[dict[str, Any]] = []
    for review_id, item in items.items():
        side_map = side_maps[review_id]
        decision = decisions[review_id]
        len15_side = side_for_source(side_map, "len15_v2")
        if not len15_side or decision.get(f"{len15_side}_severe") != "true":
            continue
        winner = decision.get("winner")
        target_side: str | None = None
        if winner in {"A", "B", "C"} and winner != len15_side:
            target_side = winner
        elif winner == "tie":
            for side in ("A", "B", "C"):
                if (
                    side != len15_side
                    and decision.get(f"{side}_severe") != "true"
                    and decision.get(f"{side}_usable") == "true"
                ):
                    target_side = side
                    break
        if not target_side:
            continue
        rows.append(
            make_training_row(
                source=item,
                summary=summary_for_side(item, target_side),
                source_kind="len15_targeted_review_repair",
                source_id=review_id,
                issue_tags=split_tags(decision.get(f"{len15_side}_issue_tags")),
                teacher_model=f"review_selected_{side_map.get(f'{target_side}_source')}",
                review_prompt_version="manual_len15_targeted_blind_review_v1",
                extra={
                    "previous_bad_summary": summary_for_side(item, len15_side),
                    "review_note": decision.get("note", ""),
                },
            )
        )
    return rows


def harvest_old_q5_reviewed(path: Path, *, cap: int) -> list[dict[str, Any]]:
    if not path.exists() or cap <= 0:
        return []
    candidates: list[dict[str, Any]] = []
    for row in load_jsonl(path):
        if not row.get("q5_severe_q8_clean"):
            continue
        candidates.append(
            make_training_row(
                source=row,
                summary=str(row.get("q8_summary", "")),
                source_kind="old_q5_review_q8_clean_repair",
                source_id=str(row.get("review_id")),
                issue_tags=split_tags(row.get("q5_issue_tags")),
                teacher_model="promoted_q8_mlx_standalone",
                review_prompt_version="q5_hardening_blind_review_v1",
                extra={
                    "previous_bad_summary": row.get("q5_summary", ""),
                    "review_note": row.get("note", ""),
                },
            )
        )
    # Prefer the four target buckets and keep deterministic diversity.
    bucket_order = {
        "number_role_binding": 0,
        "polarity_direction": 1,
        "scoped_claims": 2,
        "malformed_avoidance": 3,
        "entity_binding": 4,
        "style_anchor": 5,
    }
    candidates.sort(
        key=lambda row: (
            bucket_order.get(str(row.get("q5_final_guardrail_bucket")), 99),
            str(row.get("primary_category", "")),
            str(row.get("paper_id", "")),
        )
    )
    return candidates[:cap]


def dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    priority = {
        "kimi_qwen_verified_rewrite": 0,
        "reorient_q8_repair_for_q5_severe": 1,
        "len15_targeted_review_repair": 2,
        "old_q5_review_q8_clean_repair": 3,
        "reorient_q5_clean_anchor": 4,
        "reorient_q5_clean_win_anchor": 5,
    }
    by_paper: dict[str, dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = []
    for row in rows:
        paper_id = str(row["paper_id"])
        existing = by_paper.get(paper_id)
        if existing is None:
            by_paper[paper_id] = row
            continue
        current_priority = priority.get(str(row.get("q5_final_guardrail_source_kind")), 99)
        existing_priority = priority.get(
            str(existing.get("q5_final_guardrail_source_kind")), 99
        )
        if current_priority < existing_priority:
            dropped.append(
                {
                    "paper_id": paper_id,
                    "dropped_source_kind": existing.get("q5_final_guardrail_source_kind"),
                    "kept_source_kind": row.get("q5_final_guardrail_source_kind"),
                    "dropped_summary": existing.get("summary"),
                }
            )
            by_paper[paper_id] = row
        else:
            dropped.append(
                {
                    "paper_id": paper_id,
                    "dropped_source_kind": row.get("q5_final_guardrail_source_kind"),
                    "kept_source_kind": existing.get("q5_final_guardrail_source_kind"),
                    "dropped_summary": row.get("summary"),
                }
            )
    return list(by_paper.values()), dropped


def audit_and_split(
    rows: list[dict[str, Any]], *, max_words: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        flags = audit_summary(row, str(row.get("summary", "")), max_words=max_words)
        if flags:
            rejected.append({**row, "audit_reject_flags": flags})
        else:
            accepted.append(row)
    return accepted, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harvest high-confidence final q5 guardrail overlay rows."
    )
    parser.add_argument("--output-release", default=DEFAULT_OUTPUT_RELEASE)
    parser.add_argument("--source-split", default=DEFAULT_SOURCE_SPLIT)
    parser.add_argument("--kimi-qwen-rows", default=DEFAULT_KIMI_QWEN_ROWS)
    parser.add_argument("--reorient-root", default=DEFAULT_REORIENT_ROOT)
    parser.add_argument("--len15-targeted-root", default=DEFAULT_LEN15_TARGETED_ROOT)
    parser.add_argument("--old-q5-reviewed-rows", default=DEFAULT_Q5_REVIEWED_ROWS)
    parser.add_argument("--replay-jsonl", action="append", default=[DEFAULT_REPLAY_JSONL])
    parser.add_argument("--replay-anchor-cap", type=int, default=0)
    parser.add_argument("--old-q5-cap", type=int, default=60)
    parser.add_argument("--clean-q5-anchor-cap", type=int, default=24)
    parser.add_argument("--max-words", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_release = Path(args.output_release)
    if output_release.exists() and any(output_release.iterdir()) and not args.overwrite:
        raise SystemExit(f"{output_release} already exists; pass --overwrite")
    output_release.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict[str, Any]] = []
    source_counts_before: Counter[str] = Counter()
    sources = [
        harvest_kimi_qwen(Path(args.kimi_qwen_rows)),
        harvest_reorient(Path(args.reorient_root), max_clean_q5_anchors=args.clean_q5_anchor_cap),
        harvest_len15_targeted(Path(args.len15_targeted_root)),
        harvest_old_q5_reviewed(Path(args.old_q5_reviewed_rows), cap=args.old_q5_cap),
        harvest_len15_replay(
            [Path(path) for path in args.replay_jsonl],
            cap=args.replay_anchor_cap,
            max_words=args.max_words,
        ),
    ]
    for source_rows in sources:
        raw_rows.extend(source_rows)
        source_counts_before.update(
            str(row.get("q5_final_guardrail_source_kind")) for row in source_rows
        )

    deduped, dedupe_dropped = dedupe_rows(raw_rows)
    accepted, rejected = audit_and_split(deduped, max_words=args.max_words)
    for row in accepted:
        row["source_split"] = args.source_split
    rng = random.Random(args.seed)
    accepted.sort(key=lambda row: str(row["paper_id"]))
    rng.shuffle(accepted)

    write_jsonl(output_release / "training_rows.jsonl", accepted)
    write_jsonl(output_release / "audit_rejected_rows.jsonl", rejected)
    write_jsonl(output_release / "dedupe_dropped_rows.jsonl", dedupe_dropped)

    word_counts = [word_count(row["summary"]) for row in accepted]
    manifest = {
        "created_at": utc_now_iso(),
        "mode": "q5_final_guardrail_overlay_v1",
        "output_release": str(output_release),
        "inputs": {
            "kimi_qwen_rows": args.kimi_qwen_rows,
            "reorient_root": args.reorient_root,
            "len15_targeted_root": args.len15_targeted_root,
            "old_q5_reviewed_rows": args.old_q5_reviewed_rows,
            "old_q5_cap": args.old_q5_cap,
            "clean_q5_anchor_cap": args.clean_q5_anchor_cap,
            "replay_jsonl": args.replay_jsonl,
            "replay_anchor_cap": args.replay_anchor_cap,
        },
        "audit_thresholds": {
            "max_words": args.max_words,
            "max_title_overlap": 0.72,
            "number_policy": "target numbers must appear in title or abstract",
        },
        "summary": {
            "raw_rows_before_dedupe": len(raw_rows),
            "source_counts_before_dedupe": dict(source_counts_before),
            "dedupe_dropped_rows": len(dedupe_dropped),
            "audit_rejected_rows": len(rejected),
            "training_rows": len(accepted),
            "source_counts": dict(
                Counter(row.get("q5_final_guardrail_source_kind") for row in accepted)
            ),
            "bucket_counts": dict(
                Counter(row.get("q5_final_guardrail_bucket") for row in accepted)
            ),
            "issue_tag_counts": dict(
                Counter(
                    tag
                    for row in accepted
                    for tag in row.get("q5_final_guardrail_issue_tags", [])
                )
            ),
            "primary_category_counts": dict(
                Counter(row.get("primary_category") for row in accepted)
            ),
            "summary_word_count": {
                "min": min(word_counts) if word_counts else None,
                "max": max(word_counts) if word_counts else None,
                "avg": round(sum(word_counts) / len(word_counts), 2)
                if word_counts
                else None,
            },
            "audit_reject_reason_counts": dict(
                Counter(flag for row in rejected for flag in row.get("audit_reject_flags", []))
            ),
        },
        "outputs": {
            "training_rows": str(output_release / "training_rows.jsonl"),
            "audit_rejected_rows": str(output_release / "audit_rejected_rows.jsonl"),
            "dedupe_dropped_rows": str(output_release / "dedupe_dropped_rows.jsonl"),
        },
    }
    write_json(output_release / "run_manifest.json", manifest)
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
