#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_mlx_heuristic_10746_support_dataset import (
    DEFAULT_GOLD_DIR,
    DEFAULT_PROMPT_SPEC,
    DEFAULT_REUSE_EVAL_JSONL,
    DEFAULT_SOURCE_JSONL,
    assistant_text,
    clone_gold_holdout,
    convert_selected_row,
    load_jsonl,
    load_prompt_spec,
    score_row,
    select_rows,
    target_counts,
    tokens,
    write_json,
    write_jsonl,
)


DEFAULT_OUTPUT_DIR = Path(
    "hf_bucket/experiments/mlx_lora/"
    "granite_4_0_350m_natural_language_994_gold8_strict10746_support_v1/"
    "dataset_strict10746_support1600_title_abstract_feed_blurb_v5"
)
HYPE_COMPARISON_TERMS = {
    "best",
    "breakthrough",
    "dramatically",
    "major",
    "novel",
    "outperform",
    "outperformed",
    "outperforming",
    "outperforms",
    "revolutionary",
    "significant",
    "significantly",
    "state-of-the-art",
    "superior",
}
WEAK_OPENING_TERMS = {"new", "recent", "recently"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a strict deterministic support set from the old 10,746-row "
            "release for Granite gold-heavy replay."
        )
    )
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--prompt-spec", type=Path, default=DEFAULT_PROMPT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--exclude-eval-jsonl", type=Path, default=DEFAULT_REUSE_EVAL_JSONL)
    parser.add_argument("--target-count", type=int, default=1600)
    parser.add_argument("--max-per-primary-category", type=int, default=120)
    parser.add_argument("--min-word-count", type=int, default=9)
    parser.add_argument("--max-word-count", type=int, default=14)
    parser.add_argument("--max-title-overlap", type=int, default=2)
    parser.add_argument("--min-source-support-fraction", type=float, default=0.60)
    parser.add_argument("--min-abstract-non-title-hits", type=int, default=2)
    parser.add_argument("--max-acronym-count", type=int, default=1)
    parser.add_argument("--reject-title-trigram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reject-vague-verbs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reject-hype-comparison", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reject-weak-openings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260527)
    return parser.parse_args()


def strict_rejects(audit: dict[str, Any], args: argparse.Namespace) -> list[str]:
    rejects: list[str] = []
    words = int(audit["word_count"])
    if words < args.min_word_count:
        rejects.append("strict_too_short")
    if words > args.max_word_count:
        rejects.append("strict_too_long")
    if int(audit["title_overlap_count"]) > args.max_title_overlap:
        rejects.append("strict_title_overlap")
    if args.reject_title_trigram and audit["title_trigram"]:
        rejects.append("strict_title_trigram")
    if float(audit["source_support_fraction"]) < args.min_source_support_fraction:
        rejects.append("strict_low_source_support")
    if len(audit["abstract_non_title_hits"]) < args.min_abstract_non_title_hits:
        rejects.append("strict_low_abstract_non_title_support")
    if int(audit["acronym_count"]) > args.max_acronym_count:
        rejects.append("strict_acronym_stack")
    if args.reject_vague_verbs and audit["vague_hits"]:
        rejects.append("strict_vague_verb")
    if args.reject_hype_comparison and set(tokens(audit["summary"])) & HYPE_COMPARISON_TERMS:
        rejects.append("strict_hype_comparison")
    summary_tokens = tokens(audit["summary"])
    if args.reject_weak_openings and summary_tokens and summary_tokens[0] in WEAK_OPENING_TERMS:
        rejects.append("strict_weak_opening")
    return rejects


def add_policy_rejects(audits: list[dict[str, Any]], args: argparse.Namespace) -> None:
    for audit in audits:
        policy_rejects = strict_rejects(audit, args)
        if policy_rejects:
            audit["hard_rejects"] = list(audit["hard_rejects"]) + policy_rejects
        audit["strict_policy_rejects"] = policy_rejects


def selected_summary(selected_audits: list[dict[str, Any]]) -> dict[str, Any]:
    if not selected_audits:
        return {}
    return {
        "score": {
            "min": round(min(audit["score"] for audit in selected_audits), 3),
            "max": round(max(audit["score"] for audit in selected_audits), 3),
            "avg": round(sum(audit["score"] for audit in selected_audits) / len(selected_audits), 3),
        },
        "word_counts": dict(Counter(audit["word_count"] for audit in selected_audits).most_common()),
        "title_overlap_counts": dict(
            Counter(audit["title_overlap_count"] for audit in selected_audits).most_common()
        ),
        "acronym_counts": dict(Counter(audit["acronym_count"] for audit in selected_audits).most_common()),
        "flag_counts": dict(Counter(flag for audit in selected_audits for flag in audit["flags"]).most_common()),
    }


def main() -> None:
    args = parse_args()
    if args.target_count < 1:
        raise ValueError("--target-count must be positive")
    if args.min_word_count > args.max_word_count:
        raise ValueError("--min-word-count cannot exceed --max-word-count")

    prompt_spec = load_prompt_spec(args.prompt_spec)
    source_rows = load_jsonl(args.source_jsonl)
    gold_train = load_jsonl(args.gold_dir / "train.jsonl")
    gold_valid = load_jsonl(args.gold_dir / "valid.jsonl")
    gold_test = load_jsonl(args.gold_dir / "test.jsonl")
    gold_ids = {str(row.get("paper_id") or "") for row in gold_train + gold_valid + gold_test}
    eval_ids: set[str] = set()
    if args.exclude_eval_jsonl.exists():
        eval_ids = {str(row.get("paper_id") or "") for row in load_jsonl(args.exclude_eval_jsonl)}

    filtered_rows: list[dict[str, Any]] = []
    duplicate_excluded = 0
    eval_excluded = 0
    for row in source_rows:
        paper_id = str(row.get("paper_id") or "")
        if paper_id in gold_ids:
            duplicate_excluded += 1
            continue
        if paper_id in eval_ids:
            eval_excluded += 1
            continue
        filtered_rows.append(row)

    audits = [score_row(row) for row in filtered_rows]
    add_policy_rejects(audits, args)
    selected_indexes = set(
        select_rows(
            filtered_rows,
            audits,
            target_count=args.target_count,
            max_per_primary_category=args.max_per_primary_category,
        )
    )
    for index in selected_indexes:
        audits[index]["selected"] = True

    selected_rows = [
        convert_selected_row(
            filtered_rows[index],
            audits[index],
            prompt_spec=prompt_spec,
            source_jsonl=args.source_jsonl,
        )
        for index in sorted(
            selected_indexes,
            key=lambda item: (
                audits[item]["umbrella"],
                -audits[item]["score"],
                audits[item]["source_line_number"],
            ),
        )
    ]
    for row in selected_rows:
        row["dataset_component"] = "granite_strict_10746_support_v1"
        row["strict_support_policy"] = {
            "source": "old_10746_release",
            "selector": "scripts/build_mlx_granite_strict_support_v1_dataset.py",
            "target": "coverage/detail replay for Granite, not primary style learning",
        }

    if len(selected_rows) < args.target_count:
        print(
            f"[granite-strict-support] warning: selected {len(selected_rows)} "
            f"rows below target {args.target_count}"
        )

    valid_rows = [clone_gold_holdout(row, source_path=args.gold_dir / "valid.jsonl") for row in gold_valid]
    test_rows = [clone_gold_holdout(row, source_path=args.gold_dir / "test.jsonl") for row in gold_test]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", selected_rows)
    write_jsonl(args.output_dir / "valid.jsonl", valid_rows)
    write_jsonl(args.output_dir / "test.jsonl", test_rows)
    write_jsonl(args.output_dir / "selection_audit.jsonl", audits)

    selected_audits = [audit for audit in audits if audit["selected"]]
    rejected_audits = [audit for audit in audits if not audit["selected"]]
    write_jsonl(args.output_dir / "selected_audit.jsonl", selected_audits)
    write_jsonl(
        args.output_dir / "reject_sample.jsonl",
        sorted(
            rejected_audits,
            key=lambda audit: (
                bool(audit["hard_rejects"]),
                -audit["score"],
                audit["source_line_number"],
            ),
        )[:250],
    )

    hard_reject_counts: Counter[str] = Counter()
    policy_reject_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    for audit in audits:
        hard_reject_counts.update(audit["hard_rejects"])
        policy_reject_counts.update(audit["strict_policy_rejects"])
        flag_counts.update(audit["flags"])

    selected_bucket_counts = Counter(audit["umbrella"] for audit in selected_audits)
    selected_primary_counts = Counter(audit["primary_category"] for audit in selected_audits)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "scripts/build_mlx_granite_strict_support_v1_dataset.py",
        "purpose": (
            "Strict deterministic old-row support for a Granite gold-heavy LoRA. "
            "Support rows are coverage/detail replay, not equivalent to gold."
        ),
        "source_jsonl": str(args.source_jsonl),
        "gold_dir": str(args.gold_dir),
        "prompt_spec": str(args.prompt_spec),
        "prompt_version": prompt_spec["prompt_version"],
        "input_mode": "title_abstract",
        "output_dir": str(args.output_dir),
        "target_count": args.target_count,
        "max_per_primary_category": args.max_per_primary_category,
        "policy": {
            "min_word_count": args.min_word_count,
            "max_word_count": args.max_word_count,
            "max_title_overlap": args.max_title_overlap,
            "reject_title_trigram": args.reject_title_trigram,
            "min_source_support_fraction": args.min_source_support_fraction,
            "min_abstract_non_title_hits": args.min_abstract_non_title_hits,
            "max_acronym_count": args.max_acronym_count,
            "reject_vague_verbs": args.reject_vague_verbs,
            "reject_hype_comparison": args.reject_hype_comparison,
            "reject_weak_openings": args.reject_weak_openings,
        },
        "source_rows": len(source_rows),
        "source_rows_after_gold_eval_exclusions": len(filtered_rows),
        "excluded_gold_ids": duplicate_excluded,
        "excluded_eval_ids": eval_excluded,
        "selected_rows": len(selected_rows),
        "valid_rows": len(valid_rows),
        "test_rows": len(test_rows),
        "target_bucket_counts": target_counts(args.target_count),
        "selected_bucket_counts": dict(sorted(selected_bucket_counts.items())),
        "selected_primary_category_top30": dict(selected_primary_counts.most_common(30)),
        "hard_reject_counts": dict(hard_reject_counts.most_common()),
        "strict_policy_reject_counts": dict(policy_reject_counts.most_common()),
        "flag_counts": dict(flag_counts.most_common()),
        "selected_summary": selected_summary(selected_audits),
        "sample_selected_summaries": [
            {
                "paper_id": audit["paper_id"],
                "primary_category": audit["primary_category"],
                "word_count": audit["word_count"],
                "title_overlap_count": audit["title_overlap_count"],
                "summary": audit["summary"],
            }
            for audit in selected_audits[:40]
        ],
        "label_policy": "Existing old-release summaries only; no generated or manually written labels.",
    }
    write_json(args.output_dir / "manifest.json", manifest)

    readme = [
        "# Granite Strict 10746 Support Dataset",
        "",
        f"- Rows: `{len(selected_rows)}` selected from `{len(source_rows)}` old 10,746-release rows.",
        f"- Prompt: `{prompt_spec['prompt_version']}` / `title_abstract`.",
        "- Labels are existing old-release summaries; no generated or manually written labels were added.",
        "- `valid.jsonl` and `test.jsonl` are strict-gold holdouts from the 994-row release.",
        "- Selection is deterministic and audited in `selection_audit.jsonl`.",
        "- This support set is intended for low-frequency replay in Granite LoRA training.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(f"[granite-strict-support] wrote {len(selected_rows)} train rows to {args.output_dir / 'train.jsonl'}")
    print(f"[granite-strict-support] selected buckets={dict(sorted(selected_bucket_counts.items()))}")
    print(f"[granite-strict-support] hard rejects={dict(hard_reject_counts.most_common(10))}")
    print(f"[granite-strict-support] selected summary={json.dumps(manifest['selected_summary'], sort_keys=True)}")


if __name__ == "__main__":
    main()
