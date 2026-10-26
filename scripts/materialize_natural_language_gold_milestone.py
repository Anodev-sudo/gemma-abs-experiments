#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BASE_GOLD = Path(
    "hf_dataset/releases/"
    "recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_604_expansion_pool/"
    "training_rows.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "hf_dataset/releases/"
    "recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_800_milestone"
)
TARGET_SHARES = {
    "cs": 0.37,
    "physics": 0.28,
    "math": 0.16,
    "other": 0.19,
}
PHYSICS_PREFIXES = (
    "astro-ph",
    "cond-mat",
    "gr-qc",
    "hep-",
    "math-ph",
    "nucl-",
    "physics.",
    "quant-ph",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_training_rows(source: str) -> tuple[str, Path]:
    if "=" in source:
        name, value = source.split("=", 1)
        path = Path(value)
    else:
        path = Path(source)
        name = path.name or path.parent.name
    if path.is_dir():
        path = path / "training_rows.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Could not find training rows at {path}")
    return name, path


def collapsed_umbrella(row: dict[str, Any]) -> str:
    primary = str(row.get("primary_category", "")).strip()
    umbrella = str(row.get("umbrella_category", "")).strip()
    if primary.startswith("cs."):
        return "cs"
    if primary.startswith("math."):
        return "math"
    if umbrella == "physics" or primary.startswith(PHYSICS_PREFIXES):
        return "physics"
    return "other"


def target_counts(target_size: int) -> dict[str, int]:
    raw = {bucket: TARGET_SHARES[bucket] * target_size for bucket in TARGET_SHARES}
    counts = {bucket: math.floor(value) for bucket, value in raw.items()}
    remainder = target_size - sum(counts.values())
    for bucket, _ in sorted(
        ((bucket, raw[bucket] - counts[bucket]) for bucket in TARGET_SHARES),
        key=lambda item: (-item[1], item[0]),
    )[:remainder]:
        counts[bucket] += 1
    return counts


def summary_word_count(row: dict[str, Any]) -> int:
    return len([part for part in str(row.get("summary", "")).split() if part])


def select_diverse(rows: list[dict[str, Any]], quota: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if quota <= 0:
        return [], rows
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("primary_category") or "unknown")].append(row)
    categories = sorted(grouped)
    selected: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    offsets: Counter[str] = Counter()
    while len(selected) < quota:
        made_progress = False
        for category in categories:
            offset = offsets[category]
            if offset >= len(grouped[category]):
                continue
            selected.append(grouped[category][offset])
            offsets[category] += 1
            made_progress = True
            if len(selected) >= quota:
                break
        if not made_progress:
            break
    selected_ids = {id(row) for row in selected}
    overflow = [row for row in rows if id(row) not in selected_ids]
    return selected, overflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize an exact-ratio natural-language gold milestone."
    )
    parser.add_argument("--base-gold-jsonl", type=Path, default=DEFAULT_BASE_GOLD)
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Additional strict-audit release dir or training_rows.jsonl. Optional name=path.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-size", type=int, default=800)
    parser.add_argument("--max-summary-words", type=int, default=14)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_rows = load_jsonl(args.base_gold_jsonl)
    targets = target_counts(args.target_size)
    base_counts = Counter(collapsed_umbrella(row) for row in base_rows)
    needs = {
        bucket: max(targets[bucket] - base_counts.get(bucket, 0), 0)
        for bucket in TARGET_SHARES
    }

    seen_ids = {str(row.get("paper_id", "")).strip() for row in base_rows}
    duplicate_rows: list[dict[str, Any]] = []
    bad_shape_rows: list[dict[str, Any]] = []
    candidates_by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_summaries: list[dict[str, Any]] = []

    for source_arg in args.source:
        source_name, path = resolve_training_rows(source_arg)
        rows = load_jsonl(path)
        accepted = 0
        for index, row in enumerate(rows, start=1):
            paper_id = str(row.get("paper_id", "")).strip()
            if not paper_id:
                bad_shape_rows.append({**row, "_milestone_exclusion_reason": "missing_paper_id"})
                continue
            if paper_id in seen_ids:
                duplicate_rows.append(
                    {
                        **row,
                        "_milestone_source": source_name,
                        "_milestone_source_path": str(path),
                        "_milestone_source_index": index,
                        "_milestone_exclusion_reason": "duplicate_paper_id",
                    }
                )
                continue
            if summary_word_count(row) > args.max_summary_words:
                bad_shape_rows.append(
                    {
                        **row,
                        "_milestone_source": source_name,
                        "_milestone_source_path": str(path),
                        "_milestone_source_index": index,
                        "_milestone_exclusion_reason": "too_long",
                    }
                )
                continue
            enriched = {
                **row,
                "_milestone_source": source_name,
                "_milestone_source_path": str(path),
                "_milestone_source_index": index,
            }
            candidates_by_bucket[collapsed_umbrella(row)].append(enriched)
            seen_ids.add(paper_id)
            accepted += 1
        source_summaries.append(
            {
                "name": source_name,
                "path": str(path),
                "rows": len(rows),
                "deduped_usable_rows": accepted,
            }
        )

    selected_additions: list[dict[str, Any]] = []
    overflow_rows: list[dict[str, Any]] = []
    selected_additions_by_bucket: dict[str, int] = {}
    missing_by_bucket: dict[str, int] = {}
    for bucket in TARGET_SHARES:
        selected, overflow = select_diverse(candidates_by_bucket[bucket], needs[bucket])
        selected_additions.extend(selected)
        overflow_rows.extend(
            {
                **row,
                "_milestone_exclusion_reason": "over_target_bucket_quota",
            }
            for row in overflow
        )
        selected_additions_by_bucket[bucket] = len(selected)
        missing_by_bucket[bucket] = max(needs[bucket] - len(selected), 0)

    final_rows = [*base_rows, *selected_additions]
    final_counts = Counter(collapsed_umbrella(row) for row in final_rows)
    if args.require_complete and len(final_rows) != args.target_size:
        raise RuntimeError(
            f"Milestone incomplete: wrote {len(final_rows)} rows, target is {args.target_size}; "
            f"missing={missing_by_bucket}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "training_rows.jsonl", final_rows)
    write_jsonl(args.output_dir / "selected_additions.jsonl", selected_additions)
    write_jsonl(args.output_dir / "overflow_rows.jsonl", overflow_rows)
    write_jsonl(args.output_dir / "duplicate_rows.jsonl", duplicate_rows)
    write_jsonl(args.output_dir / "shape_excluded_rows.jsonl", bad_shape_rows)

    manifest = {
        "created_at": utc_now_iso(),
        "output_dir": str(args.output_dir),
        "target_size": args.target_size,
        "target_counts": targets,
        "base_gold_jsonl": str(args.base_gold_jsonl),
        "base_rows": len(base_rows),
        "base_counts": dict(sorted(base_counts.items())),
        "needed_additions": needs,
        "source_summaries": source_summaries,
        "selected_additions": len(selected_additions),
        "selected_additions_by_bucket": selected_additions_by_bucket,
        "missing_by_bucket": missing_by_bucket,
        "final_rows": len(final_rows),
        "final_counts": dict(sorted(final_counts.items())),
        "duplicate_rows": len(duplicate_rows),
        "shape_excluded_rows": len(bad_shape_rows),
        "overflow_rows": len(overflow_rows),
        "max_summary_words": args.max_summary_words,
        "selection_policy": "Base gold first, then diverse round-robin within each collapsed bucket.",
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(f"[gold-milestone] wrote {len(final_rows)} rows to {args.output_dir / 'training_rows.jsonl'}")
    print(f"[gold-milestone] final counts: {dict(sorted(final_counts.items()))}")
    print(f"[gold-milestone] missing: {missing_by_bucket}")


if __name__ == "__main__":
    main()
