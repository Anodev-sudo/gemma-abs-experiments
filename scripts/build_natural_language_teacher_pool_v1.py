#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("hf_dataset/corpora/recent_3y_balanced/arxiv_abstract_corpus_weighted_20000.jsonl")
DEFAULT_OUTPUT = Path(
    "hf_bucket/experiments/natural_language_teacher/deepseek_v4_pro_v1/"
    "source_candidates/natural_language_candidates.jsonl"
)
DEFAULT_MANIFEST = DEFAULT_OUTPUT.with_name("run_manifest.json")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
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


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("paper_id", "")), str(row.get("title", "")))


def load_excluded_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            paper_id = str(row.get("paper_id", "")).strip()
            if paper_id:
                excluded.add(paper_id)
    return excluded


def select_rows(
    rows: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
    full_corpus: bool,
    shard_count: int,
    shard_index: int,
    max_per_primary_category: int | None,
    excluded_ids: set[str],
) -> list[dict[str, Any]]:
    usable = [
        row
        for row in rows
        if str(row.get("paper_id", "")).strip()
        and str(row.get("abstract", "")).strip()
        and str(row.get("paper_id", "")).strip() not in excluded_ids
    ]
    if shard_count < 1:
        raise ValueError("--shard-count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--shard-index must be between 0 and shard-count - 1")
    usable.sort(key=row_key)
    if shard_count > 1:
        usable = [row for index, row in enumerate(usable) if index % shard_count == shard_index]
    if full_corpus:
        return usable

    rng = random.Random(seed)
    shuffled = list(usable)
    rng.shuffle(shuffled)
    selected: list[dict[str, Any]] = []
    primary_counts: collections.Counter[str] = collections.Counter()
    for row in shuffled:
        primary = str(row.get("primary_category") or "unknown")
        if max_per_primary_category is not None and primary_counts[primary] >= max_per_primary_category:
            continue
        selected.append(row)
        primary_counts[primary] += 1
        if len(selected) >= sample_size:
            break
    return selected


def materialize_candidates(rows: list[dict[str, Any]], *, seed: int, source_split: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        paper_id = str(row.get("paper_id", ""))
        candidate = {
            "candidate_id": f"natural_language_teacher_v1:{index:05d}:{paper_id}",
            "paper_id": paper_id,
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "primary_category": row.get("primary_category", ""),
            "umbrella_category": row.get("umbrella_category", ""),
            "categories": row.get("categories", []),
            "created_date": row.get("created_date", ""),
            "updated_date": row.get("updated_date", ""),
            "source_split": source_split,
            "selection_seed": seed,
            "selection_rank": index,
        }
        candidates.append(candidate)
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fresh source pool for plain-language teacher summary generation."
    )
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--full-corpus", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-per-primary-category", type=int, default=20)
    parser.add_argument(
        "--exclude-jsonl",
        action="append",
        type=Path,
        default=[],
        help="Optional JSONL whose paper_id values should be skipped.",
    )
    parser.add_argument(
        "--source-split",
        default="recent_3y_balanced_natural_language_teacher_v1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_jsonl)
    excluded_ids = load_excluded_ids(args.exclude_jsonl)
    max_per_primary = None if args.full_corpus else args.max_per_primary_category
    selected = select_rows(
        rows,
        sample_size=args.sample_size,
        seed=args.seed,
        full_corpus=args.full_corpus,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        max_per_primary_category=max_per_primary,
        excluded_ids=excluded_ids,
    )
    candidates = materialize_candidates(selected, seed=args.seed, source_split=args.source_split)
    write_jsonl(args.output_jsonl, candidates)
    manifest = {
        "created_at": utc_now_iso(),
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "source_rows": len(rows),
        "excluded_ids": len(excluded_ids),
        "candidate_rows": len(candidates),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "full_corpus": args.full_corpus,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "max_per_primary_category": max_per_primary,
        "source_split": args.source_split,
        "primary_category_counts": dict(collections.Counter(row.get("primary_category", "") for row in candidates)),
        "umbrella_category_counts": dict(collections.Counter(row.get("umbrella_category", "") for row in candidates)),
    }
    write_json(args.manifest, manifest)
    print(f"[natural-language-pool] wrote {len(candidates)} candidates to {args.output_jsonl}")


if __name__ == "__main__":
    main()
