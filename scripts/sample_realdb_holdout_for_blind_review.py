#!/usr/bin/env python3
"""Sample held-out paper rows from a Composer-style arXiv SQLite database."""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def umbrella_for_category(category: str) -> str:
    if not category:
        return ""
    return category.split(".", 1)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reservoir-sample random papers from a read-only SQLite database while "
            "excluding rows present in a training JSONL."
        )
    )
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--exclude-train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--min-abstract-chars", type=int, default=250)
    parser.add_argument("--split-label", default="realdb_random_heldout")
    parser.add_argument("--progress-every", type=int, default=250_000)
    return parser.parse_args()


def load_excluded_paper_ids(path: Path) -> set[str]:
    excluded: set[str] = set()
    for row in load_jsonl(path):
        paper_id = row.get("paper_id") or row.get("arxiv_id")
        if paper_id:
            excluded.add(str(paper_id))
    return excluded


def connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def reservoir_sample(
    conn: sqlite3.Connection,
    *,
    excluded_ids: set[str],
    count: int,
    seed: int,
    min_abstract_chars: int,
    progress_every: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    scanned = 0
    eligible = 0
    excluded_hits = 0

    query = """
        SELECT arxiv_id, title, abstract, created_date
        FROM papers
        WHERE is_deleted = 0
          AND length(trim(abstract)) >= ?
        ORDER BY arxiv_id
    """
    for arxiv_id, title, abstract, created_date in conn.execute(query, (min_abstract_chars,)):
        scanned += 1
        paper_id = str(arxiv_id)
        if paper_id in excluded_ids:
            excluded_hits += 1
            continue
        eligible += 1
        row = {
            "paper_id": paper_id,
            "title": normalize_text(title),
            "abstract": normalize_text(abstract),
            "created_date": str(created_date or ""),
        }
        if len(reservoir) < count:
            reservoir.append(row)
        else:
            replacement_index = rng.randrange(eligible)
            if replacement_index < count:
                reservoir[replacement_index] = row

        if progress_every > 0 and scanned % progress_every == 0:
            print(
                f"[realdb-sample] scanned={scanned} eligible={eligible} "
                f"excluded_hits={excluded_hits}",
                file=sys.stderr,
                flush=True,
            )

    if len(reservoir) < count:
        raise ValueError(
            f"Requested {count} rows, but only {len(reservoir)} eligible rows were found."
        )

    rng.shuffle(reservoir)
    return reservoir, {
        "scanned_rows": scanned,
        "eligible_rows_after_exclusion": eligible,
        "excluded_training_hits": excluded_hits,
    }


def attach_categories(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    ids = [str(row["paper_id"]) for row in rows]
    placeholders = ",".join("?" for _ in ids)
    query = f"""
        SELECT arxiv_id, category, is_primary
        FROM paper_categories
        WHERE arxiv_id IN ({placeholders})
    """
    by_id: dict[str, list[tuple[int, str]]] = {paper_id: [] for paper_id in ids}
    for arxiv_id, category, is_primary in conn.execute(query, ids):
        by_id.setdefault(str(arxiv_id), []).append((int(is_primary or 0), str(category)))

    for row in rows:
        pairs = sorted(by_id.get(str(row["paper_id"]), []), key=lambda item: (-item[0], item[1]))
        categories = [category for _, category in pairs]
        primary = next((category for is_primary, category in pairs if is_primary), "")
        if not primary and categories:
            primary = categories[0]
        row["primary_category"] = primary
        row["umbrella_category"] = umbrella_for_category(primary)
        row["categories"] = categories


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")

    excluded_ids = load_excluded_paper_ids(args.exclude_train_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with connect_readonly(args.db) as conn:
        rows, sample_counts = reservoir_sample(
            conn,
            excluded_ids=excluded_ids,
            count=args.count,
            seed=args.seed,
            min_abstract_chars=args.min_abstract_chars,
            progress_every=args.progress_every,
        )
        attach_categories(conn, rows)

    for index, row in enumerate(rows, start=1):
        row["eval_id"] = f"heldout_realdb_{index:03d}"
        row["split"] = args.split_label

    output_jsonl = args.output_dir / f"source_{len(rows)}.jsonl"
    write_jsonl(output_jsonl, rows)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "scripts/sample_realdb_holdout_for_blind_review.py",
        "count": len(rows),
        "seed": args.seed,
        "min_abstract_chars": args.min_abstract_chars,
        "split_label": args.split_label,
        "db_source": "Composer-style arXiv SQLite database supplied via --db",
        "exclude_train_jsonl": str(args.exclude_train_jsonl),
        "excluded_training_ids": len(excluded_ids),
        "output_jsonl": str(output_jsonl),
        **sample_counts,
    }
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[realdb-sample] wrote {len(rows)} rows to {output_jsonl}")


if __name__ == "__main__":
    main()
