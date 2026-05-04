#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
import random
import sqlite3
import sys
from typing import Any
from urllib.parse import quote

try:
    from scripts.build_safety_eval_set import (
        ADVERSARIAL_TEMPLATES,
        TOPIC_PATTERNS,
        build_eval_row,
        make_adversarial_row,
        match_edge_topics,
        summarize,
    )
    from scripts.export_weighted_abstract_sample import normalize_text, umbrella_for
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from build_safety_eval_set import (
        ADVERSARIAL_TEMPLATES,
        TOPIC_PATTERNS,
        build_eval_row,
        make_adversarial_row,
        match_edge_topics,
        summarize,
    )
    from export_weighted_abstract_sample import normalize_text, umbrella_for

DEFAULT_OUTPUT_DIR = "hf_dataset/evals/safety_realdb_10000_q8_v1"
DEFAULT_CREATED_SINCE = "2023-04-07"
DEFAULT_MIN_ABSTRACT_CHARS = 250
DEFAULT_SEED = 20260501
DEFAULT_NORMAL_COUNT = 6000
DEFAULT_EDGE_COUNT = 2000
DEFAULT_ADVERSARIAL_COUNT = 2000
DEFAULT_EXCLUDE_ROOT = "hf_dataset"
EXCLUSION_GLOBS = ("releases/**/*.jsonl", "evals/**/*.jsonl")
EXCLUSION_ID_KEYS = (
    "paper_id",
    "source_paper_id",
    "adversarial_source_paper_id",
)
EDGE_TOPIC_PREFILTER_TERMS = (
    "abuse",
    "adverse event",
    "anonym",
    "armed conflict",
    "attack",
    "backdoor",
    "bacteria",
    "battlefield",
    "biosecurity",
    "biosafety",
    "biometric",
    "bomb",
    "botnet",
    "cancer",
    "censorship",
    "clinical",
    "conspiracy",
    "covid",
    "cyber",
    "de-anonym",
    "diagnos",
    "disease",
    "discrimination",
    "disinformation",
    "election",
    "epidemic",
    "explosive",
    "extremis",
    "facial recognition",
    "fake news",
    "firearm",
    "gain-of-function",
    "gender bias",
    "gunshot",
    "harassment",
    "hate",
    "hateful",
    "infectious",
    "intrusion",
    "jailbreak",
    "malware",
    "military",
    "misinformation",
    "mortality",
    "opioid",
    "overdose",
    "pandemic",
    "patient",
    "pathogen",
    "phishing",
    "political polarization",
    "pornograph",
    "privacy",
    "propaganda",
    "racial",
    "racis",
    "radicali",
    "ransomware",
    "rape",
    "sars-cov-2",
    "security vulnerabilit",
    "self-harm",
    "sex trafficking",
    "sexual",
    "stereotype",
    "suicide",
    "surveillance",
    "terroris",
    "toxicity",
    "tumor",
    "tumour",
    "viral",
    "violence",
    "violent",
    "virus",
    "vulnerabilit",
    "war",
    "weapon",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a q8 safety eval set from the live Composer SQLite database."
    )
    parser.add_argument("--db", required=True, help="Path to the Composer SQLite database.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--normal-count", type=int, default=DEFAULT_NORMAL_COUNT)
    parser.add_argument("--edge-count", type=int, default=DEFAULT_EDGE_COUNT)
    parser.add_argument("--adversarial-count", type=int, default=DEFAULT_ADVERSARIAL_COUNT)
    parser.add_argument("--created-since", default=DEFAULT_CREATED_SINCE)
    parser.add_argument("--min-abstract-chars", type=int, default=DEFAULT_MIN_ABSTRACT_CHARS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--exclude-root", default=DEFAULT_EXCLUDE_ROOT)
    parser.add_argument(
        "--source-label",
        default="Composer SQLite papers table",
        help="Non-path label for the source database stored in the manifest.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(db_path.resolve()), safe='/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def schema_digest(conn: sqlite3.Connection) -> str:
    rows = conn.execute(
        """
        SELECT type, name, tbl_name, sql
        FROM sqlite_master
        WHERE tbl_name IN ('papers', 'paper_categories')
           OR name IN ('papers', 'paper_categories')
        ORDER BY type, name
        """
    ).fetchall()
    payload = "\n".join(
        json.dumps(
            {
                "type": row["type"],
                "name": row["name"],
                "tbl_name": row["tbl_name"],
                "sql": row["sql"],
            },
            sort_keys=True,
        )
        for row in rows
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def strip_adversarial_suffix(paper_id: str) -> str:
    return paper_id.split("::adv::", 1)[0]


def add_excluded_id(excluded_ids: set[str], value: Any) -> None:
    if not isinstance(value, str):
        return
    paper_id = value.strip()
    if not paper_id:
        return
    excluded_ids.add(paper_id)
    excluded_ids.add(strip_adversarial_suffix(paper_id))


def safe_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def collect_excluded_ids(
    exclude_root: Path,
    output_dir: Path,
) -> tuple[set[str], dict[str, Any]]:
    excluded_ids: set[str] = set()
    files: list[dict[str, Any]] = []
    root = exclude_root.resolve()
    resolved_output = output_dir.resolve()

    for pattern in EXCLUSION_GLOBS:
        for path in sorted(root.glob(pattern)):
            if path.is_dir():
                continue
            try:
                if path.resolve().is_relative_to(resolved_output):
                    continue
            except FileNotFoundError:
                pass

            rows_read = 0
            ids_before = len(excluded_ids)
            with path.open(encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    rows_read += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    for key in EXCLUSION_ID_KEYS:
                        add_excluded_id(excluded_ids, row.get(key))

            files.append(
                {
                    "path": safe_relative(path, root),
                    "rows_read": rows_read,
                    "new_ids": len(excluded_ids) - ids_before,
                }
            )

    return excluded_ids, {
        "root_label": exclude_root.as_posix(),
        "patterns": list(EXCLUSION_GLOBS),
        "id_keys": list(EXCLUSION_ID_KEYS),
        "files_scanned": len(files),
        "ids_excluded": len(excluded_ids),
        "files": files,
    }


def stable_score(seed: int, label: str, paper_id: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{label}:{paper_id}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big")


def consider_smallest(
    heap: list[tuple[int, str, dict[str, Any]]],
    limit: int,
    *,
    score: int,
    row: dict[str, Any],
) -> None:
    if limit <= 0:
        return
    item = (-score, str(row["paper_id"]), row)
    if len(heap) < limit:
        heapq.heappush(heap, item)
        return
    if item > heap[0]:
        heapq.heapreplace(heap, item)


def heap_rows(heap: list[tuple[int, str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]


def index_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
        [name],
    ).fetchone()
    return row is not None


def query_primary_rows(conn: sqlite3.Connection) -> str:
    papers_source = "papers p"
    categories_source = "paper_categories pc"
    if index_exists(conn, "idx_papers_feed_order"):
        papers_source = "papers p INDEXED BY idx_papers_feed_order"
    if index_exists(conn, "idx_paper_categories_arxiv_primary"):
        categories_source = (
            "paper_categories pc INDEXED BY idx_paper_categories_arxiv_primary"
        )

    return f"""
        SELECT
            pc.category AS primary_category,
            p.arxiv_id,
            p.title,
            p.abstract,
            p.created_date,
            p.updated_date
        FROM {papers_source}
        JOIN {categories_source}
            ON pc.arxiv_id = p.arxiv_id
           AND pc.is_primary = 1
        WHERE p.is_deleted = 0
          AND p.created_date >= ?
          AND LENGTH(TRIM(p.abstract)) >= ?
        ORDER BY p.created_date DESC, p.arxiv_id DESC
    """


def make_record(row: sqlite3.Row) -> dict[str, Any]:
    primary_category = str(row["primary_category"])
    return {
        "paper_id": str(row["arxiv_id"]),
        "title": normalize_text(str(row["title"] or "")),
        "abstract": normalize_text(str(row["abstract"] or "")),
        "primary_category": primary_category,
        "umbrella_category": umbrella_for(primary_category),
        "categories": [primary_category],
        "created_date": str(row["created_date"] or ""),
        "updated_date": str(row["updated_date"] or ""),
    }


def maybe_match_edge_topics(row: dict[str, Any]) -> list[str]:
    text = "\n".join(
        [
            str(row.get("title", "")),
            str(row.get("abstract", "")),
            str(row.get("primary_category", "")),
            " ".join(row.get("categories", [])),
        ]
    ).lower()
    if not any(term in text for term in EDGE_TOPIC_PREFILTER_TERMS):
        return []
    return match_edge_topics(row)


def scan_candidate_reservoirs(
    conn: sqlite3.Connection,
    *,
    excluded_ids: set[str],
    normal_limit: int,
    edge_limit: int,
    min_abstract_chars: int,
    created_since: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    normal_heap: list[tuple[int, str, dict[str, Any]]] = []
    edge_heap: list[tuple[int, str, dict[str, Any]]] = []
    candidate_counts = Counter()
    category_counts = Counter()
    edge_topic_counts = Counter()
    date_min: str | None = None
    date_max: str | None = None

    cursor = conn.execute(query_primary_rows(conn), [created_since, min_abstract_chars])
    for scanned, db_row in enumerate(cursor, start=1):
        if scanned % 50_000 == 0:
            print(
                f"[realdb-safety] scanned {scanned:,} eligible primary rows...",
                file=sys.stderr,
                flush=True,
            )
        record = make_record(db_row)
        paper_id = str(record["paper_id"])
        candidate_counts["eligible_primary_rows"] += 1
        created_date = str(record["created_date"])
        if created_date:
            date_min = created_date if date_min is None else min(date_min, created_date)
            date_max = created_date if date_max is None else max(date_max, created_date)

        if paper_id in excluded_ids or strip_adversarial_suffix(paper_id) in excluded_ids:
            candidate_counts["excluded_prior_ids"] += 1
            continue

        topics = maybe_match_edge_topics(record)
        record["edge_topics"] = topics
        category_counts[str(record["primary_category"])] += 1
        if topics:
            candidate_counts["edge_candidates"] += 1
            edge_topic_counts.update(topics)
            consider_smallest(
                edge_heap,
                edge_limit,
                score=stable_score(seed, "edge", paper_id),
                row=record,
            )
        else:
            candidate_counts["normal_candidates"] += 1
            consider_smallest(
                normal_heap,
                normal_limit,
                score=stable_score(seed, "normal", paper_id),
                row=record,
            )

    stats = {
        "candidate_counts": dict(candidate_counts),
        "candidate_primary_categories_top_50": dict(category_counts.most_common(50)),
        "candidate_edge_topics": dict(edge_topic_counts),
        "eligible_date_bounds": {
            "min_created_date": date_min,
            "max_created_date": date_max,
        },
        "reservoir_sizes": {
            "normal": len(normal_heap),
            "edge": len(edge_heap),
        },
    }
    return heap_rows(normal_heap), heap_rows(edge_heap), stats


def fetch_all_categories(conn: sqlite3.Connection, paper_ids: list[str]) -> dict[str, list[str]]:
    categories_by_paper: dict[str, list[str]] = {}
    chunk_size = 900
    for start in range(0, len(paper_ids), chunk_size):
        chunk = paper_ids[start : start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT arxiv_id, category
            FROM paper_categories
            WHERE arxiv_id IN ({placeholders})
            ORDER BY arxiv_id, is_primary DESC, category
            """,
            chunk,
        ).fetchall()
        for row in rows:
            categories_by_paper.setdefault(str(row["arxiv_id"]), []).append(str(row["category"]))
    return categories_by_paper


def hydrate_categories(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    categories_by_paper = fetch_all_categories(conn, [str(row["paper_id"]) for row in rows])
    for row in rows:
        categories = categories_by_paper.get(str(row["paper_id"]), [])
        if categories:
            row["categories"] = categories
        row["edge_topics"] = maybe_match_edge_topics(row)


def select_adversarial_sources(
    *,
    normal_holdout: list[dict[str, Any]],
    edge_holdout: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    edge_target = min(len(edge_holdout), max(count * 2 // 3, count - len(normal_holdout)))
    sources = edge_holdout[:edge_target]
    sources.extend(normal_holdout[: count - len(sources)])
    if len(sources) < count:
        extra_edge = edge_holdout[edge_target:]
        sources.extend(extra_edge[: count - len(sources)])
    if len(sources) < count:
        raise ValueError(
            f"Need {count} held-out adversarial source rows; found {len(sources)}"
        )
    return sources[:count]


def selected_date_bounds(rows: list[dict[str, Any]]) -> dict[str, str | None]:
    dates = [str(row.get("created_date", "")) for row in rows if row.get("created_date")]
    return {
        "min_created_date": min(dates) if dates else None,
        "max_created_date": max(dates) if dates else None,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_readme(output_dir: Path, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    text = f"""# RealDB q8 Safety Eval v1

This eval set samples recent papers from the Composer SQLite database and excludes paper IDs already present in prior dataset release/eval JSONL files.

## Contents

- `eval_inputs.jsonl`: model inputs for q8 safety generation.
- `paper_ids.json`: sampled eval paper IDs, including adversarial synthetic IDs.
- `manifest.json`: reproducibility metadata without machine-specific database paths.

## Counts

- Total rows: {summary["count"]}
- Normal rows: {summary["by_split"].get("normal", 0)}
- Sensitive-edge rows: {summary["by_split"].get("edge", 0)}
- Adversarial rows: {summary["by_split"].get("adversarial", 0)}

## Rebuild

```bash
hf_bucket/experiments/mlx_lora/runtime/py312/bin/python \\
  scripts/build_realdb_safety_eval_set.py \\
  --db <COMPOSER_DB> \\
  --output-dir {manifest["artifact_dir"]} \\
  --normal-count {manifest["requested_counts"]["normal"]} \\
  --edge-count {manifest["requested_counts"]["edge"]} \\
  --adversarial-count {manifest["requested_counts"]["adversarial"]} \\
  --created-since {manifest["created_since"]} \\
  --min-abstract-chars {manifest["min_abstract_chars"]} \\
  --seed {manifest["seed"]}
```
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{output_dir} already exists and is not empty; pass --overwrite")

    exclude_root = Path(args.exclude_root)
    excluded_ids, exclusion_manifest = collect_excluded_ids(exclude_root, output_dir)
    db_path = Path(args.db)
    normal_reservoir_limit = args.normal_count + args.adversarial_count
    edge_reservoir_limit = args.edge_count + args.adversarial_count

    conn = connect_readonly(db_path)
    try:
        digest = schema_digest(conn)
        normal_pool, edge_pool, scan_stats = scan_candidate_reservoirs(
            conn,
            excluded_ids=excluded_ids,
            normal_limit=normal_reservoir_limit,
            edge_limit=edge_reservoir_limit,
            min_abstract_chars=args.min_abstract_chars,
            created_since=args.created_since,
            seed=args.seed,
        )

        if len(normal_pool) < args.normal_count:
            raise ValueError(
                f"Need {args.normal_count} normal rows, found {len(normal_pool)}"
            )
        if len(edge_pool) < args.edge_count:
            raise ValueError(f"Need {args.edge_count} edge rows, found {len(edge_pool)}")

        normal_rows = normal_pool[: args.normal_count]
        edge_rows = edge_pool[: args.edge_count]
        adversarial_sources = select_adversarial_sources(
            normal_holdout=normal_pool[args.normal_count :],
            edge_holdout=edge_pool[args.edge_count :],
            count=args.adversarial_count,
        )

        hydrate_categories(conn, normal_rows + edge_rows + adversarial_sources)
    finally:
        conn.close()

    rng = random.Random(args.seed)
    adversarial_rows = [
        make_adversarial_row(
            source,
            sequence=index,
            template=ADVERSARIAL_TEMPLATES[(index - 1) % len(ADVERSARIAL_TEMPLATES)],
        )
        for index, source in enumerate(adversarial_sources, start=1)
    ]
    rng.shuffle(adversarial_rows)

    output_rows = [
        build_eval_row(row, "normal", index)
        for index, row in enumerate(normal_rows, start=1)
    ] + [
        build_eval_row(row, "edge", index)
        for index, row in enumerate(edge_rows, start=1)
    ] + [
        build_eval_row(row, "adversarial", index)
        for index, row in enumerate(adversarial_rows, start=1)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "eval_inputs.jsonl", output_rows)
    (output_dir / "paper_ids.json").write_text(
        json.dumps([row["paper_id"] for row in output_rows], indent=2) + "\n",
        encoding="utf-8",
    )

    summary = summarize(output_rows)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": args.output_dir,
        "source_database": {
            "label": args.source_label,
            "filename": db_path.name,
            "absolute_path_recorded": False,
        },
        "schema_sha256": digest,
        "created_since": args.created_since,
        "min_abstract_chars": args.min_abstract_chars,
        "seed": args.seed,
        "requested_counts": {
            "normal": args.normal_count,
            "edge": args.edge_count,
            "adversarial": args.adversarial_count,
        },
        "row_counts": {
            "eval_inputs": len(output_rows),
            "normal": len(normal_rows),
            "edge": len(edge_rows),
            "adversarial": len(adversarial_rows),
            "adversarial_sources": len(adversarial_sources),
        },
        "date_bounds": {
            "eligible": scan_stats["eligible_date_bounds"],
            "selected": selected_date_bounds(output_rows),
        },
        "exclusions": exclusion_manifest,
        "sampling": {
            "strategy": "stable hash reservoirs over eligible primary-category DB rows",
            "normal_reservoir_limit": normal_reservoir_limit,
            "edge_reservoir_limit": edge_reservoir_limit,
            "candidate_counts": scan_stats["candidate_counts"],
            "candidate_primary_categories_top_50": scan_stats[
                "candidate_primary_categories_top_50"
            ],
            "candidate_edge_topics": scan_stats["candidate_edge_topics"],
            "reservoir_sizes": scan_stats["reservoir_sizes"],
            "adversarial_source_mix": dict(
                Counter("edge" if row.get("edge_topics") else "normal" for row in adversarial_sources)
            ),
        },
        "edge_topic_patterns": TOPIC_PATTERNS,
        "adversarial_templates": ADVERSARIAL_TEMPLATES,
        "summary": summary,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(output_dir, manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_dataset(args)
    print(f"Wrote {manifest['row_counts']['eval_inputs']} eval rows to {args.output_dir}")
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
