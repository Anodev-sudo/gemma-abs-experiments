#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PHYSICS_EXACT = {
    "astro-ph",
    "gr-qc",
    "hep-ex",
    "hep-lat",
    "hep-ph",
    "hep-th",
    "math-ph",
    "nucl-ex",
    "nucl-th",
    "quant-ph",
}

TOP_UMBRELLA_TARGET_SHARES = {
    "cs": 0.37,
    "physics": 0.28,
    "math": 0.16,
}

NON_TOP_UMBRELLA_MAX_SHARE = 0.085

ML_CS_FOCUS_CATEGORIES = {
    "cs.AI",
    "cs.CL",
    "cs.CV",
    "cs.IR",
    "cs.LG",
    "cs.MA",
    "cs.MM",
    "cs.NE",
    "cs.RO",
    "cs.SI",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a category-diverse arXiv abstract corpus from the Composer SQLite "
            "database, with extra weight on CS and physics-related categories."
        )
    )
    parser.add_argument("--db", required=True, help="Path to the Composer SQLite database.")
    parser.add_argument(
        "--output-dir",
        default="hf_dataset/corpora/all_time_weighted_20000",
        help="Directory for the exported corpus and manifest in dataset staging.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=20_000,
        help="Number of abstracts to export.",
    )
    parser.add_argument(
        "--min-per-category",
        type=int,
        default=12,
        help="Minimum quota to reserve for each primary category when available.",
    )
    parser.add_argument(
        "--min-abstract-chars",
        type=int,
        default=250,
        help="Discard abstracts shorter than this many characters after trimming.",
    )
    parser.add_argument(
        "--created-since",
        default=None,
        help="Optional inclusive created_date cutoff in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260407,
        help="Deterministic seed for reservoir sampling.",
    )
    parser.add_argument(
        "--max-per-category",
        type=int,
        default=275,
        help="Hard cap on the number of sampled papers from any single primary category.",
    )
    return parser.parse_args()


def umbrella_for(category: str) -> str:
    if category.startswith("cs."):
        return "cs"
    if (
        category.startswith("astro-ph.")
        or category.startswith("cond-mat.")
        or category.startswith("hep-")
        or category.startswith("physics.")
        or category in PHYSICS_EXACT
        or category == "cond-mat"
    ):
        return "physics"
    if category.startswith("math."):
        return "math"
    if category.startswith("stat."):
        return "stat"
    if category.startswith("eess."):
        return "eess"
    if category.startswith("q-bio."):
        return "q-bio"
    if category.startswith("nlin."):
        return "nlin"
    if category.startswith("q-fin."):
        return "q-fin"
    if category.startswith("econ."):
        return "econ"
    return "other"


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.blake2b(f"{seed}:{label}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_primary_category_counts(
    conn: sqlite3.Connection,
    min_abstract_chars: int,
    created_since: str | None,
) -> dict[str, int]:
    query = """
        SELECT
            pc.category AS category,
            COUNT(*) AS paper_count
        FROM paper_categories pc
        JOIN papers p
            ON p.arxiv_id = pc.arxiv_id
        WHERE pc.is_primary = 1
          AND p.is_deleted = 0
          AND LENGTH(TRIM(p.abstract)) >= ?
    """
    params: list[Any] = [min_abstract_chars]
    if created_since:
        query += " AND p.created_date >= ?"
        params.append(created_since)
    query += """
        GROUP BY pc.category
        ORDER BY pc.category
    """
    rows = conn.execute(query, params).fetchall()
    return {row["category"]: int(row["paper_count"]) for row in rows}


def category_score(category: str, available_count: int) -> float:
    score = math.sqrt(available_count)
    if category in ML_CS_FOCUS_CATEGORIES:
        score *= 1.75
    elif category.startswith("cs."):
        score *= 1.15
    return score


def distribute_remaining(
    quotas: dict[str, int],
    capacities: dict[str, int],
    scores: dict[str, float],
    remaining: int,
) -> None:
    while remaining > 0:
        eligible = [category for category, capacity in capacities.items() if capacity > 0]
        if not eligible:
            break
        total_score = sum(scores[category] for category in eligible)
        if total_score <= 0:
            for category in eligible[:remaining]:
                quotas[category] += 1
                capacities[category] -= 1
                remaining -= 1
            continue

        raw_allocations: list[tuple[float, str, int]] = []
        granted = 0
        round_remaining = remaining
        for category in eligible:
            raw = round_remaining * scores[category] / total_score
            whole = min(capacities[category], int(math.floor(raw)))
            if whole > 0:
                quotas[category] += whole
                capacities[category] -= whole
                remaining -= whole
                granted += whole
            raw_allocations.append((raw - math.floor(raw), category, whole))

        if remaining == 0:
            break

        if granted == 0:
            raw_allocations.sort(key=lambda item: (-item[0], item[1]))
            granted_one = False
            for _, category, _ in raw_allocations:
                if capacities[category] <= 0:
                    continue
                quotas[category] += 1
                capacities[category] -= 1
                remaining -= 1
                granted_one = True
                if remaining == 0:
                    break
            if not granted_one:
                break


def build_umbrella_targets(
    counts: dict[str, int],
    capped_counts: dict[str, int],
    target_size: int,
) -> dict[str, int]:
    umbrella_counts = defaultdict(int)
    umbrella_capacities = defaultdict(int)
    for category, count in counts.items():
        umbrella = umbrella_for(category)
        umbrella_counts[umbrella] += count
        umbrella_capacities[umbrella] += capped_counts[category]

    non_top_umbrella_cap = int(target_size * NON_TOP_UMBRELLA_MAX_SHARE)
    umbrella_effective_capacities = {}
    for umbrella, capacity in umbrella_capacities.items():
        if umbrella in TOP_UMBRELLA_TARGET_SHARES:
            umbrella_effective_capacities[umbrella] = capacity
        else:
            umbrella_effective_capacities[umbrella] = min(capacity, non_top_umbrella_cap)

    targets = {umbrella: 0 for umbrella in umbrella_counts}
    for umbrella, share in TOP_UMBRELLA_TARGET_SHARES.items():
        desired = int(round(target_size * share))
        targets[umbrella] = min(desired, umbrella_effective_capacities.get(umbrella, 0))

    remaining = target_size - sum(targets.values())

    rest_umbrellas = sorted(umbrella for umbrella in umbrella_counts if umbrella not in TOP_UMBRELLA_TARGET_SHARES)
    rest_capacities = {
        umbrella: umbrella_effective_capacities[umbrella] - targets.get(umbrella, 0)
        for umbrella in rest_umbrellas
    }
    rest_scores = {
        umbrella: math.sqrt(umbrella_counts[umbrella]) for umbrella in rest_umbrellas
    }
    distribute_remaining(targets, rest_capacities, rest_scores, remaining)

    remaining = target_size - sum(targets.values())
    if remaining > 0:
        extra_capacities = {
            umbrella: umbrella_effective_capacities[umbrella] - targets.get(umbrella, 0)
            for umbrella in umbrella_counts
        }
        extra_scores = {
            umbrella: math.sqrt(umbrella_counts[umbrella]) for umbrella in umbrella_counts
        }
        distribute_remaining(targets, extra_capacities, extra_scores, remaining)

    return targets


def build_quotas(
    counts: dict[str, int],
    target_size: int,
    min_per_category: int,
    max_per_category: int,
) -> dict[str, int]:
    capped_counts = {category: min(count, max_per_category) for category, count in counts.items()}
    umbrella_targets = build_umbrella_targets(counts, capped_counts, target_size)
    quotas = {category: 0 for category in counts}

    umbrella_categories = defaultdict(list)
    for category in sorted(counts):
        umbrella_categories[umbrella_for(category)].append(category)

    for umbrella, categories in umbrella_categories.items():
        umbrella_target = umbrella_targets.get(umbrella, 0)
        reserved = sum(min(capped_counts[category], min_per_category) for category in categories)
        if reserved > umbrella_target:
            raise ValueError(
                f"Reserved minimum quota ({reserved}) for umbrella {umbrella} exceeds "
                f"its target allocation ({umbrella_target})."
            )

        capacities: dict[str, int] = {}
        scores: dict[str, float] = {}
        for category in categories:
            quotas[category] = min(capped_counts[category], min_per_category)
            capacities[category] = capped_counts[category] - quotas[category]
            scores[category] = category_score(category, counts[category])

        distribute_remaining(quotas, capacities, scores, umbrella_target - reserved)

    actual_total = sum(quotas.values())
    if actual_total < target_size:
        capacities = {
            category: capped_counts[category] - quotas[category] for category in counts
        }
        scores = {category: category_score(category, counts[category]) for category in counts}
        distribute_remaining(quotas, capacities, scores, target_size - actual_total)
        actual_total = sum(quotas.values())

    if actual_total != target_size:
        raise ValueError(f"Quota construction failed: expected {target_size}, got {actual_total}.")
    return quotas


def reservoir_sample_by_category(
    conn: sqlite3.Connection,
    quotas: dict[str, int],
    min_abstract_chars: int,
    created_since: str | None,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    rngs = {category: random.Random(stable_seed(seed, category)) for category in quotas}
    reservoirs: dict[str, list[dict[str, Any]]] = {category: [] for category in quotas}
    seen_counts = defaultdict(int)

    query = """
        SELECT
            pc.category AS primary_category,
            p.arxiv_id,
            p.title,
            p.abstract,
            p.created_date,
            p.updated_date
        FROM paper_categories pc
        JOIN papers p
            ON p.arxiv_id = pc.arxiv_id
        WHERE pc.is_primary = 1
          AND p.is_deleted = 0
          AND LENGTH(TRIM(p.abstract)) >= ?
    """
    params: list[Any] = [min_abstract_chars]
    if created_since:
        query += " AND p.created_date >= ?"
        params.append(created_since)
    cursor = conn.execute(query, params)

    total_scanned = 0
    for row in cursor:
        total_scanned += 1
        if total_scanned % 250_000 == 0:
            print(
                f"[sample] scanned {total_scanned:,} primary-category rows...",
                file=sys.stderr,
                flush=True,
            )
        category = str(row["primary_category"])
        quota = quotas.get(category, 0)
        if quota <= 0:
            continue

        seen_counts[category] += 1
        record = {
            "paper_id": row["arxiv_id"],
            "title": normalize_text(row["title"]),
            "abstract": normalize_text(row["abstract"]),
            "primary_category": category,
            "umbrella_category": umbrella_for(category),
            "created_date": row["created_date"],
            "updated_date": row["updated_date"],
        }

        bucket = reservoirs[category]
        if len(bucket) < quota:
            bucket.append(record)
            continue

        index = rngs[category].randrange(seen_counts[category])
        if index < quota:
            bucket[index] = record

    return reservoirs, dict(seen_counts)


def fetch_all_categories(conn: sqlite3.Connection, paper_ids: list[str]) -> dict[str, list[str]]:
    categories_by_paper: dict[str, list[str]] = defaultdict(list)
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
            categories_by_paper[row["arxiv_id"]].append(row["category"])
    return categories_by_paper


def flatten_reservoirs(reservoirs: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for category in sorted(reservoirs):
        records.extend(
            sorted(reservoirs[category], key=lambda record: str(record["paper_id"]))
        )
    return records


def write_outputs(
    output_dir: Path,
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    breakdown_rows: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"arxiv_abstract_corpus_weighted_{manifest['exported_records']}"

    jsonl_path = output_dir / f"{stem}.jsonl"
    manifest_path = output_dir / f"{stem}_manifest.json"
    breakdown_path = output_dir / f"{stem}_breakdown.csv"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with breakdown_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "primary_category",
                "umbrella_category",
                "available_count",
                "quota",
                "selected_count",
            ],
        )
        writer.writeheader()
        writer.writerows(breakdown_rows)


def build_manifest(
    counts: dict[str, int],
    quotas: dict[str, int],
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected_by_category = defaultdict(int)
    selected_by_umbrella = defaultdict(int)
    available_by_umbrella = defaultdict(int)

    for category, count in counts.items():
        available_by_umbrella[umbrella_for(category)] += count

    for record in records:
        selected_by_category[str(record["primary_category"])] += 1
        selected_by_umbrella[str(record["umbrella_category"])] += 1

    breakdown_rows = []
    for category in sorted(counts):
        breakdown_rows.append(
            {
                "primary_category": category,
                "umbrella_category": umbrella_for(category),
                "available_count": counts[category],
                "quota": quotas.get(category, 0),
                "selected_count": selected_by_category.get(category, 0),
            }
        )

    manifest = {
        "exported_records": len(records),
        "target_size": args.target_size,
        "min_per_category": args.min_per_category,
        "max_per_category": args.max_per_category,
        "min_abstract_chars": args.min_abstract_chars,
        "created_since": args.created_since,
        "seed": args.seed,
        "category_count": len(counts),
        "top_umbrella_target_shares": TOP_UMBRELLA_TARGET_SHARES,
        "non_top_umbrella_max_share": NON_TOP_UMBRELLA_MAX_SHARE,
        "available_by_umbrella": dict(sorted(available_by_umbrella.items())),
        "selected_by_umbrella": dict(sorted(selected_by_umbrella.items())),
        "quotas_by_category": {category: quotas[category] for category in sorted(quotas)},
    }
    return manifest, breakdown_rows


def main() -> None:
    args = parse_args()
    conn = connect(args.db)
    try:
        print("[export] loading primary-category counts...", file=sys.stderr, flush=True)
        counts = fetch_primary_category_counts(
            conn,
            args.min_abstract_chars,
            args.created_since,
        )
        print(
            f"[export] found {len(counts)} primary categories; building quotas for {args.target_size:,} abstracts...",
            file=sys.stderr,
            flush=True,
        )
        quotas = build_quotas(
            counts,
            args.target_size,
            args.min_per_category,
            args.max_per_category,
        )
        print("[export] sampling abstracts by primary category...", file=sys.stderr, flush=True)
        reservoirs, seen_counts = reservoir_sample_by_category(
            conn,
            quotas,
            args.min_abstract_chars,
            args.created_since,
            args.seed,
        )
        print("[export] fetching secondary category metadata...", file=sys.stderr, flush=True)
        records = flatten_reservoirs(reservoirs)
        categories_by_paper = fetch_all_categories(
            conn,
            [str(record["paper_id"]) for record in records],
        )
        for record in records:
            record["categories"] = categories_by_paper.get(str(record["paper_id"]), [])

        manifest, breakdown_rows = build_manifest(counts, quotas, records, args)
        manifest["scanned_records_by_category"] = {
            category: seen_counts.get(category, 0) for category in sorted(counts)
        }

        print("[export] writing jsonl, manifest, and breakdown files...", file=sys.stderr, flush=True)
        write_outputs(Path(args.output_dir), records, manifest, breakdown_rows)
        print(
            f"[export] wrote {len(records):,} records to {args.output_dir}",
            file=sys.stderr,
            flush=True,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
