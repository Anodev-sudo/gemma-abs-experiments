#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import synthetic_summary_runtime as runtime

DEFAULT_INPUT_JSONL = (
    "hf_dataset/corpora/recent_3y_balanced/"
    "arxiv_abstract_corpus_weighted_20000.jsonl"
)
DEFAULT_EXCLUDE_JSONL = (
    "hf_bucket/runs/recent_3y_balanced_5000_pilot_v2/sample_source_rows.jsonl"
)
DEFAULT_OUTPUT_DIR = "hf_dataset/corpora/recent_3y_balanced_incremental_7k_v1"
DEFAULT_TARGET_SIZE = 7000
DEFAULT_SEED = 20260422
EXPECTED_TARGET_QUOTAS_7000 = {
    "cs": 2635,
    "econ": 177,
    "eess": 365,
    "math": 1095,
    "nlin": 102,
    "other": 10,
    "physics": 1959,
    "q-bio": 211,
    "q-fin": 130,
    "stat": 316,
}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic incremental summary source set by excluding "
            "already reviewed papers and preserving the reviewed umbrella mix."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        default=DEFAULT_INPUT_JSONL,
        help="Path to the balanced source corpus JSONL.",
    )
    parser.add_argument(
        "--exclude-jsonl",
        default=DEFAULT_EXCLUDE_JSONL,
        help="Path to the already reviewed source rows JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the new incremental source artifact.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        help="Exact number of unseen source rows to select.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic selection seed.",
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


def stable_rank(seed: int, paper_id: str) -> tuple[int, str]:
    digest = hashlib.blake2b(
        f"{seed}:{paper_id}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big"), paper_id


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return runtime.load_jsonl(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    runtime.write_jsonl(path, rows)


def require_unique_ids(rows: list[dict[str, Any]], *, label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        paper_id = str(row.get("paper_id", ""))
        if not paper_id:
            raise ValueError(f"{label} contains a row with an empty paper_id.")
        if paper_id in seen:
            duplicates.add(paper_id)
        seen.add(paper_id)
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:10])
        raise ValueError(f"{label} contains duplicate paper_id values: {preview}")


def reference_umbrella_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        umbrella = str(row.get("umbrella_category", "")).strip()
        if not umbrella:
            umbrella = umbrella_for(str(row.get("primary_category", "")))
        counts[umbrella] += 1
    return counts


def scale_umbrella_counts(
    counts: Counter[str],
    *,
    target_size: int,
) -> dict[str, int]:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("Reference reviewed set is empty; cannot scale umbrella counts.")

    scaled_floors: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for umbrella in sorted(counts):
        scaled = counts[umbrella] * target_size / total
        floor_value = math.floor(scaled)
        scaled_floors[umbrella] = floor_value
        remainders.append((scaled - floor_value, umbrella))

    remaining = target_size - sum(scaled_floors.values())
    if remaining < 0:
        raise ValueError("Scaled umbrella allocation underflowed unexpectedly.")
    for _, umbrella in sorted(remainders, key=lambda item: (-item[0], item[1]))[:remaining]:
        scaled_floors[umbrella] += 1

    if sum(scaled_floors.values()) != target_size:
        raise ValueError("Scaled umbrella counts did not sum to the requested target size.")
    return dict(sorted(scaled_floors.items()))


def validate_expected_quotas(
    *,
    target_size: int,
    computed: dict[str, int],
) -> None:
    if target_size != DEFAULT_TARGET_SIZE:
        return
    if computed != EXPECTED_TARGET_QUOTAS_7000:
        raise ValueError(
            "Computed umbrella quotas do not match the expected fixed 7k quotas. "
            f"Expected {EXPECTED_TARGET_QUOTAS_7000}, got {computed}."
        )


def select_unseen_rows(
    *,
    input_rows: list[dict[str, Any]],
    excluded_ids: set[str],
    target_quotas: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    unseen_by_umbrella: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in input_rows:
        paper_id = str(row["paper_id"])
        if paper_id in excluded_ids:
            continue
        umbrella = str(row.get("umbrella_category", "")).strip()
        if not umbrella:
            umbrella = umbrella_for(str(row.get("primary_category", "")))
        row = dict(row)
        row["umbrella_category"] = umbrella
        unseen_by_umbrella[umbrella].append(row)

    selected_rows: list[dict[str, Any]] = []
    for umbrella, quota in target_quotas.items():
        pool = unseen_by_umbrella.get(umbrella, [])
        if len(pool) < quota:
            raise ValueError(
                f"Umbrella {umbrella} only has {len(pool)} unseen rows available, "
                f"but quota requires {quota}."
            )
        ordered_pool = sorted(
            pool,
            key=lambda row: stable_rank(seed, str(row["paper_id"])),
        )
        selected_rows.extend(ordered_pool[:quota])

    overlap = [row["paper_id"] for row in selected_rows if row["paper_id"] in excluded_ids]
    if overlap:
        preview = ", ".join(sorted(str(item) for item in overlap[:10]))
        raise ValueError(f"Selected rows still overlap with excluded paper_ids: {preview}")

    selected_rows = sorted(selected_rows, key=lambda row: str(row["paper_id"]))
    if len(selected_rows) != sum(target_quotas.values()):
        raise ValueError(
            "Selected row count does not match the requested umbrella quota total."
        )
    return selected_rows


def build_manifest(
    *,
    args: argparse.Namespace,
    input_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    target_quotas: dict[str, int],
) -> dict[str, Any]:
    excluded_ids = {str(row["paper_id"]) for row in reference_rows}
    unseen_rows = [row for row in input_rows if str(row["paper_id"]) not in excluded_ids]
    selected_by_umbrella = Counter(str(row["umbrella_category"]) for row in selected_rows)
    unseen_by_umbrella = Counter(
        str(row.get("umbrella_category") or umbrella_for(str(row["primary_category"])))
        for row in unseen_rows
    )
    selected_by_primary = Counter(str(row["primary_category"]) for row in selected_rows)
    return {
        "input_jsonl": args.input_jsonl,
        "exclude_jsonl": args.exclude_jsonl,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "target_size": args.target_size,
        "input_row_count": len(input_rows),
        "excluded_reviewed_rows": len(reference_rows),
        "unseen_remainder_rows": len(unseen_rows),
        "reference_reviewed_umbrella_counts": dict(
            sorted(reference_umbrella_counts(reference_rows).items())
        ),
        "target_umbrella_counts": target_quotas,
        "available_unseen_umbrella_counts": dict(sorted(unseen_by_umbrella.items())),
        "selected_umbrella_counts": dict(sorted(selected_by_umbrella.items())),
        "selected_primary_category_counts": dict(sorted(selected_by_primary.items())),
        "selection_order": "paper_id_sorted_after_stable_seed_membership_selection",
        "selection_policy": (
            "exclude all reviewed paper_ids, scale the reviewed umbrella mix with "
            "largest remainder rounding, then choose unseen rows within each umbrella "
            "by stable seeded paper_id ranking"
        ),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    exclude_path = Path(args.exclude_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[prepare-source] loading balanced source corpus...", flush=True)
    input_rows = load_jsonl(input_path)
    print("[prepare-source] loading reviewed exclusion set...", flush=True)
    reference_rows = load_jsonl(exclude_path)

    require_unique_ids(input_rows, label="input corpus")
    require_unique_ids(reference_rows, label="reviewed exclusion set")

    excluded_ids = {str(row["paper_id"]) for row in reference_rows}
    reference_counts = reference_umbrella_counts(reference_rows)
    target_quotas = scale_umbrella_counts(reference_counts, target_size=args.target_size)
    validate_expected_quotas(target_size=args.target_size, computed=target_quotas)

    print(
        f"[prepare-source] selecting {args.target_size} unseen rows with seed {args.seed}...",
        flush=True,
    )
    selected_rows = select_unseen_rows(
        input_rows=input_rows,
        excluded_ids=excluded_ids,
        target_quotas=target_quotas,
        seed=args.seed,
    )
    require_unique_ids(selected_rows, label="selected source rows")

    source_rows_path = output_dir / "source_rows.jsonl"
    manifest_path = output_dir / "manifest.json"
    write_jsonl(source_rows_path, selected_rows)
    manifest = build_manifest(
        args=args,
        input_rows=input_rows,
        reference_rows=reference_rows,
        selected_rows=selected_rows,
        target_quotas=target_quotas,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[prepare-source] wrote {len(selected_rows)} rows to {source_rows_path}",
        flush=True,
    )
    print(
        f"[prepare-source] wrote manifest to {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
