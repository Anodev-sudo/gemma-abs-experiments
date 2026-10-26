#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHYSICS_PREFIXES = (
    "astro-ph",
    "cond-mat",
    "gr-qc",
    "hep-ex",
    "hep-lat",
    "hep-ph",
    "hep-th",
    "math-ph",
    "nucl-ex",
    "nucl-th",
    "physics.",
    "quant-ph",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drop known low-quality math rows from a natural-language release and "
            "backfill with explicit replacement rows."
        )
    )
    parser.add_argument("--release-dir", required=True, help="Release directory to prune.")
    parser.add_argument(
        "--replacement-jsonl",
        required=True,
        help="JSONL containing replacement training rows.",
    )
    parser.add_argument(
        "--drop-paper-id",
        action="append",
        default=[],
        help="Paper ID to remove. Repeat for multiple rows.",
    )
    parser.add_argument(
        "--backfill-paper-id",
        action="append",
        default=[],
        help="Replacement paper ID to append. Repeat in desired order.",
    )
    parser.add_argument(
        "--reason",
        default="manual math natural-language quality prune",
        help="Short reason recorded in the manifest.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output release directory. Defaults to overwriting --release-dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def summary_text(row: dict[str, Any]) -> str:
    return str(row.get("summary", row.get("generated_summary", ""))).strip()


def summary_word_count(row: dict[str, Any]) -> int:
    return len([part for part in summary_text(row).split() if part])


def collapsed_umbrella(primary_category: str) -> str:
    if primary_category.startswith("cs."):
        return "cs"
    if primary_category.startswith("math."):
        return "math"
    if primary_category.startswith(PHYSICS_PREFIXES):
        return "physics"
    return "other"


def row_brief(row: dict[str, Any]) -> dict[str, str]:
    return {
        "paper_id": str(row.get("paper_id", "")),
        "primary_category": str(row.get("primary_category", "")),
        "title": str(row.get("title", "")),
        "summary": summary_text(row),
        "source_split": str(row.get("source_split", "")),
    }


def component_paths(manifest: dict[str, Any]) -> list[tuple[str, Path]]:
    components: list[tuple[str, Path]] = []
    for component in manifest.get("components", []):
        if not isinstance(component, dict):
            continue
        name = str(component.get("name", "")).strip()
        path = Path(str(component.get("path", "")).strip())
        if name and path:
            components.append((name, path))
    return components


def component_row_ids(path: Path) -> set[str]:
    training_rows = path if path.is_file() else path / "training_rows.jsonl"
    if not training_rows.exists():
        return set()
    return {str(row.get("paper_id", "")) for row in load_jsonl(training_rows)}


def recompute_component_counts(
    manifest: dict[str, Any],
    final_rows: list[dict[str, Any]],
    dropped_rows: list[dict[str, Any]],
) -> None:
    components = component_paths(manifest)
    if not components:
        return

    row_ids_by_component = {
        name: component_row_ids(path) for name, path in components
    }
    selected_counts: Counter[str] = Counter()
    for row in final_rows:
        paper_id = str(row.get("paper_id", ""))
        for name, ids in row_ids_by_component.items():
            if paper_id in ids:
                selected_counts[name] += 1
                break

    dropped_counts: Counter[str] = Counter()
    for row in dropped_rows:
        paper_id = str(row.get("paper_id", ""))
        for name, ids in row_ids_by_component.items():
            if paper_id in ids:
                dropped_counts[name] += 1
                break

    manifest["component_selected_counts"] = dict(selected_counts)
    component_counts = dict(manifest.get("component_counts", {}))
    for name, ids in row_ids_by_component.items():
        entry = dict(component_counts.get(name, {}))
        input_rows = int(entry.get("input_rows", len(ids)) or len(ids))
        added_rows = int(selected_counts.get(name, 0))
        skipped_duplicates = int(entry.get("skipped_duplicates", 0) or 0)
        skipped_math_quality = int(dropped_counts.get(name, 0))
        skipped_quota = max(
            0,
            input_rows - added_rows - skipped_duplicates - skipped_math_quality,
        )
        entry["input_rows"] = input_rows
        entry["added_rows"] = added_rows
        if skipped_math_quality:
            entry["skipped_math_quality"] = skipped_math_quality
        entry["skipped_duplicates"] = skipped_duplicates
        entry["skipped_quota"] = skipped_quota
        component_counts[name] = entry
    manifest["component_counts"] = component_counts


def update_manifest(
    manifest: dict[str, Any],
    *,
    output_dir: Path,
    final_rows: list[dict[str, Any]],
    dropped_rows: list[dict[str, Any]],
    backfill_rows: list[dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    primary_counts = Counter(str(row.get("primary_category", "")) for row in final_rows)
    umbrella_counts = Counter(
        collapsed_umbrella(str(row.get("primary_category", ""))) for row in final_rows
    )
    non_top_counts = {
        category: count
        for category, count in primary_counts.items()
        if not category.startswith(("cs.", "math.")) and collapsed_umbrella(category) != "physics"
    }
    manifest = dict(manifest)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["output_dir"] = str(output_dir)
    manifest["training_rows"] = len(final_rows)
    manifest["unique_paper_ids"] = len({str(row.get("paper_id", "")) for row in final_rows})
    manifest["max_summary_words"] = max((summary_word_count(row) for row in final_rows), default=0)
    manifest["over_14_word_summaries"] = sum(
        1 for row in final_rows if summary_word_count(row) > 14
    )
    manifest["empty_summaries"] = sum(1 for row in final_rows if not summary_text(row))
    manifest["collapsed_umbrella_counts"] = dict(sorted(umbrella_counts.items()))
    manifest["non_top_umbrella_counts"] = dict(sorted(non_top_counts.items()))
    manifest["top_primary_categories"] = primary_counts.most_common(25)
    manifest["math_quality_prune"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "dropped_rows": [row_brief(row) for row in dropped_rows],
        "backfill_rows": [row_brief(row) for row in backfill_rows],
    }
    recompute_component_counts(manifest, final_rows, dropped_rows)
    return manifest


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    output_dir = Path(args.output_dir) if args.output_dir else release_dir
    training_path = release_dir / "training_rows.jsonl"
    manifest_path = release_dir / "manifest.json"
    output_training_path = output_dir / "training_rows.jsonl"
    output_manifest_path = output_dir / "manifest.json"

    if not training_path.exists():
        raise FileNotFoundError(training_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    if output_dir.exists() and output_dir != release_dir and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite")
    if output_dir == release_dir and not args.overwrite:
        raise FileExistsError("Refusing to overwrite release in place without --overwrite")

    drop_ids = list(dict.fromkeys(args.drop_paper_id))
    backfill_ids = list(dict.fromkeys(args.backfill_paper_id))
    if not drop_ids:
        raise ValueError("At least one --drop-paper-id is required")
    if len(backfill_ids) != len(drop_ids):
        raise ValueError("--backfill-paper-id count must match --drop-paper-id count")
    if set(drop_ids).intersection(backfill_ids):
        raise ValueError("Drop and backfill paper IDs must be disjoint")

    rows = load_jsonl(training_path)
    replacements = load_jsonl(Path(args.replacement_jsonl))
    manifest = load_json(manifest_path)
    rows_by_id = {str(row.get("paper_id", "")): row for row in rows}
    replacement_by_id = {str(row.get("paper_id", "")): row for row in replacements}

    missing_drops = [paper_id for paper_id in drop_ids if paper_id not in rows_by_id]
    if missing_drops:
        raise ValueError(f"Drop IDs not present in release: {missing_drops}")
    missing_backfills = [
        paper_id for paper_id in backfill_ids if paper_id not in replacement_by_id
    ]
    if missing_backfills:
        raise ValueError(f"Backfill IDs not present in replacement source: {missing_backfills}")
    existing_backfills = [paper_id for paper_id in backfill_ids if paper_id in rows_by_id]
    if existing_backfills:
        raise ValueError(f"Backfill IDs already present in release: {existing_backfills}")

    drop_set = set(drop_ids)
    dropped_rows = [rows_by_id[paper_id] for paper_id in drop_ids]
    kept_rows = [row for row in rows if str(row.get("paper_id", "")) not in drop_set]
    backfill_rows = [replacement_by_id[paper_id] for paper_id in backfill_ids]
    final_rows = kept_rows + backfill_rows
    final_ids = [str(row.get("paper_id", "")) for row in final_rows]
    duplicate_ids = [paper_id for paper_id, count in Counter(final_ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate paper IDs after prune: {duplicate_ids}")
    if len(final_rows) != len(rows):
        raise ValueError(f"Row count changed from {len(rows)} to {len(final_rows)}")

    updated_manifest = update_manifest(
        manifest,
        output_dir=output_dir,
        final_rows=final_rows,
        dropped_rows=dropped_rows,
        backfill_rows=backfill_rows,
        reason=args.reason,
    )
    if updated_manifest["training_rows"] != updated_manifest["unique_paper_ids"]:
        raise ValueError("Final rows are not unique")
    if updated_manifest["empty_summaries"]:
        raise ValueError("Final release has empty summaries")
    if updated_manifest["over_14_word_summaries"]:
        raise ValueError("Final release has over-14-word summaries")

    write_jsonl(output_training_path, final_rows)
    write_json(output_manifest_path, updated_manifest)
    print(f"[math-release-prune] dropped {len(dropped_rows)} rows")
    print(f"[math-release-prune] backfilled {len(backfill_rows)} rows")
    print(f"[math-release-prune] wrote {output_training_path}")


if __name__ == "__main__":
    main()
