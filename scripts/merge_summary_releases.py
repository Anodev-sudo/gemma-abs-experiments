#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import synthetic_summary_runtime as runtime

DEFAULT_OUTPUT_DIR = "hf_dataset/releases/recent_3y_balanced_10746_v1"

REQUIRED_ROW_FIELDS = (
    "paper_id",
    "title",
    "abstract",
    "summary",
    "primary_category",
    "created_date",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge promoted summary releases into one deterministic training_rows.jsonl "
            "artifact, failing on any paper_id overlap."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help=(
            "Release directory or training_rows.jsonl file to merge. Pass once per "
            "source release."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Merged release directory to write.",
    )
    return parser.parse_args()


def resolve_release(source_arg: str) -> tuple[Path, Path]:
    source_path = Path(source_arg)
    if source_path.is_dir():
        training_rows_path = source_path / "training_rows.jsonl"
        release_dir = source_path
    else:
        training_rows_path = source_path
        release_dir = source_path.parent
    if not training_rows_path.exists():
        raise FileNotFoundError(f"Could not find training_rows.jsonl at {training_rows_path}")
    return release_dir, training_rows_path


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def validate_source_rows(rows: list[dict[str, Any]], release_dir: Path) -> None:
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        missing = [field for field in REQUIRED_ROW_FIELDS if not row.get(field)]
        if missing:
            raise ValueError(
                f"{release_dir} row {index} is missing required fields: {', '.join(missing)}"
            )
        paper_id = str(row["paper_id"])
        if paper_id in seen_ids:
            raise ValueError(f"{release_dir} contains duplicate paper_id {paper_id}")
        seen_ids.add(paper_id)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def consistency_summary(values: set[str]) -> dict[str, Any]:
    ordered = sorted(values)
    return {"consistent": len(ordered) == 1, "values": ordered}


def main() -> None:
    args = parse_args()
    if len(args.source) < 2:
        raise ValueError("Pass at least two --source releases to merge.")

    resolved_sources = [resolve_release(source_arg) for source_arg in args.source]
    output_dir = Path(args.output_dir)

    source_release_summaries: list[dict[str, Any]] = []
    merged_rows_by_id: dict[str, dict[str, Any]] = {}
    paper_id_to_release: dict[str, str] = {}
    overlapping_paper_ids: list[str] = []

    generator_prompt_versions: set[str] = set()
    review_prompt_versions: set[str] = set()
    teacher_models: set[str] = set()
    source_splits: set[str] = set()
    primary_category_counts: Counter[str] = Counter()

    for release_dir, training_rows_path in resolved_sources:
        rows = runtime.load_jsonl(training_rows_path)
        validate_source_rows(rows, release_dir)
        release_manifest = load_json_if_exists(release_dir / "run_manifest.json")

        source_release_summaries.append(
            {
                "release_dir": str(release_dir),
                "training_rows_path": str(training_rows_path),
                "row_count": len(rows),
                "run_manifest_path": str(release_dir / "run_manifest.json"),
            }
        )

        for row in rows:
            paper_id = str(row["paper_id"])
            if paper_id in merged_rows_by_id:
                overlapping_paper_ids.append(paper_id)
                continue
            merged_rows_by_id[paper_id] = row
            paper_id_to_release[paper_id] = str(release_dir)
            generator_prompt_versions.add(str(row.get("generator_prompt_version", "")))
            review_prompt_versions.add(str(row.get("review_prompt_version", "")))
            teacher_models.add(str(row.get("teacher_model", "")))
            source_splits.add(str(row.get("source_split", "")))
            primary_category_counts[str(row["primary_category"])] += 1

        manifest_source_split = release_manifest.get("source_split")
        if manifest_source_split:
            source_splits.add(str(manifest_source_split))

    if overlapping_paper_ids:
        preview = sorted(set(overlapping_paper_ids))[:10]
        raise ValueError(
            f"paper_id overlap detected across releases: {preview!r} "
            f"(total overlaps={len(set(overlapping_paper_ids))})"
        )

    merged_rows = [merged_rows_by_id[paper_id] for paper_id in sorted(merged_rows_by_id)]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "training_rows.jsonl", merged_rows)

    manifest = {
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "mode": "merged_release",
        "output_dir": str(output_dir),
        "source_releases": source_release_summaries,
        "source_row_counts": {
            summary["release_dir"]: summary["row_count"]
            for summary in source_release_summaries
        },
        "merged_row_count": len(merged_rows),
        "paper_id_overlap_count": 0,
        "paper_id_overlap_ids": [],
        "required_row_fields": list(REQUIRED_ROW_FIELDS),
        "generator_prompt_version_check": consistency_summary(generator_prompt_versions),
        "review_prompt_version_check": consistency_summary(review_prompt_versions),
        "teacher_model_check": consistency_summary(teacher_models),
        "source_split_lineage": sorted(source_splits),
        "deterministic_ordering": "Rows are sorted by paper_id ascending.",
        "top_primary_categories": primary_category_counts.most_common(20),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"[merge-summary-releases] wrote {output_dir / 'training_rows.jsonl'}")
    print(f"[merge-summary-releases] wrote {output_dir / 'run_manifest.json'}")
    print(f"[merge-summary-releases] merged rows: {len(merged_rows)}")


if __name__ == "__main__":
    main()
