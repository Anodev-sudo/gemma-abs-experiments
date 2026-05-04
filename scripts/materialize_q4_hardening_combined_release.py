#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

try:
    from scripts.synthetic_summary_runtime import load_jsonl
except ImportError:  # pragma: no cover - script execution from scripts/
    from synthetic_summary_runtime import load_jsonl


DEFAULT_STRICT_RELEASE = (
    "hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_distill_6000_v1"
)
DEFAULT_KIMI_RELEASE = (
    "hf_dataset/releases/"
    "recent_3y_balanced_10746_q4_kimi_correction_salvage_no_reasoning_v1"
)
DEFAULT_OUTPUT_DIR = (
    "hf_dataset/releases/recent_3y_balanced_10746_q4_hardening_combined_v1"
)

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
            "Materialize the q4 hardening combined release, preferring "
            "Qwen-accepted Kimi corrections on paper_id overlap."
        )
    )
    parser.add_argument("--strict-release", default=DEFAULT_STRICT_RELEASE)
    parser.add_argument("--kimi-release", default=DEFAULT_KIMI_RELEASE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-row-count", type=int, default=148)
    parser.add_argument("--expected-overlap-count", type=int, default=18)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_training_rows(source: str) -> tuple[Path, Path]:
    source_path = Path(source)
    if source_path.is_dir():
        release_dir = source_path
        training_rows_path = source_path / "training_rows.jsonl"
    else:
        release_dir = source_path.parent
        training_rows_path = source_path
    if not training_rows_path.exists():
        raise FileNotFoundError(f"Missing training rows at {training_rows_path}")
    return release_dir, training_rows_path


def validate_rows(rows: list[dict[str, Any]], release_dir: Path) -> None:
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        missing = [field for field in REQUIRED_ROW_FIELDS if not row.get(field)]
        if missing:
            raise ValueError(
                f"{release_dir} row {index} is missing required fields: "
                + ", ".join(missing)
            )
        paper_id = str(row["paper_id"])
        if paper_id in seen_ids:
            raise ValueError(f"{release_dir} contains duplicate paper_id {paper_id}")
        seen_ids.add(paper_id)


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
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def annotate_row(row: dict[str, Any], source_name: str, release_dir: Path) -> dict[str, Any]:
    copied = dict(row)
    copied["q4_hardening_combined_source"] = source_name
    copied["q4_hardening_combined_source_release"] = str(release_dir)
    copied["source_split"] = "q4_hardening_combined_v1"
    return copied


def materialize_combined_rows(
    *,
    strict_rows: list[dict[str, Any]],
    strict_release_dir: Path,
    kimi_rows: list[dict[str, Any]],
    kimi_release_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    overlap_rows: list[dict[str, Any]] = []

    for row in strict_rows:
        rows_by_id[str(row["paper_id"])] = annotate_row(row, "strict_6k_qa", strict_release_dir)

    for row in kimi_rows:
        paper_id = str(row["paper_id"])
        kimi_row = annotate_row(row, "kimi_qwen_verified", kimi_release_dir)
        if paper_id in rows_by_id:
            overlap_rows.append(
                {
                    "paper_id": paper_id,
                    "chosen_source": "kimi_qwen_verified",
                    "replaced_source": rows_by_id[paper_id].get(
                        "q4_hardening_combined_source"
                    ),
                    "strict_summary": rows_by_id[paper_id].get("summary", ""),
                    "kimi_summary": kimi_row.get("summary", ""),
                }
            )
        rows_by_id[paper_id] = kimi_row

    return [rows_by_id[paper_id] for paper_id in sorted(rows_by_id)], overlap_rows


def summarize(
    *,
    strict_rows: list[dict[str, Any]],
    kimi_rows: list[dict[str, Any]],
    combined_rows: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "strict_input_rows": len(strict_rows),
        "kimi_input_rows": len(kimi_rows),
        "overlap_count": len(overlap_rows),
        "combined_rows": len(combined_rows),
        "source_counts": dict(
            Counter(row.get("q4_hardening_combined_source", "") for row in combined_rows)
        ),
        "selection_bucket_counts": dict(
            Counter(
                row.get("q4_hardening_selection_bucket") or row.get("selection_bucket")
                for row in combined_rows
            )
        ),
        "q4_issue_tag_counts": dict(
            Counter(tag for row in combined_rows for tag in row.get("q4_issue_tags", []))
        ),
    }


def main() -> None:
    args = parse_args()
    strict_release_dir, strict_training_path = resolve_training_rows(args.strict_release)
    kimi_release_dir, kimi_training_path = resolve_training_rows(args.kimi_release)
    strict_rows = load_jsonl(strict_training_path)
    kimi_rows = load_jsonl(kimi_training_path)
    validate_rows(strict_rows, strict_release_dir)
    validate_rows(kimi_rows, kimi_release_dir)

    combined_rows, overlap_rows = materialize_combined_rows(
        strict_rows=strict_rows,
        strict_release_dir=strict_release_dir,
        kimi_rows=kimi_rows,
        kimi_release_dir=kimi_release_dir,
    )

    if args.expected_row_count is not None and len(combined_rows) != args.expected_row_count:
        raise ValueError(
            f"Combined row count is {len(combined_rows)}, expected "
            f"{args.expected_row_count}"
        )
    if (
        args.expected_overlap_count is not None
        and len(overlap_rows) != args.expected_overlap_count
    ):
        raise ValueError(
            f"Overlap count is {len(overlap_rows)}, expected "
            f"{args.expected_overlap_count}"
        )

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "training_rows.jsonl", combined_rows)
    write_jsonl(output_dir / "overlap_rows.jsonl", overlap_rows)
    manifest = {
        "created_at": utc_now_iso(),
        "mode": "q4_hardening_combined_v1",
        "output_dir": str(output_dir),
        "strict_release": str(strict_release_dir),
        "strict_training_rows": str(strict_training_path),
        "kimi_release": str(kimi_release_dir),
        "kimi_training_rows": str(kimi_training_path),
        "overlap_policy": "Prefer kimi_qwen_verified rows on duplicate paper_id.",
        "deterministic_ordering": "Rows are sorted by paper_id ascending.",
        "summary": summarize(
            strict_rows=strict_rows,
            kimi_rows=kimi_rows,
            combined_rows=combined_rows,
            overlap_rows=overlap_rows,
        ),
        "outputs": {
            "training_rows": str(output_dir / "training_rows.jsonl"),
            "overlap_rows": str(output_dir / "overlap_rows.jsonl"),
            "run_manifest": str(output_dir / "run_manifest.json"),
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
