#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import synthetic_summary_runtime as runtime

DEFAULT_RUN_ROOT = "hf_bucket/runs/recent_3y_balanced_incremental_7k_v1"
DEFAULT_EXCLUDE_JSONL = (
    "hf_bucket/runs/recent_3y_balanced_5000_pilot_v2/sample_source_rows.jsonl"
)
DEFAULT_JSON_OUTPUT_NAME = "aggregate_summary.json"
DEFAULT_TEXT_OUTPUT_NAME = "aggregate_summary.txt"

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
            "Summarize incremental synthetic-summary shard runs without merging "
            "them into a release artifact."
        )
    )
    parser.add_argument(
        "--run-root",
        default=DEFAULT_RUN_ROOT,
        help="Root directory that contains shard_* run directories.",
    )
    parser.add_argument(
        "--exclude-jsonl",
        default=DEFAULT_EXCLUDE_JSONL,
        help="Reviewed source rows that should not overlap with shard selections.",
    )
    parser.add_argument(
        "--json-output-name",
        default=DEFAULT_JSON_OUTPUT_NAME,
        help="Filename for the aggregate JSON summary written under the run root.",
    )
    parser.add_argument(
        "--text-output-name",
        default=DEFAULT_TEXT_OUTPUT_NAME,
        help="Filename for the readable text summary written under the run root.",
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


def load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return runtime.load_jsonl(path)


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_shard(shard_dir: Path) -> dict[str, Any]:
    manifest = load_json_if_exists(shard_dir / "run_manifest.json")
    sample_rows = load_jsonl_if_exists(shard_dir / "sample_source_rows.jsonl")
    generation_rows = load_jsonl_if_exists(shard_dir / "generation_outputs.jsonl")
    review_rows = load_jsonl_if_exists(shard_dir / "review_outputs.jsonl")
    training_rows = load_jsonl_if_exists(shard_dir / "training_rows.jsonl")
    dropped_rows = load_jsonl_if_exists(shard_dir / "dropped_rows.jsonl")
    return {
        "shard_dir": shard_dir.name,
        "mode": manifest.get("mode", "missing"),
        "selected_rows": len(sample_rows),
        "generated_rows": len(generation_rows),
        "reviewed_rows": len(review_rows),
        "kept_rows": len(training_rows),
        "dropped_rows": len(dropped_rows),
        "sample_rows": sample_rows,
        "training_rows": training_rows,
    }


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    json_output_path = run_root / args.json_output_name
    text_output_path = run_root / args.text_output_name

    excluded_rows = load_jsonl_if_exists(Path(args.exclude_jsonl))
    excluded_ids = {str(row["paper_id"]) for row in excluded_rows}
    shard_dirs = sorted(
        path for path in run_root.glob("shard_*") if path.is_dir()
    )

    shard_summaries: list[dict[str, Any]] = []
    sample_id_to_shards: dict[str, set[str]] = defaultdict(set)
    overlap_with_excluded: set[str] = set()
    selected_umbrella_counts: Counter[str] = Counter()
    kept_umbrella_counts: Counter[str] = Counter()
    kept_primary_counts: Counter[str] = Counter()

    totals = Counter(
        {
            "selected_rows": 0,
            "generated_rows": 0,
            "reviewed_rows": 0,
            "kept_rows": 0,
            "dropped_rows": 0,
        }
    )

    for shard_dir in shard_dirs:
        shard_summary = summarize_shard(shard_dir)
        shard_summaries.append(
            {
                key: value
                for key, value in shard_summary.items()
                if key not in {"sample_rows", "training_rows"}
            }
        )
        totals.update(
            {
                "selected_rows": shard_summary["selected_rows"],
                "generated_rows": shard_summary["generated_rows"],
                "reviewed_rows": shard_summary["reviewed_rows"],
                "kept_rows": shard_summary["kept_rows"],
                "dropped_rows": shard_summary["dropped_rows"],
            }
        )

        for row in shard_summary["sample_rows"]:
            paper_id = str(row["paper_id"])
            sample_id_to_shards[paper_id].add(shard_dir.name)
            umbrella = str(row.get("umbrella_category") or umbrella_for(str(row["primary_category"])))
            selected_umbrella_counts[umbrella] += 1
            if paper_id in excluded_ids:
                overlap_with_excluded.add(paper_id)

        for row in shard_summary["training_rows"]:
            umbrella = umbrella_for(str(row["primary_category"]))
            kept_umbrella_counts[umbrella] += 1
            kept_primary_counts[str(row["primary_category"])] += 1

    duplicate_sample_ids = {
        paper_id: sorted(shards)
        for paper_id, shards in sample_id_to_shards.items()
        if len(shards) > 1
    }

    summary = {
        "run_root": str(run_root),
        "exclude_jsonl": args.exclude_jsonl,
        "shard_count": len(shard_dirs),
        "totals": dict(totals),
        "selected_umbrella_counts": dict(sorted(selected_umbrella_counts.items())),
        "kept_umbrella_counts": dict(sorted(kept_umbrella_counts.items())),
        "top_kept_primary_categories": kept_primary_counts.most_common(20),
        "duplicate_sample_id_count": len(duplicate_sample_ids),
        "duplicate_sample_ids": dict(sorted(duplicate_sample_ids.items())),
        "overlap_with_excluded_count": len(overlap_with_excluded),
        "overlap_with_excluded_ids": sorted(overlap_with_excluded),
        "shards": shard_summaries,
    }

    json_output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"Run root: {run_root}",
        f"Shards found: {len(shard_dirs)}",
        "",
        "Totals:",
        f"- selected_rows: {totals['selected_rows']}",
        f"- generated_rows: {totals['generated_rows']}",
        f"- reviewed_rows: {totals['reviewed_rows']}",
        f"- kept_rows: {totals['kept_rows']}",
        f"- dropped_rows: {totals['dropped_rows']}",
        "",
        f"Duplicate sample paper_ids: {len(duplicate_sample_ids)}",
        f"Overlap with excluded reviewed set: {len(overlap_with_excluded)}",
        "",
        "Selected umbrella counts:",
    ]
    lines.extend(
        f"- {umbrella}: {count}"
        for umbrella, count in sorted(selected_umbrella_counts.items())
    )
    lines.append("")
    lines.append("Kept umbrella counts:")
    lines.extend(
        f"- {umbrella}: {count}"
        for umbrella, count in sorted(kept_umbrella_counts.items())
    )
    lines.append("")
    lines.append("Top kept primary categories:")
    lines.extend(f"- {category}: {count}" for category, count in kept_primary_counts.most_common(20))
    text_output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[summarize-runs] wrote {json_output_path}")
    print(f"[summarize-runs] wrote {text_output_path}")


if __name__ == "__main__":
    main()
