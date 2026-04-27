#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import synthetic_summary_runtime as runtime

DEFAULT_RUN_ROOT = "hf_bucket/runs/recent_3y_balanced_incremental_7k_v1"
DEFAULT_OUTPUT_DIR = "hf_dataset/releases/recent_3y_balanced_incremental_6249_v1"
DEFAULT_EXCLUDE_JSONL = (
    "hf_bucket/runs/recent_3y_balanced_5000_pilot_v2/sample_source_rows.jsonl"
)

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
            "Promote a completed synthetic-summary shard run into a flat release "
            "directory with training_rows.jsonl and run_manifest.json."
        )
    )
    parser.add_argument(
        "--run-root",
        default=DEFAULT_RUN_ROOT,
        help="Root directory that contains shard_* run directories.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Release directory to write.",
    )
    parser.add_argument(
        "--exclude-jsonl",
        default=DEFAULT_EXCLUDE_JSONL,
        help=(
            "Reviewed source rows that the promoted release should not overlap "
            "with. Pass an empty string to skip this check."
        ),
    )
    parser.add_argument(
        "--expected-reviewed",
        type=int,
        default=None,
        help="Optional exact reviewed-row count check.",
    )
    parser.add_argument(
        "--expected-kept",
        type=int,
        default=None,
        help="Optional exact kept-row count check.",
    )
    parser.add_argument(
        "--expected-dropped",
        type=int,
        default=None,
        help="Optional exact dropped-row count check.",
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return runtime.load_jsonl(path)


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")
    return path


def summarize_shard(shard_dir: Path) -> dict[str, Any]:
    manifest = load_json(require_path(shard_dir / "run_manifest.json"))
    if manifest.get("mode") != "full_run":
        raise ValueError(
            f"{shard_dir} is not a completed full_run shard: mode={manifest.get('mode')!r}"
        )

    sample_rows = load_jsonl(require_path(shard_dir / "sample_source_rows.jsonl"))
    generation_rows = load_jsonl(require_path(shard_dir / "generation_outputs.jsonl"))
    review_rows = load_jsonl(require_path(shard_dir / "review_outputs.jsonl"))
    training_rows = load_jsonl(require_path(shard_dir / "training_rows.jsonl"))
    dropped_rows = load_jsonl(require_path(shard_dir / "dropped_rows.jsonl"))

    expected_counts = {
        "selected_rows": int(manifest.get("selected_row_count", len(sample_rows))),
        "generated_rows": int(manifest.get("generated_rows", len(generation_rows))),
        "reviewed_rows": int(manifest.get("reviewed_rows", len(review_rows))),
        "kept_rows": int(manifest.get("final_training_rows", len(training_rows))),
        "dropped_rows": int(manifest.get("dropped_rows", len(dropped_rows))),
    }
    actual_counts = {
        "selected_rows": len(sample_rows),
        "generated_rows": len(generation_rows),
        "reviewed_rows": len(review_rows),
        "kept_rows": len(training_rows),
        "dropped_rows": len(dropped_rows),
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"{shard_dir} file counts do not match its run_manifest.json: "
            f"actual={actual_counts} expected={expected_counts}"
        )

    return {
        "manifest": manifest,
        "sample_rows": sample_rows,
        "generation_rows": generation_rows,
        "review_rows": review_rows,
        "training_rows": training_rows,
        "dropped_rows": dropped_rows,
    }


def assert_optional_exact(label: str, actual: int, expected: int | None) -> None:
    if expected is None:
        return
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected}, found {actual}")


def collect_single_value(name: str, manifests: list[dict[str, Any]]) -> Any:
    values = {manifest.get(name) for manifest in manifests}
    if len(values) != 1:
        raise ValueError(f"Inconsistent {name} across shards: {sorted(values)!r}")
    return values.pop()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def build_manifest(
    *,
    args: argparse.Namespace,
    run_root: Path,
    output_dir: Path,
    shard_dirs: list[Path],
    shard_payloads: list[dict[str, Any]],
    selected_umbrella_counts: Counter[str],
    kept_umbrella_counts: Counter[str],
    kept_primary_counts: Counter[str],
    duplicate_sample_ids: dict[str, list[str]],
    duplicate_kept_ids: dict[str, list[str]],
    overlap_with_excluded: list[str],
) -> dict[str, Any]:
    manifests = [payload["manifest"] for payload in shard_payloads]
    totals = Counter(
        {
            "selected_rows": 0,
            "generated_rows": 0,
            "reviewed_rows": 0,
            "kept_rows": 0,
            "dropped_rows": 0,
        }
    )
    shard_summaries: list[dict[str, Any]] = []
    for shard_dir, payload in zip(shard_dirs, shard_payloads, strict=True):
        shard_summary = {
            "shard_dir": shard_dir.name,
            "mode": payload["manifest"]["mode"],
            "selected_rows": len(payload["sample_rows"]),
            "generated_rows": len(payload["generation_rows"]),
            "reviewed_rows": len(payload["review_rows"]),
            "kept_rows": len(payload["training_rows"]),
            "dropped_rows": len(payload["dropped_rows"]),
        }
        shard_summaries.append(shard_summary)
        totals.update(
            {
                "selected_rows": shard_summary["selected_rows"],
                "generated_rows": shard_summary["generated_rows"],
                "reviewed_rows": shard_summary["reviewed_rows"],
                "kept_rows": shard_summary["kept_rows"],
                "dropped_rows": shard_summary["dropped_rows"],
            }
        )

    return {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "mode": "promoted_release",
        "source_run_root": str(run_root),
        "output_dir": str(output_dir),
        "selection_mode": collect_single_value("selection_mode", manifests),
        "source_split": collect_single_value("source_split", manifests),
        "input_jsonl": collect_single_value("input_jsonl", manifests),
        "input_row_count": collect_single_value("input_row_count", manifests),
        "seed": collect_single_value("seed", manifests),
        "shard_count": len(shard_dirs),
        "selected_row_count": totals["selected_rows"],
        "generated_rows": totals["generated_rows"],
        "reviewed_rows": totals["reviewed_rows"],
        "final_training_rows": totals["kept_rows"],
        "dropped_rows": totals["dropped_rows"],
        "review_decisions": {
            "accept": totals["kept_rows"],
            "drop": totals["dropped_rows"],
        },
        "generator_model": collect_single_value("generator_model", manifests),
        "generator_prompt_version": collect_single_value(
            "generator_prompt_version", manifests
        ),
        "generator_max_output_tokens": collect_single_value(
            "generator_max_output_tokens", manifests
        ),
        "generation_workers": collect_single_value("generation_workers", manifests),
        "reviewer_model": collect_single_value("reviewer_model", manifests),
        "review_prompt_version": collect_single_value("review_prompt_version", manifests),
        "reviewer_max_output_tokens": collect_single_value(
            "reviewer_max_output_tokens", manifests
        ),
        "review_workers": collect_single_value("review_workers", manifests),
        "reviewer_mode": collect_single_value("reviewer_mode", manifests),
        "reviewer_role": collect_single_value("reviewer_role", manifests),
        "reviewer_reasoning_enabled": collect_single_value(
            "reviewer_reasoning_enabled", manifests
        ),
        "reviewer_reasoning_effort": collect_single_value(
            "reviewer_reasoning_effort", manifests
        ),
        "selected_umbrella_counts": dict(sorted(selected_umbrella_counts.items())),
        "kept_umbrella_counts": dict(sorted(kept_umbrella_counts.items())),
        "top_kept_primary_categories": kept_primary_counts.most_common(20),
        "duplicate_sample_id_count": len(duplicate_sample_ids),
        "duplicate_sample_ids": dict(sorted(duplicate_sample_ids.items())),
        "duplicate_kept_paper_id_count": len(duplicate_kept_ids),
        "duplicate_kept_paper_ids": dict(sorted(duplicate_kept_ids.items())),
        "overlap_with_excluded_count": len(overlap_with_excluded),
        "overlap_with_excluded_ids": overlap_with_excluded,
        "exclude_jsonl": args.exclude_jsonl or None,
        "expected_reviewed_rows": args.expected_reviewed,
        "expected_kept_rows": args.expected_kept,
        "expected_dropped_rows": args.expected_dropped,
        "shards": shard_summaries,
    }


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    exclude_path = Path(args.exclude_jsonl) if args.exclude_jsonl else None

    shard_dirs = sorted(path for path in run_root.glob("shard_*") if path.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_* directories found under {run_root}")

    excluded_ids: set[str] = set()
    if exclude_path:
        excluded_rows = load_jsonl(require_path(exclude_path))
        excluded_ids = {str(row["paper_id"]) for row in excluded_rows}

    shard_payloads = [summarize_shard(shard_dir) for shard_dir in shard_dirs]

    sample_id_to_shards: dict[str, set[str]] = defaultdict(set)
    kept_id_to_shards: dict[str, set[str]] = defaultdict(set)
    overlap_with_excluded: set[str] = set()
    selected_umbrella_counts: Counter[str] = Counter()
    kept_umbrella_counts: Counter[str] = Counter()
    kept_primary_counts: Counter[str] = Counter()
    promoted_rows: list[dict[str, Any]] = []

    reviewed_total = 0
    kept_total = 0
    dropped_total = 0

    for shard_dir, payload in zip(shard_dirs, shard_payloads, strict=True):
        reviewed_total += len(payload["review_rows"])
        kept_total += len(payload["training_rows"])
        dropped_total += len(payload["dropped_rows"])
        promoted_rows.extend(payload["training_rows"])

        for row in payload["sample_rows"]:
            paper_id = str(row["paper_id"])
            sample_id_to_shards[paper_id].add(shard_dir.name)
            umbrella = str(
                row.get("umbrella_category") or umbrella_for(str(row["primary_category"]))
            )
            selected_umbrella_counts[umbrella] += 1
            if paper_id in excluded_ids:
                overlap_with_excluded.add(paper_id)

        for row in payload["training_rows"]:
            paper_id = str(row["paper_id"])
            kept_id_to_shards[paper_id].add(shard_dir.name)
            umbrella = umbrella_for(str(row["primary_category"]))
            kept_umbrella_counts[umbrella] += 1
            kept_primary_counts[str(row["primary_category"])] += 1

    duplicate_sample_ids = {
        paper_id: sorted(shards)
        for paper_id, shards in sample_id_to_shards.items()
        if len(shards) > 1
    }
    duplicate_kept_ids = {
        paper_id: sorted(shards)
        for paper_id, shards in kept_id_to_shards.items()
        if len(shards) > 1
    }
    if duplicate_kept_ids:
        raise ValueError(
            f"Duplicate kept paper_ids across shards: {sorted(duplicate_kept_ids)[:10]!r}"
        )

    assert_optional_exact("reviewed_rows", reviewed_total, args.expected_reviewed)
    assert_optional_exact("kept_rows", kept_total, args.expected_kept)
    assert_optional_exact("dropped_rows", dropped_total, args.expected_dropped)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "training_rows.jsonl", promoted_rows)

    manifest = build_manifest(
        args=args,
        run_root=run_root,
        output_dir=output_dir,
        shard_dirs=shard_dirs,
        shard_payloads=shard_payloads,
        selected_umbrella_counts=selected_umbrella_counts,
        kept_umbrella_counts=kept_umbrella_counts,
        kept_primary_counts=kept_primary_counts,
        duplicate_sample_ids=duplicate_sample_ids,
        duplicate_kept_ids=duplicate_kept_ids,
        overlap_with_excluded=sorted(overlap_with_excluded),
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"[promote-summary-run] wrote {output_dir / 'training_rows.jsonl'}")
    print(f"[promote-summary-run] wrote {output_dir / 'run_manifest.json'}")
    print(
        "[promote-summary-run] totals: "
        f"reviewed={reviewed_total} kept={kept_total} dropped={dropped_total}"
    )


if __name__ == "__main__":
    main()
