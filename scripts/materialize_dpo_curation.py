#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PRIORITY_TIER_RANK = {"A": 0, "B": 1, "C": 2, "D": 3}
CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}
PREFERENCE_GAP_RANK = {"large": 0, "medium": 1, "small": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize a curated DPO candidate set from a reviewed bundle."
    )
    parser.add_argument(
        "--bundle-dir",
        required=True,
        help="Curation bundle directory containing all_candidates.jsonl and decisions/.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <bundle-dir>/curated_dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in a non-empty output directory.",
    )
    return parser.parse_args()


def ensure_new_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        existing = list(path.iterdir())
        if existing and not overwrite:
            raise FileExistsError(
                f"Output directory {path} is not empty. Use --overwrite to replace it."
            )
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_decisions(bundle_dir: Path) -> dict[str, dict[str, str]]:
    decisions: dict[str, dict[str, str]] = {}
    decisions_dir = bundle_dir / "decisions"
    for path in sorted(decisions_dir.glob("*.tsv")):
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                candidate_id = str(row.get("candidate_id", "")).strip()
                decision = str(row.get("decision", "")).strip().lower()
                if not candidate_id or decision not in {"keep", "drop"}:
                    continue
                decisions[candidate_id] = {
                    "decision": decision,
                    "confidence": str(row.get("confidence", "")).strip().lower(),
                    "chosen_quality": str(row.get("chosen_quality", "")).strip().lower(),
                    "preference_gap": str(row.get("preference_gap", "")).strip().lower(),
                    "priority_tier": str(row.get("priority_tier", "")).strip().upper(),
                    "note": str(row.get("note", "")).strip(),
                }
    return decisions


def materialize_sort_key(row: dict[str, Any]) -> tuple[int, int, int, int]:
    decision = row["decision_metadata"]
    return (
        PRIORITY_TIER_RANK.get(str(decision.get("priority_tier", "")), 99),
        CONFIDENCE_RANK.get(str(decision.get("confidence", "")), 99),
        PREFERENCE_GAP_RANK.get(str(decision.get("preference_gap", "")), 99),
        int(row.get("axis_rank", 0)),
    )


def main() -> None:
    args = parse_args()
    bundle_dir = Path(args.bundle_dir)
    output_dir = Path(args.output_dir) if args.output_dir else bundle_dir / "curated_dataset"

    ensure_new_output_dir(output_dir, overwrite=args.overwrite)

    manifest = load_json(bundle_dir / "manifest.json")
    all_candidates = load_jsonl(bundle_dir / "all_candidates.jsonl")
    decisions = load_decisions(bundle_dir)
    axis_targets = {
        str(axis): int(count)
        for axis, count in manifest.get("axis_targets", {}).items()
    }

    kept_by_axis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in all_candidates:
        decision = decisions.get(str(candidate.get("candidate_id", "")))
        if not decision or decision["decision"] != "keep":
            continue
        materialized = dict(candidate)
        materialized["decision_metadata"] = decision
        kept_by_axis[str(candidate["failure_axis"])].append(materialized)

    selected_rows: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    underfilled_axes: dict[str, int] = {}

    for axis in sorted(set(kept_by_axis) | set(axis_targets)):
        ranked = sorted(kept_by_axis[axis], key=materialize_sort_key)
        target = int(axis_targets.get(axis, len(ranked)))
        chosen = ranked[:target]
        for index, row in enumerate(chosen, start=1):
            row["curation_rank_within_axis"] = index
        selected_rows.extend(chosen)
        selected_counts[axis] = len(chosen)
        if len(chosen) < target:
            underfilled_axes[axis] = target - len(chosen)

    selected_rows.sort(
        key=lambda row: (str(row["failure_axis"]), int(row["curation_rank_within_axis"]))
    )
    write_jsonl(output_dir / "curated_candidates.jsonl", selected_rows)

    materialized_manifest = {
        "source_bundle_dir": str(bundle_dir),
        "output_dir": str(output_dir),
        "selected_row_count": len(selected_rows),
        "selected_counts_by_axis": dict(sorted(selected_counts.items())),
        "underfilled_axes": dict(sorted(underfilled_axes.items())),
        "axis_targets": dict(sorted(axis_targets.items())),
        "selected_gold_source_counts": dict(
            sorted(Counter(str(row.get("gold_source", "")) for row in selected_rows).items())
        ),
        "selected_source_kind_counts": dict(
            sorted(Counter(str(row.get("source_kind", "")) for row in selected_rows).items())
        ),
        "selected_priority_tier_counts": dict(
            sorted(
                Counter(
                    str(row["decision_metadata"].get("priority_tier", ""))
                    for row in selected_rows
                ).items()
            )
        ),
        "selected_confidence_counts": dict(
            sorted(
                Counter(
                    str(row["decision_metadata"].get("confidence", ""))
                    for row in selected_rows
                ).items()
            )
        ),
    }
    write_json(output_dir / "manifest.json", materialized_manifest)

    print(f"[materialize-dpo-curation] wrote output to {output_dir}")
    print(
        "[materialize-dpo-curation] selected counts: "
        f"{dict(sorted(selected_counts.items()))}"
    )
    if underfilled_axes:
        print(
            "[materialize-dpo-curation] underfilled axes: "
            f"{dict(sorted(underfilled_axes.items()))}"
        )


if __name__ == "__main__":
    main()
