#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dpo_utils import (
    DEFAULT_INPUT_MODE,
    DEFAULT_PROMPT_SPEC,
    build_prompt_messages,
    ensure_new_output_dir,
    load_jsonl,
    load_prompt_spec,
    write_jsonl,
)

DEFAULT_INPUT_JSONL = (
    "hf_bucket/experiments/hard_negative_pack_v1/"
    "correction_pack_25_per_axis_v1_20260420/hard_negative_pack.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "hf_dataset/releases/recent_3y_balanced_5000_pilot_v2/dpo/"
    "title_abstract_preference_hc100_v1"
)
EXPECTED_AXES = (
    "invented_number",
    "title_restatement",
    "too_generic",
    "wrong_number_binding",
)
SPLIT_COUNTS = {
    "train": 20,
    "valid": 2,
    "test": 3,
}
DEFAULT_ALLOWED_GOLD_SOURCES = ("release_training_row",)
DEFAULT_SPLIT_RATIOS = {
    "train": 0.80,
    "valid": 0.08,
    "test": 0.12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the clean high-confidence DPO preference dataset from the "
            "hard-negative pack."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        default=DEFAULT_INPUT_JSONL,
        help="Hard-negative pack JSONL source.",
    )
    parser.add_argument(
        "--extra-input-jsonl",
        action="append",
        default=[],
        help="Additional hard-negative pack JSONL sources to merge in.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the prepared DPO dataset.",
    )
    parser.add_argument(
        "--prompt-spec",
        default=DEFAULT_PROMPT_SPEC,
        help="Prompt spec JSON used to construct prompt messages.",
    )
    parser.add_argument(
        "--input-mode",
        default=DEFAULT_INPUT_MODE,
        choices=("abstract_only", "title_abstract"),
        help="Prompt input mode to prepare.",
    )
    parser.add_argument(
        "--allowed-gold-sources",
        default=",".join(DEFAULT_ALLOWED_GOLD_SOURCES),
        help=(
            "Comma-separated gold_source values to allow, in priority order. "
            "Rows from earlier sources win ties."
        ),
    )
    parser.add_argument(
        "--axis-target",
        action="append",
        default=[],
        help=(
            "Optional axis=count target. Repeat to build a custom DPO mix. "
            "Without any axis targets, the legacy 100-row high-confidence slice is used."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in a non-empty output directory.",
    )
    return parser.parse_args()


def stable_axis_sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    curated_rank = row.get("curation_rank_within_axis")
    if curated_rank is not None:
        return (
            f"{int(curated_rank):06d}",
            str(row.get("paper_id", "")),
            str(row.get("candidate_origin", "")),
        )
    return (
        str(row.get("paper_id", "")),
        str(row.get("candidate_origin", "")),
        str(row.get("gold_summary", "")),
    )


def validate_filtered_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows_by_axis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        axis = str(row.get("failure_axis", ""))
        rows_by_axis[axis].append(row)

    missing = [axis for axis in EXPECTED_AXES if axis not in rows_by_axis]
    unexpected = sorted(set(rows_by_axis) - set(EXPECTED_AXES))
    if missing or unexpected:
        problems: list[str] = []
        if missing:
            problems.append(f"missing axes: {', '.join(missing)}")
        if unexpected:
            problems.append(f"unexpected axes: {', '.join(unexpected)}")
        raise ValueError("Filtered high-confidence rows do not match expectations: " + "; ".join(problems))

    for axis in EXPECTED_AXES:
        count = len(rows_by_axis[axis])
        if count != 25:
            raise ValueError(
                f"Expected exactly 25 rows for axis {axis}, found {count}."
            )
        rows_by_axis[axis].sort(key=stable_axis_sort_key)
    return rows_by_axis


def parse_allowed_gold_sources(raw_value: str) -> list[str]:
    values = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not values:
        raise ValueError("allowed_gold_sources must not be empty")
    return values


def parse_axis_targets(raw_targets: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for raw in raw_targets:
        if "=" not in raw:
            raise ValueError(f"Invalid --axis-target value {raw!r}; expected axis=count")
        axis, raw_count = raw.split("=", 1)
        axis = axis.strip()
        raw_count = raw_count.strip()
        if not axis or not raw_count:
            raise ValueError(f"Invalid --axis-target value {raw!r}; expected axis=count")
        count = int(raw_count)
        if count <= 0:
            raise ValueError(f"Axis target for {axis} must be positive, got {count}")
        parsed[axis] = count
    return parsed


def normalize_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("paper_id", "")),
        str(row.get("failure_axis", "")),
        str(row.get("bad_summary", "")).strip(),
    )


def merge_source_rows(paths: list[Path]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        for row in load_jsonl(path):
            # Candidate-side artifacts include prep failures and rejected rows.
            # Only the canonical hard_negative_pack rows are accepted without an
            # explicit keep flag. Curated materialized rows already represent
            # selected keeps and carry decision metadata instead of accepted=true.
            if (
                path.name != "hard_negative_pack.jsonl"
                and row.get("accepted") is not True
                and not isinstance(row.get("decision_metadata"), dict)
            ):
                continue
            key = normalize_key(row)
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
    return merged


def source_priority_index(allowed_gold_sources: list[str], row: dict[str, Any]) -> int:
    value = str(row.get("gold_source", ""))
    try:
        return allowed_gold_sources.index(value)
    except ValueError:
        return len(allowed_gold_sources)


def custom_axis_sort_key(
    row: dict[str, Any],
    *,
    allowed_gold_sources: list[str],
) -> tuple[int, str, str, str]:
    curated_rank = row.get("curation_rank_within_axis")
    if curated_rank is not None:
        return (
            -1,
            f"{int(curated_rank):06d}",
            str(row.get("paper_id", "")),
            str(row.get("candidate_origin", "")),
        )
    return (
        source_priority_index(allowed_gold_sources, row),
        str(row.get("paper_id", "")),
        str(row.get("candidate_origin", "")),
        str(row.get("gold_summary", "")),
    )


def split_counts_for_total(total_count: int) -> dict[str, int]:
    valid = round(total_count * DEFAULT_SPLIT_RATIOS["valid"])
    test = round(total_count * DEFAULT_SPLIT_RATIOS["test"])
    train = total_count - valid - test
    if train <= 0 or valid < 0 or test < 0:
        raise ValueError(f"Invalid split counts for total_count={total_count}")
    return {
        "train": train,
        "valid": valid,
        "test": test,
    }


def validate_prompt_source_row(
    row: dict[str, Any],
    *,
    input_mode: str,
) -> None:
    paper_id = str(row.get("paper_id", "")).strip() or "<unknown>"
    title = str(row.get("title", "")).strip()
    abstract = str(row.get("abstract", "")).strip()
    if input_mode == "title_abstract":
        if not title:
            raise ValueError(
                f"Row for paper_id={paper_id} is missing title for input_mode=title_abstract."
            )
        if not abstract:
            raise ValueError(
                f"Row for paper_id={paper_id} is missing abstract for input_mode=title_abstract."
            )
    elif input_mode == "abstract_only":
        if not abstract:
            raise ValueError(
                f"Row for paper_id={paper_id} is missing abstract for input_mode=abstract_only."
            )


def build_dpo_row(
    row: dict[str, Any],
    *,
    prompt_spec: dict[str, Any],
    input_mode: str,
) -> dict[str, Any]:
    validate_prompt_source_row(row, input_mode=input_mode)
    prompt = build_prompt_messages(row, prompt_spec, input_mode)
    return {
        "paper_id": str(row["paper_id"]),
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": str(row["gold_summary"]).strip()}],
        "rejected": [{"role": "assistant", "content": str(row["bad_summary"]).strip()}],
        "failure_axis": str(row["failure_axis"]),
        "gold_source": str(row["gold_source"]),
        "source_kind": str(row.get("source_kind", "")),
        "candidate_origin": str(row.get("candidate_origin", "")),
        "curation_priority_tier": str(
            row.get("decision_metadata", {}).get("priority_tier", "")
        ),
        "curation_confidence": str(
            row.get("decision_metadata", {}).get("confidence", "")
        ),
        "curation_preference_gap": str(
            row.get("decision_metadata", {}).get("preference_gap", "")
        ),
        "prompt_version": str(prompt_spec["prompt_version"]),
    }


def main() -> None:
    args = parse_args()

    input_paths = [Path(args.input_jsonl)] + [Path(path) for path in args.extra_input_jsonl]
    output_dir = Path(args.output_dir)
    prompt_spec_path = Path(args.prompt_spec)
    allowed_gold_sources = parse_allowed_gold_sources(args.allowed_gold_sources)
    axis_targets = parse_axis_targets(args.axis_target)

    ensure_new_output_dir(output_dir, overwrite=args.overwrite)

    prompt_spec = load_prompt_spec(prompt_spec_path)
    source_rows = merge_source_rows(input_paths)
    filtered_rows = [
        row
        for row in source_rows
        if str(row.get("gold_source", "")) in allowed_gold_sources
    ]

    if axis_targets:
        rows_by_axis: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in filtered_rows:
            rows_by_axis[str(row.get("failure_axis", ""))].append(row)
        selected_rows: list[dict[str, Any]] = []
        split_plan: dict[str, dict[str, int]] = {}
        for axis, target_count in axis_targets.items():
            axis_rows = sorted(
                rows_by_axis.get(axis, []),
                key=lambda row: custom_axis_sort_key(
                    row, allowed_gold_sources=allowed_gold_sources
                ),
            )
            if len(axis_rows) < target_count:
                raise ValueError(
                    f"Axis {axis} has only {len(axis_rows)} eligible rows, "
                    f"but target_count={target_count}."
                )
            chosen_rows = axis_rows[:target_count]
            selected_rows.extend(chosen_rows)
            split_plan[axis] = split_counts_for_total(target_count)
        rows_by_axis = defaultdict(list)
        for row in selected_rows:
            rows_by_axis[str(row.get("failure_axis", ""))].append(row)
    else:
        if allowed_gold_sources != list(DEFAULT_ALLOWED_GOLD_SOURCES):
            raise ValueError(
                "Custom allowed_gold_sources require explicit --axis-target values."
            )
        if len(filtered_rows) != 100:
            raise ValueError(
                f"Expected 100 high-confidence rows, found {len(filtered_rows)} across {input_paths}."
            )
        rows_by_axis = validate_filtered_rows(filtered_rows)
        split_plan = {axis: dict(SPLIT_COUNTS) for axis in EXPECTED_AXES}

    selected_rows = []
    for axis_rows in rows_by_axis.values():
        selected_rows.extend(axis_rows)

    for row in selected_rows:
        validate_prompt_source_row(row, input_mode=args.input_mode)

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for axis in sorted(rows_by_axis):
        axis_rows = rows_by_axis[axis]
        axis_split_counts = split_plan[axis]
        axis_slices = {
            "train": axis_rows[: axis_split_counts["train"]],
            "valid": axis_rows[
                axis_split_counts["train"] : axis_split_counts["train"] + axis_split_counts["valid"]
            ],
            "test": axis_rows[-axis_split_counts["test"] :],
        }
        for split_name, rows in axis_slices.items():
            if len(rows) != axis_split_counts[split_name]:
                raise ValueError(
                    f"Axis {axis} produced {len(rows)} {split_name} rows; "
                    f"expected {axis_split_counts[split_name]}."
                )
            split_rows[split_name].extend(
                build_dpo_row(row, prompt_spec=prompt_spec, input_mode=args.input_mode)
                for row in rows
            )

    for split_name in ("train", "valid", "test"):
        split_rows[split_name].sort(
            key=lambda row: (str(row["failure_axis"]), str(row["paper_id"]))
        )
        write_jsonl(output_dir / f"{split_name}.jsonl", split_rows[split_name])

    manifest = {
        "source_paths": [str(path) for path in input_paths],
        "prompt_spec_path": str(prompt_spec_path),
        "prompt_version": str(prompt_spec["prompt_version"]),
        "input_mode": args.input_mode,
        "allowed_gold_sources": allowed_gold_sources,
        "accepted_filter_rule": (
            "canonical hard_negative_pack rows are always kept; candidate-side rows are "
            "kept only when accepted == true"
        ),
        "eligible_pair_count": len(filtered_rows),
        "selected_pair_count": len(selected_rows),
        "expected_axes": sorted(rows_by_axis),
        "axis_counts": dict(
            sorted(Counter(str(row["failure_axis"]) for row in selected_rows).items())
        ),
        "selected_gold_source_counts": dict(
            sorted(Counter(str(row.get("gold_source", "")) for row in selected_rows).items())
        ),
        "split_rule": (
            "per-axis deterministic split after gold-source-priority sorting; "
            "default profile uses legacy 20/2/3, custom profile uses 80/8/12-style rounded counts"
        ),
        "split_counts": {
            split_name: len(rows) for split_name, rows in split_rows.items()
        },
        "per_axis_split_counts": split_plan,
        "axis_targets": axis_targets or {axis: len(rows) for axis, rows in rows_by_axis.items()},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"[prepare-dpo] wrote dataset to {output_dir}")
    print(f"[prepare-dpo] eligible rows: {len(filtered_rows)}")
    print(f"[prepare-dpo] selected rows: {len(selected_rows)}")
    print(f"[prepare-dpo] split counts: {manifest['split_counts']}")
    print(f"[prepare-dpo] axis counts: {manifest['axis_counts']}")


if __name__ == "__main__":
    main()
