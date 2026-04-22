#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_PATHS = (
    "hf_bucket/experiments/hard_negative_pack_v1/"
    "correction_pack_25_per_axis_v1_20260420/hard_negative_pack.jsonl",
    "hf_bucket/experiments/hard_negative_pack_v1/"
    "correction_pack_25_per_axis_v1_20260420/generated_negative_candidates.jsonl",
    "hf_bucket/experiments/hard_negative_pack_v1/"
    "correction_pack_25_per_axis_v1_20260420/natural_negative_candidates.jsonl",
)
DEFAULT_REFERENCE_DATASET_DIR = (
    "hf_dataset/releases/recent_3y_balanced_5000_pilot_v2/dpo/"
    "title_abstract_preference_balanced200_v1"
)
DEFAULT_OUTPUT_DIR = (
    "hf_bucket/experiments/dpo/curation/"
    "title_abstract_preference_balanced200_pool_v1"
)
DEFAULT_GOLD_SOURCE_PRIORITY = (
    "release_training_row",
    "generated_correction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a manual-curation bundle for the current DPO candidate pool."
        )
    )
    parser.add_argument(
        "--source-jsonl",
        action="append",
        default=[],
        help=(
            "Source JSONL files to merge. Defaults to the canonical hard-negative "
            "pack plus accepted generated/natural candidate files."
        ),
    )
    parser.add_argument(
        "--reference-dataset-dir",
        default=DEFAULT_REFERENCE_DATASET_DIR,
        help=(
            "Prepared DPO dataset directory used to derive current target counts "
            "and mark currently selected rows."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the curation bundle.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in a non-empty output directory.",
    )
    parser.add_argument(
        "--seed-bundle-dir",
        default=None,
        help=(
            "Optional prior curation bundle directory. Matching decisions will be "
            "carried forward into the new decisions/*.tsv files by selection key."
        ),
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalize_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("paper_id", "")),
        str(row.get("failure_axis", "")),
        str(row.get("bad_summary", "")).strip(),
    )


def normalize_selection_key(selection_key: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(selection_key.get("paper_id", "")),
        str(selection_key.get("failure_axis", "")),
        str(selection_key.get("bad_summary", "")).strip(),
    )


def source_priority_index(row: dict[str, Any]) -> int:
    gold_source = str(row.get("gold_source", ""))
    try:
        return DEFAULT_GOLD_SOURCE_PRIORITY.index(gold_source)
    except ValueError:
        return len(DEFAULT_GOLD_SOURCE_PRIORITY)


def axis_sort_key(row: dict[str, Any]) -> tuple[int, str, str, str]:
    return (
        source_priority_index(row),
        str(row.get("paper_id", "")),
        str(row.get("candidate_origin", "")),
        str(row.get("gold_summary", "")),
    )


def load_reference_dataset(
    dataset_dir: Path,
) -> tuple[dict[str, int], set[tuple[str, str, str]], dict[str, int]]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    axis_targets = {
        str(axis): int(count)
        for axis, count in manifest.get("axis_targets", {}).items()
    }
    selected_keys: set[tuple[str, str, str]] = set()
    reference_selected_counts: Counter[str] = Counter()
    for split_name in ("train", "valid", "test"):
        for row in load_jsonl(dataset_dir / f"{split_name}.jsonl"):
            axis = str(row.get("failure_axis", ""))
            reference_selected_counts[axis] += 1
            rejected = row.get("rejected", [])
            rejected_text = ""
            if rejected:
                rejected_text = str(rejected[0].get("content", "")).strip()
            selected_keys.add(
                (
                    str(row.get("paper_id", "")),
                    axis,
                    rejected_text,
                )
            )
    return axis_targets, selected_keys, dict(sorted(reference_selected_counts.items()))


def merge_curatable_rows(paths: list[Path]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in paths:
        source_kind = (
            "canonical_pack" if path.name == "hard_negative_pack.jsonl" else path.stem
        )
        for row in load_jsonl(path):
            if path.name != "hard_negative_pack.jsonl" and row.get("accepted") is not True:
                continue
            if not str(row.get("paper_id", "")).strip():
                continue
            if not str(row.get("failure_axis", "")).strip():
                continue
            if not str(row.get("gold_summary", "")).strip():
                continue
            if not str(row.get("bad_summary", "")).strip():
                continue
            key = normalize_key(row)
            if key in seen:
                continue
            seen.add(key)
            copied = dict(row)
            copied["_source_kind"] = source_kind
            merged.append(copied)
    return merged


def decision_tsv_header() -> str:
    return "\t".join(
        [
            "candidate_id",
            "decision",
            "confidence",
            "chosen_quality",
            "preference_gap",
            "priority_tier",
            "note",
        ]
    )


def load_seeded_decisions(bundle_dir: Path | None) -> tuple[dict[tuple[str, str, str], dict[str, str]], dict[str, int]]:
    if bundle_dir is None:
        return {}, {}
    all_candidates_path = bundle_dir / "all_candidates.jsonl"
    decisions_dir = bundle_dir / "decisions"
    if not all_candidates_path.exists() or not decisions_dir.exists():
        raise FileNotFoundError(
            f"Seed bundle {bundle_dir} must contain all_candidates.jsonl and decisions/."
        )

    selection_key_by_candidate_id: dict[str, tuple[str, str, str]] = {}
    axis_by_candidate_id: dict[str, str] = {}
    for row in load_jsonl(all_candidates_path):
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not candidate_id:
            continue
        selection_key = row.get("selection_key")
        if not isinstance(selection_key, dict):
            continue
        selection_key_by_candidate_id[candidate_id] = normalize_selection_key(selection_key)
        axis_by_candidate_id[candidate_id] = str(row.get("failure_axis", "")).strip()

    seeded_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    seeded_counts: Counter[str] = Counter()
    for decision_path in sorted(decisions_dir.glob("*.tsv")):
        with decision_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                candidate_id = str(row.get("candidate_id", "")).strip()
                if not candidate_id:
                    continue
                selection_key = selection_key_by_candidate_id.get(candidate_id)
                if selection_key is None:
                    continue
                decision = str(row.get("decision", "")).strip().lower()
                if decision not in {"keep", "drop"}:
                    continue
                seeded_by_key[selection_key] = {
                    "decision": decision,
                    "confidence": str(row.get("confidence", "")).strip().lower(),
                    "chosen_quality": str(row.get("chosen_quality", "")).strip().lower(),
                    "preference_gap": str(row.get("preference_gap", "")).strip().lower(),
                    "priority_tier": str(row.get("priority_tier", "")).strip().upper(),
                    "note": str(row.get("note", "")).strip(),
                }
                seeded_counts[axis_by_candidate_id.get(candidate_id, "")] += 1
    return seeded_by_key, dict(sorted(seeded_counts.items()))


def build_readme(
    *,
    axis_counts: dict[str, int],
    axis_targets: dict[str, int],
    current_selected_counts: dict[str, int],
    reference_selected_counts: dict[str, int],
    reference_selected_missing_from_pool: dict[str, int],
    seeded_decision_counts: dict[str, int],
) -> str:
    lines = [
        "# DPO Curation Bundle",
        "",
        "Purpose:",
        "- Review the current curatable DPO pool by axis.",
        "- Mark which pairs should survive into the next hand-curated dataset.",
        "- Prefer pair quality and teaching value over sheer row count.",
        "",
        "Decision rubric:",
        "- `keep` only if the pair teaches the intended failure axis clearly.",
        "- Drop rows where the chosen summary is weak, noisy, or only marginally better.",
        "- Prefer faithful but plain chosen summaries over sharper but risky ones.",
        "- Prefer rows with a clear preference gap between chosen and rejected.",
        "- Penalize repeated papers when the axis already has plenty of strong examples.",
        "",
        "Decision file schema:",
        "- `candidate_id`",
        "- `decision`: `keep` or `drop`",
        "- `confidence`: `high`, `medium`, or `low`",
        "- `chosen_quality`: `strong`, `acceptable`, or `weak`",
        "- `preference_gap`: `large`, `medium`, or `small`",
        "- `priority_tier`: `A`, `B`, `C`, or `D`",
        "- `note`: one short sentence",
        "",
        "Axis counts and current targets:",
    ]
    for axis in sorted(axis_counts):
        lines.append(
            f"- `{axis}`: {axis_counts[axis]} candidates, "
            f"current target {axis_targets.get(axis, 0)}, "
            f"strict-pool overlap {current_selected_counts.get(axis, 0)}, "
            f"reference selected {reference_selected_counts.get(axis, 0)}, "
            f"reference-only rows excluded {reference_selected_missing_from_pool.get(axis, 0)}, "
            f"seeded prior decisions {seeded_decision_counts.get(axis, 0)}"
        )
    lines.extend(
        [
            "",
            "Files:",
            "- `all_candidates.jsonl`: the full curatable pool",
            "- `by_axis/<axis>.jsonl`: candidates for one axis",
            "- `decisions/<axis>.tsv`: fill this in during review",
            "- `manifest.json`: counts, targets, and source mix",
            "",
            "Notes:",
            "- `currently_selected_balanced200=true` means the row is in the current 200-pair dataset.",
            "- `reference-only rows excluded` are rows present in the current balanced200 dataset but not in this stricter curation pool.",
            "- `axis_rank` is deterministic, not a quality judgment.",
            "- `source_priority_rank` only reflects gold-source preference, not pair quality.",
            "- Matching prior decisions are prefilled into `decisions/*.tsv` when `--seed-bundle-dir` is used.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    source_paths = [
        Path(path) for path in (args.source_jsonl or list(DEFAULT_SOURCE_PATHS))
    ]
    reference_dataset_dir = Path(args.reference_dataset_dir)
    output_dir = Path(args.output_dir)
    seed_bundle_dir = Path(args.seed_bundle_dir) if args.seed_bundle_dir else None

    ensure_new_output_dir(output_dir, overwrite=args.overwrite)
    by_axis_dir = output_dir / "by_axis"
    decisions_dir = output_dir / "decisions"
    by_axis_dir.mkdir(parents=True, exist_ok=True)
    decisions_dir.mkdir(parents=True, exist_ok=True)

    axis_targets, selected_keys, reference_selected_counts = load_reference_dataset(
        reference_dataset_dir
    )
    seeded_decisions_by_key, seeded_decision_counts = load_seeded_decisions(seed_bundle_dir)
    merged_rows = merge_curatable_rows(source_paths)

    rows_by_axis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in merged_rows:
        rows_by_axis[str(row["failure_axis"])].append(row)

    all_output_rows: list[dict[str, Any]] = []
    axis_counts: dict[str, int] = {}
    current_selected_counts: dict[str, int] = {}
    unique_papers_by_axis: dict[str, int] = {}
    reference_selected_missing_from_pool: dict[str, int] = {}

    for axis in sorted(rows_by_axis):
        axis_rows = sorted(rows_by_axis[axis], key=axis_sort_key)
        axis_counts[axis] = len(axis_rows)
        unique_papers_by_axis[axis] = len(
            {str(row.get("paper_id", "")) for row in axis_rows}
        )
        current_selected_counts[axis] = 0

        rendered_rows: list[dict[str, Any]] = []
        for index, row in enumerate(axis_rows, start=1):
            key = normalize_key(row)
            current_selected = key in selected_keys
            if current_selected:
                current_selected_counts[axis] += 1
            rendered = {
                "candidate_id": f"{axis}:{index:03d}",
                "axis_rank": index,
                "failure_axis": axis,
                "paper_id": str(row.get("paper_id", "")),
                "title": str(row.get("title", "")),
                "gold_summary": str(row.get("gold_summary", "")).strip(),
                "bad_summary": str(row.get("bad_summary", "")).strip(),
                "gold_source": str(row.get("gold_source", "")),
                "candidate_origin": str(row.get("candidate_origin", "")),
                "source_kind": str(row.get("_source_kind", "")),
                "currently_selected_balanced200": current_selected,
                "target_keep_count_for_axis": int(axis_targets.get(axis, 0)),
                "source_priority_rank": source_priority_index(row),
                "selection_key": {
                    "paper_id": str(row.get("paper_id", "")),
                    "failure_axis": axis,
                    "bad_summary": str(row.get("bad_summary", "")).strip(),
                },
            }
            rendered_rows.append(rendered)
            all_output_rows.append(rendered)

        reference_selected_missing_from_pool[axis] = max(
            0,
            int(reference_selected_counts.get(axis, 0))
            - int(current_selected_counts.get(axis, 0)),
        )

        write_jsonl(by_axis_dir / f"{axis}.jsonl", rendered_rows)
        decision_lines = [decision_tsv_header()]
        for rendered in rendered_rows:
            seeded = seeded_decisions_by_key.get(normalize_selection_key(rendered["selection_key"]))
            if not seeded:
                continue
            decision_lines.append(
                "\t".join(
                    [
                        rendered["candidate_id"],
                        seeded["decision"],
                        seeded["confidence"],
                        seeded["chosen_quality"],
                        seeded["preference_gap"],
                        seeded["priority_tier"],
                        seeded["note"],
                    ]
                )
            )
        (decisions_dir / f"{axis}.tsv").write_text(
            "\n".join(decision_lines) + "\n",
            encoding="utf-8",
        )

    all_output_rows.sort(key=lambda row: (row["failure_axis"], row["axis_rank"]))
    write_jsonl(output_dir / "all_candidates.jsonl", all_output_rows)

    manifest = {
        "source_paths": [str(path) for path in source_paths],
        "reference_dataset_dir": str(reference_dataset_dir),
        "seed_bundle_dir": str(seed_bundle_dir) if seed_bundle_dir else "",
        "total_candidates": len(all_output_rows),
        "axis_counts": axis_counts,
        "axis_targets": axis_targets,
        "current_selected_counts": current_selected_counts,
        "reference_selected_counts": reference_selected_counts,
        "reference_selected_missing_from_pool": reference_selected_missing_from_pool,
        "seeded_decision_counts": seeded_decision_counts,
        "unique_papers_by_axis": unique_papers_by_axis,
        "selected_gold_source_counts": dict(
            sorted(Counter(row["gold_source"] for row in all_output_rows).items())
        ),
        "source_kind_counts": dict(
            sorted(Counter(row["source_kind"] for row in all_output_rows).items())
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        build_readme(
            axis_counts=axis_counts,
            axis_targets=axis_targets,
            current_selected_counts=current_selected_counts,
            reference_selected_counts=reference_selected_counts,
            reference_selected_missing_from_pool=reference_selected_missing_from_pool,
            seeded_decision_counts=seeded_decision_counts,
        ),
        encoding="utf-8",
    )

    print(f"[prepare-dpo-curation] wrote bundle to {output_dir}")
    print(f"[prepare-dpo-curation] total candidates: {len(all_output_rows)}")
    print(f"[prepare-dpo-curation] axis counts: {axis_counts}")


if __name__ == "__main__":
    main()
