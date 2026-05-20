#!/usr/bin/env python3
"""Prepare retry inputs and consolidate grounded-shape rewrite passes."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


ROOT = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_residual218_grounded_shape_scaffold_v1"
)
DEFAULT_ORIGINAL = ROOT / "rewrite_candidate_pack_v1/grounded_shape_rewrite_review_input.jsonl"
DEFAULT_OUT = ROOT / "rewrite_candidate_pack_v1/grounded_shape_rewrite_retry2_review_input.jsonl"
DEFAULT_CONSOLIDATED = ROOT / "grounded_shape_rewrite_consolidated_v1"
DEFAULT_PASS_ROOTS = [
    ROOT / "grounded_shape_rewrite_kimi_qwen_v1",
    ROOT / "grounded_shape_rewrite_retry_kimi_qwen_v1",
]

ROW_STRATEGIES = {
    "2405.10684": (
        "Third-pass focus: prior accepted-looking rewrite was too close to the brittle "
        "changing-look target, then the second rewrite was locally too long. Write a "
        "different 10-14 word label. Avoid exact catalog/source names, PKS, FGL, "
        "IceCube, redshift, line wavelengths, and the phrase changing-look. Preserve "
        "the source-supported object-level result: optical spectroscopy of a gamma-ray "
        "blazar shows increased continuum, faded emission lines, and high jet activity. "
        "If this is retried again, use exactly 10-12 whitespace-separated words and "
        "prioritize continuum plus emission-line changes over listing every detail. "
        "For a final retry, drop long openers like optical spectroscopy of a; a concise "
        "object-first gamma-ray blazar spectrum label is preferred."
    ),
    "2404.18022": (
        "Third-pass focus: Qwen rejected labels that mention optical filaments because "
        "they risk conflating the binary's shell with nearby remnant filaments. Write a "
        "label centered on the oxygen-bright shell around CI Cam or an X-ray binary star. "
        "Avoid neutron star, remnant causality, and optical filaments unless the caveat "
        "that they are only nearby is unmistakable."
    ),
    "2501.11917": (
        "Third-pass focus: Qwen rejected local fields and real-or-complex fields as "
        "semantic drift. Keep the mathematical object source-grounded: twisted Shalika "
        "periods or functionals for smooth Archimedean representations, connected to "
        "L-parameters if useful. Avoid local fields, real fields, complex fields, and "
        "deformed wording such as archimedes. If this is retried again, do not switch "
        "the object to Archimedean groups; keep the object as smooth Archimedean "
        "representations or drop the row. For a final retry, preserve the exact object "
        "phrase smooth Archimedean representations if using the rare adjective."
    ),
    "2506.20088": (
        "Third-pass focus: previous rewrites were conceptually useful but locally too "
        "long. Write a 10-14 word label. Avoid exact source names, exact times, neutron "
        "star binary claims, and thermal-origin overinterpretation. Preserve unusual "
        "Galactic-center X-ray transient, rapid fading or brightness changes, and "
        "unusual iron emission lines. If this is retried again, use exactly 10-12 "
        "whitespace-separated words and omit secondary phrasing before dropping the "
        "transient, fading, or iron-line facts."
    ),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def norm(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def materialized_paths(pass_roots: list[Path], name: str) -> list[Path]:
    return [root / "materialized" / name for root in pass_roots]


def load_by_paper(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    rows_by_pid: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for path in paths:
        for row in load_jsonl(path):
            row = dict(row)
            row["_source_file"] = str(path)
            rows_by_pid[str(row["paper_id"])].append(row)
    return dict(rows_by_pid)


def summarize_rejections(
    rejected_after: dict[str, list[dict[str, Any]]],
    rejected_before: dict[str, list[dict[str, Any]]],
    pid: str,
) -> str:
    bits: list[str] = []
    for row in rejected_after.get(pid, []):
        summary = norm(row.get("repair_summary", ""))
        tags = ", ".join(row.get("verifier_issue_tags", []))
        reason = norm(row.get("verifier_reason_span", ""))
        if summary:
            bits.append(f"Qwen rejected candidate: {summary}")
        if tags:
            bits.append(f"Qwen tags: {tags}.")
        if reason:
            bits.append(f"Qwen reason: {reason}.")
    for row in rejected_before.get(pid, []):
        summary = norm(row.get("repair_summary", ""))
        flags = ", ".join(row.get("writer_hard_filter_flags", []))
        if summary:
            bits.append(f"Local screen rejected candidate: {summary}")
        if flags:
            bits.append(f"Local flags: {flags}.")
    return " ".join(bits)


def command_retry(args: argparse.Namespace) -> None:
    pass_roots = [Path(path) for path in args.pass_roots]
    original_rows = load_jsonl(args.original)
    accepted = load_by_paper(materialized_paths(pass_roots, "accepted_repairs.jsonl"))
    rejected_after = load_by_paper(materialized_paths(pass_roots, "rejected_after_qwen.jsonl"))
    rejected_before = load_by_paper(materialized_paths(pass_roots, "rejected_before_qwen.jsonl"))

    retry_rows: list[dict[str, Any]] = []
    for row in original_rows:
        pid = str(row["paper_id"])
        if accepted.get(pid):
            continue
        if pid not in ROW_STRATEGIES:
            continue
        retry = dict(row)
        history = summarize_rejections(rejected_after, rejected_before, pid)
        extra_note = ROW_STRATEGIES[pid]
        retry["category_note"] = norm(
            f"{row.get('category_note', '')} {history} {extra_note}"
        )
        retry["reviewer_note"] = norm(
            f"{row.get('reviewer_note', '')} THIRD PASS: {extra_note}"
        )
        retry["retry_source"] = "q5_grounded_shape_rewrite_retry2_v1"
        retry_rows.append(retry)

    write_jsonl(args.out, sorted(retry_rows, key=lambda item: item["paper_id"]))
    write_json(
        args.out.with_suffix(".manifest.json"),
        {
            "created_by": "scripts/prepare_q5_grounded_shape_rewrite_retry_v1.py retry",
            "original": str(args.original),
            "pass_roots": [str(path) for path in pass_roots],
            "out": str(args.out),
            "accepted_before_retry": sorted(accepted),
            "retry_rows": len(retry_rows),
            "retry_paper_ids": [row["paper_id"] for row in sorted(retry_rows, key=lambda item: item["paper_id"])],
        },
    )
    print(f"[retry] wrote {len(retry_rows)} rows to {args.out}")


def command_consolidate(args: argparse.Namespace) -> None:
    pass_roots = [Path(path) for path in args.pass_roots]
    accepted_rows: dict[str, dict[str, Any]] = {}
    accepted_sources: dict[str, str] = {}
    for path in materialized_paths(pass_roots, "training_rows.jsonl"):
        for row in load_jsonl(path):
            pid = str(row["paper_id"])
            if pid in accepted_rows:
                continue
            accepted_rows[pid] = row
            accepted_sources[pid] = str(path)

    original_rows = load_jsonl(args.original)
    accepted_pids = set(accepted_rows)
    unresolved_rows = [row for row in original_rows if str(row["paper_id"]) not in accepted_pids]

    out_dir = args.consolidated
    rows = [accepted_rows[pid] for pid in sorted(accepted_rows)]
    write_jsonl(out_dir / "training_rows.jsonl", rows)
    write_jsonl(out_dir / "unresolved_rows.jsonl", sorted(unresolved_rows, key=lambda row: str(row["paper_id"])))
    write_json(
        out_dir / "run_manifest.json",
        {
            "created_by": "scripts/prepare_q5_grounded_shape_rewrite_retry_v1.py consolidate",
            "pass_roots": [str(path) for path in pass_roots],
            "accepted_rows": len(rows),
            "accepted_paper_ids": sorted(accepted_pids),
            "accepted_sources": accepted_sources,
            "unresolved_rows": len(unresolved_rows),
            "unresolved_paper_ids": [str(row["paper_id"]) for row in sorted(unresolved_rows, key=lambda item: str(item["paper_id"]))],
            "failure_family_counts": dict(collections.Counter(row.get("failure_family", "") for row in rows)),
        },
    )

    lines = [
        "# q5 Grounded-Shape Rewrite Consolidation",
        "",
        f"- Accepted rows: {len(rows)}",
        f"- Unresolved rows: {len(unresolved_rows)}",
        "",
        "## Accepted",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['paper_id']}`: {row['summary']}")
    lines.extend(["", "## Unresolved", ""])
    for row in sorted(unresolved_rows, key=lambda item: str(item["paper_id"])):
        lines.append(f"- `{row['paper_id']}`: {row.get('category', '')}")
    (out_dir / "REWRITE_TAKEAWAY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[consolidate] accepted={len(rows)} unresolved={len(unresolved_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["retry", "consolidate"])
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--consolidated", type=Path, default=DEFAULT_CONSOLIDATED)
    parser.add_argument(
        "--pass-roots",
        type=Path,
        nargs="+",
        default=DEFAULT_PASS_ROOTS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "retry":
        command_retry(args)
    elif args.command == "consolidate":
        command_consolidate(args)


if __name__ == "__main__":
    main()
