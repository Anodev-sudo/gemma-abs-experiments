#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

DEFAULT_GENERATIONS_JSONL = (
    "hf_bucket/experiments/mlx_lora/review_comparisons/"
    "q8_vs_q4_hardening_mine_1000_v1/generations.jsonl"
)
DEFAULT_OUTPUT_DIR = "hf_bucket/experiments/mlx_lora/q4_hardening_distill_v1/review_packets"
DEFAULT_SEED = 20260501
DEFAULT_PACKET_SIZE = 25

ISSUE_TAGS = (
    "invented_number",
    "wrong_number_binding",
    "unsupported_entity",
    "unsupported_mechanism",
    "causal_drift",
    "polarity_flip",
    "overstrong_claim",
    "misses_point",
    "malformed_repetition",
    "non_answer",
)
DECISION_HEADERS = (
    "review_id",
    "winner",
    "left_severe",
    "right_severe",
    "left_issue_tags",
    "right_issue_tags",
    "left_target_usable",
    "right_target_usable",
    "confidence",
    "note",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare blinded q4-hardening review packets from paired generations."
    )
    parser.add_argument("--generations-jsonl", default=DEFAULT_GENERATIONS_JSONL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--q8-label", default="q8_promoted")
    parser.add_argument("--q4-label", default="q4_lineage")
    parser.add_argument("--packet-size", type=int, default=DEFAULT_PACKET_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_blank_decisions(path: Path, review_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECISION_HEADERS, delimiter="\t")
        writer.writeheader()
        for review_id in review_ids:
            writer.writerow({"review_id": review_id})


def side_payload(row: dict[str, Any], source: str) -> dict[str, str]:
    if source == "left":
        return {
            "label": str(row["left_label"]),
            "summary": str(row["left_summary"]),
            "summary_raw": str(row.get("left_summary_raw", "")),
        }
    if source == "right":
        return {
            "label": str(row["right_label"]),
            "summary": str(row["right_summary"]),
            "summary_raw": str(row.get("right_summary_raw", "")),
        }
    raise ValueError(f"Unknown side source {source!r}")


def label_to_side(row: dict[str, Any], label: str) -> str:
    matches = [
        side
        for side in ("left", "right")
        if str(row.get(f"{side}_label", "")) == label
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one side with label {label!r} for paper {row.get('paper_id')}"
        )
    return matches[0]


def build_review_rows(
    generation_rows: list[dict[str, Any]],
    *,
    q8_label: str,
    q4_label: str,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    review_rows: list[dict[str, Any]] = []
    side_maps: list[dict[str, Any]] = []
    for index, row in enumerate(generation_rows, start=1):
        q8_source = label_to_side(row, q8_label)
        q4_source = label_to_side(row, q4_label)
        q8_payload = side_payload(row, q8_source)
        q4_payload = side_payload(row, q4_source)
        q8_on_left = rng.choice([True, False])
        left_payload = q8_payload if q8_on_left else q4_payload
        right_payload = q4_payload if q8_on_left else q8_payload
        review_id = f"q4_hardening:{index:04d}"
        review_rows.append(
            {
                "review_id": review_id,
                "paper_id": row["paper_id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "primary_category": row.get("primary_category"),
                "categories": row.get("categories", []),
                "created_date": row.get("created_date"),
                "selection_bucket": row.get("selection_bucket") or row.get("split"),
                "q4_hardening_axes": row.get("q4_hardening_axes", []),
                "prompt_version": row.get("prompt_version"),
                "input_mode": row.get("input_mode"),
                "left_summary": left_payload["summary"],
                "right_summary": right_payload["summary"],
                "review_instruction": (
                    "Blinded pairwise review. Choose the better feed summary, mark severe "
                    "faithfulness/corruption separately per side, and mark each side target_usable "
                    "only if that side is faithful enough to train from unchanged."
                ),
                "issue_tag_options": list(ISSUE_TAGS),
            }
        )
        side_maps.append(
            {
                "review_id": review_id,
                "paper_id": row["paper_id"],
                "left_source": "q8" if q8_on_left else "q4",
                "right_source": "q4" if q8_on_left else "q8",
                "q8_original_side": q8_source,
                "q4_original_side": q4_source,
                "q8_label": q8_payload["label"],
                "q4_label": q4_payload["label"],
                "q8_summary": q8_payload["summary"],
                "q4_summary": q4_payload["summary"],
            }
        )
    return review_rows, side_maps


def packetize(rows: list[dict[str, Any]], packet_size: int) -> list[list[dict[str, Any]]]:
    if packet_size <= 0:
        raise ValueError("--packet-size must be positive")
    return [rows[start : start + packet_size] for start in range(0, len(rows), packet_size)]


def build_readme(manifest: dict[str, Any]) -> str:
    return (
        "# q4 Hardening Blinded Review Packets\n\n"
        "Review objective: find q4-only severe failures while preserving blinded q8/q4 labels.\n\n"
        "Fill each `decisions.tsv` row with:\n"
        "- `winner`: `left`, `right`, `tie`, or `both_bad`\n"
        "- `left_severe` / `right_severe`: `yes` or `no`\n"
        "- `left_issue_tags` / `right_issue_tags`: comma-separated tags from the packet\n"
        "- `left_target_usable` / `right_target_usable`: `yes` only if the side is faithful enough to train from unchanged\n"
        "- `confidence`: `high`, `medium`, or `low`\n"
        "- `note`: short reason, especially for severe rows\n\n"
        "Issue tag guide:\n"
        "- `invented_number`: explicit number, quantity, date, unit, or percentage is absent from the source\n"
        "- `wrong_number_binding`: source number is real, but attached to the wrong metric, entity, period, comparison, or direction\n"
        "- `unsupported_entity`: named entity, method, acronym, domain, or actor is absent or swapped\n"
        "- `unsupported_mechanism`: method, pathway, or mechanism is not supported by the source\n"
        "- `causal_drift`: effect, cause, improvement, reduction, or explanation is too strong or misattributed\n"
        "- `polarity_flip`: direction is reversed, negated incorrectly, or says the opposite of the source\n"
        "- `overstrong_claim`: claim is stronger or broader than the source supports\n"
        "- `misses_point`: summary is about a real side detail but misses the main point\n"
        "- `malformed_repetition`: repeated, garbled, truncated, or mechanically malformed text\n"
        "- `non_answer`: does not answer the summary task\n\n"
        "Do not infer which side is q8 or q4. The side map is intentionally separate.\n\n"
        f"- Review rows: `{manifest['review_rows']}`\n"
        f"- Packet size: `{manifest['packet_size']}`\n"
        f"- Primary packets: `{manifest['primary_packets']}`\n"
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{output_dir} already exists and is not empty; pass --overwrite")

    generation_rows = load_jsonl(Path(args.generations_jsonl))
    review_rows, side_maps = build_review_rows(
        generation_rows,
        q8_label=args.q8_label,
        q4_label=args.q4_label,
        seed=args.seed,
    )
    packets = packetize(review_rows, args.packet_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "review_items.jsonl", review_rows)
    write_jsonl(output_dir / "side_map.jsonl", side_maps)
    for packet_index, packet_rows in enumerate(packets):
        packet_dir = output_dir / "primary" / f"packet_{packet_index:02d}"
        write_jsonl(packet_dir / "review_items.jsonl", packet_rows)
        write_blank_decisions(
            packet_dir / "decisions.tsv",
            [str(row["review_id"]) for row in packet_rows],
        )

    manifest = {
        "created_at": utc_now_iso(),
        "generations_jsonl": args.generations_jsonl,
        "output_dir": str(output_dir),
        "q8_label": args.q8_label,
        "q4_label": args.q4_label,
        "seed": args.seed,
        "packet_size": args.packet_size,
        "review_rows": len(review_rows),
        "primary_packets": len(packets),
        "side_counts": dict(Counter(row["left_source"] for row in side_maps)),
        "decision_headers": DECISION_HEADERS,
        "issue_tags": ISSUE_TAGS,
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "README.md").write_text(build_readme(manifest), encoding="utf-8")
    print(f"[q4-hardening-packets] wrote {len(review_rows)} review rows to {output_dir}")
    print(f"[q4-hardening-packets] primary packets: {len(packets)}")


if __name__ == "__main__":
    main()
