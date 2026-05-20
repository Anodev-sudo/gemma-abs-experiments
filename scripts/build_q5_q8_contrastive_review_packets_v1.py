#!/usr/bin/env python3
"""Build manual review packets from q8-vs-q5 contrastive heuristic hits."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "hf_bucket/experiments/sft/"
    "q5_promoted218_vs_q8_contrastive_mine_10000_v1"
)
DEFAULT_SCORED = DEFAULT_ROOT / "contrastive_heuristics_v1_tuned/scored_rows.jsonl"
DEFAULT_HP = DEFAULT_ROOT / "contrastive_heuristics_v1_tuned/high_precision_candidates.jsonl"
DEFAULT_OUT = DEFAULT_ROOT / "manual_review_hp_and_fog_union_gpt54mini"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build disjoint manual review shards for high-precision plus semantic-fog rows."
    )
    parser.add_argument("--scored-jsonl", default=str(DEFAULT_SCORED))
    parser.add_argument("--high-precision-jsonl", default=str(DEFAULT_HP))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--shards", type=int, default=10)
    return parser.parse_args()


def bucket_for(row: dict[str, Any], high_precision_ids: set[str]) -> str:
    paper_id = str(row["paper_id"])
    is_hp = paper_id in high_precision_ids
    is_fog = "semantic_fog_abstraction" in row.get("flags", [])
    if is_hp and is_fog:
        return "both"
    if is_hp:
        return "high_precision_only"
    if is_fog:
        return "semantic_fog_only"
    raise ValueError(f"Row {paper_id} is neither high-precision nor semantic-fog.")


def compact_review_row(row: dict[str, Any], *, review_id: str, source_bucket: str) -> dict[str, Any]:
    evidence = row.get("evidence", {})
    return {
        "review_id": review_id,
        "paper_id": str(row["paper_id"]),
        "source_bucket": source_bucket,
        "score": row.get("score"),
        "flags": row.get("flags", []),
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "q8_summary": row.get("q8_summary", ""),
        "q5_summary": row.get("q5_summary", ""),
        "missing_q8_source_terms": evidence.get("missing_q8_source_terms", []),
        "q5_unanchored_terms": evidence.get("q5_unanchored_terms", []),
    }


def write_prompt(path: Path, shard_count: int) -> None:
    text = f"""# Manual Review Instructions

You are reviewing q5 summaries against title, abstract, and q8 reference summaries.

For each row, judge whether the q5 summary is an actual failure. Do not trust the
heuristic flags; they are only how the row was selected.

Mark `actual_failure` as true when q5 is materially worse because it invents a
mechanism/entity, drops the main result, changes value or polarity, uses malformed
technical wording, or becomes vague semantic fog instead of a useful source-grounded
summary.

Mark `actual_failure` as false when q5 is acceptable, even if q8 is sharper or the
heuristic terms look suspicious. If q8 is wrong and q5 is fine, this is not a q5
failure.

Write one JSON object per input row to your assigned output file. Required fields:

- `review_id`
- `paper_id`
- `actual_failure`: boolean
- `failure_type`: one of `invented_or_wrong_entity`, `lost_main_result`,
  `value_or_polarity_error`, `semantic_fog`, `malformed_terms`, `digit_or_metric_issue`,
  `not_a_failure`, `both_bad_or_unclear`
- `severity`: one of `none`, `minor`, `major`
- `preferred`: one of `q8`, `q5`, `neither`, `tie`
- `note`: very concise, ideally under twelve words

There are {shard_count} shards. Review only your assigned shard and write only your
assigned output file.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    scored_rows = load_jsonl(Path(args.scored_jsonl))
    hp_rows = load_jsonl(Path(args.high_precision_jsonl))
    high_precision_ids = {str(row["paper_id"]) for row in hp_rows}
    fog_ids = {
        str(row["paper_id"])
        for row in scored_rows
        if "semantic_fog_abstraction" in row.get("flags", [])
    }
    selected_ids = high_precision_ids | fog_ids
    by_id = {str(row["paper_id"]): row for row in scored_rows}

    selected: list[dict[str, Any]] = []
    for paper_id in sorted(
        selected_ids,
        key=lambda pid: (
            -float(by_id[pid].get("score", 0.0)),
            by_id[pid].get("paper_id", ""),
        ),
    ):
        row = by_id[paper_id]
        source_bucket = bucket_for(row, high_precision_ids)
        selected.append(
            compact_review_row(
                row,
                review_id=f"hp_fog_{len(selected) + 1:04d}",
                source_bucket=source_bucket,
            )
        )

    out_dir = Path(args.output_dir)
    shards_dir = out_dir / "shards"
    results_dir = out_dir / "results"
    shards_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "review_rows_union.jsonl", selected)

    shard_count = args.shards
    if shard_count <= 0:
        raise ValueError("--shards must be positive")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for index, row in enumerate(selected):
        shards[index % shard_count].append(row)

    shard_manifests: list[dict[str, Any]] = []
    for shard_index, shard_rows in enumerate(shards, start=1):
        shard_path = shards_dir / f"review_shard_{shard_index:02d}.jsonl"
        result_path = results_dir / f"review_results_shard_{shard_index:02d}.jsonl"
        write_jsonl(shard_path, shard_rows)
        shard_manifests.append(
            {
                "shard": shard_index,
                "rows": len(shard_rows),
                "input_jsonl": str(shard_path),
                "result_jsonl": str(result_path),
                "bucket_counts": dict(Counter(row["source_bucket"] for row in shard_rows)),
            }
        )

    write_prompt(out_dir / "REVIEW_PROMPT.md", shard_count)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scored_jsonl": args.scored_jsonl,
        "high_precision_jsonl": args.high_precision_jsonl,
        "output_dir": str(out_dir),
        "rows": len(selected),
        "high_precision_rows": len(high_precision_ids),
        "semantic_fog_rows": len(fog_ids),
        "overlap_rows": len(high_precision_ids & fog_ids),
        "union_rows": len(selected_ids),
        "bucket_counts": dict(Counter(row["source_bucket"] for row in selected)),
        "shards": shard_manifests,
        "review_prompt": str(out_dir / "REVIEW_PROMPT.md"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[review-packets] wrote {len(selected)} unique rows to {out_dir}")
    print(f"[review-packets] bucket counts: {manifest['bucket_counts']}")
    print(f"[review-packets] shards: {shard_count}")


if __name__ == "__main__":
    main()
