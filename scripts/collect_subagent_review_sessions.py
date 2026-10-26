#!/usr/bin/env python3
"""Collect JSONL review decisions from Codex subagent session logs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_DECISIONS = {"silver_candidate", "bronze_only", "needs_rewrite", "reject"}


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def find_session(session_dir: Path, agent_id: str) -> Path:
    matches = sorted(session_dir.rglob(f"*{agent_id}.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"expected one session log for {agent_id}, found {len(matches)}")
    return matches[0]


def final_answer_text(session_path: Path) -> str:
    final_text = ""
    with session_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            payload = record.get("payload") or {}
            if (
                record.get("type") == "response_item"
                and payload.get("type") == "message"
                and payload.get("role") == "assistant"
                and payload.get("phase") == "final_answer"
            ):
                parts = payload.get("content") or []
                texts = [part.get("text", "") for part in parts if part.get("type") == "output_text"]
                final_text = "\n".join(texts)
    if not final_text:
        raise ValueError(f"no assistant final_answer found in {session_path}")
    return final_text


def parse_review_text(text: str, agent_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            raise ValueError(f"{agent_id}: line {line_number} is not JSONL: {line[:80]}")
        row = json.loads(line)
        if row.get("decision") not in ALLOWED_DECISIONS:
            raise ValueError(f"{agent_id}: invalid decision {row.get('decision')!r}")
        if not row.get("candidate_id"):
            raise ValueError(f"{agent_id}: missing candidate_id on line {line_number}")
        if not isinstance(row.get("issue_tags", []), list):
            raise ValueError(f"{agent_id}: issue_tags must be a list on line {line_number}")
        rows.append(row)
    return rows


def parse_mapping(values: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for value in values:
        if ":" not in value:
            raise ValueError(f"mapping must look like AGENT_ID:SHARD_INDEX, got {value!r}")
        agent_id, shard_text = value.rsplit(":", 1)
        mapping[agent_id] = int(shard_text)
    return mapping


def collect_existing_reviews(review_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(review_dir.glob("review_shard_*.jsonl")):
        rows.extend(iter_jsonl(path))
    return rows


def validate_existing_reviews(run_root: Path, review_dir: Path) -> list[str]:
    errors: list[str] = []
    for review_path in sorted(review_dir.glob("review_shard_*.jsonl")):
        suffix = review_path.stem.removeprefix("review_shard_")
        shard_path = run_root / f"shard_{suffix}.jsonl"
        if not shard_path.exists():
            errors.append(f"{review_path}: matching shard is missing")
            continue
        shard_ids = [str(row["candidate_id"]) for row in iter_jsonl(shard_path)]
        review_rows = iter_jsonl(review_path)
        review_ids = [str(row.get("candidate_id", "")) for row in review_rows]
        if review_ids != shard_ids:
            missing = sorted(set(shard_ids) - set(review_ids))
            extra = sorted(set(review_ids) - set(shard_ids))
            errors.append(
                f"{review_path}: candidate_id mismatch missing={len(missing)} extra={len(extra)}"
            )
        for row_index, row in enumerate(review_rows, 1):
            if row.get("decision") not in ALLOWED_DECISIONS:
                errors.append(f"{review_path}:{row_index}: invalid decision {row.get('decision')!r}")
            if not isinstance(row.get("issue_tags", []), list):
                errors.append(f"{review_path}:{row_index}: issue_tags must be a list")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, default=Path.home() / ".codex" / "sessions")
    parser.add_argument("--map", action="append", default=[], help="AGENT_ID:SHARD_INDEX")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--combined-name", default="combined_reviewed.jsonl")
    parser.add_argument("--combine-only", action="store_true")
    args = parser.parse_args()

    if not args.map and not args.combine_only:
        raise SystemExit("at least one --map AGENT_ID:SHARD_INDEX is required")

    review_dir = args.output_dir or args.run_root / "reviews"
    manifest_path = args.run_root / "shard_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}

    collected: dict[str, Any] = {}
    for agent_id, shard_index in parse_mapping(args.map).items():
        shard_path = args.run_root / f"shard_{shard_index:03d}.jsonl"
        shard_rows = iter_jsonl(shard_path)
        shard_ids = [str(row["candidate_id"]) for row in shard_rows]

        session_path = find_session(args.session_dir, agent_id)
        review_rows = parse_review_text(final_answer_text(session_path), agent_id)
        review_ids = [str(row["candidate_id"]) for row in review_rows]
        if review_ids != shard_ids:
            missing = sorted(set(shard_ids) - set(review_ids))
            extra = sorted(set(review_ids) - set(shard_ids))
            raise ValueError(
                f"candidate_id mismatch for shard {shard_index:03d}: "
                f"missing={len(missing)} extra={len(extra)}"
            )

        review_path = review_dir / f"review_shard_{shard_index:03d}.jsonl"
        write_jsonl(review_path, review_rows)
        collected[f"shard_{shard_index:03d}"] = {
            "agent_id": agent_id,
            "session_path": session_path.as_posix(),
            "review_path": review_path.as_posix(),
            "rows": len(review_rows),
            "decision_counts": dict(sorted(Counter(row["decision"] for row in review_rows).items())),
        }

    validation_errors = validate_existing_reviews(args.run_root, review_dir)
    if validation_errors:
        raise SystemExit("\n".join(validation_errors))

    combined_rows = collect_existing_reviews(review_dir)
    duplicate_count = len(combined_rows) - len({str(row["candidate_id"]) for row in combined_rows})
    combined_path = review_dir / args.combined_name
    write_jsonl(combined_path, combined_rows)

    reviewed_shards = sorted(path.stem.removeprefix("review_shard_") for path in review_dir.glob("review_shard_*.jsonl"))
    expected_shard_count = int(manifest.get("shard_count") or 0)
    missing_shards: list[str] = []
    if expected_shard_count:
        expected = {f"{index:03d}" for index in range(expected_shard_count)}
        missing_shards = sorted(expected - set(reviewed_shards))

    collection_manifest = {
        "run_root": args.run_root.as_posix(),
        "review_dir": review_dir.as_posix(),
        "combined_review_jsonl": combined_path.as_posix(),
        "collected_this_run": collected,
        "reviewed_rows": len(combined_rows),
        "decision_counts": dict(sorted(Counter(row["decision"] for row in combined_rows).items())),
        "duplicate_candidate_ids": duplicate_count,
        "reviewed_shards": reviewed_shards,
        "missing_shards": missing_shards,
    }
    collection_manifest_path = review_dir / "review_collection_manifest.json"
    collection_manifest_path.write_text(
        json.dumps(collection_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(collection_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
