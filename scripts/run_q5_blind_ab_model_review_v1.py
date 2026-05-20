#!/usr/bin/env python3
"""Run a model-based blind A/B review over q5 blind-review shards."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from synthetic_summary_runtime import (
    load_dotenv,
    request_json_with_retries,
)


DEFAULT_MODEL = "qwen/qwen3-235b-a22b-2507"
CHOICES = {"summary_a", "summary_b", "tie", "neither"}


RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "q5_blind_ab_review_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "reviews": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "review_id": {"type": "string"},
                            "paper_id": {"type": "string"},
                            "preferred_summary": {
                                "type": "string",
                                "enum": ["summary_a", "summary_b", "tie", "neither"],
                            },
                            "more_accurate_summary": {
                                "type": "string",
                                "enum": ["summary_a", "summary_b", "tie", "neither"],
                            },
                            "notes": {"type": "string"},
                        },
                        "required": [
                            "review_id",
                            "paper_id",
                            "preferred_summary",
                            "more_accurate_summary",
                            "notes",
                        ],
                    },
                }
            },
            "required": ["reviews"],
        },
    },
}


SYSTEM_PROMPT = """You are reviewing generated paper summaries without knowing which model produced which side.

For each row, read the title and abstract, then compare Summary A and Summary B.

Judge naturally. Do not apply mechanical heuristics, and do not prefer or penalize a side just because it has a digit, is shorter, or uses a particular phrase. Prefer the summary that is more useful in a broad mobile paper feed while staying faithful to the source.

Choose:
- preferred_summary: the one you would rather show in a broad mobile paper feed
- more_accurate_summary: the one better supported by the title and abstract

Allowed values are summary_a, summary_b, tie, and neither.
Use tie when both are similarly good or similarly flawed.
Use neither only when both are clearly bad.
Notes must be concise, ideally under twelve words.
Return one JSON object only."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_messages(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    review_rows = [
        {
            "review_id": row["review_id"],
            "paper_id": row["paper_id"],
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "summary_a": row.get("summary_a", ""),
            "summary_b": row.get("summary_b", ""),
        }
        for row in rows
    ]
    user_prompt = (
        "Return JSON only with a reviews array containing exactly one review object "
        "for every input row. Copy review_id and paper_id exactly from each input row; "
        "do not infer or alter identifiers.\n\n"
        f"INPUT_ROWS:\n{json.dumps(review_rows, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def validate_reviews(source_rows: list[dict[str, Any]], parsed: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {(str(row["review_id"]), str(row["paper_id"])) for row in source_rows}
    expected_by_review_id = {str(row["review_id"]): str(row["paper_id"]) for row in source_rows}
    reviews = parsed.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("Response missing reviews array")
    output_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for idx, row in enumerate(reviews, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Review {idx} is not an object")
        review_id = str(row.get("review_id", ""))
        paper_id = str(row.get("paper_id", ""))
        key = (review_id, paper_id)
        if key not in expected:
            expected_paper_id = expected_by_review_id.get(review_id)
            if expected_paper_id is None:
                raise ValueError(f"Unexpected review key {key}")
            paper_id = expected_paper_id
            key = (review_id, paper_id)
        if key in seen:
            raise ValueError(f"Duplicate review key {key}")
        seen.add(key)
        preferred = str(row.get("preferred_summary", ""))
        accurate = str(row.get("more_accurate_summary", ""))
        notes = str(row.get("notes", "")).strip()
        if preferred not in CHOICES:
            raise ValueError(f"Bad preferred_summary for {review_id}: {preferred}")
        if accurate not in CHOICES:
            raise ValueError(f"Bad more_accurate_summary for {review_id}: {accurate}")
        if not notes:
            raise ValueError(f"Missing notes for {review_id}")
        output_rows.append(
            {
                "review_id": review_id,
                "paper_id": paper_id,
                "preferred_summary": preferred,
                "more_accurate_summary": accurate,
                "notes": notes,
            }
        )
    missing = expected - seen
    if missing:
        raise ValueError(f"Missing {len(missing)} reviews; first={sorted(missing)[:3]}")
    output_rows.sort(key=lambda row: row["review_id"])
    return output_rows


def review_shard(
    *,
    shard_path: Path,
    output_path: Path,
    api_base: str,
    api_key: str,
    model: str,
    max_output_tokens: int,
    timeout: int,
    max_retries: int,
    retry_backoff_seconds: float,
    chunk_size: int,
    force: bool,
) -> dict[str, Any]:
    rows = load_jsonl(shard_path)
    if output_path.exists() and not force:
        existing = load_jsonl(output_path)
        if len(existing) == len(rows):
            return {"shard": str(shard_path), "output": str(output_path), "rows": len(existing), "reused": True}

    all_reviewed: list[dict[str, Any]] = []
    chunks = [rows[index : index + chunk_size] for index in range(0, len(rows), chunk_size)]
    for chunk_index, chunk_rows in enumerate(chunks, start=1):
        messages = build_messages(chunk_rows)
        chunk_label = f"{shard_path.name}:chunk{chunk_index:02d}"
        chunk_out = output_path.parent / "chunk_reviews" / f"{output_path.stem}_chunk_{chunk_index:02d}.jsonl"
        if chunk_out.exists() and not force:
            existing_chunk = load_jsonl(chunk_out)
            if len(existing_chunk) == len(chunk_rows):
                all_reviewed.extend(existing_chunk)
                continue

        last_error: Exception | None = None
        for attempt in range(1, max_retries + 2):
            try:
                parsed, raw_text, raw_response = request_json_with_retries(
                    label=f"blind-review {chunk_label}",
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_output_tokens=max_output_tokens,
                    timeout=timeout,
                    max_retries=1,
                    retry_backoff_seconds=retry_backoff_seconds,
                    response_format=RESPONSE_FORMAT,
                    reasoning={"effort": "none", "exclude": True},
                    attempt_log_dir=output_path.parent / "raw_review_attempts",
                )
                reviewed = validate_reviews(chunk_rows, parsed)
                write_jsonl(chunk_out, reviewed)
                raw_dir = output_path.parent / "raw_review_responses"
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / f"{output_path.stem}_chunk_{chunk_index:02d}.json").write_text(
                    json.dumps(
                        {
                            "model": model,
                            "shard": str(shard_path),
                            "chunk_index": chunk_index,
                            "output": str(chunk_out),
                            "raw_text": raw_text,
                            "raw_response_id": raw_response.get("id", ""),
                        },
                        indent=2,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                all_reviewed.extend(reviewed)
                break
            except Exception as exc:
                last_error = exc
                print(
                    f"[blind-review:error] {chunk_label} attempt {attempt}/{max_retries + 1}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            raise RuntimeError(f"Failed {chunk_label} after retries: {last_error}")

    all_reviewed.sort(key=lambda row: row["review_id"])
    write_jsonl(output_path, all_reviewed)
    return {"shard": str(shard_path), "output": str(output_path), "rows": len(all_reviewed), "reused": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model-based blind A/B review.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=9000)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    api_base = args.api_base or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")

    manifest_path = args.root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = [Path(path) for path in manifest["shard_paths"]]
    expected_outputs = [Path(path) for path in manifest["expected_output_paths"]]
    if len(shards) != len(expected_outputs):
        raise SystemExit("manifest shard_paths and expected_output_paths length mismatch")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                review_shard,
                shard_path=shard,
                output_path=output,
                api_base=api_base,
                api_key=api_key,
                model=args.model,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
                chunk_size=args.chunk_size,
                force=args.force,
            )
            for shard, output in zip(shards, expected_outputs)
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[blind-review] {Path(result['output']).name} rows={result['rows']} reused={result['reused']}")

    results.sort(key=lambda row: row["output"])
    (args.root / "model_review_manifest.json").write_text(
        json.dumps(
            {
                "created_by": "scripts/run_q5_blind_ab_model_review_v1.py",
                "root": str(args.root),
                "model": args.model,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
