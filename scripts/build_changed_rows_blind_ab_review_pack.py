#!/usr/bin/env python3
"""Build a blind A/B packet only for rows whose candidate output changed."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def norm(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def output_by_pid(path: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for row in load_jsonl(path):
        paper_id = str(row.get("paper_id", ""))
        if not paper_id:
            continue
        outputs[paper_id] = norm(row.get("normalized_output") or row.get("raw_output"))
    return outputs


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Blind A/B Summary Review",
        "",
        "For each row, read the title and abstract, then compare Summary A and Summary B.",
        "",
        "Choose:",
        "- preferred summary: the one you would rather show in a broad mobile paper feed",
        "- more accurate summary: the one better supported by the title and abstract",
        "",
        "Do not apply mechanical rules. Judge usefulness, naturalness, specificity, and faithfulness.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['review_id']} `{row['paper_id']}`",
                f"Title: {row.get('title', '')}",
                "",
                f"Abstract: {row.get('abstract', '')}",
                "",
                f"Summary A: {row['summary_a']}",
                f"Summary B: {row['summary_b']}",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_subagent_prompt(path: Path, row_count: int, shards: int) -> None:
    prompt = f"""# Changed-Row Blind A/B Review Prompt

You are reviewing generated paper summaries without knowing which model produced which side.

For each row, read the title and abstract, then compare Summary A and Summary B.

Judge naturally. Do not apply mechanical heuristics, and do not prefer or penalize a side just because it has a digit, is shorter, or uses a particular phrase. Prefer the summary that is more useful in a broad mobile paper feed while staying faithful to the source.

For every input row, write exactly one JSONL object:

```json
{{"review_id":"blind_ab_001","paper_id":"...","preferred_summary":"summary_a","more_accurate_summary":"summary_b","notes":"A is smoother, B preserves result"}}
```

Allowed values:

- `preferred_summary`: `summary_a`, `summary_b`, `tie`, or `neither`
- `more_accurate_summary`: `summary_a`, `summary_b`, `tie`, or `neither`
- `notes`: very concise, ideally under twelve words

Use `tie` when both are similarly good or similarly flawed. Use `neither` only when both are clearly bad.

There are {row_count} changed rows across {shards} shards. Each reviewer owns one shard and one output file.
"""
    path.write_text(prompt, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path, required=True)
    parser.add_argument("--reference-candidate-output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--shard-size", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    source_rows = {str(row["paper_id"]): row for row in load_jsonl(args.source)}
    baseline = output_by_pid(args.baseline_output)
    reference_candidate = output_by_pid(args.reference_candidate_output)
    candidate = output_by_pid(args.candidate_output)

    changed_pids = [
        paper_id
        for paper_id in sorted(source_rows)
        if paper_id in baseline
        and paper_id in reference_candidate
        and paper_id in candidate
        and reference_candidate[paper_id] != candidate[paper_id]
    ]
    review_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for paper_id in changed_pids:
        source = source_rows[paper_id]
        candidate_is_a = rng.random() < 0.5
        summary_a = candidate[paper_id] if candidate_is_a else baseline[paper_id]
        summary_b = baseline[paper_id] if candidate_is_a else candidate[paper_id]
        review_id = str(source.get("review_id") or source.get("eval_id") or paper_id)
        review_rows.append(
            {
                "review_id": review_id,
                "paper_id": paper_id,
                "title": source.get("title", ""),
                "abstract": source.get("abstract", ""),
                "summary_a": summary_a,
                "summary_b": summary_b,
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "paper_id": paper_id,
                "summary_a_model": args.candidate_label if candidate_is_a else args.base_label,
                "summary_b_model": args.base_label if candidate_is_a else args.candidate_label,
                "split": source.get("split", ""),
            }
        )

    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    row_count = len(review_rows)
    review_path = out / f"blind_review_{row_count}.jsonl"
    key_path = out / f"blind_key_{row_count}.jsonl"
    write_jsonl(review_path, review_rows)
    write_jsonl(key_path, key_rows)

    shard_paths: list[str] = []
    expected_output_paths: list[str] = []
    for shard_index, offset in enumerate(range(0, row_count, args.shard_size), start=1):
        shard = review_rows[offset : offset + args.shard_size]
        shard_path = out / f"blind_review_shard_{shard_index:02d}.jsonl"
        result_path = out / f"blind_review_results_shard_{shard_index:02d}.jsonl"
        write_jsonl(shard_path, shard)
        write_markdown(out / f"BLIND_REVIEW_SHARD_{shard_index:02d}.md", shard)
        shard_paths.append(str(shard_path))
        expected_output_paths.append(str(result_path))
    write_subagent_prompt(out / "SUBAGENT_BLIND_AB_REVIEW_PROMPT.md", row_count, len(shard_paths))

    manifest = {
        "created_by": "scripts/build_changed_rows_blind_ab_review_pack.py",
        "source": str(args.source),
        "baseline_output": str(args.baseline_output),
        "reference_candidate_output": str(args.reference_candidate_output),
        "candidate_output": str(args.candidate_output),
        "output_root": str(out),
        "review_path": str(review_path),
        "key_path": str(key_path),
        "shard_paths": shard_paths,
        "expected_output_paths": expected_output_paths,
        "count": row_count,
        "shard_size": args.shard_size,
        "seed": args.seed,
        "base_label": args.base_label,
        "candidate_label": args.candidate_label,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[changed-blind-ab] wrote {review_path}")
    print(f"[changed-blind-ab] wrote {key_path}")
    print(f"[changed-blind-ab] wrote {len(shard_paths)} shards for {row_count} changed rows")


if __name__ == "__main__":
    main()
