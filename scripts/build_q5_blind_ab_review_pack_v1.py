#!/usr/bin/env python3
"""Build a blind A/B review pack for q5 checkpoint comparison."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_hard_attractor_booster_v1_lr1e6/eval/heuristic_inputs/combined_454.jsonl")
DEFAULT_A = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/eval/style_norm_combined_454_outputs/checkpoint700_outputs_454.jsonl")
DEFAULT_B = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin150_style_replay_v1_lr5e7_steps225/eval/combined_454_outputs/checkpoint_25_outputs_454.jsonl")
DEFAULT_OUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin150_style_replay_v1_lr5e7_steps225/eval/blind_ab_ckpt700_vs_replay25_200")

DIGIT_RE = re.compile(r"\d")
NUMBER_DENSE_RE = re.compile(r"(?:\d.*){2,}")
FORMULAISH_RE = re.compile(r"[$<>≤≥≈]|\\[A-Za-z]+|\^|_|<sup>|</sup>")
STRESS_IDS = {"1901.06548", "2111.10635", "2303.09722", "2406.04373"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def norm(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def output_by_pid(path: Path) -> dict[str, str]:
    return {
        str(row["paper_id"]): norm(row.get("normalized_output") or row.get("raw_output"))
        for row in load_jsonl(path)
        if row.get("paper_id")
    }


def score_adversarial(row: dict[str, Any], base: str, candidate: str) -> int:
    text = " ".join([row.get("title", ""), row.get("abstract", ""), base, candidate])
    score = 0
    if row.get("paper_id") in STRESS_IDS:
        score += 100
    if row.get("split") == "numeric_ckpt700_risk_pool350":
        score += 20
    if row.get("split") == "fresh_recent_normal_100":
        score += 5
    if DIGIT_RE.search(candidate):
        score += 30
    if DIGIT_RE.search(base):
        score += 15
    if NUMBER_DENSE_RE.search(base) or NUMBER_DENSE_RE.search(candidate):
        score += 10
    if FORMULAISH_RE.search(text):
        score += 8
    if len(base.split()) > 15 or len(candidate.split()) > 15:
        score += 4
    return score


def select_rows(
    source_rows: list[dict[str, Any]],
    base_outputs: dict[str, str],
    candidate_outputs: dict[str, str],
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    usable = [row for row in source_rows if str(row.get("paper_id")) in base_outputs and str(row.get("paper_id")) in candidate_outputs]
    by_pid = {str(row["paper_id"]): row for row in usable}
    selected: list[str] = []

    def add_many(pids: list[str], limit: int | None = None) -> None:
        nonlocal selected
        added = 0
        for pid in pids:
            if pid in by_pid and pid not in selected:
                selected.append(pid)
                added += 1
                if limit is not None and added >= limit:
                    break

    add_many(sorted(STRESS_IDS))

    candidate_digit = sorted(pid for pid, out in candidate_outputs.items() if pid in by_pid and DIGIT_RE.search(out))
    add_many(candidate_digit)

    numeric = [row for row in usable if row.get("split") == "numeric_ckpt700_risk_pool350"]
    numeric_scored = sorted(
        numeric,
        key=lambda row: (
            score_adversarial(row, base_outputs[str(row["paper_id"])], candidate_outputs[str(row["paper_id"])]),
            str(row["paper_id"]),
        ),
        reverse=True,
    )
    add_many([str(row["paper_id"]) for row in numeric_scored], limit=120)

    fresh = [row for row in usable if row.get("split") == "fresh_recent_normal_100"]
    rng.shuffle(fresh)
    add_many([str(row["paper_id"]) for row in fresh], limit=50)

    remaining = [str(row["paper_id"]) for row in usable if str(row["paper_id"]) not in selected]
    rng.shuffle(remaining)
    add_many(remaining)
    selected = selected[:count]

    if len(selected) != count:
        raise SystemExit(f"Selected {len(selected)} rows, expected {count}")

    selected_rows = [by_pid[pid] for pid in selected]
    manifest = {
        "selected_rows": len(selected_rows),
        "stress_rows": sum(1 for row in selected_rows if str(row["paper_id"]) in STRESS_IDS),
        "candidate_digit_rows": sum(1 for row in selected_rows if DIGIT_RE.search(candidate_outputs[str(row["paper_id"])])),
        "base_digit_rows": sum(1 for row in selected_rows if DIGIT_RE.search(base_outputs[str(row["paper_id"])])),
        "split_counts": {},
    }
    for row in selected_rows:
        split = str(row.get("split", ""))
        manifest["split_counts"][split] = manifest["split_counts"].get(split, 0) + 1
    return selected_rows, manifest


def build_blind_rows(
    selected_rows: list[dict[str, Any]],
    base_outputs: dict[str, str],
    candidate_outputs: dict[str, str],
    seed: int,
    base_label: str,
    candidate_label: str,
    previous_key: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed + 17)
    review_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for index, source in enumerate(selected_rows, start=1):
        pid = str(source["paper_id"])
        base = base_outputs[pid]
        candidate = candidate_outputs[pid]
        review_id = f"blind_ab_{index:03d}"
        if previous_key:
            previous = previous_key[review_id]
            candidate_is_a = str(previous["summary_a_model"]).startswith("candidate_")
        else:
            candidate_is_a = rng.random() < 0.5
        summary_a = candidate if candidate_is_a else base
        summary_b = base if candidate_is_a else candidate
        key_rows.append(
            {
                "review_id": review_id,
                "paper_id": pid,
                "summary_a_model": candidate_label if candidate_is_a else base_label,
                "summary_b_model": base_label if candidate_is_a else candidate_label,
                "split": source.get("split", ""),
            }
        )
        review_rows.append(
            {
                "review_id": review_id,
                "paper_id": pid,
                "title": source.get("title", ""),
                "abstract": source.get("abstract", ""),
                "summary_a": summary_a,
                "summary_b": summary_b,
            }
        )
    return review_rows, key_rows


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
    prompt = f"""# q5 Blind A/B Review Prompt

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

There are {row_count} rows across {shards} shards. Each reviewer owns one shard and one output file.
"""
    path.write_text(prompt, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build blind q5 A/B review packet.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint700-output", type=Path, default=DEFAULT_A)
    parser.add_argument("--candidate-output", type=Path, default=DEFAULT_B)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--shard-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--reuse-key", type=Path, default=None, help="Optional prior blind_key JSONL; reuses paper order and candidate side.")
    parser.add_argument("--base-label", default="checkpoint_700")
    parser.add_argument("--candidate-label", default="candidate_replay_checkpoint_150")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = load_jsonl(args.source)
    checkpoint700 = output_by_pid(args.checkpoint700_output)
    candidate = output_by_pid(args.candidate_output)
    previous_key = None
    if args.reuse_key:
        previous_key_rows = load_jsonl(args.reuse_key)
        previous_key = {str(row["review_id"]): row for row in previous_key_rows}
        source_by_pid = {str(row["paper_id"]): row for row in source_rows}
        selected_rows = [source_by_pid[str(row["paper_id"])] for row in previous_key_rows]
        missing_outputs = [
            str(row["paper_id"])
            for row in selected_rows
            if str(row["paper_id"]) not in checkpoint700 or str(row["paper_id"]) not in candidate
        ]
        if missing_outputs:
            raise SystemExit(f"Missing outputs for reused rows: {missing_outputs[:10]}")
        selection_manifest = {
            "selected_rows": len(selected_rows),
            "stress_rows": sum(1 for row in selected_rows if str(row["paper_id"]) in STRESS_IDS),
            "candidate_digit_rows": sum(1 for row in selected_rows if DIGIT_RE.search(candidate[str(row["paper_id"])])),
            "base_digit_rows": sum(1 for row in selected_rows if DIGIT_RE.search(checkpoint700[str(row["paper_id"])])),
            "split_counts": {},
            "reused_key": str(args.reuse_key),
        }
        for row in selected_rows:
            split = str(row.get("split", ""))
            selection_manifest["split_counts"][split] = selection_manifest["split_counts"].get(split, 0) + 1
    else:
        selected_rows, selection_manifest = select_rows(source_rows, checkpoint700, candidate, args.count, args.seed)
    review_rows, key_rows = build_blind_rows(
        selected_rows,
        checkpoint700,
        candidate,
        args.seed,
        base_label=args.base_label,
        candidate_label=args.candidate_label,
        previous_key=previous_key,
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
    for shard_index, offset in enumerate(range(0, len(review_rows), args.shard_size), start=1):
        shard = review_rows[offset : offset + args.shard_size]
        shard_path = out / f"blind_review_shard_{shard_index:02d}.jsonl"
        result_path = out / f"blind_review_results_shard_{shard_index:02d}.jsonl"
        write_jsonl(shard_path, shard)
        write_markdown(out / f"BLIND_REVIEW_SHARD_{shard_index:02d}.md", shard)
        shard_paths.append(str(shard_path))
        expected_output_paths.append(str(result_path))
    write_subagent_prompt(out / "SUBAGENT_BLIND_AB_REVIEW_PROMPT.md", row_count, len(shard_paths))

    manifest = {
        "created_by": "scripts/build_q5_blind_ab_review_pack_v1.py",
        "source": str(args.source),
        "checkpoint700_output": str(args.checkpoint700_output),
        "candidate_output": str(args.candidate_output),
        "output_root": str(out),
        "review_path": str(review_path),
        "key_path": str(key_path),
        "shard_paths": shard_paths,
        "expected_output_paths": expected_output_paths,
        "count": args.count,
        "shard_size": args.shard_size,
        "seed": args.seed,
        **selection_manifest,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[blind-ab] wrote {review_path}")
    print(f"[blind-ab] wrote {key_path}")
    print(f"[blind-ab] wrote {len(shard_paths)} shards")


if __name__ == "__main__":
    main()
