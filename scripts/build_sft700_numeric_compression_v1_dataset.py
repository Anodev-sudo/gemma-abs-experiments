#!/usr/bin/env python3
"""Build second-stage compression dataset from numeric checkpoint overlong outputs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def wc(text: str) -> int:
    return len(str(text).split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source-dataset",
        default="hf_bucket/experiments/sft/q5_step2400_sft700_numeric_role_curriculum_v1/dataset_majority_synthetic_v1/train.jsonl",
    )
    ap.add_argument(
        "--numeric-checkpoint-outputs",
        default="hf_bucket/experiments/sft/q5_step2400_sft700_numeric_compression_v1/ckpt175_outputs_on_numeric_curriculum_428.jsonl",
    )
    ap.add_argument(
        "--output-dir",
        default="hf_bucket/experiments/sft/q5_step2400_sft700_numeric_compression_v1/dataset_over13_v1",
    )
    ap.add_argument("--min-output-words", type=int, default=14)
    args = ap.parse_args()

    source_rows = load_jsonl(Path(args.source_dataset))
    outputs = load_jsonl(Path(args.numeric_checkpoint_outputs))
    by_pid = {row["paper_id"]: row for row in source_rows}

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for out in outputs:
        pid = out["paper_id"]
        row = by_pid[pid]
        pred = out.get("normalized_output", "")
        pred_wc = wc(pred)
        target_wc = wc(row.get("summary", ""))
        if pred_wc < args.min_output_words:
            skipped.append({"paper_id": pid, "reason": "numeric_checkpoint_output_not_over_threshold", "output_word_count": pred_wc})
            continue
        if not 10 <= target_wc <= 15:
            skipped.append({"paper_id": pid, "reason": "target_not_10_15", "target_word_count": target_wc})
            continue
        copied = dict(row)
        copied["source_split"] = "numeric_ckpt175_compression_over13_v1"
        copied["patch_component"] = "numeric_ckpt175_over13_compression"
        copied["curriculum_component"] = "numeric_ckpt175_over13_compression"
        copied["numeric_checkpoint_base"] = "q5_step2400_sft700_numeric_role_curriculum_v1/checkpoint-175"
        copied["numeric_checkpoint_output"] = pred
        copied["numeric_checkpoint_output_word_count"] = pred_wc
        copied["compression_target_word_count"] = target_wc
        copied["compression_word_delta"] = pred_wc - target_wc
        rows.append(copied)

    rows.sort(key=lambda r: (-int(r.get("compression_word_delta", 0)), r.get("paper_id", "")))
    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "train.jsonl", rows)
    write_jsonl(out_dir / "skipped_rows.jsonl", skipped)

    manifest = {
        "dataset_id": "q5_step2400_sft700_numeric_compression_over13_v1",
        "purpose": "Second-stage compression patch from numeric checkpoint-175: train only cases where checkpoint-175 became >13 words back to existing verified concise labels.",
        "source_dataset": args.source_dataset,
        "numeric_checkpoint_outputs": args.numeric_checkpoint_outputs,
        "selection_rule": f"numeric checkpoint output word count >= {args.min_output_words}; existing target is 10-15 words",
        "row_count": len(rows),
        "skipped_count": len(skipped),
        "component_counts": dict(Counter(r.get("patch_component", "") for r in rows)),
        "original_component_counts": dict(Counter(r.get("curriculum_component", "") for r in source_rows if r.get("paper_id") in {x.get("paper_id") for x in rows})),
        "target_word_count_counts": dict(sorted(Counter(wc(r.get("summary", "")) for r in rows).items())),
        "numeric_checkpoint_output_word_count_counts": dict(sorted(Counter(r.get("numeric_checkpoint_output_word_count", 0) for r in rows).items())),
        "avg_target_words": sum(wc(r.get("summary", "")) for r in rows) / len(rows) if rows else 0,
        "avg_numeric_checkpoint_output_words": sum(r.get("numeric_checkpoint_output_word_count", 0) for r in rows) / len(rows) if rows else 0,
        "notes": [
            "No new teacher labels are introduced; targets are the existing Kimi/Qwen-reviewed real labels or deterministic current-failed synthetic labels.",
            "The numeric checkpoint output is stored for audit only and is not part of the student inference prompt.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Numeric compression over13 v1\n\n"
        "Second-stage dataset from numeric checkpoint-175 overlong outputs back to existing concise targets.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
