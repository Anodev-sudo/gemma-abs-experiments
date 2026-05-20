#!/usr/bin/env python3
"""Build the checkpoint-218 grounded-shape rewrite training dataset.

This appends Kimi-authored, Qwen-verified grounded-shape rewrites to the
existing low-anchor scaffold. It does not author labels.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any

from generate_llama_stream import SYSTEM_PROMPT, USER_TEMPLATE


ROOT = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_residual218_grounded_shape_rewrite_v1_lr2e7"
)
DEFAULT_SCAFFOLD = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_residual218_grounded_shape_scaffold_v1/"
    "dataset_193residual_77hard_600selfanchor_18groundedkeep_v1/train.jsonl"
)
DEFAULT_SCAFFOLD_MANIFEST = DEFAULT_SCAFFOLD.with_name("manifest.json")
DEFAULT_REWRITES = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_residual218_grounded_shape_scaffold_v1/"
    "grounded_shape_rewrite_consolidated_v1/training_rows.jsonl"
)

DIGIT_RE = re.compile(r"\d")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")
SENTENCE_END_RE = re.compile(r"""[.!?]+(?:['")\]]+)?(?=\s|$)""")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def norm(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def sentence_count(text: str) -> int:
    text = norm(text)
    if not text:
        return 0
    endings = SENTENCE_END_RE.findall(text)
    return len(endings) if endings else 1


def messages(title: str, abstract: str, summary: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(title=title, abstract=abstract)},
        {"role": "assistant", "content": summary},
    ]


def loss_weight(weight: float, policy: str) -> dict[str, Any]:
    return {
        "enabled": weight != 1.0,
        "base_weight": weight,
        "spans": [],
        "policy": policy,
    }


def ensure_clean_label(row: dict[str, Any], *, min_words: int, max_words: int) -> None:
    summary = norm(row.get("summary", ""))
    pid = row.get("paper_id", "")
    reasons: list[str] = []
    if not summary:
        reasons.append("empty_summary")
    if DIGIT_RE.search(summary):
        reasons.append("digit")
    if sentence_count(summary) != 1:
        reasons.append("not_single_sentence")
    wc = word_count(summary)
    if wc < min_words:
        reasons.append("too_short")
    if wc > max_words:
        reasons.append("too_long")
    if reasons:
        raise SystemExit(f"Label failed rewrite dataset screen: {pid} {reasons} {summary!r}")


def normalize_rewrite_row(row: dict[str, Any], *, repeat_index: int, weight: float) -> dict[str, Any]:
    source_pid = str(row["paper_id"])
    summary = norm(row.get("summary", ""))
    title = norm(row.get("title", ""))
    abstract = norm(row.get("abstract", ""))
    if not title or not abstract:
        raise SystemExit(f"Missing title/abstract for rewrite row {source_pid}")
    out = dict(row)
    out["source_paper_id"] = source_pid
    out["paper_id"] = f"{source_pid}::grounded_shape_rewrite_{repeat_index:02d}"
    out["summary"] = summary
    out["grounded_shape_rewrite_repeat_index"] = repeat_index
    out["weighted_numeric_patch_row_type"] = "grounded_shape_rewrite_repair"
    out["dataset_component"] = "grounded_shape_rewrite"
    out["messages"] = messages(title, abstract, summary)
    out["loss_weight_hints"] = loss_weight(
        weight,
        f"grounded_shape_rewrite_v1 full assistant target weight {weight:g}",
    )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build q5 residual218 grounded-shape rewrite dataset.")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--scaffold-train", type=Path, default=DEFAULT_SCAFFOLD)
    parser.add_argument("--scaffold-manifest", type=Path, default=DEFAULT_SCAFFOLD_MANIFEST)
    parser.add_argument("--rewrite-rows", type=Path, default=DEFAULT_REWRITES)
    parser.add_argument("--rewrite-repeat", type=int, default=3)
    parser.add_argument("--rewrite-weight", type=float, default=1.5)
    parser.add_argument("--min-words", type=int, default=8)
    parser.add_argument("--max-words", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rewrite_repeat < 1:
        raise SystemExit("--rewrite-repeat must be >= 1")

    scaffold_rows = load_jsonl(args.scaffold_train)
    rewrite_source_rows = load_jsonl(args.rewrite_rows)
    for row in rewrite_source_rows:
        ensure_clean_label(row, min_words=args.min_words, max_words=args.max_words)

    rewrite_rows: list[dict[str, Any]] = []
    for row in sorted(rewrite_source_rows, key=lambda item: str(item["paper_id"])):
        for repeat_index in range(args.rewrite_repeat):
            rewrite_rows.append(
                normalize_rewrite_row(row, repeat_index=repeat_index, weight=args.rewrite_weight)
            )

    final = [*scaffold_rows, *rewrite_rows]
    duplicate_ids = [
        pid for pid, count in collections.Counter(str(row["paper_id"]) for row in final).items() if count > 1
    ]
    if duplicate_ids:
        raise SystemExit(f"Duplicate paper IDs in final dataset: {duplicate_ids[:20]}")
    digit_labels = [str(row["paper_id"]) for row in final if DIGIT_RE.search(str(row.get("summary", "")))]
    if digit_labels:
        raise SystemExit(f"Digit labels in final dataset: {digit_labels[:20]}")

    dataset_dir = (
        args.output_root
        / f"dataset_193residual_77hard_600selfanchor_18groundedkeep_"
        f"{len(rewrite_rows)}groundedrewrite_v1"
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(dataset_dir / "train.jsonl", final)
    write_jsonl(dataset_dir / "grounded_shape_rewrite_rows.jsonl", rewrite_rows)
    write_jsonl(dataset_dir / "unique_grounded_shape_rewrite_rows.jsonl", rewrite_source_rows)

    scaffold_manifest = {}
    if args.scaffold_manifest.exists():
        scaffold_manifest = json.loads(args.scaffold_manifest.read_text(encoding="utf-8"))

    manifest = {
        "created_by": "scripts/build_q5_residual218_grounded_shape_rewrite_v1_dataset.py",
        "dataset_path": str(dataset_dir / "train.jsonl"),
        "scaffold_train": str(args.scaffold_train),
        "scaffold_manifest": str(args.scaffold_manifest),
        "rewrite_rows_source": str(args.rewrite_rows),
        "start_adapter_intended": (
            "hf_bucket/experiments/sft/"
            "q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7/checkpoint-218"
        ),
        "rows": len(final),
        "scaffold_rows": len(scaffold_rows),
        "grounded_shape_rewrite_unique_rows": len(rewrite_source_rows),
        "grounded_shape_rewrite_rows": len(rewrite_rows),
        "grounded_shape_rewrite_repeat": args.rewrite_repeat,
        "digit_label_rows": 0,
        "duplicate_paper_ids": 0,
        "row_type_counts": dict(collections.Counter(row.get("weighted_numeric_patch_row_type", "") for row in final)),
        "component_counts": dict(collections.Counter(row.get("dataset_component", "") for row in final)),
        "rewrite_failure_family_counts": dict(collections.Counter(row.get("failure_family", "") for row in rewrite_rows)),
        "weight_policy": {
            **dict(scaffold_manifest.get("weight_policy", {})),
            "grounded_shape_rewrites": args.rewrite_weight,
        },
        "policy_notes": [
            "No Codex-authored labels are used.",
            "Grounded-shape rewrite rows are Kimi-authored and Qwen-verified.",
            "Rewrite rows use moderate pressure and repeat-specific training IDs.",
            "Checkpoint-218 self-anchors remain low weight from the scaffold dataset.",
        ],
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[grounded-shape-rewrite] wrote {len(final)} train rows to {dataset_dir / 'train.jsonl'}")
    print(f"[grounded-shape-rewrite] rewrite rows={len(rewrite_rows)} unique={len(rewrite_source_rows)}")


if __name__ == "__main__":
    main()
