#!/usr/bin/env python3
"""Build a compact q8-win cleanup dataset after broad cleanup checkpoint 30."""
from __future__ import annotations

import collections
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(
    "hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/"
    "cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30"
)
OUT_DIR = ROOT / "dataset_78cleanup_x3_v1"
SOURCE = Path("hf_dataset/releases/recent_3y_balanced_10746_q5_broad_cleanup30_q8win_v1/training_rows.jsonl")

SYSTEM_PROMPT = (
    "You write one-sentence paper summaries for a mobile research feed. "
    "Keep them concrete, readable, and useful. Output only the summary sentence."
)

USER_TEMPLATE = (
    "The app already shows the title, author list, and date. "
    "Write one sentence that tells a reader why they might open this paper. "
    "Aim for 10 to 15 words. Avoid repeating the title. "
    "Avoid filler like \"This paper\" or \"We propose\". "
    "Do not mention authors or dates. Keep the wording natural and specific.\n\n"
    "Title:\n{title}\n\n"
    "Abstract:\n{abstract}"
)

REPEAT = 3
WEIGHT = 2.0


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


def messages(title: str, abstract: str, summary: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(title=title, abstract=abstract)},
        {"role": "assistant", "content": summary},
    ]


def weighted_copy(row: dict[str, Any], repeat_index: int) -> dict[str, Any]:
    out = deepcopy(row)
    source_pid = str(row.get("paper_id", ""))
    title = norm(row.get("title"))
    abstract = norm(row.get("abstract"))
    summary = norm(row.get("summary"))
    if not source_pid or not title or not abstract or not summary:
        raise ValueError(f"Missing required field in cleanup row paper_id={source_pid!r}")
    out["source_paper_id"] = source_pid
    out["paper_id"] = f"{source_pid}::broad_cleanup30_q8win_{repeat_index:02d}"
    out["title"] = title
    out["abstract"] = abstract
    out["summary"] = summary
    out["dataset_component"] = "q5_broad_cleanup30_q8win_kimi_qwen"
    out["weighted_numeric_patch_row_type"] = "q5_broad_cleanup30_q8win_kimi_qwen"
    out["source_split"] = "q5_broad_cleanup30_q8win_v1"
    out["cleanup_repeat_index"] = repeat_index
    out["messages"] = messages(title, abstract, summary)
    out["loss_weight_hints"] = {
        "enabled": True,
        "base_weight": WEIGHT,
        "spans": [],
        "policy": f"q5_broad_cleanup30_q8win_v1 repeat{REPEAT} weight {WEIGHT:g}",
    }
    return out


def main() -> None:
    source_rows = load_jsonl(SOURCE)
    rows: list[dict[str, Any]] = []
    for row in source_rows:
        for repeat_index in range(REPEAT):
            rows.append(weighted_copy(row, repeat_index))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "train.jsonl", rows)
    manifest = {
        "created_by": "scripts/build_q5_broad_cleanup30_q8win_v1_dataset.py",
        "source": str(SOURCE),
        "output": str(OUT_DIR / "train.jsonl"),
        "source_rows": len(source_rows),
        "training_rows": len(rows),
        "repeat": REPEAT,
        "weight": WEIGHT,
        "component_counts": dict(collections.Counter(row["dataset_component"] for row in rows)),
        "split_counts": dict(collections.Counter(row.get("blind_review_split", "") for row in rows)),
        "issue_tag_counts": dict(collections.Counter(tag for row in rows for tag in row.get("q5_issue_tags", []))),
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[broad-cleanup30-q8win-dataset] wrote {len(rows)} rows to {OUT_DIR / 'train.jsonl'}")


if __name__ == "__main__":
    main()
