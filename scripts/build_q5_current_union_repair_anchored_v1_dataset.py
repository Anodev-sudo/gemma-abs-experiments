#!/usr/bin/env python3
"""Build a conservative union of Kimi/Qwen q5 repair labels for current-q5 SFT."""
from __future__ import annotations

import collections
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


OUT_DIR = Path(
    "hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/"
    "cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/"
    "anchored_union_after_ckpt50/dataset_union_repairs_anchored_v1"
)
OUT_PATH = OUT_DIR / "train.jsonl"

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

SOURCES = [
    {
        "component": "broad2500_cleanup_anchor",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_broad2500_cleanup_v1/training_rows.jsonl"),
        "priority": 10,
        "repeat": 1,
        "weight": 0.8,
    },
    {
        "component": "q8win_cleanup_anchor",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_broad_cleanup30_q8win_v1/training_rows.jsonl"),
        "priority": 20,
        "repeat": 1,
        "weight": 1.1,
    },
    {
        "component": "q8win_round2_repair",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_broad_q8win50_round2_v1/training_rows.jsonl"),
        "priority": 30,
        "repeat": 2,
        "weight": 1.5,
    },
]


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


def normalize_source_row(row: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(row)
    paper_id = str(row.get("paper_id", "")).strip()
    title = norm(row.get("title"))
    abstract = norm(row.get("abstract"))
    summary = norm(row.get("summary"))
    if not paper_id or not title or not abstract or not summary:
        raise ValueError(f"Missing required field in {source['path']} paper_id={paper_id!r}")
    out["paper_id"] = paper_id
    out["title"] = title
    out["abstract"] = abstract
    out["summary"] = summary
    out["dataset_component"] = source["component"]
    out["component_priority"] = source["priority"]
    out["messages"] = messages(title, abstract, summary)
    return out


def weighted_copy(row: dict[str, Any], repeat_index: int, repeat: int, weight: float) -> dict[str, Any]:
    out = deepcopy(row)
    source_pid = str(row["paper_id"])
    component = str(row["dataset_component"])
    out["source_paper_id"] = source_pid
    out["paper_id"] = f"{source_pid}::{component}_{repeat_index:02d}"
    out["cleanup_repeat_index"] = repeat_index
    out["loss_weight_hints"] = {
        "enabled": True,
        "base_weight": weight,
        "spans": [],
        "policy": f"q5_current_union_repair_anchored_v1 {component} repeat{repeat} weight {weight:g}",
    }
    return out


def main() -> None:
    selected: dict[str, dict[str, Any]] = {}
    source_counts: dict[str, int] = {}
    overridden_counts: collections.Counter[str] = collections.Counter()
    source_by_component = {source["component"]: source for source in SOURCES}
    for source in SOURCES:
        rows = load_jsonl(source["path"])
        source_counts[source["component"]] = len(rows)
        for row in rows:
            paper_id = str(row.get("paper_id", "")).strip()
            current = selected.get(paper_id)
            if current and int(current["component_priority"]) > int(source["priority"]):
                continue
            if current:
                overridden_counts[str(current["dataset_component"])] += 1
            selected[paper_id] = normalize_source_row(row, source)

    rows: list[dict[str, Any]] = []
    for row in sorted(selected.values(), key=lambda item: (str(item["dataset_component"]), str(item["paper_id"]))):
        source = source_by_component[str(row["dataset_component"])]
        for repeat_index in range(int(source["repeat"])):
            rows.append(weighted_copy(row, repeat_index, int(source["repeat"]), float(source["weight"])))

    write_jsonl(OUT_PATH, rows)
    component_counts = collections.Counter(row["dataset_component"] for row in rows)
    manifest = {
        "created_by": "scripts/build_q5_current_union_repair_anchored_v1_dataset.py",
        "purpose": "Conservative current-q5 overlay using only existing Kimi/Qwen repair labels, with broad/q8win rows as anchors.",
        "output": str(OUT_PATH),
        "source_counts": source_counts,
        "unique_source_papers": len(selected),
        "training_rows": len(rows),
        "component_counts": dict(sorted(component_counts.items())),
        "component_weights": {source["component"]: source["weight"] for source in SOURCES},
        "component_repeats": {source["component"]: source["repeat"] for source in SOURCES},
        "overridden_counts": dict(sorted(overridden_counts.items())),
        "starting_adapter": "hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/lr6e7/cleanup_after_ckpt2500/lr4e7/q8win_after_ckpt30/lr3e7_x3/checkpoint-50",
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# q5 Current Union Repair Anchored v1",
        "",
        f"- Unique source papers: {len(selected)}",
        f"- Training rows: {len(rows)}",
        "- Selection: one target per paper, with round-two q8-win labels overriding earlier repairs.",
        "",
        "| component | rows | repeat | weight |",
        "|---|---:|---:|---:|",
    ]
    for source in SOURCES:
        component = source["component"]
        lines.append(
            f"| `{component}` | {component_counts.get(component, 0)} | {source['repeat']} | {source['weight']:g} |"
        )
    lines.append("")
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[q5-current-union-repair-anchored] wrote {len(rows)} rows to {OUT_PATH}")
    print(f"[q5-current-union-repair-anchored] components={dict(sorted(component_counts.items()))}")


if __name__ == "__main__":
    main()
