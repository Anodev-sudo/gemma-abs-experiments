#!/usr/bin/env python3
"""Build broad q5 basin v2 with the latest broad/q8-win cleanup labels."""
from __future__ import annotations

import collections
import json
from pathlib import Path

from build_q5_broad_reweighted_basin_v1_dataset import (
    SOURCES as V1_SOURCES,
    load_jsonl,
    normalize_row,
    write_jsonl,
)


OUT_DIR = Path("hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v2")
OUT_PATH = OUT_DIR / "dataset_broad_base_low_repair_high_v2" / "train.jsonl"

SOURCES = [
    *V1_SOURCES,
    {
        "component": "q5_broad2500_cleanup",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_broad2500_cleanup_v1/training_rows.jsonl"),
        "priority": 91,
        "weight": 2.1,
    },
    {
        "component": "q5_broad_cleanup30_q8win",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_broad_cleanup30_q8win_v1/training_rows.jsonl"),
        "priority": 94,
        "weight": 2.35,
    },
    {
        "component": "q5_broad_q8win50_round2",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_broad_q8win50_round2_v1/training_rows.jsonl"),
        "priority": 96,
        "weight": 2.45,
    },
]


def main() -> None:
    selected = {}
    source_counts = {}
    overridden_counts: collections.Counter[str] = collections.Counter()
    missing_sources: list[str] = []

    for source in SOURCES:
        path = source["path"]
        if not path.exists():
            missing_sources.append(str(path))
            continue
        rows = load_jsonl(path)
        source_counts[source["component"]] = len(rows)
        for row in rows:
            paper_id = str(row.get("paper_id", "")).strip()
            if not paper_id:
                raise ValueError(f"Missing paper_id in {path}")
            current = selected.get(paper_id)
            if current and int(current["component_priority"]) > int(source["priority"]):
                continue
            if current:
                overridden_counts[str(current["dataset_component"])] += 1
            selected[paper_id] = normalize_row(row, source)

    if missing_sources:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(missing_sources))

    rows = sorted(selected.values(), key=lambda row: (str(row["dataset_component"]), str(row["paper_id"])))
    write_jsonl(OUT_PATH, rows)

    component_counts = collections.Counter(row["dataset_component"] for row in rows)
    weight_by_component = {
        component: next(source["weight"] for source in SOURCES if source["component"] == component)
        for component in component_counts
    }
    manifest = {
        "created_by": "scripts/build_q5_broad_reweighted_basin_v2_dataset.py",
        "purpose": "Broad q5 basin-change SFT mix plus latest broad/q8-win Kimi/Qwen cleanup labels.",
        "output_path": str(OUT_PATH),
        "row_count": len(rows),
        "selection_policy": "one target per paper_id; higher component priority overrides lower priority labels",
        "source_counts": source_counts,
        "component_counts": dict(sorted(component_counts.items())),
        "component_weights": dict(sorted(weight_by_component.items())),
        "overridden_counts": dict(sorted(overridden_counts.items())),
        "starting_adapter": "hf_bucket/experiments/mlx_lora/runtime_shift/llama_cpp_q5_clean_slate_broad_v1_20260508/gguf_adapter/peft_2400",
        "new_v2_components": [
            "q5_broad2500_cleanup",
            "q5_broad_cleanup30_q8win",
            "q5_broad_q8win50_round2",
        ],
    }
    manifest_path = OUT_PATH.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# q5 Broad Reweighted Basin v2 Dataset",
        "",
        f"- Rows: {len(rows)}",
        "- Selection: one target per paper, with higher-priority repair/style rows overriding broad base rows.",
        "- v2 adds the broad2500 cleanup, q8-win cleanup, and q8-win round-two cleanup labels.",
        "",
        "| component | rows | weight |",
        "|---|---:|---:|",
    ]
    for component, count in sorted(component_counts.items()):
        lines.append(f"| `{component}` | {count} | {weight_by_component[component]:g} |")
    lines.append("")
    (OUT_PATH.parent / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[q5-broad-reweighted-v2] wrote {len(rows)} rows to {OUT_PATH}")
    print(f"[q5-broad-reweighted-v2] components={dict(sorted(component_counts.items()))}")


if __name__ == "__main__":
    main()
