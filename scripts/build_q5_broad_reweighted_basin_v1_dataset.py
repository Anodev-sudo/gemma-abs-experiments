#!/usr/bin/env python3
"""Build a broad reweighted q5 SFT dataset from existing labels."""
from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any


OUT_DIR = Path("hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1")
OUT_PATH = OUT_DIR / "dataset_broad_base_low_repair_high_v1" / "train.jsonl"

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
        "component": "base_10746_low_weight",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_v1/training_rows.jsonl"),
        "priority": 0,
        "weight": 0.35,
    },
    {
        "component": "faithfulness_overlay_1400",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_faithfulness_overlay_1400_v2/training_rows.jsonl"),
        "priority": 20,
        "weight": 0.9,
    },
    {
        "component": "specificity_preserving_400",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_specificity_preserving_400_v1/training_rows.jsonl"),
        "priority": 25,
        "weight": 1.0,
    },
    {
        "component": "recenter_broad_overlay",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_recenter_broad_overlay_v1/training_rows.jsonl"),
        "priority": 30,
        "weight": 0.8,
    },
    {
        "component": "step2400_mined_kimi_qwen_8000",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_step2400_8000_mined_kimi_qwen_v1/training_rows.jsonl"),
        "priority": 45,
        "weight": 1.35,
    },
    {
        "component": "step2400_mined_kimi_qwen_10000",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_step2400_10000_mined_kimi_qwen_v1/training_rows.jsonl"),
        "priority": 50,
        "weight": 1.45,
    },
    {
        "component": "q5_precision_plus_q4_source_kimi_qwen",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_precision_plus_q4_source_kimi_qwen_v1/training_rows.jsonl"),
        "priority": 55,
        "weight": 1.6,
    },
    {
        "component": "q5_precision_kimi_qwen_combined",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_precision_kimi_qwen_overlay_combined_v1/training_rows.jsonl"),
        "priority": 58,
        "weight": 1.65,
    },
    {
        "component": "q5_correction_overlay_kimi_qwen",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_kimi_qwen_correction_overlay_v1/training_rows.jsonl"),
        "priority": 60,
        "weight": 1.7,
    },
    {
        "component": "q5_correction_overlay_repair_kimi_qwen",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_kimi_qwen_correction_overlay_repair_v1/training_rows.jsonl"),
        "priority": 62,
        "weight": 1.75,
    },
    {
        "component": "q5_residual218_push_kimi_qwen",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_residual218_push_285_v1/training_rows.jsonl"),
        "priority": 70,
        "weight": 1.9,
    },
    {
        "component": "q5_q8shape150_q8win_value_aware",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_q8shape150_q8win_value_aware_v1/training_rows.jsonl"),
        "priority": 80,
        "weight": 2.1,
    },
    {
        "component": "q5_value125_q8win_value_aware",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_value125_q8win_value_aware_v1/training_rows.jsonl"),
        "priority": 85,
        "weight": 2.2,
    },
    {
        "component": "q5_secondpass50_q8win_specificity",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_secondpass50_q8win_specificity_v1/training_rows.jsonl"),
        "priority": 88,
        "weight": 2.3,
    },
    {
        "component": "hard77_preserve",
        "path": Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7/dataset_193repair_600selfanchor_77hard_v1/hard77_rows_77.jsonl"),
        "priority": 90,
        "weight": 2.0,
    },
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def norm(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def messages(title: str, abstract: str, summary: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(title=title, abstract=abstract)},
        {"role": "assistant", "content": summary},
    ]


def normalize_row(row: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    title = norm(row.get("title"))
    abstract = norm(row.get("abstract"))
    summary = norm(row.get("summary"))
    if not title or not abstract or not summary:
        raise ValueError(f"Missing title, abstract, or summary in {source['path']} paper_id={row.get('paper_id')}")

    return {
        "paper_id": str(row["paper_id"]),
        "title": title,
        "abstract": abstract,
        "summary": summary,
        "primary_category": row.get("primary_category", ""),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date", ""),
        "teacher_model": row.get("teacher_model", ""),
        "generator_prompt_version": row.get("generator_prompt_version", ""),
        "review_prompt_version": row.get("review_prompt_version", ""),
        "source_split": row.get("source_split", ""),
        "dataset_component": source["component"],
        "component_priority": source["priority"],
        "messages": messages(title, abstract, summary),
        "loss_weight_hints": {
            "enabled": True,
            "base_weight": source["weight"],
            "spans": [],
            "policy": f"q5_broad_reweighted_basin_v1 {source['component']} weight {source['weight']:g}",
        },
        "provenance": {
            "source_path": str(source["path"]),
            "original_source_split": row.get("source_split", ""),
            "original_dataset_component": row.get("dataset_component", ""),
        },
    }


def main() -> None:
    selected: dict[str, dict[str, Any]] = {}
    source_counts: dict[str, int] = {}
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
        component: next(
            source["weight"] for source in SOURCES if source["component"] == component
        )
        for component in component_counts
    }
    manifest = {
        "created_by": "scripts/build_q5_broad_reweighted_basin_v1_dataset.py",
        "purpose": "Broad q5 basin-change SFT mix from existing 10k labels plus higher-weight repair/style rows.",
        "output_path": str(OUT_PATH),
        "row_count": len(rows),
        "selection_policy": "one target per paper_id; higher component priority overrides lower priority labels",
        "source_counts": source_counts,
        "component_counts": dict(sorted(component_counts.items())),
        "component_weights": dict(sorted(weight_by_component.items())),
        "overridden_counts": dict(sorted(overridden_counts.items())),
        "starting_adapter": "hf_bucket/experiments/mlx_lora/runtime_shift/llama_cpp_q5_clean_slate_broad_v1_20260508/gguf_adapter/peft_2400",
    }
    manifest_path = OUT_PATH.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# q5 Broad Reweighted Basin v1 Dataset",
        "",
        f"- Rows: {len(rows)}",
        "- Selection: one target per paper, with higher-priority repair/style rows overriding broad base rows.",
        "",
        "| component | rows | weight |",
        "|---|---:|---:|",
    ]
    for component, count in sorted(component_counts.items()):
        lines.append(f"| `{component}` | {count} | {weight_by_component[component]:g} |")
    lines.append("")
    (OUT_PATH.parent / "README.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[q5-broad-reweighted] wrote {len(rows)} rows to {OUT_PATH}")
    print(f"[q5-broad-reweighted] components={dict(sorted(component_counts.items()))}")


if __name__ == "__main__":
    main()
