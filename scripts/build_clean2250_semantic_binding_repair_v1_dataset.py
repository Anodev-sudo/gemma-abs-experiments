#!/usr/bin/env python3
"""Build a semantic-binding repair dataset for the clean-slate checkpoint-2250 lineage."""
from __future__ import annotations

import collections
import json
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path("hf_bucket/experiments/sft/gemma270m_clean_slate_under14_v1/lr1_5e5")
OUT_DIR = ROOT / "semantic_binding_repair_v1" / "dataset_semantic_binding_repair_v1"
OUT_PATH = OUT_DIR / "train.jsonl"
SEED = 20260518

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

WORD_RE = re.compile(r"\b[\w][\w'-]*\b")

SOURCES = [
    {
        "component": "under14_style_anchor_sample",
        "path": Path("hf_bucket/experiments/sft/q8_promoted_broad_reweighted_under14_v1/dataset_under14_kimi_qwen_v1/train.jsonl"),
        "priority": 0,
        "repeat": 1,
        "weight": 0.30,
        "max_rows": 2200,
    },
    {
        "component": "faithfulness_overlay_anchor",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_faithfulness_overlay_1400_v2/training_rows.jsonl"),
        "priority": 20,
        "repeat": 1,
        "weight": 0.75,
    },
    {
        "component": "specificity_preserving_anchor",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_specificity_preserving_400_v1/training_rows.jsonl"),
        "priority": 30,
        "repeat": 1,
        "weight": 0.90,
    },
    {
        "component": "q5_recenter_broad_anchor",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_recenter_broad_overlay_v1/training_rows.jsonl"),
        "priority": 35,
        "repeat": 1,
        "weight": 0.70,
    },
    {
        "component": "q5_step2400_mined_semantic_repair",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_step2400_10000_mined_kimi_qwen_v1/training_rows.jsonl"),
        "priority": 45,
        "repeat": 1,
        "weight": 1.20,
    },
    {
        "component": "q5_final_guardrail_replay",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_final_guardrail_overlay_replay_v2/training_rows.jsonl"),
        "priority": 60,
        "repeat": 1,
        "weight": 1.40,
    },
    {
        "component": "q5_precision_plus_q4_repair",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_precision_plus_q4_source_kimi_qwen_v1/training_rows.jsonl"),
        "priority": 70,
        "repeat": 2,
        "weight": 1.60,
    },
    {
        "component": "q5_precision_combined_repair",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_precision_kimi_qwen_overlay_combined_v1/training_rows.jsonl"),
        "priority": 72,
        "repeat": 1,
        "weight": 1.60,
    },
    {
        "component": "q5_correction_repair",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_kimi_qwen_correction_overlay_repair_v1/training_rows.jsonl"),
        "priority": 75,
        "repeat": 1,
        "weight": 1.50,
    },
    {
        "component": "q5_residual218_push_repair",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_residual218_push_285_v1/training_rows.jsonl"),
        "priority": 80,
        "repeat": 2,
        "weight": 1.80,
    },
    {
        "component": "q5_q8shape150_q8win_value_repair",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_q8shape150_q8win_value_aware_v1/training_rows.jsonl"),
        "priority": 82,
        "repeat": 2,
        "weight": 1.80,
    },
    {
        "component": "q5_value125_q8win_value_repair",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_value125_q8win_value_aware_v1/training_rows.jsonl"),
        "priority": 84,
        "repeat": 2,
        "weight": 1.90,
    },
    {
        "component": "q5_broad2500_cleanup_repair",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_broad2500_cleanup_v1/training_rows.jsonl"),
        "priority": 86,
        "repeat": 2,
        "weight": 1.70,
    },
    {
        "component": "q5_broad_cleanup30_q8win_repair",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_broad_cleanup30_q8win_v1/training_rows.jsonl"),
        "priority": 88,
        "repeat": 3,
        "weight": 2.00,
    },
    {
        "component": "q5_secondpass_specificity_repair",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_secondpass50_q8win_specificity_v1/training_rows.jsonl"),
        "priority": 90,
        "repeat": 3,
        "weight": 2.10,
    },
    {
        "component": "q5_broad_q8win50_round2_repair",
        "path": Path("hf_dataset/releases/recent_3y_balanced_10746_q5_broad_q8win50_round2_v1/training_rows.jsonl"),
        "priority": 92,
        "repeat": 4,
        "weight": 2.00,
    },
    {
        "component": "hard77_preserve",
        "path": Path(
            "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7/"
            "dataset_193repair_600selfanchor_77hard_v1/hard77_rows_77.jsonl"
        ),
        "priority": 94,
        "repeat": 1,
        "weight": 1.50,
    },
]

CURRENT_Q8_WIN_SOURCE = {
    "component": "clean2250_current_q8win_repair",
    "path": Path(
        "hf_bucket/experiments/sft/gemma270m_clean_slate_under14_v1/lr1_5e5/eval/"
        "blind_ab_q8_mlx_vs_clean2250_fullbase_realdb300_20260518/"
        "blind_review/blind_review_merged_unblinded.jsonl"
    ),
    "priority": 100,
    "repeat": 3,
    "weight": 2.20,
}


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


def word_count(value: Any) -> int:
    return len(WORD_RE.findall(str(value or "")))


def base_paper_id(value: Any) -> str:
    return str(value or "").split("::", 1)[0].strip()


def messages(title: str, abstract: str, summary: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(title=title, abstract=abstract)},
        {"role": "assistant", "content": summary},
    ]


def normalize_row(row: dict[str, Any], source: dict[str, Any], summary: str | None = None) -> dict[str, Any]:
    out = deepcopy(row)
    paper_id = base_paper_id(out.get("paper_id"))
    title = norm(out.get("title"))
    abstract = norm(out.get("abstract"))
    summary_text = norm(summary if summary is not None else out.get("summary"))
    if not paper_id or not title or not abstract or not summary_text:
        raise ValueError(f"Missing required field in {source['path']} paper_id={paper_id!r}")
    out["source_paper_id"] = paper_id
    out["paper_id"] = paper_id
    out["title"] = title
    out["abstract"] = abstract
    out["summary"] = summary_text
    out["summary_word_count"] = word_count(summary_text)
    out["dataset_component"] = source["component"]
    out["component_priority"] = int(source["priority"])
    out["source_split"] = out.get("source_split") or source["component"]
    out["messages"] = messages(title, abstract, summary_text)
    out["loss_weight_hints"] = {
        "enabled": True,
        "base_weight": float(source["weight"]),
        "spans": [],
        "policy": (
            "clean2250_semantic_binding_repair_v1 "
            f"{source['component']} repeat{source['repeat']} weight {source['weight']:g}"
        ),
    }
    out["provenance"] = {
        **dict(out.get("provenance") or {}),
        "source_path": str(source["path"]),
        "source_component": source["component"],
    }
    return out


def source_rows(source: dict[str, Any], rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, int]]:
    raw_rows = load_jsonl(source["path"])
    counts = {
        "raw_rows": len(raw_rows),
        "missing_required": 0,
        "too_long": 0,
        "empty_summary": 0,
        "sampled_out": 0,
    }
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        summary = norm(row.get("summary"))
        if not summary:
            counts["empty_summary"] += 1
            continue
        if word_count(summary) > 14:
            counts["too_long"] += 1
            continue
        try:
            rows.append(normalize_row(row, source))
        except ValueError:
            counts["missing_required"] += 1

    max_rows = source.get("max_rows")
    if max_rows is not None and len(rows) > int(max_rows):
        rows = sorted(rows, key=lambda item: item["source_paper_id"])
        rng.shuffle(rows)
        kept = rows[: int(max_rows)]
        counts["sampled_out"] = len(rows) - len(kept)
        rows = kept
    return rows, counts


def current_q8_win_rows(source: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    raw_rows = load_jsonl(source["path"])
    counts = {
        "raw_rows": len(raw_rows),
        "q8_wins": 0,
        "missing_required": 0,
        "too_long": 0,
        "empty_summary": 0,
    }
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        if row.get("preferred_model") != "promoted_q8_mlx_python":
            continue
        counts["q8_wins"] += 1
        summary = ""
        if row.get("summary_a_model") == "promoted_q8_mlx_python":
            summary = norm(row.get("summary_a"))
        elif row.get("summary_b_model") == "promoted_q8_mlx_python":
            summary = norm(row.get("summary_b"))
        summary = summary or norm(row.get("preferred_summary"))
        if not summary:
            counts["empty_summary"] += 1
            continue
        if word_count(summary) > 14:
            counts["too_long"] += 1
            continue
        try:
            out = normalize_row(row, source, summary=summary)
        except ValueError:
            counts["missing_required"] += 1
            continue
        out["source_split"] = "blind_ab_q8_mlx_vs_clean2250_fullbase_realdb300_20260518_q8win"
        out["repair_note"] = "Q8 promoted summary preferred over clean-slate checkpoint-2250 in blind review."
        rows.append(out)
    return rows, counts


def weighted_copy(row: dict[str, Any], repeat_index: int, repeat: int) -> dict[str, Any]:
    out = deepcopy(row)
    source_pid = str(row["source_paper_id"])
    component = str(row["dataset_component"])
    out["paper_id"] = f"{source_pid}::{component}_{repeat_index:02d}"
    out["repeat_index"] = repeat_index
    out["repeat_count"] = repeat
    return out


def main() -> None:
    rng = random.Random(SEED)
    selected: dict[str, dict[str, Any]] = {}
    selected_source_counts: collections.Counter[str] = collections.Counter()
    overridden_counts: collections.Counter[str] = collections.Counter()
    source_audits: dict[str, dict[str, int]] = {}
    missing_sources: list[str] = []

    all_sources = list(SOURCES)
    for source in all_sources:
        path = Path(source["path"])
        if not path.exists():
            missing_sources.append(str(path))
            continue
        if source is CURRENT_Q8_WIN_SOURCE:
            rows, audit = current_q8_win_rows(source)
        else:
            rows, audit = source_rows(source, rng)
        source_audits[source["component"]] = audit
        selected_source_counts[source["component"]] += len(rows)
        for row in rows:
            paper_id = str(row["source_paper_id"])
            current = selected.get(paper_id)
            if current and int(current["component_priority"]) > int(source["priority"]):
                continue
            if current:
                overridden_counts[str(current["dataset_component"])] += 1
            selected[paper_id] = row

    if missing_sources:
        raise FileNotFoundError("Missing source files:\n" + "\n".join(missing_sources))

    source_by_component = {source["component"]: source for source in all_sources}
    rows: list[dict[str, Any]] = []
    for row in sorted(selected.values(), key=lambda item: (str(item["dataset_component"]), str(item["source_paper_id"]))):
        source = source_by_component[str(row["dataset_component"])]
        repeat = int(source["repeat"])
        for repeat_index in range(repeat):
            rows.append(weighted_copy(row, repeat_index, repeat))

    write_jsonl(OUT_PATH, rows)
    write_jsonl(OUT_DIR / "selected_unique_rows.jsonl", sorted(selected.values(), key=lambda item: str(item["source_paper_id"])))

    component_counts = collections.Counter(row["dataset_component"] for row in rows)
    unique_component_counts = collections.Counter(row["dataset_component"] for row in selected.values())
    length_counts = collections.Counter(str(row["summary_word_count"]) for row in selected.values())
    manifest = {
        "created_by": "scripts/build_clean2250_semantic_binding_repair_v1_dataset.py",
        "purpose": (
            "Continue clean-slate checkpoint-2250 with compact q5-playbook semantic-binding repairs "
            "while filtering all target labels to 14 words or fewer."
        ),
        "seed": SEED,
        "output": str(OUT_PATH),
        "selected_unique_rows": str(OUT_DIR / "selected_unique_rows.jsonl"),
        "training_rows": len(rows),
        "unique_source_papers": len(selected),
        "max_summary_words": max(row["summary_word_count"] for row in selected.values()) if selected else 0,
        "avg_summary_words": (
            sum(row["summary_word_count"] for row in selected.values()) / len(selected)
            if selected
            else 0.0
        ),
        "component_counts": dict(sorted(component_counts.items())),
        "unique_component_counts": dict(sorted(unique_component_counts.items())),
        "source_audits": source_audits,
        "selected_source_counts_before_priority": dict(sorted(selected_source_counts.items())),
        "overridden_counts": dict(sorted(overridden_counts.items())),
        "component_weights": {source["component"]: source["weight"] for source in all_sources},
        "component_repeats": {source["component"]: source["repeat"] for source in all_sources},
        "length_distribution_unique": dict(sorted(length_counts.items(), key=lambda item: int(item[0]))),
        "starting_adapter": str(ROOT / "checkpoint-2250"),
        "notes": [
            "No new labels are generated; the dataset only remixes already materialized q5/q8 training releases.",
            "Rows from the 300-row clean2250 blind review are intentionally excluded from training and reserved for replay analysis.",
            "All labels above 14 words are excluded before priority selection and repeats.",
            "Priority keeps one target per source paper, so stronger repair labels override broad anchors.",
        ],
        "excluded_reference_sources": {
            "clean2250_current_q8win_repair": str(CURRENT_Q8_WIN_SOURCE["path"]),
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Clean 2250 Semantic Binding Repair v1",
        "",
        "Continuation dataset for the clean-slate full-precision PEFT lineage.",
        "",
        f"- Unique source papers: {len(selected)}",
        f"- Training rows after repeats: {len(rows)}",
        f"- Max target length: {manifest['max_summary_words']} words",
        f"- Average target length: {manifest['avg_summary_words']:.2f} words",
        "- Selection: one target per source paper, higher-priority repair labels override broad anchors.",
        "- No new labels are generated; this is a remix of existing q5/q8 training releases.",
        "- Current clean2250 blind-review q8-win rows are excluded and reserved for replay analysis.",
        "",
        "| component | unique | training rows | repeat | weight |",
        "|---|---:|---:|---:|---:|",
    ]
    for source in all_sources:
        component = source["component"]
        lines.append(
            f"| `{component}` | {unique_component_counts.get(component, 0)} | "
            f"{component_counts.get(component, 0)} | {source['repeat']} | {source['weight']:g} |"
        )
    lines.append("")
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[clean2250-semantic-binding-repair] wrote {len(rows)} rows to {OUT_PATH}")
    print(f"[clean2250-semantic-binding-repair] unique_source_papers={len(selected)} max_words={manifest['max_summary_words']}")
    print(f"[clean2250-semantic-binding-repair] components={dict(sorted(component_counts.items()))}")


if __name__ == "__main__":
    main()
