#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You write one-sentence paper summaries for a mobile research feed. "
    "Keep them concrete, readable, and useful. Output only the summary sentence."
)
USER_TEMPLATE = (
    "The app already shows the title, author list, and date. "
    "Write one sentence that tells a reader why they might open this paper. "
    "Aim for 10 to 15 words. Avoid repeating the title. "
    'Avoid filler like "This paper" or "We propose". '
    "Do not mention authors or dates. "
    "Keep the wording natural and specific.\n\n"
    "Title:\n{title}\n\n"
    "Abstract:\n{abstract}"
)

OUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_clean_stress_boost_v1_lr4e7/dataset_600normal_160numeric_40stress_clean_v1")
NORMAL = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_antagonist_clean_v1_pool350/dataset_600normal_200numeric_clean_v1/normal_selected_600.jsonl")
NUMERIC = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_antagonist_clean_v1_pool350/dataset_600normal_200numeric_clean_v1/numeric_selected_200.jsonl")
BASE_SOURCES = [
    Path("hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/train.jsonl"),
    Path("hf_dataset/releases/recent_3y_balanced_10746_v1/training_rows.jsonl"),
    Path("hf_dataset/releases/recent_3y_balanced_incremental_6249_v1/training_rows.jsonl"),
]
DIGIT_RE = re.compile(r"\d")
FORMULA_RE = re.compile(r"[$<>≤≥≈]|\\[A-Za-z]+|\bO\(|\^|<[^>]+>")

# Kimi-generated, Qwen-verified labels selected from stress drill runs.
STRESS_LABELS = {
    "1901.06548": [
        "Computing optimal tangles is NP-hard, but algorithms are provided for wire swaps.",
        "Computing optimal tangles is NP-hard, but an algorithm is given for wire swaps.",
        "Computing optimal tangles is NP-hard, and an algorithm is provided for wire swaps.",
    ],
    "2111.10635": [
        "Reinforcement learning scheduling improves throughput and reduces cost in distributed deep learning.",
        "Reinforcement learning scheduling improves throughput and reduces cost for distributed deep learning.",
        "A reinforcement learning scheduler improves throughput and reduces cost in distributed deep learning.",
    ],
    "2303.09722": [
        "Nanoprinted microfibers enable low-loss broadband coupling between tapered optical fibers and photonic waveguides.",
        "Three-dimensional nanoprinted microfibers enable efficient broadband coupling between tapered optical fibers and chip waveguides.",
        "Three-dimensional nanoprinted microfibers enable efficient broadband coupling between tapered optical fibers and photonic waveguides.",
    ],
    "2406.04373": [
        "A language model reads Verilog code to generate hardware tests that reach unexplored branches.",
        "An LLM-based Verilog reader generates hardware tests that reach unexplored code branches.",
        "LLM-aided Verilog reading generates hardware tests that reach unexplored code branches and improve coverage.",
    ],
}
STRESS_PROVENANCE = {
    "1901.06548": "stress_drill_kimi_v3_compact_qwen_verified",
    "2111.10635": "stress_drill_kimi_v3_compact_qwen_verified",
    "2303.09722": "stress_drill_kimi_v2_plus_kimi_digit_fix_qwen_verified",
    "2406.04373": "stress_drill_kimi_v2_qwen_verified",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def wc(text: str) -> int:
    return len(str(text).split())


def validate_summary(summary: str) -> None:
    if not (10 <= wc(summary) <= 15):
        raise ValueError(f"bad word count {wc(summary)}: {summary}")
    if DIGIT_RE.search(summary):
        raise ValueError(f"digit in summary: {summary}")
    if FORMULA_RE.search(summary):
        raise ValueError(f"formula/html in summary: {summary}")


def messages(title: str, abstract: str, summary: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(title=title, abstract=abstract)},
        {"role": "assistant", "content": summary},
    ]


def normal_hints() -> dict[str, Any]:
    return {"enabled": False, "base_weight": 1.0, "spans": [], "policy": "normal rows are unweighted style anchors"}


def full_target_hints(summary: str, weight: float, kind: str) -> dict[str, Any]:
    return {
        "enabled": True,
        "base_weight": 1.0,
        "spans": [{"start": 0, "end": len(summary), "text": summary, "weight": weight, "kind": kind}],
        "policy": "upweight full assistant target only; prompt/source tokens stay masked out",
    }


def base_row(row: dict[str, Any], summary: str, row_type: str, component: str, hints: dict[str, Any], mix_index: int) -> dict[str, Any]:
    validate_summary(summary)
    title = str(row.get("title") or "")
    abstract = str(row.get("abstract") or "")
    return {
        "paper_id": str(row.get("paper_id") or ""),
        "title": title,
        "abstract": abstract,
        "summary": summary,
        "primary_category": str(row.get("primary_category") or ""),
        "categories": row.get("categories") if isinstance(row.get("categories"), list) else [],
        "created_date": str(row.get("created_date") or ""),
        "teacher_model": str(row.get("teacher_model") or row.get("writer_model") or ""),
        "generator_prompt_version": str(row.get("generator_prompt_version") or row.get("writer_prompt_version") or ""),
        "review_prompt_version": str(row.get("review_prompt_version") or row.get("reviewer_prompt_version") or ""),
        "source_split": "q5_clean_stress_boost_v1_600normal_160numeric_40stress",
        "patch_component": component,
        "weighted_numeric_patch_row_type": row_type,
        "loss_weight_hints": hints,
        "mix_index": mix_index,
        "messages": messages(title, abstract, summary),
    }


def find_stress_sources() -> dict[str, dict[str, Any]]:
    found = {}
    for path in BASE_SOURCES:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            pid = row.get("paper_id")
            if pid in STRESS_LABELS and pid not in found:
                found[pid] = row
    missing = sorted(set(STRESS_LABELS) - set(found))
    if missing:
        raise SystemExit(f"Missing stress source rows: {missing}")
    return found


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    normal_rows = load_jsonl(NORMAL)
    numeric_rows = load_jsonl(NUMERIC)
    stress_sources = find_stress_sources()

    # Exclude stress IDs from broad numeric rows so the stress slice controls those targets.
    stress_ids = set(STRESS_LABELS)
    numeric_clean = []
    for row in numeric_rows:
        if row.get("paper_id") in stress_ids:
            continue
        summary = str(row.get("repair_summary") or row.get("summary") or "").strip()
        try:
            validate_summary(summary)
        except ValueError:
            continue
        numeric_clean.append(row)
    if len(numeric_clean) < 160:
        raise SystemExit(f"Need 160 numeric rows, got {len(numeric_clean)}")

    final: list[dict[str, Any]] = []
    for row in normal_rows[:600]:
        final.append(base_row(
            row,
            str(row["summary"]).strip(),
            "normal_style_anchor",
            "checkpoint700_train_digit_free_normal_anchor",
            normal_hints(),
            len(final) + 1,
        ))

    for row in numeric_clean[:160]:
        summary = str(row.get("repair_summary") or row.get("summary") or "").strip()
        final.append(base_row(
            row,
            summary,
            "numeric_antagonist_clean_digit_free",
            "kimi_writer_qwen_verified_checkpoint700_numeric_risk",
            full_target_hints(summary, 4.0, "clean_digit_free_numeric_antagonist_full_target"),
            len(final) + 1,
        ))

    # Forty stress examples: ten per paper. Cycle verified variants, with high full-target weight.
    stress_rows = []
    for pid, labels in STRESS_LABELS.items():
        src = stress_sources[pid]
        for rep in range(10):
            summary = labels[rep % len(labels)]
            row = base_row(
                {**src, "teacher_model": "moonshotai/kimi-k2.6", "generator_prompt_version": STRESS_PROVENANCE[pid], "review_prompt_version": "qwen3_235b_a22b_2507_q5_numeric_suppression_clean_label_verifier_v1"},
                summary,
                "numeric_antagonist_clean_stress_booster",
                STRESS_PROVENANCE[pid],
                full_target_hints(summary, 10.0, "clean_verified_stress_booster_full_target"),
                len(final) + len(stress_rows) + 1,
            )
            row["stress_booster_repeat_index"] = rep + 1
            row["stress_booster_verified_label_count"] = len(labels)
            stress_rows.append(row)
    final.extend(stress_rows)

    if len(final) != 800:
        raise SystemExit(f"Expected 800 rows, got {len(final)}")

    write_jsonl(OUT / "train.jsonl", final)
    write_jsonl(OUT / "normal_selected_600.jsonl", final[:600])
    write_jsonl(OUT / "numeric_selected_160.jsonl", final[600:760])
    write_jsonl(OUT / "stress_booster_40.jsonl", final[760:])

    wc_counts = Counter(wc(r["summary"]) for r in final)
    type_counts = Counter(r["weighted_numeric_patch_row_type"] for r in final)
    stress_counts = Counter(r["paper_id"] for r in stress_rows)
    digit_rows = [r for r in final if DIGIT_RE.search(r["summary"])]
    formula_rows = [r for r in final if FORMULA_RE.search(r["summary"])]
    manifest = {
        "created_by": "scripts/build_ckpt700_clean_stress_boost_v1_dataset.py",
        "rows": len(final),
        "dataset_path": str(OUT / "train.jsonl"),
        "row_type_counts": dict(type_counts),
        "stress_repeat_counts": dict(stress_counts),
        "word_count_distribution": dict(sorted(wc_counts.items())),
        "digit_summary_rows": len(digit_rows),
        "formula_symbol_summary_rows": len(formula_rows),
        "normal_weight": 1.0,
        "numeric_full_target_weight": 4.0,
        "stress_full_target_weight": 10.0,
        "label_provenance": {
            "normal": "checkpoint-700 parent SFT training rows, digit-free style anchors",
            "numeric": "fresh Kimi writer labels verified by Qwen, hard-filtered digit-free",
            "stress": "Kimi stress-drill labels verified by Qwen; 2303 digit residue fixed by Kimi and re-verified",
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    audit = [
        "# Dataset audit",
        "",
        f"Rows: {len(final)}",
        f"Row type counts: {dict(type_counts)}",
        f"Stress repeat counts: {dict(stress_counts)}",
        f"Word counts: {dict(sorted(wc_counts.items()))}",
        f"Digit-bearing summaries: {len(digit_rows)}",
        f"Formula/html summaries: {len(formula_rows)}",
        "",
        "Stress rows are repeated clean Kimi/Qwen-verified labels with full assistant-target weight 10.",
    ]
    (OUT / "DATASET_AUDIT.md").write_text("\n".join(audit) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
