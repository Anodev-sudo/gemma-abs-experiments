#!/usr/bin/env python3
"""Build the checkpoint-200 style-normalization SFT dataset."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin200_style_norm_v1_lr5e7")
NORMAL_SRC = Path("hf_bucket/experiments/sft/q5_step2400_sft700_clean_existing_targeted_weight_v1_lr4e7/dataset_600normal_200numeric_targeted_weight_clean_v1/normal_selected_600.jsonl")
IDEAL_SRC = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_ideal200_regen_v1/clean_200_after_filters_v1/training_rows_200.jsonl")
HARD77_SRC = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/dataset_hard77_repeats20_v1/eval_hard77.jsonl")
TARGETED_SRC = ROOT / "materialized/targeted_training_rows_160.jsonl"
STRESS_IDS = {"1901.06548", "2111.10635", "2303.09722", "2406.04373"}
DIGIT_RE = re.compile(r"\d")

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


def messages(title: str, abstract: str, summary: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(title=title, abstract=abstract)},
        {"role": "assistant", "content": summary},
    ]


def full_target_weight(weight: float, policy: str) -> dict[str, Any]:
    if weight == 1.0:
        return {"enabled": False, "base_weight": 1.0, "spans": [], "policy": policy}
    return {"enabled": True, "base_weight": weight, "spans": [], "policy": policy}


def normalize_row(row: dict[str, Any], row_type: str, summary_key: str = "summary", weight: float = 1.0) -> dict[str, Any]:
    summary = norm(row.get(summary_key, ""))
    out = dict(row)
    out["summary"] = summary
    out["weighted_numeric_patch_row_type"] = row_type
    out["loss_weight_hints"] = full_target_weight(weight, f"style_norm_v1 full assistant target weight {weight}")
    out["messages"] = messages(out.get("title", ""), out.get("abstract", ""), summary)
    out["source_split"] = "q5_burnin200_style_norm_v1"
    return out


def hard77_row(row: dict[str, Any]) -> dict[str, Any]:
    pid = str(row["paper_id"])
    weight = 2.0 if pid in STRESS_IDS else 1.25
    out = {
        "paper_id": pid,
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "summary": row.get("target_summary", ""),
        "primary_category": row.get("primary_category", ""),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date", ""),
        "teacher_model": row.get("teacher_model", "moonshotai/kimi-k2.6"),
        "generator_prompt_version": row.get("generator_prompt_version", ""),
        "review_prompt_version": row.get("review_prompt_version", ""),
        "hard77_source_eval_id": row.get("eval_id", ""),
    }
    return normalize_row(out, "hard77_preserve_stress" if pid in STRESS_IDS else "hard77_preserve", weight=weight)


def assert_zero_digits(rows: list[dict[str, Any]], label: str) -> None:
    bad = [row["paper_id"] for row in rows if DIGIT_RE.search(row.get("summary", ""))]
    if bad:
        raise SystemExit(f"{label} has digit summaries: {bad[:20]}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build q5 checkpoint-200 style-normalization SFT rows.")
    ap.add_argument("--output-root", type=Path, default=ROOT)
    ap.add_argument("--normal-src", type=Path, default=NORMAL_SRC)
    ap.add_argument("--ideal-src", type=Path, default=IDEAL_SRC)
    ap.add_argument("--hard77-src", type=Path, default=HARD77_SRC)
    ap.add_argument("--targeted-src", type=Path, default=TARGETED_SRC)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    normal_raw = load_jsonl(args.normal_src)
    ideal_raw = load_jsonl(args.ideal_src)
    hard77_raw = load_jsonl(args.hard77_src)
    targeted_raw = load_jsonl(args.targeted_src)
    if len(normal_raw) != 600:
        raise SystemExit(f"Expected 600 normal rows, got {len(normal_raw)}")
    if len(ideal_raw) != 200:
        raise SystemExit(f"Expected 200 ideal numeric rows, got {len(ideal_raw)}")
    if len(hard77_raw) != 77:
        raise SystemExit(f"Expected 77 hard77 rows, got {len(hard77_raw)}")
    if len(targeted_raw) != 160:
        raise SystemExit(f"Expected 160 targeted rows, got {len(targeted_raw)}")

    normal = [normalize_row(row, "normal_style_anchor", weight=1.0) for row in normal_raw]
    ideal = [normalize_row(row, "numeric_ideal200_anchor", weight=1.0) for row in ideal_raw]
    hard77 = [hard77_row(row) for row in hard77_raw]
    targeted = [normalize_row(row, f"style_norm_targeted_{row.get('normalization_category', 'unknown')}", weight=2.0) for row in targeted_raw]

    assert_zero_digits(ideal, "ideal numeric")
    assert_zero_digits(hard77, "hard77")
    assert_zero_digits(targeted, "targeted style normalization")

    final = [*normal, *ideal, *hard77, *targeted]
    if len(final) != 1037:
        raise SystemExit(f"Expected 1037 final rows, got {len(final)}")

    out = args.output_root / "dataset_600normal_200ideal_77hard_160style_v1"
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", final)
    write_jsonl(out / "normal_selected_600.jsonl", normal)
    write_jsonl(out / "ideal_selected_200.jsonl", ideal)
    write_jsonl(out / "hard77_selected_77.jsonl", hard77)
    write_jsonl(out / "targeted_selected_160.jsonl", targeted)
    manifest = {
        "created_by": "scripts/build_q5_burnin200_style_norm_v1_dataset.py",
        "rows": len(final),
        "dataset_path": str(out / "train.jsonl"),
        "row_type_counts": dict(Counter(row["weighted_numeric_patch_row_type"] for row in final)),
        "normal_rows": len(normal),
        "ideal_rows": len(ideal),
        "hard77_rows": len(hard77),
        "targeted_rows": len(targeted),
        "targeted_category_counts": dict(Counter(row.get("normalization_category", "") for row in targeted)),
        "digit_rows_checked": {
            "ideal": 0,
            "hard77": 0,
            "targeted": 0,
            "normal": sum(1 for row in normal if DIGIT_RE.search(row.get("summary", ""))),
        },
        "weight_policy": {
            "normal": 1.0,
            "ideal": 1.0,
            "hard77": 1.25,
            "stress": 2.0,
            "targeted": 2.0,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[style-norm-dataset] wrote {len(final)} rows to {out / 'train.jsonl'}")


if __name__ == "__main__":
    main()
