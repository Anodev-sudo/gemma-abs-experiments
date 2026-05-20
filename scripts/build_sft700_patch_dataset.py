#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

SYSTEM = (
    "You write one-sentence paper summaries for a mobile research feed. "
    "Keep them concrete, readable, and useful. Output only the summary sentence."
)
USER_TMPL = (
    "The app already shows the title, author list, and date. "
    "Write one sentence that tells a reader why they might open this paper. "
    "Aim for 10 to 15 words. Avoid repeating the title. "
    'Avoid filler like "This paper" or "We propose". '
    "Do not mention authors or dates. "
    "Keep the wording natural and specific.\n\n"
    "Title:\n{title}\n\n"
    "Abstract:\n{abstract}"
)

DIGIT_RE = re.compile(r"\d")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_word_count(row: dict[str, Any]) -> int:
    return len(str(row.get("summary", "")).split())


def has_digit(row: dict[str, Any]) -> bool:
    return bool(DIGIT_RE.search(str(row.get("summary", ""))))


def normalize_row(row: dict[str, Any], source_component: str) -> dict[str, Any]:
    out = {
        "paper_id": row["paper_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "summary": row["summary"],
        "primary_category": row.get("primary_category", ""),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date", ""),
        "teacher_model": row.get("teacher_model", ""),
        "generator_prompt_version": row.get("generator_prompt_version", ""),
        "review_prompt_version": row.get("review_prompt_version", ""),
        "source_split": row.get("source_split", source_component),
        "patch_component": source_component,
    }
    # Preserve useful repair audit metadata when present.
    for key in (
        "bad_summary_checkpoint700",
        "pre_dpo_summary",
        "failure_tags",
        "number_policy",
        "repair_candidate_id",
        "source_input_eval_id",
    ):
        if key in row:
            out[key] = row[key]
    out["messages"] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TMPL.format(title=out["title"], abstract=out["abstract"])},
        {"role": "assistant", "content": out["summary"]},
    ]
    return out


def sample_faithfulness(rows: list[dict[str, Any]], numeric_n: int, non_numeric_n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    numeric = [r for r in rows if has_digit(r) and 10 <= row_word_count(r) <= 15]
    non_numeric = [r for r in rows if not has_digit(r) and 10 <= row_word_count(r) <= 15]
    rng.shuffle(numeric)
    rng.shuffle(non_numeric)
    return numeric[:numeric_n] + non_numeric[:non_numeric_n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targeted-repairs", default="hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/failure_mining/checkpoint700_10000_v1/repairs_targeted_v1/materialized/training_rows.jsonl")
    ap.add_argument("--numeric-hardening", default="hf_dataset/releases/recent_3y_balanced_10746_q5_numeric_hardening_overlay_q4mix_len15_v2/training_rows.jsonl")
    ap.add_argument("--faithfulness", default="hf_dataset/releases/recent_3y_balanced_10746_faithfulness_overlay_1400_v2/training_rows.jsonl")
    ap.add_argument("--output-dir", default="hf_bucket/experiments/sft/q5_step2400_sft700_patch_v1/dataset")
    ap.add_argument("--faithfulness-numeric", type=int, default=250)
    ap.add_argument("--faithfulness-non-numeric", type=int, default=250)
    ap.add_argument("--seed", type=int, default=20260512)
    args = ap.parse_args()

    targeted = load_jsonl(Path(args.targeted_repairs))
    numeric = [
        r for r in load_jsonl(Path(args.numeric_hardening))
        if 10 <= row_word_count(r) <= 15
    ]
    faith = load_jsonl(Path(args.faithfulness))
    faith_sample = sample_faithfulness(faith, args.faithfulness_numeric, args.faithfulness_non_numeric, args.seed)

    # Component priority: targeted repairs > numeric hardening > faithfulness stabilizer.
    components = [
        ("targeted_sft700_repair_v1", targeted),
        ("numeric_hardening_overlay_q4mix_len15_v2", numeric),
        ("faithfulness_overlay_1400_v2_sample", faith_sample),
    ]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates = []
    for component, comp_rows in components:
        for row in comp_rows:
            pid = row["paper_id"]
            if pid in seen:
                duplicates.append({"paper_id": pid, "dropped_component": component})
                continue
            seen.add(pid)
            rows.append(normalize_row(row, component))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", rows)
    write_jsonl(out_dir / "dedupe_dropped_rows.jsonl", duplicates)

    wc = [row_word_count(r) for r in rows]
    component_counts = Counter(r["patch_component"] for r in rows)
    digit_counts = Counter(r["patch_component"] for r in rows if has_digit(r))
    manifest = {
        "dataset_id": "q5_step2400_sft700_patch_v1",
        "purpose": "Targeted SFT patch from checkpoint-700 numeric/semantic repairs plus stabilizing numeric-hardening and faithfulness rows.",
        "seed": args.seed,
        "source_files": {
            "targeted_repairs": args.targeted_repairs,
            "numeric_hardening": args.numeric_hardening,
            "faithfulness": args.faithfulness,
        },
        "row_count": len(rows),
        "dedupe_dropped_rows": len(duplicates),
        "component_counts": dict(component_counts),
        "digit_summary_counts": dict(digit_counts),
        "word_count": {
            "avg": sum(wc) / len(wc),
            "min": min(wc),
            "max": max(wc),
            "under10": sum(w < 10 for w in wc),
            "over15": sum(w > 15 for w in wc),
            "in_10_15": sum(10 <= w <= 15 for w in wc),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
