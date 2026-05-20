#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def wc(text: str) -> int:
    return len(str(text).split())


def normalize(row: dict[str, Any], component: str) -> dict[str, Any]:
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
        "source_split": row.get("source_split", component),
        "patch_component": component,
    }
    for key in (
        "repair_id",
        "failure_tags",
        "number_policy",
        "current_checkpoint_700_summary",
        "v3_checkpoint_110_summary",
        "writer_evidence_span",
        "bad_summary_checkpoint700",
        "pre_dpo_summary",
    ):
        if key in row:
            out[key] = row[key]
    user = SYSTEM + "\n\n" + USER_TMPL.format(title=out["title"], abstract=out["abstract"])
    out["messages"] = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": out["summary"]},
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kimi-distill", default="hf_bucket/experiments/sft/q5_step2400_sft700_curated_v3/kimi_v3_win_concise_repairs_v1/combined_materialized/training_rows.jsonl")
    ap.add_argument("--accepted-repairs", default="hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/failure_mining/checkpoint700_10000_v1/repairs_targeted_v1/materialized/training_rows.jsonl")
    ap.add_argument("--output-dir", default="hf_bucket/experiments/sft/q5_step2400_sft700_v4_kimi_distill/dataset")
    args = ap.parse_args()

    kimi = load_jsonl(Path(args.kimi_distill))
    accepted = load_jsonl(Path(args.accepted_repairs))

    rows_by_pid: dict[str, dict[str, Any]] = {}

    # Kimi distillation rows are highest priority: v3 wins compressed by Kimi.
    for row in kimi:
        if 10 <= wc(row["summary"]) <= 13:
            rows_by_pid[row["paper_id"]] = normalize(row, "kimi_v3_win_concise_distill")

    # Add concise accepted repairs from the earlier Kimi+Qwen targeted repair set.
    for row in accepted:
        if row["paper_id"] in rows_by_pid:
            continue
        if 10 <= wc(row["summary"]) <= 13:
            rows_by_pid[row["paper_id"]] = normalize(row, "accepted_targeted_repair_le13")

    rows = list(rows_by_pid.values())
    rows.sort(key=lambda r: (r["patch_component"], r["paper_id"]))
    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "train.jsonl", rows)

    wcs = [wc(r["summary"]) for r in rows]
    manifest = {
        "dataset_id": "q5_step2400_sft700_v4_kimi_distill",
        "purpose": "Assistant-only v4 patch combining Kimi-distilled v3 wins with concise accepted targeted repairs; no worker-generated labels.",
        "row_count": len(rows),
        "source_files": {
            "kimi_distill": args.kimi_distill,
            "accepted_repairs": args.accepted_repairs,
        },
        "component_counts": dict(Counter(r["patch_component"] for r in rows)),
        "digit_summary_count": sum(bool(DIGIT_RE.search(r["summary"])) for r in rows),
        "number_policy_counts": dict(Counter(r.get("number_policy", "") for r in rows if r.get("number_policy"))),
        "failure_tag_counts": dict(Counter(t for r in rows for t in r.get("failure_tags", []))),
        "word_count": {
            "avg": sum(wcs) / len(wcs),
            "min": min(wcs),
            "max": max(wcs),
            "under10": sum(w < 10 for w in wcs),
            "over13": sum(w > 13 for w in wcs),
            "over15": sum(w > 15 for w in wcs),
            "counts": dict(Counter(wcs)),
        },
        "caveat": "Kimi distillation rows come from old 200-row review cases; use fresh/uncontaminated holdout for promotion decisions.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# q5 Step-2400 SFT700 v4 Kimi Distill Dataset\n\n"
        "Kimi-distilled v3 wins plus concise accepted targeted repairs. No worker-generated labels.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
