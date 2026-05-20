#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
TARGETS = {
    "1901.06548": "Height-optimal tangles are NP-hard, with faster algorithms for structured swap lists.",
    "2111.10635": "HeterPS improves DNN training throughput 14.5x while cutting costs 312.3%.",
    "2303.09722": "A 3D microfiber coupler achieves 97% efficiency across broad bandwidths.",
    "2406.04373": "LLM-guided Verilog reading generates tests that outperform random testing.",
}
REPEATS = {
    "1901.06548": 40,
    "2111.10635": 40,
    "2303.09722": 8,
    "2406.04373": 8,
}


def load_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("paper_id") in TARGETS:
                rows[row["paper_id"]] = row
    missing = sorted(set(TARGETS) - set(rows))
    if missing:
        raise SystemExit(f"missing rows: {missing}")
    return rows


def messages(title: str, abstract: str, summary: str) -> list[dict[str, str]]:
    user = SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE.format(title=title, abstract=abstract)
    return [{"role": "user", "content": user}, {"role": "assistant", "content": summary}]


def make(row: dict[str, Any], replica: int) -> dict[str, Any]:
    pid = row["paper_id"]
    summary = TARGETS[pid]
    return {
        "paper_id": pid if replica == 0 else f"{pid}#replica{replica}",
        "source_paper_id": pid,
        "title": row["title"],
        "abstract": row["abstract"],
        "summary": summary,
        "primary_category": row.get("primary_category", ""),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date", ""),
        "teacher_model": "human_numeric_poc_exact_prompt_label",
        "generator_prompt_version": "gemma_270m_it_abstract_to_glanceable_summary_v1_exact",
        "review_prompt_version": "manual_numeric_binding_poc_exact_prompt_v1",
        "source_split": "numeric_binding_poc_train_on_stress_cases_exact_prompt",
        "patch_component": "exact_prompt_repeated_stress" if pid in {"1901.06548", "2111.10635"} else "exact_prompt_repeated_support",
        "messages": messages(row["title"], row["abstract"], summary),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="hf_bucket/experiments/dpo/q5_step2400_dpo_mined10000_v2_valid35/behavior_compare_final_vs_pre/non_trained_holdout_pair_rows_200.jsonl")
    ap.add_argument("--output-dir", default="hf_bucket/experiments/sft/q5_step2400_sft700_numeric_binding_poc_exact_prompt_v1/dataset")
    args = ap.parse_args()
    source = load_rows(Path(args.source))
    rows = []
    for pid, repeats in REPEATS.items():
        for replica in range(repeats):
            rows.append(make(source[pid], replica))
    out = Path(args.output_dir)
    write_jsonl(out / "train.jsonl", rows)
    write_jsonl(out / "stress_eval_rows.jsonl", [source[pid] for pid in TARGETS])
    manifest = {
        "dataset_id": "q5_step2400_sft700_numeric_binding_poc_exact_prompt_v1",
        "purpose": "Exact-prompt overfit diagnostic: training messages match scripts/generate_llama_stream.py prompt exactly.",
        "row_count": len(rows),
        "repeat_counts": REPEATS,
        "component_counts": dict(Counter(r["patch_component"] for r in rows)),
        "stress_targets": TARGETS,
        "important_caveat": "Deliberately trains on and repeats stress/eval cases; diagnostic only.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
