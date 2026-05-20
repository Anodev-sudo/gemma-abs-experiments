#!/usr/bin/env python3
"""Build majority-synthetic numeric-role SFT curriculum from reviewed real rows."""

from __future__ import annotations

import argparse
import json
import random
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def wc(text: str) -> int:
    return len(str(text).split())


def normalize(row: dict[str, Any], component: str) -> dict[str, Any]:
    out = dict(row)
    out["patch_component"] = component
    out["curriculum_component"] = component
    user = SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE.format(
        title=out.get("title", ""),
        abstract=out.get("abstract", ""),
    )
    out["messages"] = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": out["summary"]},
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--real-reviewed",
        default="hf_bucket/experiments/sft/q5_step2400_sft700_numeric_role_curriculum_v1/real_mined_numeric_v1/subagent_review/materialized/training_rows.jsonl",
    )
    ap.add_argument(
        "--synthetic-screened",
        default="hf_bucket/experiments/sft/q5_step2400_sft700_numeric_role_curriculum_v1/current_sft700_screen_v5_600/materialized_failed_only/training_rows.jsonl",
    )
    ap.add_argument(
        "--output-dir",
        default="hf_bucket/experiments/sft/q5_step2400_sft700_numeric_role_curriculum_v1/dataset_majority_synthetic_v1",
    )
    ap.add_argument("--seed", type=int, default=20260515)
    args = ap.parse_args()

    real_raw = load_jsonl(Path(args.real_reviewed))
    synth_raw = load_jsonl(Path(args.synthetic_screened))

    real = [normalize(row, "real_mined_numeric_repair_subagent_accepted") for row in real_raw]
    synth = [normalize(row, "synthetic_numeric_role_current_failed_v5") for row in synth_raw]

    bad_lengths = [
        {"paper_id": r.get("paper_id"), "component": r.get("patch_component"), "word_count": wc(r.get("summary", ""))}
        for r in real + synth
        if not 10 <= wc(r.get("summary", "")) <= 15
    ]
    if bad_lengths:
        raise ValueError(f"Found rows outside 10-15 words: {bad_lengths[:5]}")

    seen: set[str] = set()
    duplicates: list[str] = []
    rows: list[dict[str, Any]] = []
    for row in real + synth:
        pid = str(row.get("paper_id", ""))
        if pid in seen:
            duplicates.append(pid)
            continue
        seen.add(pid)
        rows.append(row)
    if duplicates:
        raise ValueError(f"Duplicate paper_id values: {duplicates[:10]}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    out = Path(args.output_dir)
    write_jsonl(out / "train.jsonl", rows)

    component_counts = Counter(r["patch_component"] for r in rows)
    synth_count = component_counts["synthetic_numeric_role_current_failed_v5"]
    real_count = component_counts["real_mined_numeric_repair_subagent_accepted"]
    total = len(rows)
    manifest = {
        "dataset_id": "q5_step2400_sft700_numeric_role_curriculum_majority_synthetic_v1",
        "purpose": "Numeric-only assistant-loss patch from checkpoint-700 using subagent-accepted real mined repairs plus current-failed synthetic numeric-role rows.",
        "row_count": total,
        "source_files": {
            "real_reviewed": args.real_reviewed,
            "synthetic_screened": args.synthetic_screened,
        },
        "seed": args.seed,
        "component_counts": dict(component_counts),
        "synthetic_fraction": synth_count / total if total else 0,
        "real_fraction": real_count / total if total else 0,
        "majority_synthetic": synth_count > real_count,
        "minimum_synthetic_needed_for_majority_with_real_count": real_count + 1,
        "extra_synthetic_above_bare_majority": synth_count - (real_count + 1),
        "word_count_counts": dict(sorted(Counter(wc(r.get("summary", "")) for r in rows).items())),
        "real_number_policy_counts": dict(Counter(r.get("number_policy", "") for r in real)),
        "real_failure_tag_counts": dict(Counter(t for r in real for t in r.get("failure_tags", []))),
        "synthetic_pattern_counts": dict(Counter(r.get("numeric_pattern", "") for r in synth)),
        "synthetic_number_policy_counts": dict(Counter(r.get("number_policy", "") for r in synth)),
        "notes": [
            "Worker review decisions are used only for accept/reject filtering; no worker-generated labels are included.",
            "Synthetic rows were admitted only after current checkpoint-700 failed row-specific numeric-policy checks under llama.cpp Q5 runtime.",
            "All rows are formatted with the exact generate_llama_stream.py title+abstract prompt.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        "# q5 Step-2400 SFT700 Numeric-role Curriculum Majority Synthetic v1\n\n"
        "Numeric-only SFT patch dataset from checkpoint-700. Real mined rows are included only after subagent accept/reject review. "
        "Synthetic rows are included only if current checkpoint-700 failed their numeric policy under llama.cpp Q5 runtime.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
