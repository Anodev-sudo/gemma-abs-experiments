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

HAND_TARGETS = {
    "1901.06548": "Height-optimal tangle realization is NP-hard but admits faster exact algorithms.",
    "2111.10635": "Paddle-HeterPS delivers 14.5x higher throughput and 312.3% lower cost via RL scheduling.",
    "2304.08391": "mizar-rs checks Mizar proofs in 11.8 minutes with a 4.8x speedup.",
    "2310.19460": "Diffusion priors improve wireless image reconstruction without lowering information rates.",
    "2406.04373": "VerilogReader uses LLM code understanding to outperform random hardware test generation.",
}


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def wc(text: str) -> int:
    return len(text.split())


def add_messages(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    user = SYSTEM + "\n\n" + USER_TMPL.format(title=out["title"], abstract=out["abstract"])
    out["messages"] = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": out["summary"]},
    ]
    return out


def from_holdout_row(row: dict[str, Any], summary: str, component: str) -> dict[str, Any]:
    return add_messages({
        "paper_id": row["paper_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "summary": summary,
        "primary_category": row.get("primary_category", ""),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date", ""),
        "source_split": "q5_sft700_curated_v3_holdout_informed",
        "teacher_model": "manual_or_blind_review_selected",
        "generator_prompt_version": "manual_curated_v3",
        "review_prompt_version": "blind_worker_review_or_manual",
        "patch_component": component,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted-repairs", default="hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/failure_mining/checkpoint700_10000_v1/repairs_targeted_v1/materialized/training_rows.jsonl")
    ap.add_argument("--holdout", default="hf_bucket/experiments/dpo/q5_step2400_dpo_mined10000_v2_valid35/behavior_compare_final_vs_pre/non_trained_holdout_pair_rows_200.jsonl")
    ap.add_argument("--decoded-review", default="hf_bucket/experiments/sft/q5_step2400_sft700_targeted_only_v2/blind_review_ckpt700_vs_v2ckpt50_100/decoded_worker_blind_review.jsonl")
    ap.add_argument("--output-dir", default="hf_bucket/experiments/sft/q5_step2400_sft700_curated_v3/dataset")
    args = ap.parse_args()

    accepted = load_jsonl(Path(args.accepted_repairs))
    holdout_by_pid = {r["paper_id"]: r for r in load_jsonl(Path(args.holdout), 200)}
    decoded = load_jsonl(Path(args.decoded_review))

    rows_by_pid: dict[str, dict[str, Any]] = {}

    # 1. Concise accepted repairs only. This removes the 14-15 word bias from v2.
    for row in accepted:
        if wc(row["summary"]) <= 13:
            out = add_messages({**row, "patch_component": "concise_accepted_repair_le13"})
            rows_by_pid[out["paper_id"]] = out

    # 2. Blind-review-selected concise wins from v2 checkpoint-50.
    for row in decoded:
        if row.get("decoded_decision") != "targeted_only_v2_checkpoint_50":
            continue
        summary = row["summary_a"] if row["summary_a_model"] == "targeted_only_v2_checkpoint_50" else row["summary_b"]
        if 10 <= wc(summary) <= 13 and row["paper_id"] in holdout_by_pid:
            rows_by_pid[row["paper_id"]] = from_holdout_row(
                holdout_by_pid[row["paper_id"]], summary, "blind_v2_win_concise"
            )

    # 3. Blind-review-selected concise current wins as anti-overlength/anti-drift counterpressure.
    for row in decoded:
        if row.get("decoded_decision") != "current_sft700_checkpoint_700":
            continue
        summary = row["summary_a"] if row["summary_a_model"] == "current_sft700_checkpoint_700" else row["summary_b"]
        if 10 <= wc(summary) <= 13 and row["paper_id"] in holdout_by_pid:
            rows_by_pid[row["paper_id"]] = from_holdout_row(
                holdout_by_pid[row["paper_id"]], summary, "blind_current_win_concise_counterpressure"
            )

    # 4. Manual exact-failure targets override everything else.
    # 2304.08391 is from the accepted repair source, not the holdout.
    accepted_by_pid = {r["paper_id"]: r for r in accepted}
    for pid, summary in HAND_TARGETS.items():
        if pid in holdout_by_pid:
            src = holdout_by_pid[pid]
            rows_by_pid[pid] = from_holdout_row(src, summary, "manual_exact_failure_target")
        elif pid in accepted_by_pid:
            src = accepted_by_pid[pid]
            rows_by_pid[pid] = add_messages({**src, "summary": summary, "patch_component": "manual_exact_failure_target"})
        else:
            raise ValueError(f"No source row found for hand target {pid}")

    rows = list(rows_by_pid.values())
    rows.sort(key=lambda r: (r["patch_component"], r["paper_id"]))

    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "train.jsonl", rows)
    wcs = [wc(r["summary"]) for r in rows]
    manifest = {
        "dataset_id": "q5_step2400_sft700_curated_v3",
        "purpose": "Curated assistant-only patch: concise accepted repairs, blind-reviewed concise wins, current-win counterpressure, and exact numeric failure targets.",
        "row_count": len(rows),
        "source_files": {
            "accepted_repairs": args.accepted_repairs,
            "holdout": args.holdout,
            "decoded_review": args.decoded_review,
        },
        "caveat": "Includes selected rows from the old first-100 holdout; use a fresh holdout for promotion evaluation.",
        "component_counts": dict(Counter(r["patch_component"] for r in rows)),
        "number_policy_counts": dict(Counter(r.get("number_policy", "") for r in rows if r.get("number_policy"))),
        "failure_tag_counts": dict(Counter(t for r in rows for t in r.get("failure_tags", []))),
        "digit_summary_count": sum(bool(DIGIT_RE.search(r["summary"])) for r in rows),
        "word_count": {
            "avg": sum(wcs) / len(wcs),
            "min": min(wcs),
            "max": max(wcs),
            "under10": sum(w < 10 for w in wcs),
            "over13": sum(w > 13 for w in wcs),
            "over15": sum(w > 15 for w in wcs),
            "in_10_13": sum(10 <= w <= 13 for w in wcs),
        },
        "hand_targets": HAND_TARGETS,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# q5 Step-2400 SFT700 Curated v3 Dataset\n\n"
        "Assistant-only patch dataset with concise targeted repairs and exact failure targets.\n\n"
        "This intentionally includes selected rows from the old first-100 holdout; use a fresh holdout for promotion evaluation.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
