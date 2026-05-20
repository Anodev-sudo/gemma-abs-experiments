#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_ideal200_regen_v1")
RUNS = [
    ("full_v1", ROOT / "kimi_label_run_v7_full"),
    ("overflow70_v1", ROOT / "kimi_label_run_v7_overflow70"),
    ("full_v2_reverify", ROOT / "qwen_v2_reverify_full_v1rejects"),
    ("overflow70_v2_reverify", ROOT / "qwen_v2_reverify_overflow70_v1rejects"),
]
OUT = ROOT / "clean_200_after_filters_v1"
DIGIT_RE = re.compile(r"\d")
FORMULA_RE = re.compile(r"[$<>≤≥≈]|\\[A-Za-z]+|\bO\(|\^|_")
HIDDEN_RE = re.compile(r"\b(most|often|widespread|best|highest|largest|substantial|dramatic|major|state-of-the-art|outperforms|superior|strong|highly|significant)\b", re.I)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def hard_clean(row: dict[str, Any]) -> tuple[bool, list[str]]:
    s = str(row.get("repair_summary", ""))
    reasons = []
    if not (10 <= len(s.split()) <= 15): reasons.append(f"wc={len(s.split())}")
    if DIGIT_RE.search(s): reasons.append("digit")
    if FORMULA_RE.search(s): reasons.append("formula")
    if HIDDEN_RE.search(s): reasons.append("hidden_quant")
    if not s.endswith("."): reasons.append("period")
    if row.get("verifier_decision") != "accept": reasons.append("not_accept")
    return not reasons, reasons


def training_row(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": r["paper_id"],
        "title": r.get("title", ""),
        "abstract": r.get("abstract", ""),
        "summary": r.get("repair_summary", ""),
        "primary_category": r.get("primary_category", ""),
        "categories": r.get("categories", []) if isinstance(r.get("categories"), list) else [],
        "created_date": r.get("created_date", ""),
        "teacher_model": r.get("writer_model", ""),
        "generator_prompt_version": r.get("writer_prompt_version", ""),
        "review_prompt_version": r.get("reviewer_prompt_version", ""),
        "source_split": "q5_numeric_ideal200_clean_after_filters_v1",
        "bad_summary_checkpoint700": r.get("bad_summary", ""),
        "pre_dpo_summary": r.get("pre_dpo_summary", ""),
        "failure_tags": r.get("failure_tags", []),
        "number_policy": r.get("number_policy", ""),
        "repair_candidate_id": r.get("candidate_id", ""),
        "audit_status": r.get("audit_status", ""),
        "selection_score": r.get("selection_score", 0),
        "reviewer_model": r.get("reviewer_model", ""),
    }


def main() -> None:
    all_hard = []
    excluded = []
    for run_label, run_root in RUNS:
        for r in load_jsonl(run_root / "materialized" / "accepted_repairs.jsonl"):
            ok, reasons = hard_clean(r)
            rr = {**r, "source_run_label": run_label, "source_run_root": str(run_root)}
            if ok:
                all_hard.append(rr)
            else:
                excluded.append({**rr, "hard_exclude_reasons": reasons})

    by_pid: dict[str, dict[str, Any]] = {}
    # Prefer original full-200 rows, then v1 accept over v2 where duplicates could exist.
    priority = {"full_v1": 0, "full_v2_reverify": 1, "overflow70_v1": 2, "overflow70_v2_reverify": 3}
    for r in sorted(all_hard, key=lambda x: (priority.get(x["source_run_label"], 9), x.get("candidate_id", ""))):
        by_pid.setdefault(r["paper_id"], r)

    full_rows = [r for r in by_pid.values() if str(r.get("candidate_id", "")).startswith("ideal200:")]
    overflow_rows = [r for r in by_pid.values() if str(r.get("candidate_id", "")).startswith("ideal200_overflow70:")]
    overflow_rows = sorted(overflow_rows, key=lambda r: (-int(r.get("selection_score", 0) or 0), r.get("candidate_id", "")))
    selected = list(sorted(full_rows, key=lambda r: r.get("candidate_id", "")))
    needed = 200 - len(selected)
    if needed < 0:
        selected = selected[:200]
        needed = 0
    selected.extend(overflow_rows[:needed])
    if len(selected) != 200:
        raise SystemExit(f"Could not select 200 rows; got {len(selected)} from full={len(full_rows)} overflow={len(overflow_rows)}")
    selected_pids = {r["paper_id"] for r in selected}
    surplus = [r for r in by_pid.values() if r["paper_id"] not in selected_pids]

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "accepted_repairs_200.jsonl", selected)
    write_jsonl(OUT / "training_rows_200.jsonl", [training_row(r) for r in selected])
    write_jsonl(OUT / "surplus_hard_clean.jsonl", surplus)
    write_jsonl(OUT / "hard_excluded_accepted_repairs.jsonl", excluded)

    current_pool_pids = {r["paper_id"] for r in load_jsonl(ROOT / "kimi_label_run" / "candidate_pool" / "candidates.jsonl")}
    accepted_full_pids = {r["paper_id"] for r in full_rows}
    manifest = {
        "created_by": "scripts/materialize_numeric_ideal200_clean_pool_v1.py",
        "selected_rows": len(selected),
        "selected_unique_papers": len(selected_pids),
        "full_current_accepted_rows": len(full_rows),
        "overflow_replacement_rows": sum(1 for r in selected if str(r.get("candidate_id", "")).startswith("ideal200_overflow70:")),
        "surplus_hard_clean_rows": len(surplus),
        "hard_excluded_accepted_rows": len(excluded),
        "missing_original_current_paper_ids": sorted(current_pool_pids - accepted_full_pids),
        "selected_audit_status_counts": dict(Counter(r.get("audit_status", "") for r in selected)),
        "selected_source_run_counts": dict(Counter(r.get("source_run_label", "") for r in selected)),
        "selected_review_prompt_counts": dict(Counter(r.get("reviewer_prompt_version", "") for r in selected)),
        "selected_generator_prompt_counts": dict(Counter(r.get("writer_prompt_version", "") for r in selected)),
        "selected_digit_rows": sum(1 for r in selected if DIGIT_RE.search(r.get("repair_summary", ""))),
        "selected_formula_rows": sum(1 for r in selected if FORMULA_RE.search(r.get("repair_summary", ""))),
        "selected_hidden_quant_rows": sum(1 for r in selected if HIDDEN_RE.search(r.get("repair_summary", ""))),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
