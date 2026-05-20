#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_hard_attractor_booster_v1")
SURGICAL_RUN = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_ideal200_surgical_weight_v1_lr1e6")
SWEEP_ROWS = SURGICAL_RUN / "eval" / "heuristic_sweep_rows.jsonl"
EVAL_INPUTS = SURGICAL_RUN / "eval" / "heuristic_inputs" / "combined_454.jsonl"
OUTPUT = ROOT / "kimi_qwen_label_run" / "candidate_pool" / "candidates.jsonl"
TARGET_ROWS = 84
STRESS_IDS = ["1901.06548", "2111.10635", "2303.09722", "2406.04373"]
UNIT_TRAP_RE = re.compile(r"%|\bnm\b|\bmm\b|\bcm\b|\bkg\b|\bx\b|×|\bbits?\b|\bbit-length\b|\bF1\b|\bAUC\b|\bBLEU\b|\baccuracy\b|\blatency\b|\bcost", re.I)
BENCHMARK_RE = re.compile(r"\boutperform|\bbest\b|\bstate-of-the-art|\baccuracy\b|\bbenchmark|\bcoverage\b", re.I)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def failure_tags(row: dict[str, Any]) -> list[str]:
    h = row.get("heuristics", {}) or {}
    flags = list(h.get("flags", []))
    tags = list(flags)
    summary = row.get("summary", "")
    baseline = row.get("baseline_summary", "")
    if summary and baseline and summary == baseline:
        tags.append("exact_unchanged_vs_ckpt700")
    if UNIT_TRAP_RE.search(summary):
        tags.append("unit_or_metric_trap")
    if BENCHMARK_RE.search(summary):
        tags.append("benchmark_headline_trap")
    nums = h.get("numbers", []) or []
    if len(nums) == 1 and "has_digit" in flags:
        tags.append("single_number_role_risk")
    return sorted(set(tags))


def score(row: dict[str, Any]) -> int:
    tags = set(failure_tags(row))
    s = 0
    if row.get("paper_id") in STRESS_IDS: s += 10000
    if "exact_unchanged_vs_ckpt700" in tags: s += 120
    if "unsupported_number" in tags: s += 100
    if "number_dense" in tags: s += 80
    if "unit_or_metric_trap" in tags: s += 70
    if "benchmark_headline_trap" in tags: s += 55
    if "has_digit" in tags: s += 40
    if "over15" in tags: s += 25
    if "formula_or_html" in tags: s += 20
    if "single_number_role_risk" in tags: s += 15
    s += min(len(row.get("heuristics", {}).get("numbers", []) or []), 5) * 5
    return s


def candidate(row: dict[str, Any], idx: int, source_kind: str) -> dict[str, Any]:
    tags = failure_tags(row)
    h = row.get("heuristics", {}) or {}
    short = []
    if "exact_unchanged_vs_ckpt700" in tags: short.append("still exact-unchanged from checkpoint-700")
    if "unsupported_number" in tags: short.append("contains unsupported or risky number")
    if "number_dense" in tags: short.append("dense numeric attractor")
    if "unit_or_metric_trap" in tags: short.append("unit or metric token trap")
    if "benchmark_headline_trap" in tags: short.append("benchmark headline wording trap")
    if not short: short.append("current q5 output remains numerically risky")
    return {
        "candidate_id": f"hard_attractor_booster:{idx:04d}",
        "paper_id": row["paper_id"],
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "primary_category": row.get("primary_category", ""),
        "bad_summary": row.get("summary", ""),
        "pre_dpo_summary": row.get("baseline_summary", ""),
        "triage_decision": "current_q5_hard_attractor_failure",
        "triage_severity": "high" if score(row) >= 200 else "medium",
        "failure_tags": tags,
        "short_reason": "; ".join(short),
        "current_checkpoint": "numeric_ideal200_surgical_weight_v1_lr1e6_checkpoint_200",
        "current_output_numbers": h.get("numbers", []),
        "current_unsupported_numbers": h.get("unsupported_numbers", []),
        "source_component": source_kind,
        "source_split": "q5_numeric_hard_attractor_booster_v1",
        "selection_score": score(row),
    }


def main() -> None:
    rows = load_jsonl(SWEEP_ROWS)
    inputs = {r["paper_id"]: r for r in load_jsonl(EVAL_INPUTS)}
    ck200 = [r for r in rows if r.get("checkpoint") == 200]
    # Attach primary category if available.
    for r in ck200:
        src = inputs.get(r["paper_id"], {})
        r["primary_category"] = src.get("primary_category", "")

    stress = [r for r in ck200 if r.get("paper_id") in STRESS_IDS]
    numeric = [r for r in ck200 if r.get("split") == "numeric_ckpt700_risk_pool350"]
    eligible = [r for r in numeric if r.get("heuristics", {}).get("flags")]
    eligible.sort(key=lambda r: (-score(r), r.get("paper_id", "")))

    selected: list[dict[str, Any]] = []
    seen = set()
    for r in stress + eligible:
        if r["paper_id"] in seen:
            continue
        selected.append(r)
        seen.add(r["paper_id"])
        if len(selected) >= TARGET_ROWS:
            break

    candidates = [candidate(r, i, "remembered_stress" if r["paper_id"] in STRESS_IDS else "ckpt200_numeric_survivor") for i, r in enumerate(selected, start=1)]
    write_jsonl(OUTPUT, candidates)
    manifest = {
        "created_by": "scripts/setup_numeric_hard_attractor_booster_v1.py",
        "source_sweep_rows": str(SWEEP_ROWS),
        "target_rows": TARGET_ROWS,
        "candidate_rows": len(candidates),
        "unique_papers": len({c["paper_id"] for c in candidates}),
        "stress_ids_selected": [c["paper_id"] for c in candidates if c["paper_id"] in STRESS_IDS],
        "source_component_counts": dict(Counter(c["source_component"] for c in candidates)),
        "failure_tag_counts": dict(Counter(tag for c in candidates for tag in c["failure_tags"])),
        "score_min": min(c["selection_score"] for c in candidates),
        "score_max": max(c["selection_score"] for c in candidates),
    }
    (OUTPUT.parent / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
