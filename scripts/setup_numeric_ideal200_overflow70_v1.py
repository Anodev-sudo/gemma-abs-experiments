#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

POOL = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_antagonist_clean_v1_pool350/kimi_qwen_label_run/candidate_pool/candidates.jsonl")
CURRENT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_ideal200_regen_v1/kimi_label_run/candidate_pool/candidates.jsonl")
OUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_ideal200_regen_v1/kimi_label_run_v7_overflow70")

TAG_WEIGHTS = {
    "number_dense_summary": 5,
    "changed_dropped_source_terms": 4,
    "unsupported_terms_low_anchor": 4,
    "repetition_or_tautology": 2,
    "decimal_compaction": 3,
    "unsupported_number": 1,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def score(row: dict[str, Any]) -> int:
    tags = set(row.get("failure_tags", []))
    return sum(TAG_WEIGHTS.get(tag, 0) for tag in tags)


def umbrella(cat: str) -> str:
    if cat.startswith("cs."):
        return "cs"
    if cat.startswith("math."):
        return "math"
    if cat.startswith(("astro-ph", "cond-mat", "gr-qc", "hep-", "nucl-", "physics.", "quant-ph")):
        return "physics"
    if cat.startswith("stat."):
        return "stat"
    if cat.startswith("q-"):
        return "qfin"
    return cat.split(".", 1)[0] if cat else "unknown"


def main() -> None:
    current_pids = {r["paper_id"] for r in load_jsonl(CURRENT)}
    remaining = [r for r in load_jsonl(POOL) if r.get("paper_id") not in current_pids]
    for r in remaining:
        r["selection_score"] = score(r)
        r["umbrella"] = umbrella(str(r.get("primary_category", "")))
    # Prefer high-risk shapes, but avoid taking only one discipline.
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in sorted(remaining, key=lambda x: (-x["selection_score"], x.get("candidate_id", ""))):
        buckets[r["umbrella"]].append(r)
    selected: list[dict[str, Any]] = []
    max_per_umbrella = {"cs": 28, "physics": 24, "math": 12, "stat": 6, "qfin": 4}
    # First pass with caps.
    for r in sorted(remaining, key=lambda x: (-x["selection_score"], x["umbrella"], x.get("candidate_id", ""))):
        if len(selected) >= 70:
            break
        if sum(1 for x in selected if x["umbrella"] == r["umbrella"]) >= max_per_umbrella.get(r["umbrella"], 4):
            continue
        selected.append(r)
    # Fill any remainder by score.
    selected_ids = {r["paper_id"] for r in selected}
    for r in sorted(remaining, key=lambda x: (-x["selection_score"], x.get("candidate_id", ""))):
        if len(selected) >= 70:
            break
        if r["paper_id"] not in selected_ids:
            selected.append(r)
            selected_ids.add(r["paper_id"])

    candidates = []
    for idx, r in enumerate(selected, start=1):
        guidance = (
            "Overflow candidate from checkpoint-700 numeric-risk pool, not in the current regenerated 200. "
            "Regenerate from source using the ideal no-number abstraction policy; avoid copying old clean labels."
        )
        if r.get("selection_score", 0) >= 7:
            guidance += " Prioritize role-safe abstraction because this row has dense/semantic numeric-risk tags."
        candidates.append({
            **r,
            "candidate_id": f"ideal200_overflow70:{idx:04d}",
            "failure_tags": ["ideal200_overflow70", *r.get("failure_tags", [])],
            "short_reason": f"{guidance} Original reason: {r.get('short_reason', '')}",
            "audit_status": "overflow",
            "source_split": "q5_numeric_ideal200_regen_v1_overflow70_candidate_pool",
            "old_candidate_id": r.get("candidate_id", ""),
        })
    write_jsonl(OUT / "candidate_pool" / "candidates.jsonl", candidates)
    manifest = {
        "created_by": "scripts/setup_numeric_ideal200_overflow70_v1.py",
        "source_pool": str(POOL),
        "excluded_current_pool": str(CURRENT),
        "remaining_eligible_rows": len(remaining),
        "selected_rows": len(candidates),
        "selection_policy": "Top-scored remaining checkpoint-700 numeric-risk rows with loose umbrella caps.",
        "score_weights": TAG_WEIGHTS,
        "umbrella_counts": dict(Counter(c.get("umbrella", "") for c in candidates)),
        "tag_counts": dict(Counter(tag for c in candidates for tag in c.get("failure_tags", []))),
    }
    (OUT / "candidate_pool" / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
