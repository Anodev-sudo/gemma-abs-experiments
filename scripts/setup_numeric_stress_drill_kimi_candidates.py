#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_antagonist_clean_v1_pool350/stress_drill_kimi_v1/candidate_pool")
SOURCES = [
    Path("hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/train.jsonl"),
    Path("hf_dataset/releases/recent_3y_balanced_10746_v1/training_rows.jsonl"),
    Path("hf_dataset/releases/recent_3y_balanced_incremental_6249_v1/training_rows.jsonl"),
]
STRESS = {
    "1901.06548": {
        "bad": "Tangles with 1618-bit-length optimal order are computed for n-wire lists.",
        "tags": ["decimal_corruption_risk", "formula_risk", "np_hard_scope", "stress_case"],
        "note": "Avoid formula pieces and decimal corruption; capture NP-hardness plus supported algorithmic contribution qualitatively.",
    },
    "2111.10635": {
        "bad": "Paddle-Heterogeneous Parameter Server schedules deep neural networks with heterogeneous computing, cutting training costs by 14.5x.",
        "tags": ["wrong_metric_binding_risk", "throughput_cost_role_flip", "stress_case"],
        "note": "Do not bind the throughput multiplier to cost. Say qualitatively that scheduling improves throughput and reduces cost.",
    },
    "2303.09722": {
        "bad": "A 3D microfiber coupling achieves 97% efficiency at 1542 nm wavelength.",
        "tags": ["exact_metric_risk", "wavelength_percentage_risk", "stress_case"],
        "note": "Avoid exact efficiency and wavelength. Preserve the fiber-to-chip coupling contribution qualitatively.",
    },
    "2406.04373": {
        "bad": "LLM-guided Verilog reader generates test designs that outperform random testing.",
        "tags": ["normal_style_anchor", "hardware_test_generation", "stress_case"],
        "note": "Keep the Verilog reader test-generation contribution concise and digit-free.",
    },
}


def load_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    by_id = {}
    for source in SOURCES:
        for row in load_jsonl(source):
            pid = row.get("paper_id")
            if pid in STRESS and pid not in by_id:
                by_id[pid] = row
    rows = []
    for pid, info in STRESS.items():
        src = by_id[pid]
        for variant in range(1, 7):
            rows.append({
                "candidate_id": f"stress_drill:{pid}:v{variant:02d}",
                "paper_id": pid,
                "title": src.get("title", ""),
                "abstract": src.get("abstract", ""),
                "primary_category": src.get("primary_category", ""),
                "categories": src.get("categories") if isinstance(src.get("categories"), list) else [],
                "created_date": src.get("created_date", ""),
                "bad_summary": info["bad"],
                "pre_dpo_summary": src.get("summary", ""),
                "triage_decision": "generate_stress_drill_digit_free_label",
                "triage_severity": "high",
                "failure_tags": info["tags"] + [f"variant_{variant:02d}"],
                "short_reason": info["note"],
                "source_split": "q5_numeric_stress_drill_kimi_v1",
            })
    write_jsonl(OUT / "candidates.jsonl", rows)
    manifest = {"rows": len(rows), "stress_ids": list(STRESS), "variants_per_id": 6}
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
