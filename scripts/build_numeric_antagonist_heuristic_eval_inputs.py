#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_antagonist_clean_v1_lr4e7/eval/heuristic_inputs")
CANDIDATES = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_antagonist_clean_v1_pool350/kimi_qwen_label_run/candidate_pool/candidates.jsonl")
NORMAL_CORPUS = Path("hf_dataset/corpora/recent_3y_balanced/arxiv_abstract_corpus_weighted_20000.jsonl")
TRAIN = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_antagonist_clean_v1_pool350/dataset_600normal_200numeric_clean_v1/train.jsonl")
RELEASES = [
    Path("hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/train.jsonl"),
    Path("hf_dataset/releases/recent_3y_balanced_10746_v1/training_rows.jsonl"),
    Path("hf_dataset/releases/recent_3y_balanced_incremental_6249_v1/training_rows.jsonl"),
]
STRESS_IDS = ["1901.06548", "2111.10635", "2303.09722", "2406.04373"]


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def input_row(row: dict, eval_id: str, split: str) -> dict:
    return {
        "eval_id": eval_id,
        "paper_id": row["paper_id"],
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "primary_category": row.get("primary_category", ""),
        "split": split,
    }


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    train_ids = {r["paper_id"] for r in load_jsonl(TRAIN)}

    numeric = [input_row(r, f"numeric_{i:04d}", "numeric_ckpt700_risk_pool350") for i, r in enumerate(load_jsonl(CANDIDATES), start=1)]
    write_jsonl(ROOT / "numeric_pool350.jsonl", numeric)

    normal = []
    seen = set(train_ids) | {r["paper_id"] for r in numeric}
    for r in load_jsonl(NORMAL_CORPUS):
        pid = r.get("paper_id")
        if not pid or pid in seen:
            continue
        abstract = str(r.get("abstract") or "")
        if len(abstract) < 250:
            continue
        normal.append(input_row(r, f"normal_{len(normal)+1:04d}", "fresh_recent_normal_100"))
        seen.add(pid)
        if len(normal) >= 100:
            break
    write_jsonl(ROOT / "fresh_normal_100.jsonl", normal)

    by_id = {}
    for path in RELEASES:
        if not path.exists():
            continue
        for r in load_jsonl(path):
            if r.get("paper_id") in STRESS_IDS and r.get("paper_id") not in by_id:
                by_id[r["paper_id"]] = r
    stress = [input_row(by_id[pid], f"stress_{i:02d}", "remembered_stress") for i, pid in enumerate(STRESS_IDS, start=1) if pid in by_id]
    write_jsonl(ROOT / "remembered_stress.jsonl", stress)

    combined = numeric + normal + stress
    write_jsonl(ROOT / "combined_454.jsonl", combined)
    manifest = {
        "numeric_rows": len(numeric),
        "normal_rows": len(normal),
        "stress_rows": len(stress),
        "combined_rows": len(combined),
        "stress_ids_found": [r["paper_id"] for r in stress],
        "excluded_train_ids": len(train_ids),
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
