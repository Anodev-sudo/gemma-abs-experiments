#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


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


def by_pid(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["paper_id"]: r for r in rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="hf_bucket/experiments/dpo/q5_step2400_dpo_mined10000_v2_valid35/behavior_compare_final_vs_pre/non_trained_holdout_pair_rows_200.jsonl")
    ap.add_argument("--a-outputs", default="hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/eval/checkpoint_700_outputs_100.jsonl")
    ap.add_argument("--b-outputs", default="hf_bucket/experiments/sft/q5_step2400_sft700_targeted_only_v2/eval/checkpoint_50_outputs_100.jsonl")
    ap.add_argument("--a-model", default="current_sft700_checkpoint_700")
    ap.add_argument("--b-model", default="targeted_only_v2_checkpoint_50")
    ap.add_argument("--output-dir", default="hf_bucket/experiments/sft/q5_step2400_sft700_targeted_only_v2/blind_review_ckpt700_vs_v2ckpt50_100")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260512)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    holdout = load_jsonl(Path(args.holdout), args.limit)
    a_by_pid = by_pid(load_jsonl(Path(args.a_outputs)))
    b_by_pid = by_pid(load_jsonl(Path(args.b_outputs)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    packet_path = out_dir / "blind_review_packet.jsonl"
    key_path = out_dir / "blind_key.jsonl"
    md_path = out_dir / "README.md"

    packet_rows = []
    key_rows = []
    for idx, row in enumerate(holdout, 1):
        pid = row["paper_id"]
        if pid not in a_by_pid or pid not in b_by_pid:
            raise ValueError(f"Missing output for {pid}")
        left_is_a = rng.choice([True, False])
        a_summary = a_by_pid[pid].get("normalized_output", "")
        b_summary = b_by_pid[pid].get("normalized_output", "")
        left_summary = a_summary if left_is_a else b_summary
        right_summary = b_summary if left_is_a else a_summary
        packet_rows.append({
            "review_id": f"blind_ckpt700_v2ckpt50:{idx:04d}",
            "paper_id": pid,
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "summary_a": left_summary,
            "summary_b": right_summary,
            "instructions": "Choose summary_a, summary_b, tie, or neither. Prefer faithful, specific, natural 10-15 word mobile-feed summaries. Penalize unsupported numbers, wrong metric binding, semantic drift, title rehashing, overlong summaries, and generic smoothing.",
        })
        key_rows.append({
            "review_id": f"blind_ckpt700_v2ckpt50:{idx:04d}",
            "paper_id": pid,
            "summary_a_model": args.a_model if left_is_a else args.b_model,
            "summary_b_model": args.b_model if left_is_a else args.a_model,
            "current_sft700_checkpoint_700_summary": a_summary,
            "targeted_only_v2_checkpoint_50_summary": b_summary,
            "outputs_identical": a_summary.strip() == b_summary.strip(),
        })

    with packet_path.open("w", encoding="utf-8") as f:
        for r in packet_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with key_path.open("w", encoding="utf-8") as f:
        for r in key_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "review_id": "blind_ckpt700_vs_v2ckpt50_100",
        "seed": args.seed,
        "limit": args.limit,
        "holdout": args.holdout,
        "a_outputs": args.a_outputs,
        "b_outputs": args.b_outputs,
        "a_model": args.a_model,
        "b_model": args.b_model,
        "packet_path": str(packet_path),
        "key_path": str(key_path),
        "rows": len(packet_rows),
        "identical_outputs": sum(k["outputs_identical"] for k in key_rows),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(
        "# Blind review: current checkpoint-700 vs targeted-only v2 checkpoint-50\n\n"
        "Rows: 100. Side assignment is randomized with seed 20260512.\n\n"
        "Packet: `blind_review_packet.jsonl`\n\n"
        "Key: `blind_key.jsonl`\n\n"
        "Reviewer should not read the key before completing judgments.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
