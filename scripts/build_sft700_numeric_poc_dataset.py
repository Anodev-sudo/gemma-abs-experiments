#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    "Keep numbers only when they are important, and bind every number to the correct measurement. "
    "If a number is risky or incidental, omit it rather than changing its meaning.\n\n"
    "Title:\n{title}\n\n"
    "Abstract:\n{abstract}"
)

TARGETS = {
    "1901.06548": "Height-optimal tangles are NP-hard, with faster algorithms for structured swap lists.",
    "2111.10635": "HeterPS improves DNN training throughput 14.5x while cutting costs 312.3%.",
    "2303.09722": "A 3D microfiber coupler achieves 97% efficiency across broad bandwidths.",
    "2406.04373": "LLM-guided Verilog reading generates tests that outperform random testing.",
}

SYNTHETIC = [
    (
        "synthetic.throughput_cost.001",
        "Role-binding benchmark: Training throughput versus cloud cost",
        "A distributed training scheduler assigns neural network layers to CPUs and GPUs. Experiments show 18.2x higher throughput and 41.7% lower cloud cost compared with a uniform GPU baseline. The throughput number measures speed, while the percentage measures cost reduction.",
        "Layer scheduling raises training throughput 18.2x while reducing cloud costs 41.7%.",
    ),
    (
        "synthetic.throughput_cost.002",
        "Role-binding benchmark: Speedup and memory reduction",
        "A compiler pass fuses sparse kernels for recommendation models. It achieves a 6.4x inference speedup and reduces memory traffic by 28.9%. The 6.4x value is speedup, not memory savings, and the 28.9% value is memory reduction.",
        "Sparse-kernel fusion speeds inference 6.4x while cutting memory traffic 28.9%.",
    ),
    (
        "synthetic.throughput_cost.003",
        "Role-binding benchmark: Latency versus accuracy",
        "A retrieval system prunes candidate documents before reranking. It lowers median latency by 37.5% while improving answer accuracy by 4.2 points. The percentage describes latency reduction, and the point value describes accuracy gain.",
        "Candidate pruning reduces latency 37.5% while improving accuracy 4.2 points.",
    ),
    (
        "synthetic.throughput_cost.004",
        "Role-binding benchmark: Throughput versus energy",
        "A batching policy for edge inference increases throughput by 3.7x and lowers energy per query by 22.4%. Throughput is the multiplicative speed measure; energy is the percentage reduction.",
        "Edge batching increases throughput 3.7x while lowering energy per query 22.4%.",
    ),
    (
        "synthetic.decimal.001",
        "Decimal benchmark: Golden-ratio parameter in graph search",
        "The graph-search algorithm has a running time containing phi^n, where phi is approximately 1.618. The decimal is a mathematical constant used in the bound, not a bit length or dataset size.",
        "Graph search uses a phi-based bound without turning 1.618 into capacity.",
    ),
    (
        "synthetic.decimal.002",
        "Decimal benchmark: Stability threshold near one half",
        "A numerical solver remains stable when the damping coefficient is below 0.625. The 0.625 value is a threshold coefficient, not 625 samples or 625-bit precision.",
        "The solver stays stable below a 0.625 damping threshold.",
    ),
    (
        "synthetic.decimal.003",
        "Decimal benchmark: Approximation ratio",
        "The approximation algorithm guarantees a 1.732 ratio for a packing problem. The number is an approximation ratio, not a count, date, or memory size.",
        "A packing algorithm guarantees a 1.732 approximation ratio.",
    ),
    (
        "synthetic.metric_flip.001",
        "Metric-binding benchmark: Accuracy versus cost",
        "A mobile classifier improves top-1 accuracy by 2.8 points and reduces battery usage by 19.6%. The points measure accuracy improvement; the percentage measures battery reduction.",
        "The mobile classifier improves accuracy 2.8 points while cutting battery use 19.6%.",
    ),
    (
        "synthetic.metric_flip.002",
        "Metric-binding benchmark: Bandwidth versus error",
        "A compression method reduces network bandwidth by 52.1% while increasing reconstruction error by only 0.03. The percentage binds to bandwidth reduction, and 0.03 binds to error increase.",
        "Compression reduces bandwidth 52.1% with only 0.03 added reconstruction error.",
    ),
    (
        "synthetic.metric_flip.003",
        "Metric-binding benchmark: Recall versus runtime",
        "A search index improves recall by 11.4 points and cuts query runtime by 63.0%. The point value is recall improvement; the percentage is runtime reduction.",
        "The search index improves recall 11.4 points while cutting runtime 63.0%.",
    ),
    (
        "synthetic.omit_risky.001",
        "Risky-number benchmark: Many incidental constants",
        "A proof analyzes a recurrence with constants 1.2, 3.4, and 9.8 before showing the decision problem is NP-hard. The constants are incidental to the main contribution and should not be summarized as metrics.",
        "The recurrence proof establishes NP-hardness without relying on incidental constants.",
    ),
    (
        "synthetic.omit_risky.002",
        "Risky-number benchmark: Multiple setup details",
        "A dataset contains 12 cameras, 48 sensors, and 7 collection days. The central result is that self-supervised calibration improves cross-device transfer, while the counts only describe the setup.",
        "Self-supervised calibration improves transfer across heterogeneous sensing devices.",
    ),
]


def load_source_rows(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("paper_id") in TARGETS:
                out[row["paper_id"]] = row
    missing = sorted(set(TARGETS) - set(out))
    if missing:
        raise SystemExit(f"missing source rows: {missing}")
    return out


def make_messages(title: str, abstract: str, summary: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": SYSTEM + "\n\n" + USER_TMPL.format(title=title, abstract=abstract)},
        {"role": "assistant", "content": summary},
    ]


def normalize(row: dict[str, Any], summary: str, component: str, replica: int = 0) -> dict[str, Any]:
    out = {
        "paper_id": row["paper_id"] if replica == 0 else f"{row['paper_id']}#replica{replica}",
        "source_paper_id": row["paper_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "summary": summary,
        "primary_category": row.get("primary_category", ""),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date", ""),
        "teacher_model": "human_numeric_poc_label",
        "generator_prompt_version": "manual_numeric_binding_poc_v1",
        "review_prompt_version": "manual_numeric_binding_poc_v1",
        "source_split": "numeric_binding_poc_train_on_stress_cases",
        "patch_component": component,
        "messages": make_messages(row["title"], row["abstract"], summary),
    }
    return out


def synthetic_row(pid: str, title: str, abstract: str, summary: str, replica: int = 0) -> dict[str, Any]:
    out = {
        "paper_id": pid if replica == 0 else f"{pid}#replica{replica}",
        "source_paper_id": pid,
        "title": title,
        "abstract": abstract,
        "summary": summary,
        "primary_category": "synthetic.NUM",
        "categories": ["synthetic.NUM"],
        "created_date": "2026-05-13",
        "teacher_model": "human_numeric_poc_label",
        "generator_prompt_version": "manual_numeric_binding_poc_v1",
        "review_prompt_version": "manual_numeric_binding_poc_v1",
        "source_split": "synthetic_numeric_binding_poc",
        "patch_component": "synthetic_numeric_binding_contrast",
        "messages": make_messages(title, abstract, summary),
    }
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="hf_bucket/experiments/dpo/q5_step2400_dpo_mined10000_v2_valid35/behavior_compare_final_vs_pre/non_trained_holdout_pair_rows_200.jsonl")
    ap.add_argument("--output-dir", default="hf_bucket/experiments/sft/q5_step2400_sft700_numeric_binding_poc_v1/dataset")
    args = ap.parse_args()

    source = load_source_rows(Path(args.source))
    rows: list[dict[str, Any]] = []

    # Heavy repetition is intentional for the proof-of-concept: prove the adapter can move sticky rows at all.
    for pid in ["1901.06548", "2111.10635"]:
        for replica in range(8):
            rows.append(normalize(source[pid], TARGETS[pid], "exact_stress_case_repeated", replica))

    for pid in ["2303.09722", "2406.04373"]:
        for replica in range(3):
            rows.append(normalize(source[pid], TARGETS[pid], "exact_support_case_repeated", replica))

    for item in SYNTHETIC:
        rows.append(synthetic_row(*item))

    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "train.jsonl", rows)

    eval_rows = [source[pid] for pid in ["1901.06548", "2111.10635", "2303.09722", "2406.04373"]]
    write_jsonl(out_dir / "stress_eval_rows.jsonl", eval_rows)

    manifest = {
        "dataset_id": "q5_step2400_sft700_numeric_binding_poc_v1",
        "purpose": "Proof-of-concept overfit test: can assistant-only LoRA SFT from checkpoint-700 move sticky numeric-binding failures under llama.cpp Q5 runtime?",
        "row_count": len(rows),
        "components": {
            "exact_stress_case_repeated": 16,
            "exact_support_case_repeated": 6,
            "synthetic_numeric_binding_contrast": len(SYNTHETIC),
        },
        "stress_targets": TARGETS,
        "important_caveat": "This deliberately trains on the stress/eval cases and repeats them. It is diagnostic only, not a candidate patch or promotion dataset.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Numeric Binding POC v1\n\n"
        "Tiny diagnostic dataset that deliberately overfits exact severe numeric failures and synthetic contrast rows. "
        "Use only to answer whether LoRA SFT can move these failures at all through llama.cpp Q5 runtime.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
