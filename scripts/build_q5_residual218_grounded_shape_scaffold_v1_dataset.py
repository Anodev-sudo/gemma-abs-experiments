#!/usr/bin/env python3
"""Build a conservative checkpoint-218 grounded-shape scaffold dataset.

This does not author new labels. It reuses trusted Kimi/Qwen rows and
checkpoint-218 self-anchors, while quarantining brittle grounded-abstraction
targets for a future Kimi/Qwen rewrite pass.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any

from generate_llama_stream import SYSTEM_PROMPT, USER_TEMPLATE

ROOT = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_residual218_grounded_shape_scaffold_v1"
)
RESIDUAL_DATASET = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7/"
    "dataset_193repair_600selfanchor_77hard_v1"
)
GROUNDED_EVAL = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_hard77_burnin150_residual218_grounded_abstraction_"
    "overfit14_diag_v1_lr1e5/dataset_14repair_repeats100_fulltarget8_v1/eval_repair14.jsonl"
)
FAILURE_SHAPE = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_hard77_burnin150_residual218_grounded_abstraction_"
    "overfit14_diag_v1_lr1e5/eval/repair14_failure_shape_diagnostic.jsonl"
)
SPLASH_DIAGNOSTIC = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_hard77_burnin150_residual218_grounded_abstraction_"
    "overfit14_diag_v1_lr1e5/eval/q8_packet_100_overfit14_splash_diagnostic.json"
)

STRESS_IDS = {"1901.06548", "2111.10635", "2303.09722", "2406.04373"}
TRAINABLE_GROUNDED_BUCKETS = {
    "clean_success",
    "near_success_role_verb_drift",
    "near_success_title_object_substitution",
    "partial_grounding_success",
}

DIGIT_RE = re.compile(r"\d")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")
SENTENCE_END_RE = re.compile(r"""[.!?]+(?:['")\]]+)?(?=\s|$)""")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def norm(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def sentence_count(text: str) -> int:
    text = norm(text)
    if not text:
        return 0
    endings = SENTENCE_END_RE.findall(text)
    return len(endings) if endings else 1


def messages(title: str, abstract: str, summary: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(title=title, abstract=abstract)},
        {"role": "assistant", "content": summary},
    ]


def loss_weight(weight: float, policy: str) -> dict[str, Any]:
    return {
        "enabled": weight != 1.0,
        "base_weight": weight,
        "spans": [],
        "policy": policy,
    }


def normalize_row(row: dict[str, Any], *, row_type: str, component: str, weight: float) -> dict[str, Any]:
    summary = norm(row.get("summary", ""))
    title = norm(row.get("title", ""))
    abstract = norm(row.get("abstract", ""))
    if not summary or not title or not abstract:
        raise SystemExit(f"Missing title/abstract/summary for {row.get('paper_id')}")
    out = dict(row)
    out["summary"] = summary
    out["weighted_numeric_patch_row_type"] = row_type
    out["dataset_component"] = component
    out["messages"] = messages(title, abstract, summary)
    out["loss_weight_hints"] = loss_weight(weight, f"grounded_shape_scaffold_v1 full assistant target weight {weight:g}")
    return out


def ensure_clean_label(row: dict[str, Any]) -> None:
    summary = norm(row.get("summary", ""))
    pid = row.get("paper_id", "")
    reasons: list[str] = []
    if DIGIT_RE.search(summary):
        reasons.append("digit")
    if sentence_count(summary) != 1:
        reasons.append("not_single_sentence")
    wc = word_count(summary)
    if wc < 8:
        reasons.append("very_short")
    if wc > 18:
        reasons.append("very_long")
    if reasons:
        raise SystemExit(f"Label failed scaffold screen: {pid} {reasons} {summary!r}")


def load_residual_component(path: Path, filename: str, *, row_type: str, component: str, weight: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(path / filename):
        ensure_clean_label(row)
        rows.append(normalize_row(row, row_type=row_type, component=component, weight=weight))
    return rows


def residual_hard_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(path / "hard77_rows_77.jsonl"):
        ensure_clean_label(row)
        pid = str(row["paper_id"])
        weight = 1.5 if pid in STRESS_IDS else 1.0
        row_type = "hard77_preserve_stress" if pid in STRESS_IDS else "hard77_preserve"
        rows.append(normalize_row(row, row_type=row_type, component="hard77_preserve", weight=weight))
    return rows


def grounded_rows(
    eval_path: Path,
    failure_path: Path,
    *,
    repeat: int,
    weight: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eval_by_pid = {str(row["paper_id"]): row for row in load_jsonl(eval_path)}
    diagnostics = load_jsonl(failure_path)
    trainable: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for diag in diagnostics:
        pid = str(diag["paper_id"])
        eval_row = eval_by_pid[pid]
        base = {
            "paper_id": pid,
            "title": eval_row["title"],
            "abstract": eval_row["abstract"],
            "summary": eval_row["target_summary"],
            "primary_category": eval_row.get("primary_category", ""),
            "categories": [],
            "created_date": "",
            "teacher_model": "moonshotai/kimi-k2.6",
            "generator_prompt_version": "kimi_k2_6_q5_grounded_abstraction_repair_writer_v1",
            "review_prompt_version": "qwen3_235b_a22b_2507_q5_grounded_abstraction_repair_verifier_v1",
            "source_split": "q5_grounded_shape_scaffold_v1",
            "diagnostic_bucket": diag.get("bucket", ""),
            "diagnostic_manual_verdict": diag.get("manual_verdict", ""),
            "diagnostic_next_label_guidance": diag.get("next_label_guidance", ""),
            "checkpoint218_summary": eval_row.get("checkpoint218_summary", ""),
            "q8_summary": eval_row.get("q8_summary", ""),
        }
        ensure_clean_label(base)
        if diag.get("bucket") in TRAINABLE_GROUNDED_BUCKETS:
            for idx in range(repeat):
                row = dict(base)
                row["source_paper_id"] = pid
                row["grounded_repeat_index"] = idx
                row["paper_id"] = f"{pid}::grounded_shape_keep_{idx:02d}"
                trainable.append(
                    normalize_row(
                        row,
                        row_type="grounded_shape_keep_repair",
                        component="grounded_shape_keep",
                        weight=weight,
                    )
                )
        else:
            quarantined.append(
                {
                    **base,
                    "paper_id": pid,
                    "bucket": diag.get("bucket", ""),
                    "manual_verdict": diag.get("manual_verdict", ""),
                    "likely_issue": diag.get("likely_issue", ""),
                    "next_label_guidance": diag.get("next_label_guidance", ""),
                    "target_terms_not_in_source": diag.get("target_terms_not_in_source", []),
                    "target_source_support": diag.get("target_source_support", 0.0),
                    "best_trajectory_label": diag.get("best_trajectory_label", ""),
                }
            )
    return trainable, quarantined


def low_similarity_examples(path: Path, limit_per_label: int) -> list[dict[str, Any]]:
    if not path.exists() or limit_per_label <= 0:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    examples: list[dict[str, Any]] = []
    by_label = data.get("low_similarity_untrained_examples", {})
    for label in ("checkpoint_70", "checkpoint_120"):
        for row in by_label.get(label, [])[:limit_per_label]:
            examples.append(
                {
                    "candidate_kind": "overfit_splash_low_similarity_untrained",
                    "diagnostic_label": label,
                    "paper_id": row.get("paper_id", ""),
                    "split": row.get("split", ""),
                    "start_output": row.get("start_output", ""),
                    "overfit_output": row.get("output", ""),
                    "jaccard_vs_start": row.get("jaccard_vs_start", 0.0),
                    "rewrite_note": "Use only as blast-radius evidence; do not train without fresh Kimi/Qwen review.",
                }
            )
    return examples


def write_rewrite_readme(path: Path, quarantined: list[dict[str, Any]], splash: list[dict[str, Any]]) -> None:
    lines = [
        "# Grounded Shape Rewrite Candidate Pack",
        "",
        "These rows are intentionally not in the trainable scaffold dataset.",
        "",
        "Use this pack for a future Kimi writer and Qwen verifier pass. Do not hand-author labels.",
        "",
        "## Quarantined Grounded Rows",
        "",
        "| paper | bucket | issue | guidance |",
        "|---|---|---|---|",
    ]
    for row in quarantined:
        lines.append(
            f"| `{row['paper_id']}` | {row['bucket']} | {row['likely_issue']} | {row['next_label_guidance']} |"
        )
    lines.extend(
        [
            "",
            "## Splash Examples",
            "",
            "These are blast-radius examples from the overfit diagnostic. They are useful for prompt design, not direct training.",
            "",
            f"- Included examples: {len(splash)}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build q5 residual218 grounded-shape scaffold dataset.")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--residual-dataset", type=Path, default=RESIDUAL_DATASET)
    parser.add_argument("--grounded-eval", type=Path, default=GROUNDED_EVAL)
    parser.add_argument("--failure-shape", type=Path, default=FAILURE_SHAPE)
    parser.add_argument("--splash-diagnostic", type=Path, default=SPLASH_DIAGNOSTIC)
    parser.add_argument("--grounded-repeat", type=int, default=3)
    parser.add_argument("--splash-examples-per-label", type=int, default=10)
    parser.add_argument("--self-anchor-weight", type=float, default=0.15)
    parser.add_argument("--residual-repair-weight", type=float, default=0.5)
    parser.add_argument("--grounded-keep-weight", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grounded_repeat < 1:
        raise SystemExit("--grounded-repeat must be >= 1")

    residual_repairs = load_residual_component(
        args.residual_dataset,
        "repair_rows_193.jsonl",
        row_type="residual_repair_preserve_low_weight",
        component="residual_repair_preserve",
        weight=args.residual_repair_weight,
    )
    self_anchors = load_residual_component(
        args.residual_dataset,
        "self_anchor_rows_600.jsonl",
        row_type="checkpoint218_self_anchor_very_low_weight",
        component="checkpoint218_self_anchor",
        weight=args.self_anchor_weight,
    )
    hard77 = residual_hard_rows(args.residual_dataset)
    grounded_keep, quarantined = grounded_rows(
        args.grounded_eval,
        args.failure_shape,
        repeat=args.grounded_repeat,
        weight=args.grounded_keep_weight,
    )
    splash_examples = low_similarity_examples(args.splash_diagnostic, args.splash_examples_per_label)

    final = [*residual_repairs, *hard77, *self_anchors, *grounded_keep]
    duplicate_ids = [pid for pid, count in collections.Counter(row["paper_id"] for row in final).items() if count > 1]
    if duplicate_ids:
        raise SystemExit(f"Duplicate paper IDs in final dataset: {duplicate_ids[:20]}")
    digit_labels = [row["paper_id"] for row in final if DIGIT_RE.search(row["summary"])]
    if digit_labels:
        raise SystemExit(f"Digit labels in final dataset: {digit_labels[:20]}")

    dataset_dir = (
        args.output_root
        / f"dataset_193residual_77hard_600selfanchor_{len(grounded_keep)}groundedkeep_v1"
    )
    rewrite_dir = args.output_root / "rewrite_candidate_pack_v1"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rewrite_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(dataset_dir / "train.jsonl", final)
    write_jsonl(dataset_dir / "residual_repair_preserve_rows_193.jsonl", residual_repairs)
    write_jsonl(dataset_dir / "hard77_preserve_rows_77.jsonl", hard77)
    write_jsonl(dataset_dir / "self_anchor_rows_600.jsonl", self_anchors)
    write_jsonl(dataset_dir / f"grounded_keep_rows_{len(grounded_keep)}.jsonl", grounded_keep)
    write_jsonl(rewrite_dir / "quarantined_grounded_rewrite_candidates.jsonl", quarantined)
    write_jsonl(rewrite_dir / "overfit_splash_prompt_design_examples.jsonl", splash_examples)
    write_rewrite_readme(rewrite_dir / "README.md", quarantined, splash_examples)

    manifest = {
        "created_by": "scripts/build_q5_residual218_grounded_shape_scaffold_v1_dataset.py",
        "dataset_path": str(dataset_dir / "train.jsonl"),
        "start_adapter_intended": (
            "hf_bucket/experiments/sft/"
            "q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7/checkpoint-218"
        ),
        "rows": len(final),
        "digit_label_rows": 0,
        "duplicate_paper_ids": 0,
        "residual_repair_preserve_rows": len(residual_repairs),
        "hard77_preserve_rows": len(hard77),
        "checkpoint218_self_anchor_rows": len(self_anchors),
        "grounded_keep_rows": len(grounded_keep),
        "grounded_keep_unique_papers": len({row["source_paper_id"] for row in grounded_keep}),
        "grounded_keep_repeat": args.grounded_repeat,
        "quarantined_grounded_rewrite_candidates": len(quarantined),
        "overfit_splash_prompt_design_examples": len(splash_examples),
        "row_type_counts": dict(collections.Counter(row["weighted_numeric_patch_row_type"] for row in final)),
        "component_counts": dict(collections.Counter(row["dataset_component"] for row in final)),
        "grounded_keep_bucket_counts": dict(collections.Counter(row.get("diagnostic_bucket", "") for row in grounded_keep)),
        "quarantined_bucket_counts": dict(collections.Counter(row.get("bucket", "") for row in quarantined)),
        "weight_policy": {
            "residual_repairs": args.residual_repair_weight,
            "self_anchors": args.self_anchor_weight,
            "hard77": 1.0,
            "hard77_stress": 1.5,
            "grounded_keep": args.grounded_keep_weight,
        },
        "policy_notes": [
            "No Codex-authored labels are used.",
            "Brittle grounded rows are quarantined for future Kimi/Qwen rewrite before training.",
            "Self-anchor weight is deliberately lower than the previous residual repair run.",
        ],
        "rewrite_candidate_pack": str(rewrite_dir),
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[grounded-shape-scaffold] wrote {len(final)} train rows to {dataset_dir / 'train.jsonl'}")
    print(f"[grounded-shape-scaffold] wrote rewrite pack to {rewrite_dir}")


if __name__ == "__main__":
    main()
