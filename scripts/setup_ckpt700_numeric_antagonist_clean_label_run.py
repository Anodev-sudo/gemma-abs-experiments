#!/usr/bin/env python3
"""Build a clean-label setup from checkpoint-700-era sources only.

Outputs:
- 600 normal style grounding rows copied from checkpoint-700 SFT training data.
- 200 numeric-risk candidate rows for fresh Kimi/Qwen digit-free labeling.

No generated numeric target labels are written by this script.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DIGIT_RE = re.compile(r"\d")
FORMULA_RE = re.compile(r"\$|\\[A-Za-z]+|\bO\([^)]*\)|\^|<sup>|</sup>|[≤≥≈]")
BANNED_OPENINGS = ("this paper", "the paper", "the authors", "we ", "in this work", "in this paper")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def word_count(text: str) -> int:
    return len([p for p in str(text).strip().split() if p])


def clean_normal_row(row: dict[str, Any]) -> bool:
    summary = str(row.get("summary") or "").strip()
    title = str(row.get("title") or "").strip().lower()
    low = summary.lower()
    if not (10 <= word_count(summary) <= 15):
        return False
    if DIGIT_RE.search(summary) or FORMULA_RE.search(summary):
        return False
    if any(low.startswith(prefix) for prefix in BANNED_OPENINGS):
        return False
    if title and low.rstrip(".") == title.rstrip("."):
        return False
    return True


def round_robin_by_category(rows: list[dict[str, Any]], target: int, per_category_cap: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("primary_category") or "unknown")].append(row)
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    cats = sorted(buckets, key=lambda c: (-len(buckets[c]), c))
    per_cat_counts = Counter()
    progressed = True
    while len(selected) < target and progressed:
        progressed = False
        for cat in cats:
            if len(selected) >= target:
                break
            if per_cat_counts[cat] >= per_category_cap:
                continue
            bucket = buckets[cat]
            while bucket:
                row = bucket.pop(0)
                pid = str(row.get("paper_id") or "")
                if pid and pid not in used_ids:
                    selected.append(row)
                    used_ids.add(pid)
                    per_cat_counts[cat] += 1
                    progressed = True
                    break
    if len(selected) < target:
        for row in rows:
            if len(selected) >= target:
                break
            pid = str(row.get("paper_id") or "")
            if pid and pid not in used_ids:
                selected.append(row)
                used_ids.add(pid)
    return selected


def severity_score(row: dict[str, Any]) -> tuple[int, int, int, str]:
    tags = set(row.get("numeric_tags") or []) | set(row.get("semantic_tags") or [])
    score = 0
    for tag, value in {
        "unsupported_number": 30,
        "decimal_compaction": 25,
        "metric_role_confusion": 25,
        "number_dense_summary": 20,
        "polarity_or_negation": 20,
        "unsupported_terms": 10,
        "repetition_or_tautology": 5,
    }.items():
        if tag in tags:
            score += value
    unsupported = len(row.get("unsupported_numbers") or [])
    dense = len(row.get("output_numbers") or [])
    return (-score, -unsupported, -dense, str(row.get("paper_id") or ""))


def numeric_candidate(row: dict[str, Any], idx: int, source_name: str) -> dict[str, Any]:
    numeric_tags = row.get("numeric_tags") if isinstance(row.get("numeric_tags"), list) else []
    semantic_tags = row.get("semantic_tags") if isinstance(row.get("semantic_tags"), list) else []
    source_nums = row.get("source_numbers_sample") if isinstance(row.get("source_numbers_sample"), list) else []
    bad_summary = str(row.get("summary") or "")
    short_reason = "; ".join([
        f"checkpoint-700 numeric risk tags: {', '.join(numeric_tags) or 'none'}",
        f"semantic tags: {', '.join(semantic_tags) or 'none'}",
        f"bad summary numbers: {', '.join(map(str, row.get('output_numbers') or [])) or 'none'}",
        f"source number sample: {', '.join(map(str, source_nums[:8])) or 'none'}",
    ])
    return {
        "candidate_id": f"ckpt700_numsuppress:{idx:04d}",
        "paper_id": str(row.get("paper_id") or ""),
        "title": str(row.get("title") or ""),
        "abstract": str(row.get("abstract") or ""),
        "primary_category": str(row.get("primary_category") or ""),
        "categories": [],
        "created_date": "",
        "bad_summary": bad_summary,
        "pre_dpo_summary": str(row.get("pre_dpo_summary") or ""),
        "triage_decision": "generate_clean_no_number_label",
        "triage_severity": str(row.get("severity") or "numeric_risk"),
        "failure_tags": ["numeric_risk", "no_exact_numbers", *numeric_tags, *semantic_tags],
        "short_reason": short_reason,
        "source_component": source_name,
        "source_split": "q5_numeric_antagonist_clean_labels_v1",
        "label_policy": "fresh_kimi_writer_qwen_verifier_digit_free",
        "provenance_note": "Candidate source row only. Kimi/Qwen must generate and verify the target label.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint700-train", default="hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/train.jsonl")
    ap.add_argument("--numeric-priority", default="hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/failure_mining/checkpoint700_10000_v1/heuristic_screen/numeric_priority_candidates.jsonl")
    ap.add_argument("--numeric-candidates", default="hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/failure_mining/checkpoint700_10000_v1/heuristic_screen/numeric_candidates.jsonl")
    ap.add_argument("--output-root", default="hf_bucket/experiments/sft/q5_step2400_sft700_numeric_antagonist_clean_v1")
    ap.add_argument("--normal-count", type=int, default=600)
    ap.add_argument("--numeric-count", type=int, default=200)
    ap.add_argument("--normal-category-cap", type=int, default=18)
    args = ap.parse_args()

    output_root = Path(args.output_root)
    train_rows = load_jsonl(Path(args.checkpoint700_train))
    normal_pool = [r for r in train_rows if clean_normal_row(r)]
    normal_selected = round_robin_by_category(normal_pool, args.normal_count, args.normal_category_cap)
    if len(normal_selected) != args.normal_count:
        raise SystemExit(f"Expected {args.normal_count} normal rows, got {len(normal_selected)}")

    normalized_normal = []
    for i, r in enumerate(normal_selected, start=1):
        out = {k: r.get(k) for k in ["paper_id", "title", "abstract", "summary", "primary_category", "categories", "created_date"]}
        out.update({
            "source_split": "q5_numeric_antagonist_clean_v1_normal_anchor",
            "source_component": "checkpoint700_train_digit_free_normal_anchor",
            "teacher_model": r.get("teacher_model", "checkpoint700_parent_training_mix"),
            "generator_prompt_version": r.get("generator_prompt_version", "checkpoint700_parent_training_mix"),
            "review_prompt_version": r.get("review_prompt_version", "checkpoint700_parent_training_mix"),
            "mix_index": i,
            "loss_weight_hints": {"assistant": 1.0},
        })
        normalized_normal.append(out)

    numeric_rows_by_pid: dict[str, dict[str, Any]] = {}
    ordered: list[tuple[str, dict[str, Any]]] = []
    for source_name, path in [("checkpoint700_numeric_priority", Path(args.numeric_priority)), ("checkpoint700_numeric_candidates", Path(args.numeric_candidates))]:
        for row in load_jsonl(path):
            pid = str(row.get("paper_id") or "")
            if not pid or pid in numeric_rows_by_pid:
                continue
            row = dict(row)
            row["_source_name"] = source_name
            numeric_rows_by_pid[pid] = row
            ordered.append((source_name, row))

    sorted_rows = sorted((r for _, r in ordered), key=severity_score)
    selected_numeric = sorted_rows[: args.numeric_count]
    if len(selected_numeric) != args.numeric_count:
        raise SystemExit(f"Expected {args.numeric_count} numeric candidates, got {len(selected_numeric)}")
    numeric_candidates = [numeric_candidate(r, i, str(r.get("_source_name") or "checkpoint700_numeric_candidates")) for i, r in enumerate(selected_numeric, start=1)]

    write_jsonl(output_root / "normal_anchors" / "normal_anchor_rows.jsonl", normalized_normal)
    write_jsonl(output_root / "kimi_qwen_label_run" / "candidate_pool" / "candidates.jsonl", numeric_candidates)

    manifest = {
        "created_by": "scripts/setup_ckpt700_numeric_antagonist_clean_label_run.py",
        "normal_source": args.checkpoint700_train,
        "normal_rows": len(normalized_normal),
        "normal_selection": {
            "summary_policy": "digit_free_10_to_15_words_no_formula_no_banned_opening",
            "category_cap": args.normal_category_cap,
            "candidate_pool_rows": len(normal_pool),
            "primary_category_counts": dict(Counter(r.get("primary_category") for r in normalized_normal)),
        },
        "numeric_sources": [args.numeric_priority, args.numeric_candidates],
        "numeric_candidate_rows": len(numeric_candidates),
        "numeric_source_counts": dict(Counter(c["source_component"] for c in numeric_candidates)),
        "numeric_tag_counts": dict(Counter(tag for c in numeric_candidates for tag in c.get("failure_tags", []))),
        "label_policy": "numeric candidates have no target labels until Kimi writer and Qwen verifier run",
        "intended_mix_after_generation": "600 normal anchors + accepted/replenished 200 digit-free numeric antagonist rows",
    }
    for rel in ["run_manifest.json", "kimi_qwen_label_run/candidate_pool/run_manifest.json"]:
        (output_root / rel).parent.mkdir(parents=True, exist_ok=True)
        (output_root / rel).write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    readme = f"""# q5 checkpoint-700 numeric antagonist clean-label setup\n\nStatus: setup only. No Kimi/Qwen calls have been run by this script.\n\nRecipe:\n\n- 600 normal style-grounding rows copied from checkpoint-700 parent training data.\n- 200 checkpoint-700 numeric-risk candidates for fresh Kimi/Qwen digit-free labels.\n\nNormal anchors:\n\n`normal_anchors/normal_anchor_rows.jsonl`\n\nNumeric candidate pool:\n\n`kimi_qwen_label_run/candidate_pool/candidates.jsonl`\n\nThe numeric candidate pool contains no target labels. It includes checkpoint-700 bad summaries and heuristic risk context only.\n\nWriter command:\n\n```bash\nhf_bucket/experiments/dpo/runtime/py312/bin/python scripts/run_sft700_targeted_repair_pipeline.py \\\n  --stage write \\\n  --output-root {output_root}/kimi_qwen_label_run \\\n  --writer-prompt prompts/synthetic_summary/kimi_k2_6_q5_numeric_suppression_clean_label_v1.json \\\n  --writer-model moonshotai/kimi-k2.6 \\\n  --writer-workers 4 \\\n  --temperature 0 \\\n  --writer-max-output-tokens 220 \\\n  --writer-reasoning-effort none\n```\n\nVerifier command:\n\n```bash\nhf_bucket/experiments/dpo/runtime/py312/bin/python scripts/run_sft700_targeted_repair_pipeline.py \\\n  --stage verify \\\n  --output-root {output_root}/kimi_qwen_label_run \\\n  --reviewer-prompt prompts/synthetic_summary/qwen3_235b_a22b_2507_q5_numeric_suppression_clean_label_verifier_v1.json \\\n  --reviewer-model qwen/qwen3-235b-a22b-2507 \\\n  --reviewer-workers 4 \\\n  --temperature 0 \\\n  --reviewer-max-output-tokens 256 \\\n  --reviewer-reasoning-effort none\n```\n\nMaterialize command:\n\n```bash\nhf_bucket/experiments/dpo/runtime/py312/bin/python scripts/run_sft700_targeted_repair_pipeline.py \\\n  --stage materialize \\\n  --output-root {output_root}/kimi_qwen_label_run\n```\n"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
