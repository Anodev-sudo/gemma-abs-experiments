#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

BASE_DATASET = Path("hf_bucket/experiments/sft/q5_step2400_sft700_clean_existing_targeted_weight_v1_lr4e7/dataset_600normal_200numeric_targeted_weight_clean_v1/train.jsonl")
BOOSTER_ACCEPTED = Path("hf_bucket/experiments/sft/q5_step2400_sft700_targeted_booster_clean_v1/kimi_qwen_label_run/materialized/accepted_repairs.jsonl")
PRIOR_STRESS_ACCEPTED = [
    Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_antagonist_clean_v1_pool350/stress_drill_kimi_v2/materialized/accepted_repairs.jsonl"),
    Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_antagonist_clean_v1_pool350/stress_drill_kimi_fix_2303_digit_v1/materialized/accepted_repairs.jsonl"),
]
OUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_targeted_booster_clean_v1_lr4e7/dataset_600normal_160numeric_40booster_clean_v1")

SYSTEM_PROMPT = (
    "You write one-sentence paper summaries for a mobile research feed. "
    "Keep them concrete, readable, and useful. Output only the summary sentence."
)
USER_TEMPLATE = (
    "The app already shows the title, author list, and date. "
    "Write one sentence that tells a reader why they might open this paper. "
    "Aim for 10 to 15 words. Avoid repeating the title. "
    'Avoid filler like "This paper" or "We propose". '
    "Do not mention authors or dates. "
    "Keep the wording natural and specific.\n\n"
    "Title:\n{title}\n\n"
    "Abstract:\n{abstract}"
)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")
DIGIT_RE = re.compile(r"\d")
FORMULA_RE = re.compile(r"[$<>≤≥≈]|\\[A-Za-z]+|\bO\(|\^")
DIRECTION = {
    "accelerates", "avoids", "balances", "bounds", "clarifies", "classifies", "compresses",
    "constrains", "cuts", "detects", "enables", "estimates", "exposes", "improve", "improves",
    "lower", "lowers", "preserve", "preserves", "ranks", "reconstructs", "reduces", "reveals",
    "schedules", "supports", "tighten", "tightens", "upconverts",
}
STATIC_CONCEPTS = {
    "accuracy", "algorithm", "algorithms", "architecture", "axions", "bounds", "carbon", "classification",
    "classifier", "coding", "comets", "compression", "cost", "costs", "diagnosis", "detection", "drilling",
    "efficiency", "emissions", "energy", "exploration", "fiber", "framework", "gers", "heating", "leader",
    "latency", "mizar", "murmur", "network", "networks", "pseudocode", "resource", "resources", "routing",
    "scheduling", "sensors", "serving", "spiking", "superflares", "tangles", "throughput", "training",
    "transformer", "waveguides",
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it",
    "its", "of", "on", "or", "over", "than", "that", "the", "their", "this", "through", "to", "using",
    "while", "with", "without", "within", "across", "between", "under", "via", "new", "novel", "method",
    "model", "models", "system", "systems", "approach", "results", "paper", "study", "data",
}
FAMILY_FOCUS = {
    "stress": {"tangles", "np-hard", "algorithm", "swap", "lists", "schedules", "throughput", "cost", "microfiber", "couples", "coupling", "verilog", "tests", "branches"},
    "role_binding": {"throughput", "cost", "costs", "dataset", "classifier", "scans", "sensors", "pollution", "heating", "resources"},
    "single_metric": {"efficiency", "emissions", "latency", "accuracy", "routing", "serving", "drilling", "feedback"},
    "range_template": {"bounds", "tradeoffs", "axions", "oscillator", "leader", "election", "exponents"},
    "formula_residue": {"pseudocode", "extraction", "superflares", "plasma", "code"},
    "copy_round_drift": {"classification", "murmur", "phonocardiograms", "emotion", "spiking", "coronary", "diagnosis"},
    "dimension_object_swap": {"interaction", "mimo", "radio", "reporting", "optical"},
    "normal_semantic_drift": {"bundle", "planes", "gonality", "graphs", "catalog", "binaries"},
}


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


def wc(text: str) -> int:
    return len(str(text).split())


def hard_clean(summary: str) -> bool:
    return 10 <= wc(summary) <= 15 and summary.endswith(".") and summary.count(".") == 1 and not DIGIT_RE.search(summary) and not FORMULA_RE.search(summary)


def messages(title: str, abstract: str, summary: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(title=title, abstract=abstract)},
        {"role": "assistant", "content": summary},
    ]


def source_terms(row: dict[str, Any]) -> set[str]:
    source = f"{row.get('title','')}\n{row.get('abstract','')}".lower()
    return {m.group(0).lower() for m in WORD_RE.finditer(source)}


def targeted_hints(row: dict[str, Any], *, booster: bool) -> dict[str, Any]:
    summary = str(row.get("summary") or "")
    if not booster and row.get("weighted_numeric_patch_row_type") == "normal_style_anchor":
        return {"enabled": False, "base_weight": 1.0, "spans": [], "policy": "normal style anchor"}
    src_terms = source_terms(row)
    family = str(row.get("failure_family") or "")
    focus = FAMILY_FOCUS.get(family, set())
    direction_weight = 10.0 if booster else 5.0
    concept_weight = 7.0 if booster else 4.0
    focus_weight = 9.0 if booster else 4.0
    spans = []
    for m in WORD_RE.finditer(summary):
        tok = m.group(0)
        low = tok.lower()
        weight = None
        kind = None
        if low in DIRECTION:
            weight = direction_weight
            kind = "booster_direction_or_safe_abstraction" if booster else "direction_or_safe_abstraction"
        if low in focus and (weight is None or weight < focus_weight):
            weight = focus_weight
            kind = "booster_family_focus_concept"
        is_domain = low in STATIC_CONCEPTS or (low in src_terms and low not in STOPWORDS and len(low) >= 5) or ("-" in low and low in src_terms)
        if is_domain and (weight is None or weight < concept_weight):
            weight = concept_weight
            kind = "booster_source_grounded_domain_concept" if booster else "domain_concept"
        if weight is not None:
            spans.append({"start": m.start(), "end": m.end(), "text": tok, "weight": weight, "kind": kind})
    return {
        "enabled": True,
        "base_weight": 1.0,
        "spans": spans,
        "policy": "strong targeted booster weights" if booster else "standard targeted numeric weights",
    }


def final_booster_row(src: dict[str, Any], summary: str, mix_index: int, repeat_index: int) -> dict[str, Any]:
    row = {
        "paper_id": str(src.get("paper_id") or ""),
        "title": str(src.get("title") or ""),
        "abstract": str(src.get("abstract") or ""),
        "summary": summary,
        "primary_category": str(src.get("primary_category") or ""),
        "categories": src.get("categories") if isinstance(src.get("categories"), list) else [],
        "created_date": str(src.get("created_date") or ""),
        "teacher_model": str(src.get("writer_model") or "moonshotai/kimi-k2.6"),
        "generator_prompt_version": str(src.get("writer_prompt_version") or src.get("generator_prompt_version") or ""),
        "review_prompt_version": str(src.get("reviewer_prompt_version") or src.get("review_prompt_version") or ""),
        "source_split": "q5_targeted_booster_clean_v1_600normal_160numeric_40booster",
        "patch_component": "kimi_qwen_verified_targeted_booster",
        "weighted_numeric_patch_row_type": "numeric_antagonist_clean_targeted_booster",
        "failure_family": src.get("failure_family") or "stress_prior_clean",
        "failure_tags": src.get("failure_tags", []),
        "bad_summary": src.get("bad_summary", ""),
        "pre_dpo_summary": src.get("pre_dpo_summary", ""),
        "repair_candidate_id": src.get("candidate_id", src.get("repair_candidate_id", "")),
        "booster_repeat_index": repeat_index,
        "mix_index": mix_index,
    }
    row["loss_weight_hints"] = targeted_hints(row, booster=True)
    row["messages"] = messages(row["title"], row["abstract"], summary)
    return row


def main() -> None:
    base_rows = load_jsonl(BASE_DATASET)
    normal = [r for r in base_rows if r.get("weighted_numeric_patch_row_type") == "normal_style_anchor"][:600]
    numeric_pool = [r for r in base_rows if r.get("weighted_numeric_patch_row_type") != "normal_style_anchor"]

    accepted = []
    for row in load_jsonl(BOOSTER_ACCEPTED):
        summary = str(row.get("repair_summary") or row.get("summary") or "").strip()
        if hard_clean(summary):
            accepted.append((row, summary))
    # Add prior clean Kimi/Qwen stress labels for 2303 and 2406, where this targeted run did not produce hard-clean accepted rows.
    prior_needed = {"2303.09722", "2406.04373"}
    for path in PRIOR_STRESS_ACCEPTED:
        for row in load_jsonl(path):
            if str(row.get("paper_id")) not in prior_needed:
                continue
            summary = str(row.get("repair_summary") or row.get("summary") or "").strip()
            if hard_clean(summary):
                row = dict(row)
                row["failure_family"] = "stress"
                row["failure_tags"] = ["targeted_booster", "stress", "prior_clean_kimi_qwen_verified"]
                accepted.append((row, summary))

    # Deduplicate exact paper+summary, then prioritize sticky stress/role rows for repetition.
    seen = set()
    unique = []
    for row, summary in accepted:
        key = (row.get("paper_id"), summary)
        if key in seen:
            continue
        seen.add(key)
        unique.append((row, summary))
    priority = sorted(unique, key=lambda rs: (0 if rs[0].get("failure_family") == "stress" else 1, rs[0].get("failure_family") or "", rs[0].get("paper_id") or ""))
    booster_pairs = list(priority)
    i = 0
    while len(booster_pairs) < 40 and priority:
        booster_pairs.append(priority[i % len(priority)])
        i += 1
    booster_pairs = booster_pairs[:40]

    booster_pids = {str(row.get("paper_id")) for row, _ in booster_pairs}
    numeric = [r for r in numeric_pool if str(r.get("paper_id")) not in booster_pids][:160]

    final = []
    for row in normal:
        r = dict(row)
        r["source_split"] = "q5_targeted_booster_clean_v1_600normal_160numeric_40booster"
        r["loss_weight_hints"] = {"enabled": False, "base_weight": 1.0, "spans": [], "policy": "normal style anchor"}
        r["mix_index"] = len(final) + 1
        final.append(r)
    for row in numeric:
        r = dict(row)
        r["source_split"] = "q5_targeted_booster_clean_v1_600normal_160numeric_40booster"
        r["weighted_numeric_patch_row_type"] = "numeric_antagonist_clean_digit_free_targeted_weight"
        r["loss_weight_hints"] = targeted_hints(r, booster=False)
        r["mix_index"] = len(final) + 1
        final.append(r)
    repeat_counts = Counter()
    for row, summary in booster_pairs:
        repeat_counts[(row.get("paper_id"), summary)] += 1
        final.append(final_booster_row(row, summary, len(final) + 1, repeat_counts[(row.get("paper_id"), summary)]))

    if len(final) != 800:
        raise SystemExit(f"Expected 800 rows, got {len(final)}")
    write_jsonl(OUT / "train.jsonl", final)
    write_jsonl(OUT / "normal_selected_600.jsonl", final[:600])
    write_jsonl(OUT / "numeric_selected_160.jsonl", final[600:760])
    write_jsonl(OUT / "booster_selected_40.jsonl", final[760:])
    type_counts = Counter(r.get("weighted_numeric_patch_row_type") for r in final)
    span_kinds = Counter(sp.get("kind") for r in final for sp in (r.get("loss_weight_hints") or {}).get("spans", []))
    weight_values = Counter(str(sp.get("weight")) for r in final for sp in (r.get("loss_weight_hints") or {}).get("spans", []))
    booster_family = Counter(r.get("failure_family") for r in final[760:])
    manifest = {
        "created_by": "scripts/build_q5_targeted_booster_clean_v1_dataset.py",
        "rows": len(final),
        "dataset_path": str(OUT / "train.jsonl"),
        "row_type_counts": dict(type_counts),
        "booster_unique_paper_count": len({r["paper_id"] for r in final[760:]}),
        "booster_family_counts": dict(booster_family),
        "weighted_span_kind_counts": dict(span_kinds),
        "weighted_span_weight_counts": dict(weight_values),
        "digit_summary_rows": sum(1 for r in final if DIGIT_RE.search(r.get("summary", ""))),
        "formula_symbol_summary_rows": sum(1 for r in final if FORMULA_RE.search(r.get("summary", ""))),
        "label_provenance": "normal/existing numeric labels reused from clean Kimi/Qwen dataset; booster labels are Kimi-generated and Qwen-verified, with prior clean Kimi/Qwen stress labels for missing 2303/2406 coverage",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "DATASET_AUDIT.md").write_text("# Dataset audit\n\n" + json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
