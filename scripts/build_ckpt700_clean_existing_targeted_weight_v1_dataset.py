#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SRC = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_antagonist_clean_v1_pool350/dataset_600normal_200numeric_clean_v1/train.jsonl")
OUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_clean_existing_targeted_weight_v1_lr4e7/dataset_600normal_200numeric_targeted_weight_clean_v1")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")
DIGIT_RE = re.compile(r"\d")
FORMULA_RE = re.compile(r"[$<>≤≥≈]|\\[A-Za-z]+|\bO\(|\^")

DIRECTION = {
    "accelerates", "avoids", "better", "bounds", "clarifies", "classifies", "compresses",
    "constrains", "cuts", "cutting", "detects", "enables", "estimates", "exposes",
    "faster", "higher", "improve", "improves", "lower", "lowers", "preserve", "preserves",
    "reveals", "reduces", "reduced", "reducing", "rules", "safer", "speeds", "supports",
    "tightens", "without", "outperforms", "common", "practical", "consistent",
}
# Old-mechanics seed list plus broad numeric-risk concepts. Row-specific concepts are added
# from summary words that also appear in the paper title/abstract.
STATIC_CONCEPTS = {
    "accuracy", "algorithms", "annuity", "binaries", "codes", "comets", "compression", "cost", "costs",
    "deepfake", "density", "deployment", "detection", "dnn", "eclipsing", "efficiency", "energy",
    "formalization", "fusion", "latency", "llm", "memory", "particles", "performance", "pipeline",
    "recall", "recognition", "runtime", "scheduling", "serving", "structured", "superflares", "tangles",
    "throughput", "training", "uncertainty",
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it",
    "its", "of", "on", "or", "over", "than", "that", "the", "their", "this", "through", "to", "using",
    "while", "with", "without", "within", "across", "between", "under", "via", "new", "novel", "method",
    "model", "models", "system", "systems", "approach", "results", "paper", "study", "data",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def wc(text: str) -> int:
    return len(str(text).split())


def source_terms(row: dict[str, Any]) -> set[str]:
    source = f"{row.get('title','')}\n{row.get('abstract','')}".lower()
    return {m.group(0).lower() for m in WORD_RE.finditer(source)}


def targeted_hints(row: dict[str, Any]) -> dict[str, Any]:
    summary = str(row.get("summary") or "")
    if row.get("weighted_numeric_patch_row_type") == "normal_style_anchor":
        return {"enabled": False, "base_weight": 1.0, "spans": [], "policy": "normal style anchor"}

    src_terms = source_terms(row)
    spans = []
    for m in WORD_RE.finditer(summary):
        tok = m.group(0)
        low = tok.lower()
        weight = None
        kind = None
        if low in DIRECTION:
            weight = 5.0
            kind = "direction_or_safe_abstraction"
        is_domain = (
            low in STATIC_CONCEPTS
            or (low in src_terms and low not in STOPWORDS and len(low) >= 5)
            or ("-" in low and low in src_terms)
        )
        if is_domain and (weight is None or weight < 4.0):
            weight = 4.0
            kind = "domain_concept"
        if weight is not None:
            spans.append({"start": m.start(), "end": m.end(), "text": tok, "weight": weight, "kind": kind})
    return {
        "enabled": True,
        "base_weight": 1.0,
        "spans": spans,
        "policy": "targeted numeric suppression: upweight direction/safe-abstraction tokens at 5 and source-grounded domain concepts at 4; no full-target weighting",
    }


def main() -> None:
    rows = load_jsonl(SRC)
    out_rows = []
    for i, row in enumerate(rows, start=1):
        r = dict(row)
        summary = str(r.get("summary") or "")
        if not (10 <= wc(summary) <= 15):
            raise SystemExit(f"bad word count {r.get('paper_id')} {wc(summary)} {summary}")
        if DIGIT_RE.search(summary) or FORMULA_RE.search(summary):
            raise SystemExit(f"digit/formula residue {r.get('paper_id')} {summary}")
        r["source_split"] = "q5_clean_existing_targeted_weight_v1_600normal_200numeric"
        r["mix_index"] = i
        r["loss_weight_hints"] = targeted_hints(r)
        if r.get("weighted_numeric_patch_row_type") != "normal_style_anchor":
            r["weighted_numeric_patch_row_type"] = "numeric_antagonist_clean_digit_free_targeted_weight"
            r["patch_component"] = "kimi_qwen_verified_checkpoint700_numeric_risk_targeted_weight_reuse"
        out_rows.append(r)

    write_jsonl(OUT / "train.jsonl", out_rows)
    write_jsonl(OUT / "normal_selected_600.jsonl", [r for r in out_rows if r.get("weighted_numeric_patch_row_type") == "normal_style_anchor"])
    write_jsonl(OUT / "numeric_selected_200.jsonl", [r for r in out_rows if r.get("weighted_numeric_patch_row_type") != "normal_style_anchor"])

    type_counts = Counter(r.get("weighted_numeric_patch_row_type") for r in out_rows)
    weighted_span_counts = Counter()
    weighted_tokens_by_kind = Counter()
    for r in out_rows:
        hints = r.get("loss_weight_hints") or {}
        spans = hints.get("spans") or []
        if spans:
            weighted_span_counts[r.get("weighted_numeric_patch_row_type")] += len(spans)
            weighted_tokens_by_kind.update(sp.get("kind", "unknown") for sp in spans)
    rows_without_spans = [r["paper_id"] for r in out_rows if r.get("weighted_numeric_patch_row_type") != "normal_style_anchor" and not (r.get("loss_weight_hints") or {}).get("spans")]
    manifest = {
        "created_by": "scripts/build_ckpt700_clean_existing_targeted_weight_v1_dataset.py",
        "source_dataset": str(SRC),
        "dataset_path": str(OUT / "train.jsonl"),
        "rows": len(out_rows),
        "row_type_counts": dict(type_counts),
        "normal_rows": type_counts.get("normal_style_anchor", 0),
        "numeric_rows": len(out_rows) - type_counts.get("normal_style_anchor", 0),
        "digit_summary_rows": sum(1 for r in out_rows if DIGIT_RE.search(r.get("summary", ""))),
        "formula_symbol_summary_rows": sum(1 for r in out_rows if FORMULA_RE.search(r.get("summary", ""))),
        "targeted_weight_policy": {
            "base_weight": 1.0,
            "direction_or_safe_abstraction_weight": 5.0,
            "domain_concept_weight": 4.0,
            "normal_rows_enabled": False,
            "numeric_rows_full_target_weighted": False,
        },
        "weighted_span_counts": dict(weighted_span_counts),
        "weighted_span_kind_counts": dict(weighted_tokens_by_kind),
        "numeric_rows_without_targeted_spans": len(rows_without_spans),
        "numeric_rows_without_targeted_spans_sample": rows_without_spans[:20],
        "label_provenance": "same Kimi-generated, Qwen-verified, hard-filtered checkpoint-700 numeric labels as clean broad dataset; only loss weighting changed",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    audit = [
        "# Dataset audit", "", f"Rows: {len(out_rows)}", f"Row types: {dict(type_counts)}",
        f"Weighted span kinds: {dict(weighted_tokens_by_kind)}",
        f"Numeric rows without targeted spans: {len(rows_without_spans)}",
        "", "Labels are reused clean Kimi/Qwen labels; this experiment changes weighting only.",
    ]
    (OUT / "DATASET_AUDIT.md").write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
