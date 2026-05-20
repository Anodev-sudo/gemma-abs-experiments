#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

NORMAL_SRC = Path("hf_bucket/experiments/sft/q5_step2400_sft700_clean_existing_targeted_weight_v1_lr4e7/dataset_600normal_200numeric_targeted_weight_clean_v1/normal_selected_600.jsonl")
NUMERIC_SRC = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_ideal200_regen_v1/clean_200_after_filters_v1/training_rows_200.jsonl")
OUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_ideal200_surgical_weight_v1_lr1e6/dataset_600normal_200numeric_surgical_weight_v1")

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
FORMULA_RE = re.compile(r"[$<>≤≥≈]|\\[A-Za-z]+|\bO\(|\^|_")

POLARITY_PHRASES = [
    "rules out", "rather than", "tradeoff", "tradeoffs", "without", "while", "despite",
    "candidate", "suggests", "constrains", "constraint", "constraints", "limitation", "limitations",
    "remains", "mixed", "conditional", "negative", "null", "caveat", "caveats", "uncertain",
    "tentative", "non-detection", "non-detections", "no", "not",
]
DIRECTION_WORDS = {
    "adapts", "accelerates", "automates", "balances", "classifies", "compares", "constrains",
    "detects", "enables", "estimates", "exposes", "guides", "identifies", "improves", "links",
    "maps", "preserves", "probes", "probe", "reduces", "reducing", "reveals", "supports", "tracks", "optimizes",
    "forecasts", "predicts", "reconstructs", "calibrates", "allocates", "schedules", "summarizes",
    "measures", "separates", "distinguishes", "narrows", "replaces", "relates", "frames",
    "lifts", "forms", "generates", "extracts", "infers", "establishes", "evaluates", "validates",
    "detecting", "decomposes", "decompose", "increases", "increasing", "decreases", "decreasing",
}
# Avoid benchmark-headline verbs/adjectives that can become unsafe.
DO_NOT_WEIGHT = {
    "outperforms", "best", "highest", "largest", "strong", "significant", "major", "substantial",
    "dramatic", "superior", "highly", "state-of-the-art", "often", "most", "widespread", "new",
}
METRIC_ROLE_TERMS = {
    "accuracy", "accuracies", "latency", "throughput", "cost", "costs", "parameter", "parameters",
    "dataset", "datasets", "error", "errors", "speedup", "efficiency", "emissions", "memory",
    "bandwidth", "inference", "training", "runtime", "compression", "redshift", "mass", "masses",
    "distance", "distances", "rate", "rates", "count", "counts", "time", "times", "rounds",
    "message", "messages", "termination", "quality", "fidelity", "energy", "power", "carbon",
    "delay", "delays", "stalls", "resources", "resource", "compute", "storage", "loss", "losses",
    "bounds", "bound", "limits", "limit", "signals", "signal", "classification", "calibration",
    "emittance", "brightness", "charge", "density", "temperature", "velocity", "flux", "frequency",
}
GENERIC_LOW_VALUE = {
    "paper", "study", "method", "methods", "model", "models", "system", "systems", "approach",
    "framework", "algorithm", "algorithms", "result", "results", "data", "using", "use", "uses",
    "based", "new", "novel", "different", "various", "multiple", "simple", "complex", "important",
    "technical", "task", "tasks", "problem", "problems", "performance", "work", "research", "analysis",
    "tool", "tools", "application", "applications", "information", "process", "processes",
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is",
    "it", "its", "of", "on", "or", "over", "than", "that", "the", "their", "this", "through",
    "to", "using", "via", "while", "with", "without", "within", "across", "between", "under",
    "along", "alongside", "around", "before", "after", "during", "near", "toward", "towards",
}
WEIGHTS = {
    "polarity_uncertainty_caveat": 8.0,
    "role_safe_direction": 6.0,
    "metric_role_noun": 5.0,
    "source_specific_anchor": 4.0,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def messages(title: str, abstract: str, summary: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(title=title, abstract=abstract)},
        {"role": "assistant", "content": summary},
    ]


def source_terms(row: dict[str, Any]) -> set[str]:
    source = f"{row.get('title','')}\n{row.get('abstract','')}".lower()
    return {m.group(0).lower() for m in WORD_RE.finditer(source)}


def add_span(spans: list[dict[str, Any]], start: int, end: int, text: str, weight: float, kind: str) -> None:
    if start >= end:
        return
    spans.append({"start": start, "end": end, "text": text, "weight": weight, "kind": kind})


def phrase_spans(summary: str, phrase: str) -> list[tuple[int, int, str]]:
    # Word-boundary-ish phrase match that handles hyphenated terms.
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])", re.I)
    return [(m.start(), m.end(), summary[m.start():m.end()]) for m in pat.finditer(summary)]


def surgical_hints(row: dict[str, Any]) -> dict[str, Any]:
    summary = str(row.get("summary", ""))
    src_terms = source_terms(row)
    spans: list[dict[str, Any]] = []

    # Multiword/special polarity first; trainer will use max weight on overlaps.
    for phrase in sorted(POLARITY_PHRASES, key=len, reverse=True):
        for start, end, text in phrase_spans(summary, phrase):
            add_span(spans, start, end, text, WEIGHTS["polarity_uncertainty_caveat"], "polarity_uncertainty_caveat")

    for m in WORD_RE.finditer(summary):
        tok = m.group(0)
        low = tok.lower()
        if low in DO_NOT_WEIGHT:
            continue
        if low in DIRECTION_WORDS:
            add_span(spans, m.start(), m.end(), tok, WEIGHTS["role_safe_direction"], "role_safe_direction")
        if low in METRIC_ROLE_TERMS:
            add_span(spans, m.start(), m.end(), tok, WEIGHTS["metric_role_noun"], "metric_role_noun")
        source_grounded = low in src_terms or ("-" in low and low.replace("-", "") in {t.replace("-", "") for t in src_terms})
        if source_grounded and low not in STOPWORDS and low not in GENERIC_LOW_VALUE and len(low) >= 5:
            add_span(spans, m.start(), m.end(), tok, WEIGHTS["source_specific_anchor"], "source_specific_anchor")

    # Collapse exact duplicate spans, preserving max weight/kind priority by weight.
    collapsed: dict[tuple[int, int, str], dict[str, Any]] = {}
    for sp in spans:
        key = (int(sp["start"]), int(sp["end"]), str(sp["text"]))
        old = collapsed.get(key)
        if old is None or float(sp["weight"]) > float(old["weight"]):
            collapsed[key] = sp
    spans = sorted(collapsed.values(), key=lambda s: (int(s["start"]), -float(s["weight"])))
    return {
        "enabled": True,
        "base_weight": 1.0,
        "spans": spans,
        "policy": "surgical numeric-hardening weights: polarity/caveat 8, direction 6, metric-role nouns 5, source anchors 4; max weight on overlaps",
    }


def main() -> None:
    normal = load_jsonl(NORMAL_SRC)[:600]
    numeric_src = load_jsonl(NUMERIC_SRC)
    if len(normal) != 600 or len(numeric_src) != 200:
        raise SystemExit(f"Expected 600 normal and 200 numeric, got {len(normal)} and {len(numeric_src)}")

    final: list[dict[str, Any]] = []
    for idx, row in enumerate(normal, start=1):
        r = dict(row)
        r["source_split"] = "q5_numeric_ideal200_surgical_weight_v1_600normal_200numeric"
        r["weighted_numeric_patch_row_type"] = "normal_style_anchor"
        r["loss_weight_hints"] = {"enabled": False, "base_weight": 1.0, "spans": [], "policy": "normal style anchor"}
        r["mix_index"] = idx
        if "messages" not in r or not isinstance(r.get("messages"), list):
            r["messages"] = messages(r.get("title", ""), r.get("abstract", ""), r.get("summary", ""))
        final.append(r)

    for row in numeric_src:
        r = dict(row)
        r["source_split"] = "q5_numeric_ideal200_surgical_weight_v1_600normal_200numeric"
        r["weighted_numeric_patch_row_type"] = "numeric_ideal200_surgical_weight"
        r["loss_weight_hints"] = surgical_hints(r)
        r["messages"] = messages(r.get("title", ""), r.get("abstract", ""), r.get("summary", ""))
        r["mix_index"] = len(final) + 1
        final.append(r)

    span_kinds = Counter(sp["kind"] for r in final for sp in (r.get("loss_weight_hints") or {}).get("spans", []))
    span_weights = Counter(str(sp["weight"]) for r in final for sp in (r.get("loss_weight_hints") or {}).get("spans", []))
    spans_per_numeric = [len((r.get("loss_weight_hints") or {}).get("spans", [])) for r in final if r.get("weighted_numeric_patch_row_type") == "numeric_ideal200_surgical_weight"]
    zero_span_numeric = [r["paper_id"] for r in final if r.get("weighted_numeric_patch_row_type") == "numeric_ideal200_surgical_weight" and not (r.get("loss_weight_hints") or {}).get("spans")]

    write_jsonl(OUT / "train.jsonl", final)
    write_jsonl(OUT / "normal_selected_600.jsonl", final[:600])
    write_jsonl(OUT / "numeric_selected_200_surgical_weight.jsonl", final[600:])
    audit_rows = []
    for r in final[600:]:
        audit_rows.append({
            "paper_id": r["paper_id"],
            "title": r.get("title", ""),
            "summary": r.get("summary", ""),
            "audit_status": r.get("audit_status", ""),
            "weighted_spans": (r.get("loss_weight_hints") or {}).get("spans", []),
        })
    write_jsonl(OUT / "weighted_span_audit.jsonl", audit_rows)

    manifest = {
        "created_by": "scripts/build_numeric_ideal200_surgical_weight_v1_dataset.py",
        "rows": len(final),
        "normal_rows": 600,
        "numeric_rows": 200,
        "normal_source": str(NORMAL_SRC),
        "numeric_source": str(NUMERIC_SRC),
        "dataset_path": str(OUT / "train.jsonl"),
        "digit_summary_rows": sum(1 for r in final if DIGIT_RE.search(r.get("summary", ""))),
        "formula_summary_rows": sum(1 for r in final if FORMULA_RE.search(r.get("summary", ""))),
        "weighted_span_kind_counts": dict(span_kinds),
        "weighted_span_weight_counts": dict(span_weights),
        "numeric_zero_span_rows": zero_span_numeric,
        "numeric_min_spans": min(spans_per_numeric),
        "numeric_max_spans": max(spans_per_numeric),
        "numeric_avg_spans": round(sum(spans_per_numeric) / len(spans_per_numeric), 2),
        "weight_policy": WEIGHTS,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
