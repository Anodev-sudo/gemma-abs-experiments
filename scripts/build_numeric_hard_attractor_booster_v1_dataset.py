#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

NORMAL_SRC = Path("hf_bucket/experiments/sft/q5_step2400_sft700_clean_existing_targeted_weight_v1_lr4e7/dataset_600normal_200numeric_targeted_weight_clean_v1/normal_selected_600.jsonl")
IDEAL_SRC = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_ideal200_regen_v1/clean_200_after_filters_v1/training_rows_200.jsonl")
BOOSTER_ACCEPTED = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_hard_attractor_booster_v1/kimi_qwen_label_run/materialized/accepted_repairs.jsonl")
OUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_hard_attractor_booster_v1_lr1e6/dataset_600normal_200ideal_77booster_v1")

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
HIDDEN_RE = re.compile(r"\b(most|often|widespread|best|highest|largest|substantial|dramatic|major|state-of-the-art|outperforms|outperform|superior|strong|highly|significant)\b", re.I)

POLARITY_PHRASES = [
    "rules out", "rather than", "tradeoff", "tradeoffs", "without", "while", "despite",
    "candidate", "suggests", "constrains", "constraint", "constraints", "limitation", "limitations",
    "remains", "mixed", "conditional", "negative", "null", "caveat", "caveats", "uncertain",
    "tentative", "non-detection", "non-detections", "no", "not",
]
DIRECTION_WORDS = {
    "adapts", "accelerates", "accelerate", "accelerates", "automates", "balances", "classifies", "compares", "constrains",
    "detects", "detecting", "enables", "enable", "estimates", "exposes", "guides", "identifies", "improves", "improve",
    "links", "maps", "preserves", "probes", "probe", "reduces", "reduce", "reducing", "reveals", "supports", "support",
    "tracks", "optimizes", "forecasts", "predicts", "reconstructs", "calibrates", "allocates", "schedules", "summarizes",
    "measures", "separates", "distinguishes", "narrows", "replaces", "relates", "frames", "lifts", "forms", "generates",
    "generate", "extracts", "infers", "establishes", "evaluates", "validates", "decomposes", "decompose", "increases", "increasing",
    "decreases", "decreasing", "assigns", "extends", "integrates", "projects", "eases", "breaks",
}
DO_NOT_WEIGHT = {
    "outperforms", "outperform", "best", "highest", "largest", "strong", "significant", "major", "substantial",
    "dramatic", "superior", "highly", "state-of-the-art", "often", "most", "widespread", "new",
}
METRIC_ROLE_TERMS = {
    "accuracy", "accuracies", "latency", "throughput", "cost", "costs", "parameter", "parameters", "dataset", "datasets",
    "error", "errors", "speedup", "efficiency", "emissions", "memory", "bandwidth", "inference", "training", "runtime",
    "compression", "redshift", "mass", "masses", "distance", "distances", "rate", "rates", "count", "counts", "time",
    "times", "rounds", "message", "messages", "termination", "quality", "fidelity", "energy", "power", "carbon", "delay",
    "delays", "stalls", "resources", "resource", "compute", "storage", "loss", "losses", "bounds", "bound", "limits",
    "limit", "signals", "signal", "classification", "calibration", "emittance", "brightness", "charge", "density", "temperature",
    "velocity", "flux", "frequency", "coverage", "validation", "testing", "cost", "feedback", "bandwidth",
}
GENERIC_LOW_VALUE = {
    "paper", "study", "method", "methods", "model", "models", "system", "systems", "approach", "framework",
    "algorithm", "algorithms", "result", "results", "data", "using", "use", "uses", "based", "new", "novel",
    "different", "various", "multiple", "simple", "complex", "important", "technical", "task", "tasks", "problem",
    "problems", "performance", "work", "research", "analysis", "tool", "tools", "application", "applications",
    "information", "process", "processes",
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it", "its", "of",
    "on", "or", "over", "than", "that", "the", "their", "this", "through", "to", "using", "via", "while", "with",
    "without", "within", "across", "between", "under", "along", "alongside", "around", "before", "after", "during", "near",
    "toward", "towards",
}
WEIGHTS = {
    "polarity_uncertainty_caveat": 8.0,
    "role_safe_direction": 6.0,
    "metric_role_noun": 5.0,
    "source_specific_anchor": 2.0,
}
MAX_SOURCE_ANCHORS_PER_ROW = 3


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
    if start < end:
        spans.append({"start": start, "end": end, "text": text, "weight": weight, "kind": kind})


def phrase_spans(summary: str, phrase: str) -> list[tuple[int, int, str]]:
    pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(phrase) + r"(?![A-Za-z0-9])", re.I)
    return [(m.start(), m.end(), summary[m.start():m.end()]) for m in pat.finditer(summary)]


def surgical_hints(row: dict[str, Any], row_kind: str) -> dict[str, Any]:
    summary = str(row.get("summary", ""))
    src_terms = source_terms(row)
    spans: list[dict[str, Any]] = []
    occupied_high = set()

    for phrase in sorted(POLARITY_PHRASES, key=len, reverse=True):
        for start, end, text in phrase_spans(summary, phrase):
            add_span(spans, start, end, text, WEIGHTS["polarity_uncertainty_caveat"], "polarity_uncertainty_caveat")
            occupied_high.update(range(start, end))

    source_anchor_candidates = []
    for m in WORD_RE.finditer(summary):
        tok = m.group(0)
        low = tok.lower()
        if low in DO_NOT_WEIGHT:
            continue
        if low in DIRECTION_WORDS:
            add_span(spans, m.start(), m.end(), tok, WEIGHTS["role_safe_direction"], "role_safe_direction")
            occupied_high.update(range(m.start(), m.end()))
        if low in METRIC_ROLE_TERMS:
            add_span(spans, m.start(), m.end(), tok, WEIGHTS["metric_role_noun"], "metric_role_noun")
            occupied_high.update(range(m.start(), m.end()))
        source_grounded = low in src_terms or ("-" in low and low.replace("-", "") in {t.replace("-", "") for t in src_terms})
        if source_grounded and low not in STOPWORDS and low not in GENERIC_LOW_VALUE and len(low) >= 5:
            # Avoid turning source anchors into the dominant signal: cap and skip words already receiving stronger weights.
            if not any(pos in occupied_high for pos in range(m.start(), m.end())):
                source_anchor_candidates.append((m.start(), m.end(), tok))

    for start, end, tok in source_anchor_candidates[:MAX_SOURCE_ANCHORS_PER_ROW]:
        add_span(spans, start, end, tok, WEIGHTS["source_specific_anchor"], "source_specific_anchor")

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
        "policy": f"hard-attractor v1 {row_kind}: polarity 8, direction 6, metric-role 5, source anchors 2 capped at {MAX_SOURCE_ANCHORS_PER_ROW}",
    }


def hard_clean_summary(summary: str) -> tuple[bool, list[str]]:
    reasons = []
    if not (10 <= len(summary.split()) <= 15): reasons.append(f"wc={len(summary.split())}")
    if DIGIT_RE.search(summary): reasons.append("digit")
    if FORMULA_RE.search(summary): reasons.append("formula")
    if HIDDEN_RE.search(summary): reasons.append("hidden_quant")
    if not summary.endswith("."): reasons.append("period")
    return not reasons, reasons


def booster_training_row(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": r["paper_id"], "title": r.get("title", ""), "abstract": r.get("abstract", ""),
        "summary": r.get("repair_summary", ""), "primary_category": r.get("primary_category", ""),
        "categories": [], "created_date": "", "teacher_model": r.get("writer_model", ""),
        "generator_prompt_version": r.get("writer_prompt_version", ""),
        "review_prompt_version": r.get("reviewer_prompt_version", ""),
        "source_split": "q5_numeric_hard_attractor_booster_v1",
        "bad_summary_checkpoint700": r.get("pre_dpo_summary", ""),
        "bad_summary_current_q5": r.get("bad_summary", ""),
        "failure_tags": r.get("failure_tags", []), "number_policy": r.get("number_policy", ""),
        "repair_candidate_id": r.get("candidate_id", ""),
    }


def main() -> None:
    normal = load_jsonl(NORMAL_SRC)[:600]
    ideal = load_jsonl(IDEAL_SRC)
    booster_raw = load_jsonl(BOOSTER_ACCEPTED)
    booster = []
    excluded = []
    for r in booster_raw:
        tr = booster_training_row(r)
        ok, reasons = hard_clean_summary(tr["summary"])
        if ok:
            booster.append(tr)
        else:
            excluded.append({**tr, "hard_exclude_reasons": reasons})
    if len(normal) != 600 or len(ideal) != 200:
        raise SystemExit(f"Expected 600 normal and 200 ideal; got {len(normal)}, {len(ideal)}")
    if len(booster) < 60:
        raise SystemExit(f"Too few hard-clean booster rows: {len(booster)}")

    final: list[dict[str, Any]] = []
    for row in normal:
        r = dict(row)
        r["source_split"] = "q5_numeric_hard_attractor_booster_v1_600normal_200ideal_77booster"
        r["weighted_numeric_patch_row_type"] = "normal_style_anchor"
        r["loss_weight_hints"] = {"enabled": False, "base_weight": 1.0, "spans": [], "policy": "normal style anchor"}
        if "messages" not in r or not isinstance(r.get("messages"), list):
            r["messages"] = messages(r.get("title", ""), r.get("abstract", ""), r.get("summary", ""))
        final.append(r)

    for row in ideal:
        r = dict(row)
        r["source_split"] = "q5_numeric_hard_attractor_booster_v1_600normal_200ideal_77booster"
        r["weighted_numeric_patch_row_type"] = "numeric_ideal200_reduced_anchor_weight"
        r["loss_weight_hints"] = surgical_hints(r, "ideal200")
        r["messages"] = messages(r.get("title", ""), r.get("abstract", ""), r.get("summary", ""))
        final.append(r)

    for row in booster:
        r = dict(row)
        r["source_split"] = "q5_numeric_hard_attractor_booster_v1_600normal_200ideal_77booster"
        r["weighted_numeric_patch_row_type"] = "hard_attractor_booster"
        r["loss_weight_hints"] = surgical_hints(r, "hard_attractor_booster")
        r["messages"] = messages(r.get("title", ""), r.get("abstract", ""), r.get("summary", ""))
        final.append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "train.jsonl", final)
    write_jsonl(OUT / "normal_selected_600.jsonl", final[:600])
    write_jsonl(OUT / "ideal_selected_200.jsonl", final[600:800])
    write_jsonl(OUT / "booster_selected_hard_clean.jsonl", final[800:])
    write_jsonl(OUT / "booster_hard_excluded.jsonl", excluded)
    audit = [{"paper_id": r["paper_id"], "row_type": r["weighted_numeric_patch_row_type"], "summary": r["summary"], "spans": r["loss_weight_hints"]["spans"]} for r in final[600:]]
    write_jsonl(OUT / "weighted_span_audit.jsonl", audit)

    span_kinds = Counter(sp["kind"] for r in final for sp in (r.get("loss_weight_hints") or {}).get("spans", []))
    row_types = Counter(r["weighted_numeric_patch_row_type"] for r in final)
    numeric_spans = [len((r.get("loss_weight_hints") or {}).get("spans", [])) for r in final if r["weighted_numeric_patch_row_type"] != "normal_style_anchor"]
    manifest = {
        "created_by": "scripts/build_numeric_hard_attractor_booster_v1_dataset.py",
        "rows": len(final), "normal_rows": 600, "ideal_rows": 200, "booster_rows": len(booster),
        "booster_hard_excluded_rows": len(excluded), "dataset_path": str(OUT / "train.jsonl"),
        "digit_summary_rows": sum(1 for r in final if DIGIT_RE.search(r.get("summary", ""))),
        "formula_summary_rows": sum(1 for r in final if FORMULA_RE.search(r.get("summary", ""))),
        "hidden_quant_summary_rows": sum(1 for r in final if HIDDEN_RE.search(r.get("summary", ""))),
        "row_type_counts": dict(row_types), "weighted_span_kind_counts": dict(span_kinds),
        "numeric_min_spans": min(numeric_spans), "numeric_max_spans": max(numeric_spans),
        "numeric_avg_spans": round(sum(numeric_spans)/len(numeric_spans), 2),
        "weight_policy": WEIGHTS, "max_source_anchors_per_row": MAX_SOURCE_ANCHORS_PER_ROW,
        "booster_stress_rows": [r["paper_id"] for r in booster if r["paper_id"] in {"1901.06548","2111.10635","2303.09722","2406.04373"}],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
