#!/usr/bin/env python3
"""Contrastive q8-vs-q5 heuristics for grounded-shape failure mining.

The score is a candidate selector, not a judge. It is intentionally loose: the
goal is to find rows where q5 looks polished but may have dropped q8's
source-anchored hook, invented technical wording, or flattened numeric/polarity
meaning into generic feed prose.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+'-]*")
DIGIT_RE = re.compile(r"\d")
NUMBER_RE = re.compile(
    r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?(?:\s?[%x]|e[-+]?\d+)?"
    r"|\b\d+[A-Za-z]+|\b[A-Za-z]+\d+[A-Za-z0-9-]*"
)
SENTENCE_END_RE = re.compile(r"[.!?]+(?:['\")\]]+)?(?=\s|$)")

STOPWORDS = {
    "about",
    "above",
    "across",
    "after",
    "again",
    "against",
    "also",
    "among",
    "and",
    "another",
    "any",
    "are",
    "around",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "can",
    "could",
    "does",
    "doing",
    "done",
    "during",
    "each",
    "for",
    "from",
    "had",
    "has",
    "have",
    "having",
    "her",
    "here",
    "his",
    "how",
    "into",
    "its",
    "more",
    "most",
    "new",
    "not",
    "our",
    "over",
    "paper",
    "papers",
    "per",
    "same",
    "show",
    "shows",
    "shown",
    "study",
    "such",
    "than",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "under",
    "using",
    "via",
    "was",
    "were",
    "when",
    "where",
    "which",
    "while",
    "with",
    "within",
    "without",
}

GENERIC_TERMS = {
    "approach",
    "approaches",
    "architecture",
    "data",
    "different",
    "domain",
    "domains",
    "effective",
    "efficiency",
    "efficient",
    "framework",
    "frameworks",
    "improved",
    "improves",
    "improving",
    "method",
    "methods",
    "model",
    "models",
    "novel",
    "performance",
    "results",
    "robust",
    "system",
    "systems",
    "task",
    "tasks",
    "technique",
    "techniques",
}

FOG_WORDS = {
    "allocation",
    "alignment",
    "analysis",
    "architecture",
    "characterizes",
    "coordination",
    "enables",
    "framework",
    "frameworks",
    "infrastructure",
    "mapping",
    "maps",
    "pipeline",
    "representation",
    "representations",
    "resource",
    "resources",
    "shapes",
    "structure",
    "structures",
    "supports",
    "validation",
    "workflow",
}

FOG_PATTERN_RE = re.compile(
    r"\b(enables?|supports?|characterizes?|reveals?|maps?|links?|uses?|"
    r"provides?|offers?|guides?)\b.{0,80}\b("
    r"frameworks?|structures?|mapping|allocation|validation|infrastructure|"
    r"pipeline|workflow|representations?|resources?)\b",
    re.I,
)

POLARITY_WORDS = {
    "but",
    "cannot",
    "caveat",
    "despite",
    "except",
    "fail",
    "failed",
    "fails",
    "failure",
    "however",
    "lack",
    "lacks",
    "limited",
    "limitation",
    "reduces",
    "reduction",
    "risk",
    "tradeoff",
    "uncertain",
    "underperforms",
    "without",
    "worse",
}

POSITIVE_WORDS = {
    "achieves",
    "boosts",
    "confirms",
    "enables",
    "enhances",
    "guarantees",
    "improves",
    "increases",
    "optimizes",
    "outperforms",
    "proves",
    "reduces",
    "solves",
    "supports",
}

RESULT_WORDS = {
    "accuracy",
    "cost",
    "efficiency",
    "faster",
    "improves",
    "lower",
    "outperforms",
    "reduces",
    "throughput",
    "tradeoff",
    "worse",
}

KNOWN_BAD_SUBSTRINGS = {
    "ather-proof",
    "bayron",
    "curravariance",
    "parametriding",
    "sterochastic",
}

ODD_MORPH_RE = re.compile(
    r"\b[A-Za-z]{7,}(?:proof|variance|vance|ness|istic|ality|ization|ating|"
    r"iding|ography|ology|fulness)\b",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score q5-vs-q8 paired generations for contrastive failure mining."
    )
    parser.add_argument("--source-jsonl", required=True)
    parser.add_argument("--q8-jsonl", required=True)
    parser.add_argument("--q5-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--q8-label", default="q8_mlx_python")
    parser.add_argument("--q5-label", default="q5_promoted")
    parser.add_argument(
        "--labeled-review-jsonl",
        default=None,
        help="Optional prior unblinded blind-review rows for calibration.",
    )
    parser.add_argument(
        "--high-precision-threshold",
        type=float,
        default=85.0,
        help="Tight review threshold. Calibrated to prioritize precision over recall.",
    )
    parser.add_argument(
        "--high-recall-threshold",
        type=float,
        default=45.0,
        help="Broad mining threshold. Useful for teacher repair candidate pools.",
    )
    parser.add_argument("--top-report-rows", type=int, default=80)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def norm_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def word_count(text: str) -> int:
    return len(words(text))


def has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def sentence_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return len(SENTENCE_END_RE.findall(text)) or 1


def canonical(token: str) -> str:
    token = token.lower().strip("'\".-_")
    if not token or has_digit(token):
        return ""
    if token.endswith("'s"):
        token = token[:-2]
    for suffix in ("ization", "ations", "ation", "ments", "ment", "ingly", "edly"):
        if len(token) > len(suffix) + 4 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    for suffix in ("ing", "ers", "ies", "ied", "ed", "es", "s"):
        if len(token) > len(suffix) + 4 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    if len(token) < 3 or token in STOPWORDS:
        return ""
    return token


def content_terms(text: str) -> list[tuple[str, str]]:
    seen: set[str] = set()
    terms: list[tuple[str, str]] = []
    for token in words(text):
        key = canonical(token)
        if not key or key in seen:
            continue
        seen.add(key)
        terms.append((key, token.lower()))
    return terms


def term_set(text: str) -> set[str]:
    return {key for key, _ in content_terms(text)}


def anchored_terms(summary: str, source_terms: set[str]) -> list[tuple[str, str]]:
    return [
        (key, token)
        for key, token in content_terms(summary)
        if key in source_terms and key not in GENERIC_TERMS
    ]


def title_overlap_ratio(summary: str, title: str) -> float:
    summary_terms = term_set(summary)
    if not summary_terms:
        return 0.0
    return len(summary_terms & term_set(title)) / len(summary_terms)


def number_tokens(text: str) -> list[str]:
    return NUMBER_RE.findall(text)


def has_any(text: str, terms: set[str]) -> bool:
    keys = {key for key, _ in content_terms(text)}
    return bool(keys & terms)


def unanchored_q5_terms(
    q5_summary: str,
    q8_summary: str,
    source_terms: set[str],
) -> list[str]:
    q8_terms = term_set(q8_summary)
    out: list[str] = []
    for key, token in content_terms(q5_summary):
        if key in source_terms or key in q8_terms or key in GENERIC_TERMS:
            continue
        if len(token) < 6:
            continue
        out.append(token)
    return out[:20]


def malformed_terms(q5_summary: str, source_text: str, q8_summary: str) -> list[str]:
    source_lower = source_text.lower()
    q8_lower = q8_summary.lower()
    found: list[str] = []
    lower = q5_summary.lower()
    for bad in KNOWN_BAD_SUBSTRINGS:
        if bad in lower:
            found.append(bad)
    for token in words(q5_summary):
        t = token.lower()
        if len(t) < 8:
            continue
        if t in source_lower or t in q8_lower:
            continue
        if "-" in t and t not in source_lower:
            found.append(t)
            continue
        if ODD_MORPH_RE.search(t) and t not in FOG_WORDS:
            found.append(t)
    return sorted(set(found))[:20]


def q8_source_terms_missing_from_q5(
    q8_summary: str,
    q5_summary: str,
    source_terms: set[str],
) -> list[str]:
    q5_terms = term_set(q5_summary)
    missing: list[str] = []
    seen: set[str] = set()
    for key, token in anchored_terms(q8_summary, source_terms):
        if key in q5_terms or key in seen:
            continue
        missing.append(token)
        seen.add(key)
    return missing[:24]


def fog_terms(summary: str) -> list[str]:
    terms = [token for key, token in content_terms(summary) if key in FOG_WORDS]
    return terms[:20]


def score_pair(row: dict[str, Any], q8_summary: str, q5_summary: str) -> dict[str, Any]:
    title = norm_text(row.get("title", ""))
    abstract = norm_text(row.get("abstract", ""))
    source_text = f"{title}\n{abstract}"
    source_terms = term_set(source_text)

    q8_summary = norm_text(q8_summary)
    q5_summary = norm_text(q5_summary)
    q8_anchored = anchored_terms(q8_summary, source_terms)
    q5_anchored = anchored_terms(q5_summary, source_terms)
    missing_q8_terms = q8_source_terms_missing_from_q5(q8_summary, q5_summary, source_terms)
    q5_unanchored = unanchored_q5_terms(q5_summary, q8_summary, source_terms)
    q5_malformed = malformed_terms(q5_summary, source_text, q8_summary)

    q8_anchor_count = len(q8_anchored)
    q5_anchor_count = len(q5_anchored)
    missing_ratio = len(missing_q8_terms) / q8_anchor_count if q8_anchor_count else 0.0
    q5_wc = word_count(q5_summary)
    q8_wc = word_count(q8_summary)
    q5_title_overlap = title_overlap_ratio(q5_summary, title)
    q8_title_overlap = title_overlap_ratio(q8_summary, title)
    q5_nums = number_tokens(q5_summary)
    q8_nums = number_tokens(q8_summary)
    source_nums = number_tokens(source_text)
    q5_fog_terms = fog_terms(q5_summary)

    categories: dict[str, float] = defaultdict(float)
    evidence: dict[str, Any] = {
        "missing_q8_source_terms": missing_q8_terms,
        "q5_unanchored_terms": q5_unanchored,
        "q5_malformed_terms": q5_malformed,
        "q5_fog_terms": q5_fog_terms,
        "q8_numbers": q8_nums[:12],
        "q5_numbers": q5_nums[:12],
        "q5_title_overlap": round(q5_title_overlap, 4),
        "q8_title_overlap": round(q8_title_overlap, 4),
        "q8_source_anchor_count": q8_anchor_count,
        "q5_source_anchor_count": q5_anchor_count,
    }

    if q8_anchor_count >= 4 and missing_ratio >= 0.45:
        categories["lost_q8_source_hook"] += min(35.0, 13.0 + 5.0 * len(missing_q8_terms))
    if q8_anchor_count >= 3 and q8_anchor_count - q5_anchor_count >= 3:
        categories["q8_concreteness_advantage"] += min(18.0, 4.0 * (q8_anchor_count - q5_anchor_count))
    if q5_unanchored:
        techy = [term for term in q5_unanchored if len(term) >= 8 or "-" in term]
        categories["unanchored_q5_technical_terms"] += min(25.0, 5.0 * len(techy or q5_unanchored))
    if q5_malformed:
        categories["malformed_or_invented_terms"] += min(30.0, 15.0 + 5.0 * len(q5_malformed))

    fog_pattern = bool(FOG_PATTERN_RE.search(q5_summary))
    low_specificity = q5_anchor_count <= 3 and q5_wc >= 8
    hook_drop = missing_ratio >= 0.4 and q8_anchor_count >= 4
    if (fog_pattern or len(q5_fog_terms) >= 3) and (low_specificity or hook_drop):
        categories["semantic_fog_abstraction"] += 18.0
    elif fog_pattern and q5_anchor_count <= q8_anchor_count:
        categories["semantic_fog_abstraction"] += 9.0

    source_has_polarity = has_any(source_text, POLARITY_WORDS)
    q8_has_polarity = has_any(q8_summary, POLARITY_WORDS)
    q5_has_polarity = has_any(q5_summary, POLARITY_WORDS)
    q5_has_positive = has_any(q5_summary, POSITIVE_WORDS)
    if (source_has_polarity or q8_has_polarity) and not q5_has_polarity and q5_has_positive:
        categories["value_or_polarity_binding"] += 16.0
    elif q8_has_polarity and not q5_has_polarity:
        categories["value_or_polarity_binding"] += 9.0

    q8_has_result = bool(q8_nums) or has_any(q8_summary, RESULT_WORDS)
    q5_has_result = bool(q5_nums) or has_any(q5_summary, RESULT_WORDS)
    if source_nums and q8_has_result and not q5_has_result and hook_drop:
        categories["numeric_result_drop"] += 16.0
    elif q8_nums and not q5_nums and hook_drop:
        categories["metric_to_qualitative_hook_drop"] += 8.0

    if has_digit(q5_summary):
        categories["q5_digit_or_number_residue"] += min(18.0, 10.0 + 2.0 * len(q5_nums))

    if q5_title_overlap >= 0.55 and q5_title_overlap - q8_title_overlap >= 0.15 and hook_drop:
        categories["title_rehash_missed_hook"] += 12.0
    elif q5_title_overlap >= 0.7:
        categories["title_rehash_missed_hook"] += 8.0

    if q5_wc < 8:
        categories["too_short_to_carry_hook"] += 8.0
    if q5_wc > 18:
        categories["overlong_or_dense"] += 5.0
    if sentence_count(q5_summary) > 1:
        categories["malformed_or_invented_terms"] += 8.0

    score = float(sum(categories.values()))
    flags = sorted(categories)
    if score >= 55:
        severity = "very_high"
    elif score >= 45:
        severity = "high"
    elif score >= 25:
        severity = "medium"
    elif score > 0:
        severity = "low"
    else:
        severity = "none"

    return {
        "eval_id": row.get("eval_id"),
        "paper_id": str(row.get("paper_id", "")),
        "title": title,
        "abstract": abstract,
        "primary_category": row.get("primary_category"),
        "created_date": row.get("created_date"),
        "q8_summary": q8_summary,
        "q5_summary": q5_summary,
        "score": round(score, 3),
        "severity": severity,
        "flags": flags,
        "category_scores": {key: round(value, 3) for key, value in sorted(categories.items())},
        "evidence": evidence,
        "q8_word_count": q8_wc,
        "q5_word_count": q5_wc,
        "outputs_match": q8_summary == q5_summary,
    }


def output_summary(row: dict[str, Any]) -> str:
    for key in (
        "summary",
        "normalized_output",
        "output",
        "generated_summary",
        "left_summary",
        "right_summary",
    ):
        value = row.get(key)
        if value is not None:
            return norm_text(value)
    return ""


def by_paper_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        pid = row.get("paper_id")
        if pid is None:
            continue
        out[str(pid)] = row
    return out


def build_scored_rows(
    *,
    source_jsonl: Path,
    q8_jsonl: Path,
    q5_jsonl: Path,
) -> list[dict[str, Any]]:
    source_rows = load_jsonl(source_jsonl)
    q8_rows = by_paper_id(q8_jsonl)
    q5_rows = by_paper_id(q5_jsonl)
    scored: list[dict[str, Any]] = []
    missing_q8: list[str] = []
    missing_q5: list[str] = []
    for row in source_rows:
        pid = str(row.get("paper_id", ""))
        q8_row = q8_rows.get(pid)
        q5_row = q5_rows.get(pid)
        if q8_row is None:
            missing_q8.append(pid)
            continue
        if q5_row is None:
            missing_q5.append(pid)
            continue
        scored.append(score_pair(row, output_summary(q8_row), output_summary(q5_row)))
    if missing_q8 or missing_q5:
        raise ValueError(
            f"Missing outputs for source rows: q8={len(missing_q8)} q5={len(missing_q5)}"
        )
    return scored


def scored_review_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in load_jsonl(path):
        a_model = str(row.get("summary_a_model", ""))
        b_model = str(row.get("summary_b_model", ""))
        if a_model.startswith("q8"):
            q8_summary = row.get("summary_a", "")
            q5_summary = row.get("summary_b", "")
            q5_model = b_model
        elif b_model.startswith("q8"):
            q8_summary = row.get("summary_b", "")
            q5_summary = row.get("summary_a", "")
            q5_model = a_model
        else:
            continue
        scored = score_pair(row, str(q8_summary), str(q5_summary))
        scored["human_preferred_model"] = row.get("preferred_model")
        scored["human_more_accurate_model"] = row.get("more_accurate_model")
        scored["human_q8_preferred"] = str(row.get("preferred_model", "")).startswith("q8")
        scored["human_q8_more_accurate"] = str(row.get("more_accurate_model", "")).startswith("q8")
        scored["q5_review_model"] = q5_model
        rows.append(scored)
    return rows


def precision_at_k(rows: list[dict[str, Any]], key: str, ks: list[int]) -> list[str]:
    ranked = sorted(rows, key=lambda row: row["score"], reverse=True)
    lines: list[str] = []
    total_positive = sum(1 for row in ranked if row.get(key))
    for k in ks:
        subset = ranked[: min(k, len(ranked))]
        if not subset:
            continue
        positives = sum(1 for row in subset if row.get(key))
        precision = positives / len(subset)
        recall = positives / total_positive if total_positive else 0.0
        lines.append(
            f"- top {len(subset)}: precision `{precision:.3f}`, recall `{recall:.3f}`, positives `{positives}`"
        )
    return lines


def threshold_table(rows: list[dict[str, Any]], key: str) -> list[str]:
    lines = ["| threshold | rows | precision | recall | positives |", "|---:|---:|---:|---:|---:|"]
    total_positive = sum(1 for row in rows if row.get(key))
    for threshold in (20, 25, 30, 35, 40, 45, 50, 60):
        subset = [row for row in rows if row["score"] >= threshold]
        positives = sum(1 for row in subset if row.get(key))
        precision = positives / len(subset) if subset else math.nan
        recall = positives / total_positive if total_positive else 0.0
        precision_text = "n/a" if math.isnan(precision) else f"{precision:.3f}"
        lines.append(
            f"| {threshold} | {len(subset)} | {precision_text} | {recall:.3f} | {positives} |"
        )
    return lines


def report_lines(
    rows: list[dict[str, Any]],
    *,
    q8_label: str,
    q5_label: str,
    high_precision_threshold: float,
    high_recall_threshold: float,
    top_report_rows: int,
    calibration_rows: list[dict[str, Any]] | None,
) -> list[str]:
    ranked = sorted(rows, key=lambda row: row["score"], reverse=True)
    flag_counts = Counter(flag for row in rows for flag in row["flags"])
    severity_counts = Counter(row["severity"] for row in rows)
    lines = [
        "# q8 vs q5 Contrastive Heuristic Report",
        "",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Rows scored: `{len(rows)}`",
        f"- q8 label: `{q8_label}`",
        f"- q5 label: `{q5_label}`",
        f"- High precision threshold: `{high_precision_threshold}`",
        f"- High recall threshold: `{high_recall_threshold}`",
        "",
        "## Counts",
        "",
        f"- High precision candidates: `{sum(1 for row in rows if row['score'] >= high_precision_threshold)}`",
        f"- High recall candidates: `{sum(1 for row in rows if row['score'] >= high_recall_threshold)}`",
        f"- q5 digit/number residue rows: `{sum(1 for row in rows if 'q5_digit_or_number_residue' in row['flags'])}`",
        f"- Semantic fog rows: `{sum(1 for row in rows if 'semantic_fog_abstraction' in row['flags'])}`",
        f"- Lost q8 source hook rows: `{sum(1 for row in rows if 'lost_q8_source_hook' in row['flags'])}`",
        "",
        "## Severity",
        "",
    ]
    for severity, count in sorted(severity_counts.items()):
        lines.append(f"- {severity}: `{count}`")
    lines.extend(["", "## Flag Counts", ""])
    for flag, count in flag_counts.most_common():
        lines.append(f"- {flag}: `{count}`")

    if calibration_rows is not None:
        lines.extend(
            [
                "",
                "## Calibration Against Prior Blind Review",
                "",
                "Human label used here: q8 marked more accurate in the prior review.",
                "",
                "### Precision At K",
                "",
            ]
        )
        lines.extend(precision_at_k(calibration_rows, "human_q8_more_accurate", [25, 50, 100, 150, 200, 300]))
        lines.extend(["", "### Thresholds", ""])
        lines.extend(threshold_table(calibration_rows, "human_q8_more_accurate"))

    lines.extend(["", "## Top Candidates", ""])
    for index, row in enumerate(ranked[:top_report_rows], start=1):
        evidence = row["evidence"]
        lines.extend(
            [
                f"### {index}. {row['paper_id']} score={row['score']} severity={row['severity']}",
                "",
                f"**Title**: {row['title']}",
                "",
                f"**q8**: {row['q8_summary']}",
                "",
                f"**q5**: {row['q5_summary']}",
                "",
                f"**Flags**: `{', '.join(row['flags'])}`",
                "",
                f"**Missing q8 source terms**: `{', '.join(evidence['missing_q8_source_terms'][:12])}`",
                "",
                f"**q5 unanchored terms**: `{', '.join(evidence['q5_unanchored_terms'][:12])}`",
                "",
            ]
        )
    return lines


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scored = build_scored_rows(
        source_jsonl=Path(args.source_jsonl),
        q8_jsonl=Path(args.q8_jsonl),
        q5_jsonl=Path(args.q5_jsonl),
    )
    ranked = sorted(scored, key=lambda row: row["score"], reverse=True)
    high_precision = [row for row in ranked if row["score"] >= args.high_precision_threshold]
    high_recall = [row for row in ranked if row["score"] >= args.high_recall_threshold]

    calibration_rows = (
        scored_review_rows(Path(args.labeled_review_jsonl))
        if args.labeled_review_jsonl
        else None
    )

    write_jsonl(output_dir / "scored_rows.jsonl", scored)
    write_jsonl(output_dir / "ranked_candidates.jsonl", ranked)
    write_jsonl(output_dir / "high_precision_candidates.jsonl", high_precision)
    write_jsonl(output_dir / "high_recall_candidates.jsonl", high_recall)
    if calibration_rows is not None:
        write_jsonl(output_dir / "calibration_scored_prior_review.jsonl", calibration_rows)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_jsonl": args.source_jsonl,
        "q8_jsonl": args.q8_jsonl,
        "q5_jsonl": args.q5_jsonl,
        "q8_label": args.q8_label,
        "q5_label": args.q5_label,
        "rows_scored": len(scored),
        "high_precision_threshold": args.high_precision_threshold,
        "high_recall_threshold": args.high_recall_threshold,
        "high_precision_rows": len(high_precision),
        "high_recall_rows": len(high_recall),
        "labeled_review_jsonl": args.labeled_review_jsonl,
        "outputs": {
            "scored_rows_jsonl": "scored_rows.jsonl",
            "ranked_candidates_jsonl": "ranked_candidates.jsonl",
            "high_precision_candidates_jsonl": "high_precision_candidates.jsonl",
            "high_recall_candidates_jsonl": "high_recall_candidates.jsonl",
            "heuristic_report_md": "HEURISTIC_REPORT.md",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "HEURISTIC_REPORT.md").write_text(
        "\n".join(
            report_lines(
                scored,
                q8_label=args.q8_label,
                q5_label=args.q5_label,
                high_precision_threshold=args.high_precision_threshold,
                high_recall_threshold=args.high_recall_threshold,
                top_report_rows=args.top_report_rows,
                calibration_rows=calibration_rows,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[contrastive-heuristics] scored {len(scored)} rows")
    print(f"[contrastive-heuristics] high precision candidates: {len(high_precision)}")
    print(f"[contrastive-heuristics] high recall candidates: {len(high_recall)}")
    print(f"[contrastive-heuristics] report: {output_dir / 'HEURISTIC_REPORT.md'}")


if __name__ == "__main__":
    main()
