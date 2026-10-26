#!/usr/bin/env python3
"""Build the q5 994-gold token-weighted natural-language dataset."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_GOLD_DATASET = Path(
    "hf_dataset/releases/"
    "recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/"
    "training_rows.jsonl"
)
DEFAULT_ANCHORS = Path(
    "hf_dataset/releases/recent_3y_balanced_10746_faithfulness_overlay_1400_v2/training_rows.jsonl"
)
DEFAULT_EVAL = Path(
    "hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_500_lora_v1/"
    "review_samples/ckpt120_vs_ckpt220_fresh20/generations.jsonl"
)
DEFAULT_PROMPT_SPEC = Path("prompts/mlx_lora/gemma_270m_it_abstract_to_feed_blurb_v5.json")
DEFAULT_OUTPUT_DIR = Path(
    "hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_token_weighted_gold_v1/"
    "dataset_994gold_600faithanchors_tokenweighted_v1"
)

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+'-]*")
FORMULAISH_RE = re.compile(r"[\d\\$={}^_*<>|/%]")

STOPWORDS = {
    "a",
    "across",
    "after",
    "an",
    "and",
    "are",
    "as",
    "at",
    "before",
    "between",
    "be",
    "but",
    "by",
    "can",
    "despite",
    "during",
    "except",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "instead",
    "into",
    "is",
    "it",
    "its",
    "may",
    "more",
    "new",
    "no",
    "not",
    "of",
    "on",
    "only",
    "or",
    "per",
    "rather",
    "still",
    "over",
    "than",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "to",
    "using",
    "with",
    "without",
    "while",
}
GENERIC_TOKENS = {
    "analysis",
    "algorithm",
    "algorithms",
    "approach",
    "data",
    "digit",
    "digits",
    "framework",
    "method",
    "methods",
    "model",
    "models",
    "number",
    "numbers",
    "paper",
    "problem",
    "problems",
    "research",
    "study",
    "system",
    "systems",
    "technique",
    "techniques",
}
HYPE_TOKENS = {
    "best",
    "better",
    "impressive",
    "novel",
    "outperform",
    "outperforms",
    "outperformed",
    "promising",
    "significant",
    "significantly",
    "state-of-the-art",
    "strong",
    "substantial",
}
POLARITY_PHRASES = (
    "rather than",
    "without",
    "instead",
    "despite",
    "except",
    "limited",
    "while",
    "only",
    "not",
    "but",
    "no",
)
PAYOFF_STEMS = {
    "bound",
    "classify",
    "classifi",
    "connect",
    "detect",
    "enable",
    "enabl",
    "estimate",
    "estimat",
    "explain",
    "improve",
    "improv",
    "link",
    "map",
    "predict",
    "prov",
    "prove",
    "reduc",
    "reduce",
    "reveal",
    "track",
}
MATH_RESULT_STEMS = {
    "algorithm",
    "bound",
    "classification",
    "condition",
    "construction",
    "counterexample",
    "criterion",
    "equivalence",
    "proof",
}
THEORY_HINTS = {
    "algorithm",
    "bound",
    "complexity",
    "condition",
    "construction",
    "counterexample",
    "equivalence",
    "graph",
    "proof",
    "theorem",
}


@dataclass(frozen=True)
class Token:
    text: str
    norm: str
    stem: str
    start: int
    end: int


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    weight: float
    kind: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dataset", type=Path, default=DEFAULT_GOLD_DATASET)
    parser.add_argument("--anchor-source", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--eval-jsonl", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--prompt-spec", type=Path, default=DEFAULT_PROMPT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anchor-count", type=int, default=600)
    parser.add_argument("--gold-weight", type=float, default=1.0)
    parser.add_argument("--risky-gold-weight", type=float, default=0.65)
    parser.add_argument("--risky-fallback-weight", type=float, default=0.85)
    parser.add_argument("--anchor-weight", type=float, default=0.12)
    parser.add_argument("--max-risky-fallback-frac", type=float, default=0.05)
    parser.add_argument("--audit-sample-count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260526)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def norm_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def norm_token(value: str) -> str:
    lowered = value.casefold().strip("'")
    lowered = lowered.replace("'", "")
    lowered = re.sub(r"[^a-z0-9-]+", "", lowered)
    return lowered.strip("-")


def simple_stem(value: str) -> str:
    token = norm_token(value)
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 6 and token.endswith("ing"):
        root = token[:-3]
        if root.endswith(root[-1:] * 2):
            root = root[:-1]
        return root
    if len(token) > 5 and token.endswith("ed"):
        root = token[:-2]
        if root.endswith(root[-1:] * 2):
            root = root[:-1]
        return root
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def tokens_with_offsets(text: str) -> list[Token]:
    tokens: list[Token] = []
    for match in WORD_RE.finditer(text):
        value = match.group(0)
        normalized = norm_token(value)
        if not normalized:
            continue
        tokens.append(Token(value, normalized, simple_stem(normalized), match.start(), match.end()))
    return tokens


def token_sets(text: str) -> tuple[set[str], set[str]]:
    tokens = tokens_with_offsets(text)
    return {token.norm for token in tokens}, {token.stem for token in tokens}


def content_token_set(text: str) -> set[str]:
    return {
        token.norm
        for token in tokens_with_offsets(text)
        if token.norm not in STOPWORDS and not FORMULAISH_RE.search(token.text)
    }


def contiguous_ngrams(tokens: list[Token], n: int) -> set[tuple[str, ...]]:
    return {
        tuple(token.norm for token in tokens[idx : idx + n])
        for idx in range(0, max(0, len(tokens) - n + 1))
    }


def title_overlap_stats(title: str, summary: str) -> dict[str, Any]:
    title_tokens = tokens_with_offsets(title)
    summary_tokens = tokens_with_offsets(summary)
    title_content = {
        token.norm
        for token in title_tokens
        if token.norm not in STOPWORDS and not FORMULAISH_RE.search(token.text)
    }
    summary_content = {
        token.norm
        for token in summary_tokens
        if token.norm not in STOPWORDS and not FORMULAISH_RE.search(token.text)
    }
    overlap = sorted(title_content & summary_content)
    title_trigrams = contiguous_ngrams(title_tokens, 3)
    summary_trigrams = contiguous_ngrams(summary_tokens, 3)
    trigram_hits = sorted(" ".join(gram) for gram in title_trigrams & summary_trigrams)
    return {
        "non_stop_overlap_count": len(overlap),
        "non_stop_overlap_tokens": overlap,
        "title_trigram_hits": trigram_hits,
        "title_trigram_hit": bool(trigram_hits),
        "title_overlap_risk": len(overlap) >= 3 or bool(trigram_hits),
    }


def is_forbidden_content_token(token: Token) -> bool:
    if not token.norm or token.norm in STOPWORDS:
        return True
    if token.norm in GENERIC_TOKENS or token.norm in HYPE_TOKENS:
        return True
    if FORMULAISH_RE.search(token.text):
        return True
    if len(token.text) >= 3 and token.text.isupper():
        return True
    return False


def source_grounded(token: Token, abstract_norms: set[str], abstract_stems: set[str]) -> bool:
    return token.norm in abstract_norms or token.stem in abstract_stems


def is_title_only(token: Token, title_norms: set[str], title_stems: set[str], abstract_norms: set[str], abstract_stems: set[str]) -> bool:
    in_title = token.norm in title_norms or token.stem in title_stems
    in_abstract = token.norm in abstract_norms or token.stem in abstract_stems
    return in_title and not in_abstract


def is_theory_row(row: dict[str, Any]) -> bool:
    cats = [str(row.get("primary_category") or ""), *[str(cat) for cat in row.get("categories", [])]]
    if not any(cat.startswith("math.") or cat in {"cs.CC", "cs.DM"} for cat in cats):
        return False
    haystack = " ".join([str(row.get("title") or ""), str(row.get("abstract") or ""), str(row.get("summary") or "")]).casefold()
    return any(hint in haystack for hint in THEORY_HINTS)


def phrase_spans(summary: str, phrase: str, weight: float, kind: str, reason: str) -> list[Span]:
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])", re.IGNORECASE)
    return [
        Span(match.start(), match.end(), summary[match.start() : match.end()], weight, kind, reason)
        for match in pattern.finditer(summary)
    ]


def token_span(token: Token, weight: float, kind: str, reason: str) -> Span:
    return Span(token.start, token.end, token.text, weight, kind, reason)


def collapse_spans(spans: list[Span]) -> list[Span]:
    best: dict[tuple[int, int, str], Span] = {}
    for span in spans:
        key = (span.start, span.end, span.kind)
        existing = best.get(key)
        if existing is None or span.weight > existing.weight:
            best[key] = span
    collapsed = list(best.values())
    collapsed.sort(key=lambda span: (span.start, span.end, span.kind))
    return collapsed


def select_spans(row: dict[str, Any], title_stats: dict[str, Any]) -> tuple[list[Span], list[str]]:
    del title_stats
    title = norm_text(row.get("title"))
    abstract = norm_text(row.get("abstract"))
    summary = norm_text(row.get("summary"))
    summary_tokens = tokens_with_offsets(summary)
    title_norms, title_stems = token_sets(title)
    abstract_norms, abstract_stems = token_sets(abstract)

    spans: list[Span] = []
    flags: list[str] = []

    for phrase in POLARITY_PHRASES:
        spans.extend(phrase_spans(summary, phrase, 2.5, "polarity_caveat", "caveat or polarity word"))

    for token in summary_tokens:
        if token.stem not in PAYOFF_STEMS:
            continue
        if is_forbidden_content_token(token):
            continue
        if not source_grounded(token, abstract_norms, abstract_stems):
            continue
        if is_title_only(token, title_norms, title_stems, abstract_norms, abstract_stems):
            continue
        spans.append(token_span(token, 1.8, "payoff_verb", "source-grounded payoff verb"))

    if is_theory_row(row):
        for token in summary_tokens:
            if token.stem not in MATH_RESULT_STEMS and token.norm not in MATH_RESULT_STEMS:
                continue
            if is_forbidden_content_token(token) and token.norm not in {"algorithm", "algorithms"}:
                continue
            if not source_grounded(token, abstract_norms, abstract_stems):
                continue
            if is_title_only(token, title_norms, title_stems, abstract_norms, abstract_stems):
                continue
            spans.append(token_span(token, 2.0, "math_result_shape", "theory result-shape word"))

    anchor_count = 0
    for token in summary_tokens:
        if anchor_count >= 3:
            break
        if is_forbidden_content_token(token):
            continue
        if token.norm in title_norms or token.stem in title_stems:
            continue
        if not source_grounded(token, abstract_norms, abstract_stems):
            continue
        spans.append(token_span(token, 1.5, "non_title_source_anchor", "abstract token not copied from title"))
        anchor_count += 1

    collapsed = collapse_spans(spans)
    for span in collapsed:
        if summary[span.start : span.end] != span.text:
            raise ValueError(f"Bad span offset for {row.get('paper_id')}: {span}")
    if not collapsed:
        flags.append("zero_selected_spans")
    return collapsed, flags


def load_prompt_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = ("prompt_version", "system_prompt", "user_prompt_templates")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"Prompt spec {path} missing keys: {missing}")
    if "title_abstract" not in spec["user_prompt_templates"]:
        raise ValueError(f"Prompt spec {path} is missing title_abstract template")
    return spec


def render_messages(row: dict[str, Any], summary: str, prompt_spec: dict[str, Any]) -> list[dict[str, str]]:
    title = norm_text(row.get("title"))
    abstract = norm_text(row.get("abstract"))
    if not title or not abstract or not summary:
        raise ValueError(f"Missing title, abstract, or summary for paper_id={row.get('paper_id')}")
    user = str(prompt_spec["user_prompt_templates"]["title_abstract"]).format(title=title, abstract=abstract)
    return [
        {"role": "system", "content": str(prompt_spec["system_prompt"])},
        {"role": "user", "content": user},
        {"role": "assistant", "content": summary},
    ]


def row_summary(row: dict[str, Any]) -> str:
    if row.get("summary"):
        return norm_text(row["summary"])
    messages = row.get("messages") or []
    if messages and isinstance(messages, list):
        return norm_text(messages[-1].get("content", ""))
    return ""


def converted_base_row(
    row: dict[str, Any],
    *,
    component: str,
    prompt_spec: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    summary = row_summary(row)
    out = {
        "paper_id": str(row["paper_id"]),
        "title": norm_text(row.get("title")),
        "abstract": norm_text(row.get("abstract")),
        "summary": summary,
        "primary_category": row.get("primary_category", ""),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date", ""),
        "source_split": row.get("source_split", ""),
        "teacher_model": row.get("teacher_model", ""),
        "generator_prompt_version": row.get("generator_prompt_version", ""),
        "review_prompt_version": row.get("review_prompt_version", ""),
        "dataset_component": component,
        "prompt_version": prompt_spec["prompt_version"],
        "input_mode": "title_abstract",
        "messages": render_messages(row, summary, prompt_spec),
        "provenance": {
            "source_path": str(source_path),
            "original_source_split": row.get("source_split", ""),
            "original_prompt_version": row.get("prompt_version", ""),
        },
    }
    if not out["paper_id"] or not out["title"] or not out["abstract"] or not out["summary"]:
        raise ValueError(f"Incomplete row after conversion from {source_path}: {row.get('paper_id')}")
    return out


def span_to_json(span: Span) -> dict[str, Any]:
    return {
        "start": span.start,
        "end": span.end,
        "text": span.text,
        "weight": span.weight,
        "kind": span.kind,
        "reason": span.reason,
    }


def validate_no_bad_title_only_spans(row: dict[str, Any], spans: list[Span]) -> None:
    title_norms, title_stems = token_sets(norm_text(row.get("title")))
    abstract_norms, abstract_stems = token_sets(norm_text(row.get("abstract")))
    for span in spans:
        if span.kind == "polarity_caveat":
            continue
        for token in tokens_with_offsets(span.text):
            if token.norm in STOPWORDS or is_forbidden_content_token(token):
                continue
            if is_title_only(token, title_norms, title_stems, abstract_norms, abstract_stems):
                raise ValueError(
                    f"Title-only token upweighted for {row.get('paper_id')}: "
                    f"{token.text!r} in span {span.text!r} ({span.kind})"
                )


def convert_gold_row(
    row: dict[str, Any],
    *,
    prompt_spec: dict[str, Any],
    source_path: Path,
    normal_weight: float,
    risky_weight: float,
    fallback_weight: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = converted_base_row(row, component="strict_gold_token_weighted", prompt_spec=prompt_spec, source_path=source_path)
    stats = title_overlap_stats(out["title"], out["summary"])
    spans, flags = select_spans(out, stats)
    is_risky = bool(stats["title_overlap_risk"])
    base_weight = risky_weight if is_risky else normal_weight
    if is_risky and not spans:
        base_weight = fallback_weight
        flags.append("risky_zero_span_base_fallback")
    validate_no_bad_title_only_spans(out, spans)

    out["loss_weight_hints"] = {
        "enabled": True,
        "base_weight": base_weight,
        "spans": [span_to_json(span) for span in spans],
        "policy": "q5_natural_language_994_token_weighted_gold_v1",
    }
    out["token_weight_policy"] = {
        "title_overlap_risk": is_risky,
        "selected_span_count": len(spans),
        "risk_flags": flags,
    }
    audit = {
        "paper_id": out["paper_id"],
        "dataset_component": out["dataset_component"],
        "title": out["title"],
        "summary": out["summary"],
        "primary_category": out["primary_category"],
        "title_overlap_stats": stats,
        "base_weight": base_weight,
        "risk_flags": flags,
        "selected_spans": [span_to_json(span) for span in spans],
    }
    return out, audit


def convert_anchor_row(
    row: dict[str, Any],
    *,
    prompt_spec: dict[str, Any],
    source_path: Path,
    anchor_weight: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = converted_base_row(row, component="faithfulness_anchor", prompt_spec=prompt_spec, source_path=source_path)
    stats = title_overlap_stats(out["title"], out["summary"])
    out["loss_weight_hints"] = {
        "enabled": True,
        "base_weight": anchor_weight,
        "spans": [],
        "policy": "q5_natural_language_994_token_weighted_gold_v1 low-weight faithfulness anchor",
    }
    audit = {
        "paper_id": out["paper_id"],
        "dataset_component": out["dataset_component"],
        "title": out["title"],
        "summary": out["summary"],
        "primary_category": out["primary_category"],
        "title_overlap_stats": stats,
        "base_weight": anchor_weight,
        "risk_flags": ["anchor_no_spans"],
        "selected_spans": [],
    }
    return out, audit


def write_audit_sample(path: Path, audits: list[dict[str, Any]], sample_count: int, seed: int) -> None:
    rng = random.Random(seed)
    candidates = [row for row in audits if row["dataset_component"] == "strict_gold_token_weighted"]
    sample = list(candidates)
    rng.shuffle(sample)
    sample = sample[:sample_count]
    lines = ["# Weighted Span Audit Sample", ""]
    for idx, row in enumerate(sample, 1):
        stats = row["title_overlap_stats"]
        lines.append(f"## {idx}. {row['paper_id']} / {row['primary_category']}")
        lines.append(f"- title risk: `{stats['title_overlap_risk']}` overlap={stats['non_stop_overlap_count']} base={row['base_weight']}")
        if row["risk_flags"]:
            lines.append(f"- flags: `{', '.join(row['risk_flags'])}`")
        lines.append(f"- title: {row['title']}")
        lines.append(f"- summary: {row['summary']}")
        if row["selected_spans"]:
            for span in row["selected_spans"]:
                lines.append(
                    f"  - `{span['text']}` [{span['kind']}, weight={span['weight']}, "
                    f"offset={span['start']}:{span['end']}]"
                )
        else:
            lines.append("  - no selected spans")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.anchor_count <= 0:
        raise ValueError("--anchor-count must be positive")
    for name in ("gold_weight", "risky_gold_weight", "risky_fallback_weight", "anchor_weight"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    prompt_spec = load_prompt_spec(args.prompt_spec)
    gold_rows = load_jsonl(args.gold_dataset)
    if len(gold_rows) != 994:
        raise ValueError(f"Expected 994 gold rows, found {len(gold_rows)} in {args.gold_dataset}")
    gold_ids = {str(row["paper_id"]) for row in gold_rows}
    if len(gold_ids) != len(gold_rows):
        raise ValueError("Gold dataset contains duplicate paper_id values")

    eval_ids = {str(row["paper_id"]) for row in load_jsonl(args.eval_jsonl)}
    anchor_candidates = [
        row
        for row in load_jsonl(args.anchor_source)
        if str(row.get("paper_id")) not in gold_ids and str(row.get("paper_id")) not in eval_ids
    ]
    if len(anchor_candidates) < args.anchor_count:
        raise ValueError(f"Need {args.anchor_count} anchors, found {len(anchor_candidates)}")

    rng = random.Random(args.seed)
    rng.shuffle(anchor_candidates)
    anchor_rows = anchor_candidates[: args.anchor_count]

    converted: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for row in gold_rows:
        out, audit = convert_gold_row(
            row,
            prompt_spec=prompt_spec,
            source_path=args.gold_dataset,
            normal_weight=args.gold_weight,
            risky_weight=args.risky_gold_weight,
            fallback_weight=args.risky_fallback_weight,
        )
        converted.append(out)
        audits.append(audit)
    for row in anchor_rows:
        out, audit = convert_anchor_row(
            row,
            prompt_spec=prompt_spec,
            source_path=args.anchor_source,
            anchor_weight=args.anchor_weight,
        )
        converted.append(out)
        audits.append(audit)

    expected_rows = len(gold_rows) + args.anchor_count
    if len(converted) != expected_rows:
        raise ValueError(f"Expected {expected_rows} converted rows, got {len(converted)}")
    converted_ids = [str(row["paper_id"]) for row in converted]
    if len(set(converted_ids)) != len(converted_ids):
        raise ValueError("Converted dataset contains duplicate paper_id values")

    gold_audits = [row for row in audits if row["dataset_component"] == "strict_gold_token_weighted"]
    risky_gold = [row for row in gold_audits if row["title_overlap_stats"]["title_overlap_risk"]]
    risky_fallbacks = [row for row in risky_gold if "risky_zero_span_base_fallback" in row["risk_flags"]]
    fallback_frac = len(risky_fallbacks) / max(1, len(risky_gold))
    if fallback_frac > args.max_risky_fallback_frac:
        raise ValueError(
            f"Risky zero-span fallback rate too high: {len(risky_fallbacks)}/{len(risky_gold)} "
            f"({fallback_frac:.2%})"
        )

    rng.shuffle(converted)
    output_path = args.output_dir / "train.jsonl"
    write_jsonl(output_path, converted)
    write_jsonl(args.output_dir / "weighted_span_audit.jsonl", audits)
    write_audit_sample(args.output_dir / "weighted_span_audit_sample40.md", audits, args.audit_sample_count, args.seed + 17)

    component_counts = Counter(row["dataset_component"] for row in converted)
    span_kind_counts = Counter(
        span["kind"]
        for row in audits
        for span in row["selected_spans"]
    )
    base_weight_counts = Counter(str(row["loss_weight_hints"]["base_weight"]) for row in converted)
    category_counts = Counter(norm_text(row.get("primary_category")) for row in converted)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "scripts/build_q5_natural_language_token_weighted_gold_v1_dataset.py",
        "purpose": "q5 promoted sidecar diagnostic with deterministic assistant-token weighting over 994 strict gold rows.",
        "output_path": str(output_path),
        "weighted_span_audit": str(args.output_dir / "weighted_span_audit.jsonl"),
        "weighted_span_audit_sample": str(args.output_dir / "weighted_span_audit_sample40.md"),
        "prompt_spec": str(args.prompt_spec),
        "prompt_version": prompt_spec["prompt_version"],
        "input_mode": "title_abstract",
        "row_count": len(converted),
        "component_counts": dict(sorted(component_counts.items())),
        "base_weight_counts": dict(sorted(base_weight_counts.items())),
        "span_kind_counts": dict(sorted(span_kind_counts.items())),
        "weights": {
            "strict_gold_default": args.gold_weight,
            "strict_gold_title_overlap_risk": args.risky_gold_weight,
            "strict_gold_title_overlap_risk_zero_span_fallback": args.risky_fallback_weight,
            "faithfulness_anchor": args.anchor_weight,
        },
        "sources": {
            "gold": str(args.gold_dataset),
            "anchor": str(args.anchor_source),
            "eval_excluded": str(args.eval_jsonl),
        },
        "anchor_selection": {
            "seed": args.seed,
            "requested_count": args.anchor_count,
            "candidate_count_after_exclusions": len(anchor_candidates),
            "excluded_gold_ids": len(gold_ids),
            "excluded_eval_ids": len(eval_ids),
        },
        "validation": {
            "expected_row_count": expected_rows,
            "actual_row_count": len(converted),
            "title_risk_gold_rows": len(risky_gold),
            "risky_zero_span_fallback_rows": len(risky_fallbacks),
            "risky_zero_span_fallback_fraction": round(fallback_frac, 6),
            "max_risky_zero_span_fallback_fraction": args.max_risky_fallback_frac,
            "title_only_span_check": "passed",
            "span_offset_check": "passed",
        },
        "primary_category_top20": dict(category_counts.most_common(20)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = [
        "# q5 994-Gold Token-Weighted Dataset v1",
        "",
        f"- Rows: {len(converted)}",
        f"- Strict gold rows: {component_counts['strict_gold_token_weighted']}",
        f"- Faithfulness anchors: {component_counts['faithfulness_anchor']} at base weight {args.anchor_weight:g}",
        f"- Prompt: `{prompt_spec['prompt_version']}` / `title_abstract`",
        f"- Title-risk gold rows: {len(risky_gold)}",
        f"- Risky zero-span fallbacks: {len(risky_fallbacks)} ({fallback_frac:.2%})",
        "",
        "Gold rows use deterministic assistant-token spans. Anchor rows are low-weight replay targets with no spans.",
        "See `weighted_span_audit.jsonl` and `weighted_span_audit_sample40.md` before training.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(f"[token-weighted-gold] wrote {len(converted)} rows to {output_path}")
    print(f"[token-weighted-gold] components={dict(sorted(component_counts.items()))}")
    print(f"[token-weighted-gold] span kinds={dict(sorted(span_kind_counts.items()))}")
    print(
        "[token-weighted-gold] risky fallback="
        f"{len(risky_fallbacks)}/{len(risky_gold)} ({fallback_frac:.2%})"
    )


if __name__ == "__main__":
    main()
