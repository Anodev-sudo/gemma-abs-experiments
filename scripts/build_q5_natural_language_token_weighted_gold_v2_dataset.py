#!/usr/bin/env python3
"""Build the q5 994-gold specificity-preserving token-weighted dataset."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_q5_natural_language_token_weighted_gold_v1_dataset as v1


POLICY_NAME = "q5_natural_language_994_token_weighted_gold_v2_specificity"
DEFAULT_OUTPUT_DIR = Path(
    "hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_994_token_weighted_gold_v2/"
    "dataset_994gold_600faithanchors_specificity_v2"
)

SCOPE_QUALIFIERS = ("suggests", "estimated", "limited", "appears", "likely", "could", "may")
LOW_SPECIFICITY_TOKENS = {
    "additional",
    "better",
    "complex",
    "different",
    "down",
    "effective",
    "far",
    "first",
    "important",
    "large",
    "larger",
    "main",
    "major",
    "new",
    "robust",
    "simple",
    "smaller",
    "stable",
    "strong",
    "time",
    "up",
    "useful",
}
BENCHMARK_TOKENS = {
    "accuracy",
    "accurate",
    "benchmark",
    "benchmarks",
    "faster",
    "performance",
    "speedup",
    "state-of-the-art",
}
MATH_RESULT_TERMS = {
    "algorithm",
    "bound",
    "classification",
    "cohomology",
    "condition",
    "construction",
    "counterexample",
    "criterion",
    "equivalence",
    "functor",
    "genus",
    "graph",
    "invariant",
    "module",
    "proof",
    "semigroup",
    "variety",
}
SYSTEMS_CS_CATEGORIES = {
    "cs.AR",
    "cs.CR",
    "cs.DB",
    "cs.DC",
    "cs.HC",
    "cs.IR",
    "cs.IT",
    "cs.NI",
    "cs.RO",
    "cs.SE",
    "cs.SY",
}


@dataclass(frozen=True)
class Candidate:
    token: v1.Token
    score: float
    weight: float
    kind: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dataset", type=Path, default=v1.DEFAULT_GOLD_DATASET)
    parser.add_argument("--anchor-source", type=Path, default=v1.DEFAULT_ANCHORS)
    parser.add_argument("--eval-jsonl", type=Path, default=v1.DEFAULT_EVAL)
    parser.add_argument("--prompt-spec", type=Path, default=v1.DEFAULT_PROMPT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anchor-count", type=int, default=600)
    parser.add_argument("--gold-weight", type=float, default=1.0)
    parser.add_argument("--risky-gold-weight", type=float, default=0.75)
    parser.add_argument("--risky-fallback-weight", type=float, default=0.90)
    parser.add_argument("--under10-base-cap", type=float, default=0.85)
    parser.add_argument("--anchor-weight", type=float, default=0.12)
    parser.add_argument("--max-risky-fallback-frac", type=float, default=0.05)
    parser.add_argument("--audit-sample-count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260527)
    return parser.parse_args()


def word_count(text: str) -> int:
    return len(v1.WORD_RE.findall(text))


def categories_for(row: dict[str, Any]) -> list[str]:
    cats = [str(row.get("primary_category") or "")]
    cats.extend(str(cat) for cat in row.get("categories", []) if cat)
    return [cat for cat in cats if cat]


def umbrella_category(category: str) -> str:
    if category.startswith("cs."):
        return "cs"
    if category.startswith("math."):
        return "math"
    if category.startswith(("astro-ph", "cond-mat", "gr-qc", "hep-", "nucl-", "physics.")) or category == "quant-ph":
        return "physics"
    return category.split(".")[0] if category else "unknown"


def is_domain_object_row(row: dict[str, Any]) -> bool:
    cats = set(categories_for(row))
    primary = str(row.get("primary_category") or "")
    if primary.startswith(("math.", "physics.", "astro-ph", "cond-mat", "gr-qc", "hep-", "nucl-")):
        return True
    if primary in {"quant-ph"} or primary.startswith(("eess.", "q-bio.")):
        return True
    return bool(cats & SYSTEMS_CS_CATEGORIES)


def copied_title_trigram_indexes(title: str, summary: str) -> set[int]:
    title_grams = v1.contiguous_ngrams(v1.tokens_with_offsets(title), 3)
    if not title_grams:
        return set()
    summary_tokens = v1.tokens_with_offsets(summary)
    copied: set[int] = set()
    for idx in range(max(0, len(summary_tokens) - 2)):
        gram = tuple(token.norm for token in summary_tokens[idx : idx + 3])
        if gram in title_grams:
            copied.update({idx, idx + 1, idx + 2})
    return copied


def token_index_by_offset(tokens: list[v1.Token]) -> dict[tuple[int, int], int]:
    return {(token.start, token.end): idx for idx, token in enumerate(tokens)}


def span_has_copied_title_token(span: v1.Span, summary_tokens: list[v1.Token], copied_indexes: set[int]) -> bool:
    for idx, token in enumerate(summary_tokens):
        if idx not in copied_indexes:
            continue
        if token.start < span.end and span.start < token.end:
            return True
    return False


def token_is_copied_title_trigram(token: v1.Token, token_indexes: dict[tuple[int, int], int], copied_indexes: set[int]) -> bool:
    idx = token_indexes.get((token.start, token.end))
    return idx in copied_indexes if idx is not None else False


def bad_content_token(token: v1.Token, *, allow_acronym: bool = False, allow_algorithm: bool = False) -> bool:
    if not token.norm or token.norm in v1.STOPWORDS:
        return True
    if token.norm in LOW_SPECIFICITY_TOKENS or token.norm in BENCHMARK_TOKENS:
        return True
    if token.norm in v1.HYPE_TOKENS:
        return True
    if token.norm in v1.GENERIC_TOKENS and not (allow_algorithm and token.norm in {"algorithm", "algorithms"}):
        return True
    if v1.FORMULAISH_RE.search(token.text):
        return True
    if len(token.text) >= 3 and token.text.isupper() and not allow_acronym:
        return True
    return False


def shared_by_title_and_abstract(
    token: v1.Token,
    title_norms: set[str],
    title_stems: set[str],
    abstract_norms: set[str],
    abstract_stems: set[str],
) -> bool:
    in_title = token.norm in title_norms or token.stem in title_stems
    in_abstract = token.norm in abstract_norms or token.stem in abstract_stems
    return in_title and in_abstract


def source_count(token: v1.Token, source_tokens: list[v1.Token]) -> int:
    return sum(1 for candidate in source_tokens if candidate.norm == token.norm or candidate.stem == token.stem)


def specificity_score(token: v1.Token, abstract_tokens: list[v1.Token], *, shared: bool = False) -> float:
    score = 0.0
    if len(token.norm) >= 9:
        score += 2.0
    elif len(token.norm) >= 7:
        score += 1.0
    if "-" in token.text:
        score += 1.5
    if any(ch.isupper() for ch in token.text[1:]):
        score += 1.0
    if token.norm.endswith(("tion", "ity", "ics", "ism", "ology", "ance", "ence")):
        score += 1.0
    count = source_count(token, abstract_tokens)
    if count == 1:
        score += 1.0
    elif count >= 3:
        score -= 0.5
    if shared:
        score -= 0.25
    return score


def is_acronym_like(token: v1.Token) -> bool:
    return 2 <= len(token.text) <= 8 and token.text.isupper() and token.text.isalpha()


def has_mixed_case(token: v1.Token) -> bool:
    return any(ch.isupper() for ch in token.text[1:]) and any(ch.islower() for ch in token.text)


def has_case_evidence(token: v1.Token, title_tokens: list[v1.Token], abstract_tokens: list[v1.Token]) -> bool:
    if not token.text[:1].isupper() or len(token.norm) < 4:
        return False
    for source_token in [*title_tokens, *abstract_tokens]:
        if source_token.norm == token.norm and source_token.text[:1].isupper():
            return True
    return False


def domain_candidate_score(token: v1.Token, title_tokens: list[v1.Token], abstract_tokens: list[v1.Token]) -> float | None:
    if is_acronym_like(token):
        return 5.0 + specificity_score(token, abstract_tokens)
    if has_mixed_case(token):
        return 4.0 + specificity_score(token, abstract_tokens)
    if has_case_evidence(token, title_tokens, abstract_tokens):
        return 3.0 + specificity_score(token, abstract_tokens)
    if "-" in token.text and len(token.norm) >= 5:
        return 2.5 + specificity_score(token, abstract_tokens)
    return None


def choose_candidates(candidates: list[Candidate], cap: int, used_offsets: set[tuple[int, int]], *, one_acronym: bool = False) -> list[v1.Span]:
    chosen: list[v1.Span] = []
    acronym_used = False
    ranked = sorted(candidates, key=lambda item: (-item.score, item.token.start, item.kind))
    for candidate in ranked:
        offset = (candidate.token.start, candidate.token.end)
        if offset in used_offsets:
            continue
        if one_acronym and is_acronym_like(candidate.token):
            if acronym_used:
                continue
            acronym_used = True
        chosen.append(v1.token_span(candidate.token, candidate.weight, candidate.kind, candidate.reason))
        used_offsets.add(offset)
        if len(chosen) >= cap:
            break
    return chosen


def collapse_spans(spans: list[v1.Span]) -> list[v1.Span]:
    priority = {
        "polarity_caveat": 7,
        "scope_qualifier": 6,
        "math_result_shape": 5,
        "domain_object_term": 4,
        "payoff_verb": 3,
        "source_specific_anchor": 2,
        "source_backed_shared_key_term": 1,
    }
    best: dict[tuple[int, int], v1.Span] = {}
    for span in spans:
        key = (span.start, span.end)
        existing = best.get(key)
        if existing is None:
            best[key] = span
            continue
        if (span.weight, priority.get(span.kind, 0)) > (existing.weight, priority.get(existing.kind, 0)):
            best[key] = span
    collapsed = list(best.values())
    collapsed.sort(key=lambda span: (span.start, span.end, span.kind))
    return collapsed


def select_spans(row: dict[str, Any], title_stats: dict[str, Any]) -> tuple[list[v1.Span], list[str]]:
    del title_stats
    title = v1.norm_text(row.get("title"))
    abstract = v1.norm_text(row.get("abstract"))
    summary = v1.norm_text(row.get("summary"))
    title_tokens = v1.tokens_with_offsets(title)
    abstract_tokens = v1.tokens_with_offsets(abstract)
    summary_tokens = v1.tokens_with_offsets(summary)
    title_norms, title_stems = v1.token_sets(title)
    abstract_norms, abstract_stems = v1.token_sets(abstract)
    copied_indexes = copied_title_trigram_indexes(title, summary)
    summary_index = token_index_by_offset(summary_tokens)

    spans: list[v1.Span] = []
    flags: list[str] = []
    used_offsets: set[tuple[int, int]] = set()

    for phrase in v1.POLARITY_PHRASES:
        spans.extend(v1.phrase_spans(summary, phrase, 2.5, "polarity_caveat", "caveat or polarity word"))
    for phrase in SCOPE_QUALIFIERS:
        spans.extend(v1.phrase_spans(summary, phrase, 2.1, "scope_qualifier", "scope or uncertainty qualifier"))
    used_offsets.update((span.start, span.end) for span in spans)

    payoff_candidates: list[Candidate] = []
    math_candidates: list[Candidate] = []
    domain_candidates: list[Candidate] = []
    specific_candidates: list[Candidate] = []
    shared_candidates: list[Candidate] = []

    for token in summary_tokens:
        copied = token_is_copied_title_trigram(token, summary_index, copied_indexes)
        if copied:
            continue
        title_only = v1.is_title_only(token, title_norms, title_stems, abstract_norms, abstract_stems)
        grounded = v1.source_grounded(token, abstract_norms, abstract_stems)
        shared = shared_by_title_and_abstract(token, title_norms, title_stems, abstract_norms, abstract_stems)
        non_title = token.norm not in title_norms and token.stem not in title_stems

        if token.stem in v1.PAYOFF_STEMS and grounded and not title_only:
            if not bad_content_token(token):
                payoff_candidates.append(Candidate(token, specificity_score(token, abstract_tokens), 1.6, "payoff_verb", "source-grounded payoff verb"))

        if v1.is_theory_row(row) and grounded and not title_only:
            if token.stem in MATH_RESULT_TERMS or token.norm in MATH_RESULT_TERMS:
                if not bad_content_token(token, allow_algorithm=True):
                    math_candidates.append(Candidate(token, specificity_score(token, abstract_tokens, shared=shared), 2.0, "math_result_shape", "theory result-shape or object term"))

        if is_domain_object_row(row) and grounded and not title_only:
            score = domain_candidate_score(token, title_tokens, abstract_tokens)
            if score is not None and not bad_content_token(token, allow_acronym=True):
                domain_candidates.append(Candidate(token, score, 1.65, "domain_object_term", "source-grounded central domain object"))

        if grounded and non_title and not bad_content_token(token):
            score = specificity_score(token, abstract_tokens)
            if score > 0:
                specific_candidates.append(Candidate(token, score, 1.55, "source_specific_anchor", "scored abstract-backed non-title token"))

        if grounded and shared and not title_only and not bad_content_token(token):
            score = specificity_score(token, abstract_tokens, shared=True)
            if score > 0:
                shared_candidates.append(Candidate(token, score, 1.35, "source_backed_shared_key_term", "shared title term independently backed by abstract"))

    for candidates, cap, one_acronym in (
        (math_candidates, 3, False),
        (domain_candidates, 2, True),
        (payoff_candidates, 3, False),
        (specific_candidates, 4, False),
        (shared_candidates, 2, False),
    ):
        spans.extend(choose_candidates(candidates, cap, used_offsets, one_acronym=one_acronym))

    collapsed = collapse_spans(spans)
    for span in collapsed:
        if summary[span.start : span.end] != span.text:
            raise ValueError(f"Bad span offset for {row.get('paper_id')}: {span}")
    copied_bad = [
        span for span in collapsed
        if span.kind not in {"polarity_caveat", "scope_qualifier"}
        and span_has_copied_title_token(span, summary_tokens, copied_indexes)
    ]
    if copied_bad:
        details = ", ".join(f"{span.text}/{span.kind}" for span in copied_bad)
        raise ValueError(f"Copied title trigram token upweighted for {row.get('paper_id')}: {details}")
    if copied_indexes:
        flags.append("summary_contains_title_trigram")
    if not collapsed:
        flags.append("zero_selected_spans")
    return collapsed, flags


def validate_no_bad_title_or_copy_spans(row: dict[str, Any], spans: list[v1.Span]) -> None:
    title = v1.norm_text(row.get("title"))
    abstract = v1.norm_text(row.get("abstract"))
    summary = v1.norm_text(row.get("summary"))
    title_norms, title_stems = v1.token_sets(title)
    abstract_norms, abstract_stems = v1.token_sets(abstract)
    summary_tokens = v1.tokens_with_offsets(summary)
    copied_indexes = copied_title_trigram_indexes(title, summary)
    for span in spans:
        if span.kind not in {"polarity_caveat", "scope_qualifier"} and span_has_copied_title_token(span, summary_tokens, copied_indexes):
            raise ValueError(f"Copied title trigram token upweighted for {row.get('paper_id')}: {span.text!r}")
        if span.kind in {"polarity_caveat", "scope_qualifier"}:
            continue
        for token in v1.tokens_with_offsets(span.text):
            if token.norm in v1.STOPWORDS or bad_content_token(token, allow_acronym=True, allow_algorithm=True):
                continue
            if v1.is_title_only(token, title_norms, title_stems, abstract_norms, abstract_stems):
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
    under10_base_cap: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = v1.converted_base_row(row, component="strict_gold_token_weighted_specificity_v2", prompt_spec=prompt_spec, source_path=source_path)
    stats = v1.title_overlap_stats(out["title"], out["summary"])
    spans, flags = select_spans(out, stats)
    is_risky = bool(stats["title_overlap_risk"])
    base_weight = risky_weight if is_risky else normal_weight
    if is_risky and not spans:
        base_weight = fallback_weight
        flags.append("risky_zero_span_base_fallback")
    has_scope_or_polarity = any(span.kind in {"polarity_caveat", "scope_qualifier"} for span in spans)
    summary_words = word_count(out["summary"])
    if summary_words < 10 and not has_scope_or_polarity and base_weight > under10_base_cap:
        base_weight = under10_base_cap
        flags.append("under10_base_capped")
    validate_no_bad_title_or_copy_spans(out, spans)

    out["loss_weight_hints"] = {
        "enabled": True,
        "base_weight": base_weight,
        "spans": [v1.span_to_json(span) for span in spans],
        "policy": POLICY_NAME,
    }
    out["token_weight_policy"] = {
        "title_overlap_risk": is_risky,
        "selected_span_count": len(spans),
        "risk_flags": flags,
    }
    span_counts = Counter(span.kind for span in spans)
    audit = {
        "paper_id": out["paper_id"],
        "dataset_component": out["dataset_component"],
        "title": out["title"],
        "summary": out["summary"],
        "summary_word_count": summary_words,
        "primary_category": out["primary_category"],
        "title_overlap_stats": stats,
        "base_weight": base_weight,
        "risk_flags": flags,
        "selected_spans": [v1.span_to_json(span) for span in spans],
        "span_kind_counts": dict(sorted(span_counts.items())),
        "shared_key_term_count": span_counts.get("source_backed_shared_key_term", 0),
        "domain_object_term_count": span_counts.get("domain_object_term", 0),
        "under10_base_capped": "under10_base_capped" in flags,
    }
    return out, audit


def convert_anchor_row(
    row: dict[str, Any],
    *,
    prompt_spec: dict[str, Any],
    source_path: Path,
    anchor_weight: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = v1.converted_base_row(row, component="faithfulness_anchor", prompt_spec=prompt_spec, source_path=source_path)
    stats = v1.title_overlap_stats(out["title"], out["summary"])
    out["loss_weight_hints"] = {
        "enabled": True,
        "base_weight": anchor_weight,
        "spans": [],
        "policy": f"{POLICY_NAME} low-weight faithfulness anchor",
    }
    audit = {
        "paper_id": out["paper_id"],
        "dataset_component": out["dataset_component"],
        "title": out["title"],
        "summary": out["summary"],
        "summary_word_count": word_count(out["summary"]),
        "primary_category": out["primary_category"],
        "title_overlap_stats": stats,
        "base_weight": anchor_weight,
        "risk_flags": ["anchor_no_spans"],
        "selected_spans": [],
        "span_kind_counts": {},
        "shared_key_term_count": 0,
        "domain_object_term_count": 0,
        "under10_base_capped": False,
    }
    return out, audit


def write_audit_sample(path: Path, audits: list[dict[str, Any]], sample_count: int, seed: int) -> None:
    rng = random.Random(seed)
    gold = [row for row in audits if row["dataset_component"] == "strict_gold_token_weighted_specificity_v2"]
    priority = [
        row for row in gold
        if "risky_zero_span_base_fallback" in row["risk_flags"]
        or row["shared_key_term_count"]
        or row["domain_object_term_count"]
    ]
    rng.shuffle(priority)
    rest = [row for row in gold if row not in priority]
    rng.shuffle(rest)
    sample: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in [*priority, *rest]:
        if row["paper_id"] in seen:
            continue
        sample.append(row)
        seen.add(row["paper_id"])
        if len(sample) >= sample_count:
            break

    lines = ["# Weighted Span Audit Sample v2", ""]
    for idx, row in enumerate(sample, 1):
        stats = row["title_overlap_stats"]
        lines.append(f"## {idx}. {row['paper_id']} / {row['primary_category']}")
        lines.append(
            f"- title risk: `{stats['title_overlap_risk']}` overlap={stats['non_stop_overlap_count']} "
            f"base={row['base_weight']} words={row['summary_word_count']}"
        )
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
    for name in ("gold_weight", "risky_gold_weight", "risky_fallback_weight", "under10_base_cap", "anchor_weight"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    prompt_spec = v1.load_prompt_spec(args.prompt_spec)
    gold_rows = v1.load_jsonl(args.gold_dataset)
    if len(gold_rows) != 994:
        raise ValueError(f"Expected 994 gold rows, found {len(gold_rows)} in {args.gold_dataset}")
    gold_ids = {str(row["paper_id"]) for row in gold_rows}
    if len(gold_ids) != len(gold_rows):
        raise ValueError("Gold dataset contains duplicate paper_id values")

    eval_ids = {str(row["paper_id"]) for row in v1.load_jsonl(args.eval_jsonl)}
    anchor_candidates = [
        row
        for row in v1.load_jsonl(args.anchor_source)
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
            under10_base_cap=args.under10_base_cap,
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

    gold_audits = [row for row in audits if row["dataset_component"] == "strict_gold_token_weighted_specificity_v2"]
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
    v1.write_jsonl(output_path, converted)
    v1.write_jsonl(args.output_dir / "weighted_span_audit.jsonl", audits)
    write_audit_sample(args.output_dir / "weighted_span_audit_sample60.md", audits, args.audit_sample_count, args.seed + 17)

    component_counts = Counter(row["dataset_component"] for row in converted)
    span_kind_counts = Counter(span["kind"] for row in audits for span in row["selected_spans"])
    base_weight_counts = Counter(str(row["loss_weight_hints"]["base_weight"]) for row in converted)
    category_counts = Counter(v1.norm_text(row.get("primary_category")) for row in converted)
    category_span_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in audits:
        if row["dataset_component"] != "strict_gold_token_weighted_specificity_v2":
            continue
        bucket = umbrella_category(str(row.get("primary_category") or ""))
        for span in row["selected_spans"]:
            category_span_counts[bucket][span["kind"]] += 1

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "scripts/build_q5_natural_language_token_weighted_gold_v2_dataset.py",
        "purpose": "q5 promoted sidecar diagnostic with specificity-preserving deterministic assistant-token weighting.",
        "output_path": str(output_path),
        "weighted_span_audit": str(args.output_dir / "weighted_span_audit.jsonl"),
        "weighted_span_audit_sample": str(args.output_dir / "weighted_span_audit_sample60.md"),
        "prompt_spec": str(args.prompt_spec),
        "prompt_version": prompt_spec["prompt_version"],
        "input_mode": "title_abstract",
        "row_count": len(converted),
        "component_counts": dict(sorted(component_counts.items())),
        "base_weight_counts": dict(sorted(base_weight_counts.items())),
        "span_kind_counts": dict(sorted(span_kind_counts.items())),
        "category_span_kind_counts": {key: dict(sorted(value.items())) for key, value in sorted(category_span_counts.items())},
        "v2_audit_counts": {
            "shared_key_term_spans": span_kind_counts.get("source_backed_shared_key_term", 0),
            "domain_object_term_spans": span_kind_counts.get("domain_object_term", 0),
            "under10_base_capped_rows": sum(1 for row in gold_audits if row["under10_base_capped"]),
        },
        "weights": {
            "strict_gold_default": args.gold_weight,
            "strict_gold_title_overlap_risk": args.risky_gold_weight,
            "strict_gold_title_overlap_risk_zero_span_fallback": args.risky_fallback_weight,
            "strict_gold_under10_base_cap": args.under10_base_cap,
            "faithfulness_anchor": args.anchor_weight,
            "polarity_caveat": 2.5,
            "scope_qualifier": 2.1,
            "math_result_shape": 2.0,
            "domain_object_term": 1.65,
            "payoff_verb": 1.6,
            "source_specific_anchor": 1.55,
            "source_backed_shared_key_term": 1.35,
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
            "title_trigram_span_check": "passed",
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
        "# q5 994-Gold Specificity-Preserving Token-Weighted Dataset v2",
        "",
        f"- Rows: {len(converted)}",
        f"- Strict gold rows: {component_counts['strict_gold_token_weighted_specificity_v2']}",
        f"- Faithfulness anchors: {component_counts['faithfulness_anchor']} at base weight {args.anchor_weight:g}",
        f"- Prompt: `{prompt_spec['prompt_version']}` / `title_abstract`",
        f"- Title-risk gold rows: {len(risky_gold)}",
        f"- Risky zero-span fallbacks: {len(risky_fallbacks)} ({fallback_frac:.2%})",
        f"- Under-10 base-capped rows: {manifest['v2_audit_counts']['under10_base_capped_rows']}",
        "",
        "Gold rows use deterministic assistant-token spans that prioritize source-backed specificity.",
        "Anchor rows are low-weight replay targets with no spans.",
        "See `weighted_span_audit.jsonl` and `weighted_span_audit_sample60.md` before training.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(f"[token-weighted-gold-v2] wrote {len(converted)} rows to {output_path}")
    print(f"[token-weighted-gold-v2] components={dict(sorted(component_counts.items()))}")
    print(f"[token-weighted-gold-v2] span kinds={dict(sorted(span_kind_counts.items()))}")
    print(
        "[token-weighted-gold-v2] risky fallback="
        f"{len(risky_fallbacks)}/{len(risky_gold)} ({fallback_frac:.2%})"
    )
    print(f"[token-weighted-gold-v2] under10 capped={manifest['v2_audit_counts']['under10_base_capped_rows']}")


if __name__ == "__main__":
    main()
