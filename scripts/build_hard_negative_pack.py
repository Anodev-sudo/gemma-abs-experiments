#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synthetic_summary_runtime import (
    build_messages,
    iter_bounded_futures,
    load_dotenv,
    load_jsonl,
    load_prompt_spec,
    local_filter_flags,
    normalize_text,
    render_prompt,
    request_json_with_retries,
    summary_flags,
    word_count,
    write_jsonl,
)

DEFAULT_RELEASE_DIR = "hf_dataset/releases/recent_3y_balanced_5000_pilot_v2"
DEFAULT_RUN_DIR = "hf_bucket/runs/recent_3y_balanced_5000_pilot_v2"
DEFAULT_HOLDOUT_REVIEW_JSONL = (
    "hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_v1/"
    "review_samples/holdout_100_20260419-075505/generations.jsonl"
)
DEFAULT_SEED_PAPERS_JSON = "experiments/hard_negative_pack_v1/semantic_seed_papers_v1.json"
DEFAULT_NEGATIVE_GENERATOR_PROMPT = (
    "prompts/synthetic_summary/deepseek_v32_hard_negative_generator_v2_precision.json"
)
DEFAULT_NEGATIVE_VERIFIER_PROMPT = (
    "prompts/synthetic_summary/qwen3_235b_a22b_2507_hard_negative_verifier_v6_precision_numeric.json"
)
DEFAULT_GOLD_REWRITE_PROMPT = "prompts/synthetic_summary/deepseek_v32_reviewer_v3.json"
DEFAULT_GOLD_VERIFIER_PROMPT = (
    "prompts/synthetic_summary/qwen3_235b_a22b_2507_reviewer_classifier_v3_production.json"
)
DEFAULT_CORRECTION_PROMPT = "prompts/mlx_lora/gemma_270m_it_summary_correction_v1.json"
DEFAULT_OUTPUT_ROOT = "hf_bucket/experiments/hard_negative_pack_v1"
DEFAULT_GENERATOR_MODEL = "deepseek/deepseek-v3.2"
DEFAULT_VERIFIER_MODEL = "qwen/qwen3-235b-a22b-2507"
DEFAULT_GOLD_MODEL = "deepseek/deepseek-v3.2"
DEFAULT_GENERATION_MAX_OUTPUT_TOKENS = 64
DEFAULT_VERIFICATION_MAX_OUTPUT_TOKENS = 96
DEFAULT_GOLD_MAX_OUTPUT_TOKENS = 96
DEFAULT_REQUEST_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 4
DEFAULT_RETRY_BACKOFF_SECONDS = 5.0
DEFAULT_VALID_RATIO = 0.05
DEFAULT_TEST_RATIO = 0.05
DEFAULT_SEED = 20260420
DEFAULT_ATTEMPT_MULTIPLIER = 3
AXIS_GENERATION_ATTEMPT_MULTIPLIER: dict[str, int] = {
    "invented_number": 8,
    "side_detail_fixation": 5,
    "wrong_number_binding": 5,
    "wrong_entity": 5,
}

MINED_FIRST_AXES = (
    "title_restatement",
    "too_generic",
    "side_detail_fixation",
)
GENERATED_FIRST_AXES = (
    "wrong_number_binding",
    "invented_number",
    "reversed_polarity",
    "wrong_entity",
)
ALL_AXES = MINED_FIRST_AXES + GENERATED_FIRST_AXES
STRUCTURAL_AXES = (
    "too_long",
    "too_short",
    "not_single_sentence",
    "formula_heavy",
)
ALL_KNOWN_AXES = ALL_AXES + STRUCTURAL_AXES
AXIS_REQUIRES_SOURCE_NUMBERS = {"wrong_number_binding"}
AXIS_NUMERIC_POSTCHECKS = {"wrong_number_binding", "invented_number"}
REGENERATED_MINED_AXES = ("side_detail_fixation",)
SEMANTIC_AXES = frozenset(GENERATED_FIRST_AXES)
DERIVED_SEMANTIC_SOURCE_AXES = ("too_generic", "side_detail_fixation")
DERIVED_SEMANTIC_TARGET_AXES = ("reversed_polarity", "wrong_entity")
GENERATED_AXIS_SOURCE_PRIORITY: dict[str, dict[str, int]] = {
    "wrong_number_binding": {
        "release_training": 0,
        "holdout_review_100": 1,
    },
    "invented_number": {
        "holdout_review_100": 0,
        "release_training": 1,
    },
    "reversed_polarity": {
        "derived_mined_semantic": 0,
        "holdout_review_100": 1,
        "release_training": 2,
    },
    "wrong_entity": {
        "derived_mined_semantic": 0,
        "holdout_review_100": 1,
        "release_training": 2,
    },
}
AXIS_DETAIL_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "using",
        "via",
        "with",
    }
)
TOKEN_STRIP_CHARS = ".,;:!?()[]{}<>\"'`“”‘’"
YEAR_LIKE_RE = re.compile(r"^(?:18|19|20)\d{2}$")
FORMULA_LIKE_NUMBER_RE = re.compile(r"(?:\^|=|<=|>=|!=|\([^)]*\d[^)]*\)|\b\d+\s*/\s*\w)")
SCIENTIFIC_NOTATION_RE = re.compile(r"^\d+(?:\.\d+)?e[+-]?\d+$", re.IGNORECASE)
NUMBER_WORD_TO_CANONICAL = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
}


def parse_axis_list(raw_value: str | None, default_axes: tuple[str, ...]) -> list[str]:
    if not raw_value:
        return list(default_axes)
    axes = [item.strip() for item in raw_value.split(",") if item.strip()]
    unknown = sorted(set(axes) - set(ALL_AXES))
    if unknown:
        raise ValueError(f"Unknown failure axes: {', '.join(unknown)}")
    return axes


def generation_attempt_cap(axis: str, needed: int) -> int:
    multiplier = AXIS_GENERATION_ATTEMPT_MULTIPLIER.get(axis, DEFAULT_ATTEMPT_MULTIPLIER)
    return max(needed * multiplier, needed)


def training_fallback_sort_key(axis: str, row: dict[str, Any]) -> tuple[Any, ...]:
    title_text = str(row.get("title", ""))
    abstract_text = str(row.get("abstract", ""))
    summary_text = str(row.get("summary", ""))
    source_text = " ".join([title_text, abstract_text, summary_text])
    numeric_count = len(extract_numeric_mentions(source_text))
    summary_numeric_count = len(meaningful_numeric_mentions(summary_text))
    source_numeric_count = len(meaningful_numeric_mentions(" ".join([title_text, abstract_text])))
    if axis == "invented_number":
        return (
            0 if summary_numeric_count == 0 else 1,
            summary_numeric_count,
            -source_numeric_count,
            numeric_count,
            str(row.get("paper_id", "")),
        )
    if axis == "wrong_number_binding":
        return (-numeric_count, str(row.get("paper_id", "")))
    return (str(row.get("paper_id", "")),)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a first-pass hard-negative pack for summary correction fine-tuning."
        )
    )
    parser.add_argument(
        "--release-dir",
        default=DEFAULT_RELEASE_DIR,
        help="Promoted release directory containing training_rows.jsonl.",
    )
    parser.add_argument(
        "--run-dir",
        default=DEFAULT_RUN_DIR,
        help="Pipeline run directory containing review_outputs.jsonl and dropped_rows.jsonl.",
    )
    parser.add_argument(
        "--seed-papers-json",
        default=DEFAULT_SEED_PAPERS_JSON,
        help="Tracked semantic seed-paper config JSON.",
    )
    parser.add_argument(
        "--negative-generator-prompt",
        default=DEFAULT_NEGATIVE_GENERATOR_PROMPT,
        help="Prompt spec JSON for semantic hard-negative generation.",
    )
    parser.add_argument(
        "--negative-verifier-prompt",
        default=DEFAULT_NEGATIVE_VERIFIER_PROMPT,
        help="Prompt spec JSON for hard-negative verification.",
    )
    parser.add_argument(
        "--gold-rewrite-prompt",
        default=DEFAULT_GOLD_REWRITE_PROMPT,
        help="Prompt spec JSON for generating corrected gold summaries.",
    )
    parser.add_argument(
        "--generator-model",
        default=None,
        help="Model override for semantic bad-summary generation.",
    )
    parser.add_argument(
        "--verifier-model",
        default=None,
        help="Model override for hard-negative verification and gold gating.",
    )
    parser.add_argument(
        "--gold-model",
        default=None,
        help="Model override for corrected gold-summary generation.",
    )
    parser.add_argument(
        "--target-per-axis",
        type=int,
        default=100,
        help="Target number of verified rows per failure axis.",
    )
    parser.add_argument(
        "--generated-axes",
        default=",".join(GENERATED_FIRST_AXES),
        help="Comma-separated generated-first axes to target.",
    )
    parser.add_argument(
        "--mined-axes",
        default=",".join(MINED_FIRST_AXES),
        help="Comma-separated mined-first axes to target.",
    )
    parser.add_argument(
        "--generation-workers",
        type=int,
        default=2,
        help="Maximum concurrent generation or gold-rewrite requests.",
    )
    parser.add_argument(
        "--verification-workers",
        type=int,
        default=2,
        help="Maximum concurrent verifier requests.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare pools and manifests without making model calls.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Explicit output directory. Defaults to a timestamped run under hf_bucket/experiments.",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv(
            "OPENROUTER_BASE_URL",
            os.getenv("DEEPSEEK_BASE_URL", "https://openrouter.ai/api/v1"),
        ),
        help="Base URL for an OpenAI-compatible chat completions API.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing the API key.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="HTTP timeout in seconds for each request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum retries for retryable request failures.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_SECONDS,
        help="Base retry backoff in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic seed for shuffling, sampling, and splits.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.target_per_axis <= 0:
        raise ValueError("target_per_axis must be positive")
    if args.generation_workers <= 0:
        raise ValueError("generation_workers must be positive")
    if args.verification_workers <= 0:
        raise ValueError("verification_workers must be positive")
    args.generated_axes = parse_axis_list(args.generated_axes, GENERATED_FIRST_AXES)
    args.mined_axes = parse_axis_list(args.mined_axes, MINED_FIRST_AXES)
    overlap = sorted(set(args.generated_axes) & set(args.mined_axes))
    if overlap:
        raise ValueError(f"Axes cannot be both generated-first and mined-first: {', '.join(overlap)}")
    args.generator_model = args.generator_model or DEFAULT_GENERATOR_MODEL
    args.verifier_model = args.verifier_model or DEFAULT_VERIFIER_MODEL
    args.gold_model = args.gold_model or DEFAULT_GOLD_MODEL


def resolve_output_dir(output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path(DEFAULT_OUTPUT_ROOT) / f"hard_negative_pack_v1_{timestamp}"


def load_correction_prompt_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = ("prompt_version", "system_prompt", "user_prompt_template")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(
            f"Correction prompt spec {path} is missing required keys: {', '.join(missing)}"
        )
    return spec


def contains_digit(value: str) -> bool:
    return bool(re.search(r"\d", value))


def normalize_axis_name(value: Any) -> str:
    axis = normalize_text(str(value)).casefold()
    if axis in ALL_KNOWN_AXES or axis == "none":
        return axis
    return ""


def normalize_axis_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        axis = normalize_axis_name(item)
        if not axis or axis == "none" or axis in seen:
            continue
        normalized.append(axis)
        seen.add(axis)
    return normalized


def axis_word_tokens(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", normalize_text(value).casefold())


def axis_content_tokens(value: str) -> list[str]:
    return [
        token
        for token in axis_word_tokens(value)
        if token not in AXIS_DETAIL_STOPWORDS and not token.isdigit()
    ]


def shared_prefix_length(left: list[str], right: list[str]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def shared_suffix_length(left: list[str], right: list[str]) -> int:
    count = 0
    for left_token, right_token in zip(reversed(left), reversed(right)):
        if left_token != right_token:
            break
        count += 1
    return count


def overlap_ratio(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def coverage_ratio(source: set[str], reference: set[str]) -> float:
    if not source or not reference:
        return 0.0
    return len(source & reference) / len(source)


def rough_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    for raw in normalize_text(value).split():
        token = raw.strip(TOKEN_STRIP_CHARS)
        if token:
            tokens.append(token)
    return tokens


def canonical_numeric_token(token: str) -> str | None:
    cleaned = token.casefold().strip(TOKEN_STRIP_CHARS).replace(",", "")
    if not cleaned:
        return None
    ordinal_match = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", cleaned)
    if ordinal_match:
        return ordinal_match.group(1)
    mapped = NUMBER_WORD_TO_CANONICAL.get(cleaned)
    if mapped is not None:
        return mapped
    if re.fullmatch(r"\d+(?:\.\d+)?%?", cleaned):
        return cleaned
    numeric_prefix_match = re.match(r"(\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?%?)", cleaned)
    if numeric_prefix_match:
        return numeric_prefix_match.group(1)
    if any(char.isdigit() for char in cleaned):
        return cleaned
    return None


def extract_numeric_mentions(value: str) -> list[dict[str, Any]]:
    tokens = rough_tokens(value)
    mentions: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        canonical = canonical_numeric_token(token)
        if canonical is None:
            continue
        phrase_variants: set[str] = set()
        for left in range(max(0, index - 2), index + 1):
            for right in range(index + 1, min(len(tokens), index + 3) + 1):
                phrase_tokens = tokens[left:right]
                if len(phrase_tokens) < 3:
                    continue
                phrase_variants.add(normalize_text(" ".join(phrase_tokens)).casefold())
        mentions.append(
            {
                "canonical": canonical,
                "raw": token,
                "phrase_variants": sorted(phrase_variants),
            }
        )
    return mentions


def numeric_token_set(value: str) -> set[str]:
    return {mention["canonical"] for mention in extract_numeric_mentions(value)}


def tokenized_meaningful_numeric_mentions(value: str) -> list[dict[str, Any]]:
    tokens = normalize_text(value).split()
    mentions: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        cleaned = token.strip(TOKEN_STRIP_CHARS)
        canonical = canonical_numeric_token(cleaned)
        if canonical is None:
            continue
        if YEAR_LIKE_RE.fullmatch(str(canonical)):
            continue
        if is_formula_like_numeric_text(cleaned):
            continue
        mentions.append(
            {
                "index": index,
                "raw": cleaned,
                "canonical": str(canonical),
                "prev_token": tokens[index - 1].strip(TOKEN_STRIP_CHARS).casefold()
                if index > 0
                else "",
                "next_token": tokens[index + 1].strip(TOKEN_STRIP_CHARS).casefold()
                if index + 1 < len(tokens)
                else "",
            }
        )
    return mentions


def numeric_style(raw: str) -> str:
    cleaned = raw.casefold().strip(TOKEN_STRIP_CHARS).replace(",", "")
    if not cleaned:
        return "other"
    if SCIENTIFIC_NOTATION_RE.fullmatch(cleaned):
        return "scientific"
    if cleaned.endswith("%"):
        return "percent"
    if cleaned.endswith("x"):
        return "multiplier"
    if "-" in cleaned and any(char.isdigit() for char in cleaned):
        return "range"
    if cleaned.endswith(("+", "k", "m")):
        return "compact"
    if re.fullmatch(r"\d+\.\d+", cleaned):
        return "decimal"
    if re.fullmatch(r"\d+", cleaned):
        return "integer"
    return "other"


def numeric_styles_compatible(receiver_raw: str, donor_raw: str) -> bool:
    receiver_style = numeric_style(receiver_raw)
    donor_style = numeric_style(donor_raw)
    if receiver_style == donor_style:
        return True
    scalar_styles = {"integer", "decimal", "compact"}
    return receiver_style in scalar_styles and donor_style in scalar_styles


def structured_wrong_number_binding_candidate(seed_row: dict[str, Any]) -> dict[str, Any] | None:
    gold_summary = normalize_text(str(seed_row.get("gold_summary", "")))
    title = str(seed_row.get("title", ""))
    abstract = str(seed_row.get("abstract", ""))
    if not gold_summary:
        return None

    summary_tokens = normalize_text(gold_summary).split()
    receiver_mentions = tokenized_meaningful_numeric_mentions(gold_summary)
    if not receiver_mentions:
        return None

    source_text = " ".join([title, abstract])
    donor_mentions = tokenized_meaningful_numeric_mentions(source_text)
    if not donor_mentions:
        return None

    summary_canonicals = {mention["canonical"] for mention in receiver_mentions}
    candidate_rows: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    seen_summaries: set[str] = set()

    for receiver in receiver_mentions:
        for donor in donor_mentions:
            if donor["canonical"] == receiver["canonical"]:
                continue
            if numeric_style(donor["raw"]) == "scientific":
                continue
            if not numeric_styles_compatible(receiver["raw"], donor["raw"]):
                continue
            new_tokens = list(summary_tokens)
            new_tokens[receiver["index"]] = donor["raw"]
            bad_summary = normalize_text(" ".join(new_tokens))
            if not bad_summary or bad_summary == gold_summary:
                continue
            if bad_summary in seen_summaries:
                continue
            seen_summaries.add(bad_summary)
            candidate_row = {
                "paper_id": seed_row["paper_id"],
                "title": title,
                "abstract": abstract,
                "gold_summary": gold_summary,
                "bad_summary": bad_summary,
                "failure_axis": "wrong_number_binding",
            }
            postcheck_reason = axis_postcheck_reason(candidate_row)
            if postcheck_reason:
                continue
            score = (
                0 if donor["canonical"] not in summary_canonicals else 1,
                0 if donor["next_token"] != receiver["next_token"] else 1,
                0 if donor["prev_token"] != receiver["prev_token"] else 1,
                abs(len(donor["raw"]) - len(receiver["raw"])),
                receiver["index"],
                donor["canonical"],
            )
            candidate_rows.append((score, candidate_row))

    if not candidate_rows:
        return None
    candidate_rows.sort(key=lambda item: item[0])
    return candidate_rows[0][1]


def seed_summary_text(seed_row: dict[str, Any]) -> str:
    return normalize_text(
        str(seed_row.get("gold_summary") or seed_row.get("seed_candidate_summary") or "")
    )


def source_text_for_seed(seed_row: dict[str, Any]) -> str:
    return " ".join(
        [
            str(seed_row.get("title", "")),
            str(seed_row.get("abstract", "")),
            seed_summary_text(seed_row),
        ]
    )


def is_formula_like_numeric_text(value: str) -> bool:
    text = normalize_text(value)
    if not text:
        return False
    return bool(FORMULA_LIKE_NUMBER_RE.search(text) or "<" in value or ">" in value)


def meaningful_numeric_mentions(value: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    for mention in extract_numeric_mentions(value):
        canonical = str(mention["canonical"])
        raw = str(mention["raw"])
        if YEAR_LIKE_RE.fullmatch(canonical):
            continue
        if is_formula_like_numeric_text(raw):
            continue
        mentions.append(mention)
    return mentions


def seed_numeric_profile(seed_row: dict[str, Any]) -> dict[str, Any]:
    summary_text = seed_summary_text(seed_row)
    source_text = source_text_for_seed(seed_row)
    source_mentions = meaningful_numeric_mentions(source_text)
    summary_mentions = meaningful_numeric_mentions(summary_text)
    source_distinct = {mention["canonical"] for mention in source_mentions}
    summary_distinct = {mention["canonical"] for mention in summary_mentions}
    return {
        "source_distinct_count": len(source_distinct),
        "summary_distinct_count": len(summary_distinct),
        "summary_local_flag_count": len(
            local_filter_flags(summary_text, str(seed_row.get("title", "")))
        )
        if summary_text
        else 0,
        "summary_formula_like": is_formula_like_numeric_text(summary_text),
        "has_direct_gold": bool(seed_row.get("gold_summary")),
    }


def seed_row_supports_generated_axis(seed_row: dict[str, Any], axis: str) -> bool:
    profile = seed_numeric_profile(seed_row)
    if axis in AXIS_NUMERIC_POSTCHECKS and profile["summary_formula_like"]:
        return False
    if axis == "wrong_number_binding":
        return (
            profile["summary_distinct_count"] >= 1
            and profile["source_distinct_count"] >= 2
        )
    if axis == "invented_number":
        return True
    return True


def axis_seed_sort_key(axis: str, seed_row: dict[str, Any]) -> tuple[Any, ...]:
    source_pool = str(seed_row.get("source_pool", ""))
    source_priority = GENERATED_AXIS_SOURCE_PRIORITY.get(axis, {}).get(source_pool, 99)
    paper_id = str(seed_row.get("paper_id", ""))
    profile = seed_numeric_profile(seed_row)
    if axis == "invented_number":
        return (
            0 if not profile["summary_formula_like"] else 1,
            0 if profile["summary_distinct_count"] == 0 else 1,
            source_priority,
            0 if seed_row.get("gold_summary") else 1,
            profile["summary_distinct_count"],
            -profile["source_distinct_count"],
            paper_id,
        )
    if axis == "wrong_number_binding":
        return (
            0 if seed_row_supports_generated_axis(seed_row, axis) else 1,
            0 if profile["has_direct_gold"] else 1,
            profile["summary_local_flag_count"],
            source_priority,
            -profile["summary_distinct_count"],
            -profile["source_distinct_count"],
            paper_id,
        )
    return (
        source_priority,
        0 if seed_row.get("gold_summary") else 1,
        paper_id,
    )


def order_seed_pool_for_axis(axis: str, pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if axis not in AXIS_NUMERIC_POSTCHECKS and axis not in GENERATED_AXIS_SOURCE_PRIORITY:
        return pool
    return sorted(pool, key=lambda row: axis_seed_sort_key(axis, row))


def find_supported_number_phrase(candidate_summary: str, source_text: str) -> str:
    source_signature = normalize_text(source_text).casefold()
    for mention in extract_numeric_mentions(candidate_summary):
        for phrase in mention["phrase_variants"]:
            if phrase and phrase in source_signature:
                return phrase
    return ""


def mention_has_supported_phrase(mention: dict[str, Any], source_text: str) -> bool:
    source_signature = normalize_text(source_text).casefold()
    return any(
        phrase and phrase in source_signature for phrase in mention.get("phrase_variants", [])
    )


def axis_postcheck_reason(candidate_row: dict[str, Any]) -> str:
    axis = str(candidate_row.get("failure_axis", ""))
    gold_summary = normalize_text(str(candidate_row.get("gold_summary", "")))
    bad_summary = normalize_text(str(candidate_row.get("bad_summary", "")))
    if not gold_summary or not bad_summary:
        return ""

    if axis in AXIS_NUMERIC_POSTCHECKS:
        if FORMULA_LIKE_NUMBER_RE.search(bad_summary):
            return "numeric axis used formula-like notation instead of a clean feed claim"
        mentions = meaningful_numeric_mentions(bad_summary)
        if not mentions:
            return "numeric axis lacked an explicit number"
        source_text = " ".join(
            [
                str(candidate_row.get("title", "")),
                str(candidate_row.get("abstract", "")),
                gold_summary,
            ]
        )
        source_tokens = {mention["canonical"] for mention in meaningful_numeric_mentions(source_text)}
        candidate_tokens = {mention["canonical"] for mention in mentions}
        if axis == "invented_number":
            if candidate_tokens & source_tokens:
                return "invented_number reused a source-supported number"
            return ""
        novel_tokens = candidate_tokens - source_tokens
        if novel_tokens:
            return "wrong_number_binding introduced a new number"
        if len(mentions) == 1 and YEAR_LIKE_RE.fullmatch(str(mentions[0]["canonical"])):
            return "wrong_number_binding used a source year instead of a misbound metric"
        unsupported_mentions = [
            mention
            for mention in mentions
            if not mention_has_supported_phrase(mention, source_text)
        ]
        if not unsupported_mentions:
            return "wrong_number_binding copied only source-supported number phrases"
        return ""

    gold_tokens = axis_word_tokens(gold_summary)
    bad_tokens = axis_word_tokens(bad_summary)
    if not gold_tokens or not bad_tokens:
        return ""

    gold_content = set(axis_content_tokens(gold_summary))
    bad_content = set(axis_content_tokens(bad_summary))
    title_content = set(axis_content_tokens(str(candidate_row.get("title", ""))))
    prefix_len = shared_prefix_length(gold_tokens, bad_tokens)
    suffix_len = shared_suffix_length(gold_tokens, bad_tokens)
    shorter_len = max(min(len(gold_tokens), len(bad_tokens)), 1)

    if axis == "side_detail_fixation":
        novel_content = bad_content - gold_content
        shared_content = gold_content & bad_content
        if prefix_len / shorter_len >= 0.8 or suffix_len / shorter_len >= 0.8:
            return "side_detail_fixation stayed too close to the gold framing"
        if len(novel_content) < 2 and len(shared_content) >= max(len(bad_content) - 1, 1):
            return "side_detail_fixation did not pivot to a secondary detail"
        return ""

    if axis == "wrong_entity":
        token_overlap = overlap_ratio(set(gold_tokens), set(bad_tokens))
        if token_overlap < 0.35:
            return "wrong_entity rewrote too much of the sentence frame"
        if max(prefix_len, suffix_len) < 2 and len(gold_content & bad_content) < 4:
            return "wrong_entity did not preserve the original sentence frame"
        return ""

    if axis == "title_restatement":
        title_content_coverage = coverage_ratio(bad_content, title_content)
        gold_content_coverage = coverage_ratio(bad_content, gold_content)
        extra_title_details = len(bad_content - title_content)
        if gold_content_coverage >= 0.75 and (
            title_content_coverage < 0.55 or extra_title_details >= 3
        ):
            return "title_restatement stayed too close to the gold summary"
        return ""

    if axis == "too_generic":
        title_content_coverage = coverage_ratio(bad_content, title_content)
        gold_content_coverage = coverage_ratio(bad_content, gold_content)
        missing_gold_details = len(gold_content - bad_content)
        if gold_content_coverage >= 0.75 and missing_gold_details <= 6:
            return "too_generic stayed too close to the gold summary"
        if (
            title_content_coverage <= 0.1
            and gold_content_coverage >= 0.5
            and len(bad_content) >= 7
        ):
            return "too_generic remained too specific"
        return ""

    return ""


def evaluate_negative_policy(
    *,
    candidate_row: dict[str, Any],
    parsed: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    requested_axis = str(candidate_row.get("failure_axis", ""))
    valid_negative = bool(parsed.get("valid_negative"))
    best_axis = normalize_axis_name(parsed.get("best_matching_axis", ""))
    axis_matches_requested = bool(parsed.get("failure_axis_matches"))
    reason_span = normalize_text(str(parsed.get("reason_span", ""))).casefold()

    if not bool(parsed.get("needs_correction", valid_negative)):
        failures.append("no_correction_needed")
    if "best_matching_axis" in parsed and best_axis != requested_axis:
        failures.append("best_axis_mismatch")
    if not axis_matches_requested:
        failures.append("requested_axis_not_primary")

    other_axes = [
        axis
        for axis in normalize_axis_list(parsed.get("other_major_issue_axes", []))
        if axis != requested_axis
    ]
    if other_axes:
        failures.append("multiple_axes")
    if not bool(parsed.get("single_axis_only", not other_axes)):
        failures.append("multiple_axes")

    if not bool(parsed.get("still_feed_style")):
        failures.append("not_feed_style")
    if not bool(parsed.get("meaningfully_differs_from_gold", True)):
        failures.append("too_close_to_gold")
    if requested_axis in SEMANTIC_AXES and bool(parsed.get("still_true_to_source", False)):
        failures.append("semantic_negative_still_true")
    if not valid_negative:
        failures.append("verifier_marked_invalid")
    if requested_axis == "wrong_entity":
        for axis in ALL_AXES:
            if axis == requested_axis:
                continue
            if axis in reason_span or axis.replace("_", " ") in reason_span:
                failures.append("reason_span_axis_mismatch")
                break
    if requested_axis in {"side_detail_fixation", "title_restatement", "too_generic"}:
        acceptable_markers = (
            "acceptable as written",
            "still acceptable",
            "acceptable final",
            "clear value beyond the title",
            "still adds clear value",
        )
        if any(marker in reason_span for marker in acceptable_markers):
            failures.append("still_acceptable_as_written")

    deduped_failures: list[str] = []
    for failure in failures:
        if failure not in deduped_failures:
            deduped_failures.append(failure)
    return deduped_failures


def build_policy_rejection_reason(
    *,
    parsed: dict[str, Any],
    policy_failures: list[str],
    axis_postcheck_reason: str,
) -> str:
    reason_span = normalize_text(str(parsed.get("reason_span", "")))
    details: list[str] = []
    if axis_postcheck_reason:
        details.append(axis_postcheck_reason)
    if "verifier_marked_invalid" in policy_failures:
        details.append("verifier marked invalid negative")
    if "no_correction_needed" in policy_failures:
        details.append("summary does not need correction")
    if "requested_axis_not_primary" in policy_failures:
        details.append("requested axis was not the main issue")
    if "best_axis_mismatch" in policy_failures:
        best_axis = normalize_axis_name(parsed.get("best_matching_axis", "")) or "none"
        details.append(f"best axis was {best_axis}")
    if "multiple_axes" in policy_failures:
        other_axes = normalize_axis_list(parsed.get("other_major_issue_axes", []))
        if other_axes:
            details.append(f"multiple axes: {', '.join(other_axes)}")
        else:
            details.append("multiple major issues")
    if "not_feed_style" in policy_failures:
        details.append("not feed style")
    if "too_close_to_gold" in policy_failures:
        details.append("too close to gold summary")
    if "semantic_negative_still_true" in policy_failures:
        details.append("semantic negative is still supported by title and abstract")
    if "reason_span_axis_mismatch" in policy_failures:
        details.append("reason span named another axis")
    if "still_acceptable_as_written" in policy_failures:
        details.append("verifier said the summary was still acceptable as written")

    if reason_span and details:
        return f"{reason_span}; {'; '.join(details)}"
    if reason_span:
        return reason_span
    return "; ".join(details)


def normalize_key(*parts: str) -> str:
    cleaned = [normalize_text(part).casefold() for part in parts]
    return "||".join(cleaned)


def candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    review_source = str(row.get("review_source", ""))
    review_rank = 0 if review_source == "model" else 1
    return (
        review_rank,
        str(row.get("paper_id", "")),
        normalize_text(str(row.get("bad_summary", ""))).casefold(),
    )


def sample_rows_for_axis(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    shuffled = sorted(rows, key=lambda row: str(row.get("paper_id", "")))
    random.Random(seed).shuffle(shuffled)
    return shuffled[: min(count, len(shuffled))]


def split_rows(
    rows: list[dict[str, Any]],
    seed: int,
    valid_ratio: float = DEFAULT_VALID_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    valid_count = round(len(shuffled) * valid_ratio)
    test_count = round(len(shuffled) * test_ratio)
    train_count = len(shuffled) - valid_count - test_count
    train_rows = shuffled[:train_count]
    valid_rows = shuffled[train_count : train_count + valid_count]
    test_rows = shuffled[train_count + valid_count : train_count + valid_count + test_count]
    return train_rows, valid_rows, test_rows


def build_correction_user_prompt(
    row: dict[str, Any],
    prompt_spec: dict[str, Any],
) -> str:
    return prompt_spec["user_prompt_template"].format(
        title=str(row["title"]).strip(),
        abstract=str(row["abstract"]).strip(),
        bad_summary=str(row["bad_summary"]).strip(),
    )


def build_correction_chat_example(
    row: dict[str, Any],
    prompt_spec: dict[str, Any],
) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": prompt_spec["system_prompt"],
            },
            {
                "role": "user",
                "content": build_correction_user_prompt(row, prompt_spec),
            },
            {
                "role": "assistant",
                "content": str(row["gold_summary"]).strip(),
            },
        ],
        "paper_id": row["paper_id"],
        "failure_axis": row["failure_axis"],
        "source_kind": row["source_kind"],
        "gold_source": row["gold_source"],
        "prompt_version": prompt_spec["prompt_version"],
    }


def request_prompt_json(
    *,
    spec: dict[str, Any],
    model: str,
    api_base: str,
    api_key: str,
    timeout: int,
    max_retries: int,
    retry_backoff_seconds: float,
    max_output_tokens: int,
    record: dict[str, Any],
    candidate_summary: str | None = None,
    extra_variables: dict[str, Any] | None = None,
    label: str,
) -> dict[str, Any]:
    prompt = render_prompt(
        spec,
        record,
        candidate_summary=candidate_summary,
        extra_variables=extra_variables,
    )
    parsed, raw_text, raw_response = request_json_with_retries(
        label=label,
        api_base=api_base,
        api_key=api_key,
        model=model,
        messages=build_messages(spec, prompt),
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        response_format=spec.get("response_format"),
        plugins=spec.get("plugins"),
        assistant_prefill=spec.get("assistant_prefill"),
    )
    return {
        "parsed": parsed,
        "raw_text": raw_text,
        "api_response_id": raw_response.get("id"),
    }


def map_drop_to_axis(review_row: dict[str, Any]) -> str | None:
    drop_reason = normalize_text(str(review_row.get("drop_reason", ""))).casefold()
    local_flags = set(review_row.get("local_filter_flags", []))
    if "formula_shorthand" in local_flags or "formula-heavy notation" in drop_reason:
        return "formula_heavy"
    if "too_long" in local_flags or "over 15 words" in drop_reason:
        return "too_long"
    if "too_short" in local_flags or drop_reason == "too short":
        return "too_short"
    if "not_single_sentence" in local_flags or "not exactly one sentence" in drop_reason:
        return "not_single_sentence"
    if "title_duplicate" in local_flags or "title" in drop_reason:
        return "title_restatement"
    if "too vague" in drop_reason or "too generic" in drop_reason:
        return "too_generic"
    if "misses the main payoff" in drop_reason or "too narrow" in drop_reason:
        return "side_detail_fixation"
    return None


def build_natural_candidate_pool(
    *,
    review_outputs: list[dict[str, Any]],
    dropped_rows: list[dict[str, Any]],
    run_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dropped_by_id = {row["paper_id"]: row for row in dropped_rows}
    axis_counts: Counter[str] = Counter()
    structural_counts: Counter[str] = Counter()
    unmapped_drop_reasons: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []

    for review_row in review_outputs:
        if review_row.get("decision") != "drop":
            continue
        paper_id = str(review_row["paper_id"])
        dropped = dropped_by_id.get(paper_id)
        if dropped is None:
            unmapped_drop_reasons["missing_dropped_row"] += 1
            continue
        axis = map_drop_to_axis(review_row)
        if axis is None:
            unmapped_drop_reasons[normalize_text(str(review_row.get("drop_reason", ""))) or "unmapped"] += 1
            continue
        if axis in STRUCTURAL_AXES:
            structural_counts[axis] += 1
            continue
        axis_counts[axis] += 1
        candidates.append(
            {
                "paper_id": paper_id,
                "title": dropped["title"],
                "abstract": dropped["abstract"],
                "primary_category": dropped.get("primary_category"),
                "categories": dropped.get("categories", []),
                "created_date": dropped.get("created_date"),
                "bad_summary": normalize_text(str(review_row.get("candidate_summary", ""))),
                "failure_axis": axis,
                "source_kind": "mined",
                "source_run": str(run_dir),
                "candidate_origin": (
                    f"{review_row.get('review_source', 'unknown')}:"
                    f"{normalize_text(str(review_row.get('drop_reason', '')))}"
                ),
                "teacher_model": dropped.get("teacher_model", ""),
                "generator_prompt_version": dropped.get("generator_prompt_version", ""),
                "verifier_prompt_version": "",
                "gold_source": "",
                "gold_summary": "",
                "notes": normalize_text(str(review_row.get("drop_reason", ""))),
                "review_source": review_row.get("review_source", ""),
                "review_prompt_version": dropped.get("review_prompt_version", ""),
                "review_drop_reason": review_row.get("drop_reason", ""),
                "local_filter_flags": review_row.get("local_filter_flags", []),
            }
        )
    candidates.sort(key=candidate_sort_key)
    stats = {
        "natural_pool_axis_counts": dict(sorted(axis_counts.items())),
        "structural_drop_counts": dict(sorted(structural_counts.items())),
        "unmapped_drop_reasons": dict(sorted(unmapped_drop_reasons.items())),
    }
    return candidates, stats


def load_seed_entries(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Seed config {path} must contain a top-level array.")
    for index, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Seed config row {index} must be an object.")
        if not row.get("paper_id"):
            raise ValueError(f"Seed config row {index} is missing paper_id.")
        preferred_axes = row.get("preferred_axes", [])
        unknown_axes = sorted(set(preferred_axes) - set(ALL_AXES))
        if unknown_axes:
            raise ValueError(
                f"Seed config row {index} includes unknown axes: {', '.join(unknown_axes)}"
            )
    return data


def build_holdout_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    return {row["paper_id"]: row for row in rows}


def build_seed_pool(
    *,
    seed_entries: list[dict[str, Any]],
    training_by_id: dict[str, dict[str, Any]],
    holdout_by_id: dict[str, dict[str, Any]],
    holdout_path: Path,
    release_dir: Path,
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for entry in seed_entries:
        paper_id = str(entry["paper_id"])
        source_pool = str(entry.get("source_pool", "holdout_review_100"))
        preferred_axes = list(entry.get("preferred_axes", []))
        if source_pool == "release_training":
            row = training_by_id.get(paper_id)
            if row is None:
                raise ValueError(
                    f"Seed paper {paper_id} references release_training but was not found in training_rows.jsonl"
                )
            resolved.append(
                {
                    "paper_id": paper_id,
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "primary_category": row.get("primary_category"),
                    "categories": row.get("categories", []),
                    "created_date": row.get("created_date"),
                    "gold_summary": normalize_text(str(row["summary"])),
                    "gold_source": "release_training_row",
                    "teacher_model": row.get("teacher_model", ""),
                    "seed_candidate_summary": "",
                    "source_run": str(release_dir),
                    "source_pool": source_pool,
                    "preferred_axes": preferred_axes,
                    "notes": str(entry.get("notes", "")),
                }
            )
            continue
        if source_pool != "holdout_review_100":
            raise ValueError(f"Unsupported source_pool in seed config: {source_pool}")
        holdout_row = holdout_by_id.get(paper_id)
        if holdout_row is None:
            raise ValueError(
                f"Seed paper {paper_id} was not found in holdout review set {holdout_path}"
            )
        summary_field = str(entry.get("candidate_summary_source", "tuned_summary"))
        seed_candidate_summary = normalize_text(str(holdout_row.get(summary_field, "")))
        if not seed_candidate_summary:
            fallback_fields = ("tuned_summary", "base_summary")
            for field in fallback_fields:
                seed_candidate_summary = normalize_text(str(holdout_row.get(field, "")))
                if seed_candidate_summary:
                    break
        resolved.append(
            {
                "paper_id": paper_id,
                "title": holdout_row["title"],
                "abstract": holdout_row["abstract"],
                "primary_category": holdout_row.get("primary_category"),
                "categories": holdout_row.get("categories", []),
                "created_date": holdout_row.get("created_date"),
                "gold_summary": "",
                "gold_source": "",
                "teacher_model": "",
                "seed_candidate_summary": seed_candidate_summary,
                "source_run": str(holdout_path),
                "source_pool": source_pool,
                "preferred_axes": preferred_axes,
                "notes": str(entry.get("notes", "")),
            }
        )
    resolved.sort(key=lambda row: (str(row.get("paper_id", "")), str(row.get("source_pool", ""))))
    return resolved


def build_derived_semantic_seed_pool(
    natural_pool: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    derived_rows: list[dict[str, Any]] = []
    source_axis_counts: Counter[str] = Counter()
    unique_source_axis_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    prioritized = sorted(natural_pool, key=candidate_sort_key)
    for row in prioritized:
        source_axis = str(row.get("failure_axis", ""))
        if source_axis not in DERIVED_SEMANTIC_SOURCE_AXES:
            continue
        source_axis_counts[source_axis] += 1
        paper_id = str(row["paper_id"])
        if paper_id in seen_ids:
            continue
        seed_candidate_summary = normalize_text(str(row.get("bad_summary", "")))
        if not seed_candidate_summary:
            continue
        derived_rows.append(
            {
                "paper_id": paper_id,
                "title": row["title"],
                "abstract": row["abstract"],
                "primary_category": row.get("primary_category"),
                "categories": row.get("categories", []),
                "created_date": row.get("created_date"),
                "gold_summary": "",
                "gold_source": "",
                "teacher_model": "",
                "seed_candidate_summary": seed_candidate_summary,
                "source_run": row["source_run"],
                "source_pool": "derived_mined_semantic",
                "preferred_axes": list(DERIVED_SEMANTIC_TARGET_AXES),
                "notes": f"derived from {source_axis}: {normalize_text(str(row.get('notes', '')))}".strip(": "),
                "derived_from_failure_axis": source_axis,
                "derived_from_candidate_origin": row.get("candidate_origin", ""),
            }
        )
        unique_source_axis_counts[source_axis] += 1
        seen_ids.add(paper_id)
    stats = {
        "derived_semantic_seed_rows": len(derived_rows),
        "derived_semantic_source_axis_counts": dict(sorted(source_axis_counts.items())),
        "derived_semantic_unique_source_axis_counts": dict(
            sorted(unique_source_axis_counts.items())
        ),
    }
    return derived_rows, stats


def select_mined_candidates(
    natural_pool: list[dict[str, Any]],
    mined_axes: list[str],
    target_per_axis: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_axis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in natural_pool:
        by_axis[row["failure_axis"]].append(row)
    for axis in mined_axes:
        limit_per_axis = generation_attempt_cap(axis, target_per_axis)
        selected.extend(by_axis.get(axis, [])[:limit_per_axis])
    return selected


def build_regenerated_mined_seed_pool(
    selected_natural_candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    seed_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    axis_counts: Counter[str] = Counter()
    seen_ids: set[tuple[str, str]] = set()
    for candidate in selected_natural_candidates:
        axis = str(candidate.get("failure_axis", ""))
        if axis not in REGENERATED_MINED_AXES:
            continue
        paper_id = str(candidate["paper_id"])
        dedupe_key = (paper_id, axis)
        if dedupe_key in seen_ids:
            continue
        seen_ids.add(dedupe_key)
        axis_counts[axis] += 1
        seed_rows.append(
            {
                "paper_id": paper_id,
                "title": candidate["title"],
                "abstract": candidate["abstract"],
                "primary_category": candidate.get("primary_category"),
                "categories": candidate.get("categories", []),
                "created_date": candidate.get("created_date"),
                "gold_summary": "",
                "gold_source": "",
                "teacher_model": "",
                "seed_candidate_summary": normalize_text(str(candidate.get("bad_summary", ""))),
                "source_run": candidate["source_run"],
                "source_pool": "regenerated_mined_seed",
                "preferred_axes": [axis],
                "notes": normalize_text(str(candidate.get("notes", ""))),
                "derived_from_failure_axis": axis,
                "derived_from_candidate_origin": candidate.get("candidate_origin", ""),
            }
        )
        audit_rows.append(
            {
                **candidate,
                "gold_summary": "",
                "gold_source": "",
                "status": "converted_to_seed",
                "accepted": False,
                "rejection_reason": "regenerated from mined seed",
            }
        )
    stats = {
        "regenerated_mined_seed_rows": len(seed_rows),
        "regenerated_mined_seed_axis_counts": dict(sorted(axis_counts.items())),
    }
    return seed_rows, audit_rows, stats


def build_generated_seed_pool_by_axis(
    *,
    training_rows: list[dict[str, Any]],
    resolved_seed_pool: list[dict[str, Any]],
    regenerated_mined_seed_pool: list[dict[str, Any]],
    derived_seed_pool: list[dict[str, Any]],
    requested_axes: list[str],
    release_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    pool_by_axis: dict[str, list[dict[str, Any]]] = {}
    for axis in requested_axes:
        training_fallback = sorted(
            training_rows,
            key=lambda row: training_fallback_sort_key(axis, row),
        )
        seen_ids: set[str] = set()
        pool: list[dict[str, Any]] = []
        for seed_row in regenerated_mined_seed_pool:
            if axis not in seed_row.get("preferred_axes", []):
                continue
            if not seed_row_supports_generated_axis(seed_row, axis):
                continue
            paper_id = str(seed_row["paper_id"])
            if paper_id in seen_ids:
                continue
            pool.append(dict(seed_row))
            seen_ids.add(paper_id)
        for seed_row in resolved_seed_pool:
            if axis not in seed_row.get("preferred_axes", []):
                continue
            if not seed_row_supports_generated_axis(seed_row, axis):
                continue
            paper_id = str(seed_row["paper_id"])
            if paper_id in seen_ids:
                continue
            pool.append(dict(seed_row))
            seen_ids.add(paper_id)
        for seed_row in derived_seed_pool:
            if axis not in seed_row.get("preferred_axes", []):
                continue
            if not seed_row_supports_generated_axis(seed_row, axis):
                continue
            paper_id = str(seed_row["paper_id"])
            if paper_id in seen_ids:
                continue
            pool.append(dict(seed_row))
            seen_ids.add(paper_id)
        for row in training_fallback:
            paper_id = str(row["paper_id"])
            if paper_id in seen_ids:
                continue
            if axis in AXIS_REQUIRES_SOURCE_NUMBERS:
                joined = " ".join(
                    [
                        str(row.get("title", "")),
                        str(row.get("abstract", "")),
                        str(row.get("summary", "")),
                    ]
                )
                if not contains_digit(joined):
                    continue
            fallback_seed_row = {
                "paper_id": paper_id,
                "title": row["title"],
                "abstract": row["abstract"],
                "primary_category": row.get("primary_category"),
                "categories": row.get("categories", []),
                "created_date": row.get("created_date"),
                "gold_summary": normalize_text(str(row["summary"])),
                "gold_source": "release_training_row",
                "teacher_model": row.get("teacher_model", ""),
                "seed_candidate_summary": "",
                "source_run": str(release_dir),
                "source_pool": "release_training",
                "preferred_axes": [axis],
                "notes": "",
            }
            if not seed_row_supports_generated_axis(fallback_seed_row, axis):
                continue
            pool.append(fallback_seed_row)
            seen_ids.add(paper_id)
        pool_by_axis[axis] = order_seed_pool_for_axis(axis, pool)
    return pool_by_axis


def ensure_seed_gold(
    *,
    seed_row: dict[str, Any],
    gold_cache: dict[str, dict[str, Any]],
    gold_rewrite_spec: dict[str, Any],
    gold_verifier_spec: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
) -> dict[str, Any]:
    paper_id = str(seed_row["paper_id"])
    if paper_id in gold_cache:
        return gold_cache[paper_id]
    if seed_row.get("gold_summary"):
        result = {
            "paper_id": paper_id,
            "accepted": True,
            "gold_summary": normalize_text(str(seed_row["gold_summary"])),
            "gold_source": str(seed_row["gold_source"]),
            "teacher_model": str(seed_row.get("teacher_model", "")),
            "rewrite_prompt_version": "",
            "rewrite_model": "",
            "rewrite_raw_text": "",
            "rewrite_api_response_id": None,
            "gate_prompt_version": gold_verifier_spec["prompt_version"],
            "gate_model": args.verifier_model,
            "gate_raw_text": "",
            "gate_api_response_id": None,
            "gate_reason_span": "",
        }
        gold_cache[paper_id] = result
        return result

    candidate_summary = normalize_text(str(seed_row.get("seed_candidate_summary", "")))
    if not candidate_summary:
        result = {
            "paper_id": paper_id,
            "accepted": False,
            "gold_summary": "",
            "gold_source": "",
            "teacher_model": "",
            "rewrite_prompt_version": gold_rewrite_spec["prompt_version"],
            "rewrite_model": args.gold_model,
            "rewrite_raw_text": "",
            "rewrite_api_response_id": None,
            "gate_prompt_version": gold_verifier_spec["prompt_version"],
            "gate_model": args.verifier_model,
            "gate_raw_text": "",
            "gate_api_response_id": None,
            "gate_reason_span": "missing candidate summary for gold rewrite",
        }
        gold_cache[paper_id] = result
        return result

    rewrite_response = request_prompt_json(
        spec=gold_rewrite_spec,
        model=args.gold_model,
        api_base=args.api_base,
        api_key=api_key,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        max_output_tokens=DEFAULT_GOLD_MAX_OUTPUT_TOKENS,
        record=seed_row,
        candidate_summary=candidate_summary,
        label=f"gold-rewrite {paper_id}",
    )
    rewrite_decision = normalize_text(str(rewrite_response["parsed"].get("decision", "")).lower())
    gold_summary = normalize_text(str(rewrite_response["parsed"].get("summary", "")))
    if rewrite_decision not in {"rewrite", "accept"} or not gold_summary:
        result = {
            "paper_id": paper_id,
            "accepted": False,
            "gold_summary": "",
            "gold_source": "",
            "teacher_model": "",
            "rewrite_prompt_version": gold_rewrite_spec["prompt_version"],
            "rewrite_model": args.gold_model,
            "rewrite_raw_text": rewrite_response["raw_text"],
            "rewrite_api_response_id": rewrite_response["api_response_id"],
            "gate_prompt_version": gold_verifier_spec["prompt_version"],
            "gate_model": args.verifier_model,
            "gate_raw_text": "",
            "gate_api_response_id": None,
            "gate_reason_span": "gold rewrite did not yield a usable summary",
        }
        gold_cache[paper_id] = result
        return result

    gate_response = request_prompt_json(
        spec=gold_verifier_spec,
        model=args.verifier_model,
        api_base=args.api_base,
        api_key=api_key,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        max_output_tokens=DEFAULT_VERIFICATION_MAX_OUTPUT_TOKENS,
        record=seed_row,
        candidate_summary=gold_summary,
        label=f"gold-gate {paper_id}",
    )
    gate_decision = normalize_text(str(gate_response["parsed"].get("decision", "")).lower())
    accepted = gate_decision == "accept"
    result = {
        "paper_id": paper_id,
        "accepted": accepted,
        "gold_summary": gold_summary if accepted else "",
        "gold_source": "generated_correction" if accepted else "",
        "teacher_model": args.gold_model if accepted else "",
        "rewrite_prompt_version": gold_rewrite_spec["prompt_version"],
        "rewrite_model": args.gold_model,
        "rewrite_raw_text": rewrite_response["raw_text"],
        "rewrite_api_response_id": rewrite_response["api_response_id"],
        "gate_prompt_version": gold_verifier_spec["prompt_version"],
        "gate_model": args.verifier_model,
        "gate_raw_text": gate_response["raw_text"],
        "gate_api_response_id": gate_response["api_response_id"],
        "gate_reason_span": normalize_text(str(gate_response["parsed"].get("reason_span", ""))),
    }
    gold_cache[paper_id] = result
    return result


def verify_negative_candidate(
    *,
    candidate_row: dict[str, Any],
    verifier_spec: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
) -> dict[str, Any]:
    response = request_prompt_json(
        spec=verifier_spec,
        model=args.verifier_model,
        api_base=args.api_base,
        api_key=api_key,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        max_output_tokens=DEFAULT_VERIFICATION_MAX_OUTPUT_TOKENS,
        record=candidate_row,
        candidate_summary=str(candidate_row["bad_summary"]),
        extra_variables={
            "failure_axis": candidate_row["failure_axis"],
            "gold_summary": candidate_row["gold_summary"],
        },
        label=f"negative-verify {candidate_row['paper_id']} {candidate_row['failure_axis']}",
    )
    parsed = response["parsed"]
    policy_failures = evaluate_negative_policy(candidate_row=candidate_row, parsed=parsed)
    postcheck_reason = axis_postcheck_reason(candidate_row) if not policy_failures else ""
    accepted = not policy_failures and not postcheck_reason
    rejection_reason = build_policy_rejection_reason(
        parsed=parsed,
        policy_failures=policy_failures,
        axis_postcheck_reason=postcheck_reason,
    )
    return {
        **candidate_row,
        "verifier_prompt_version": verifier_spec["prompt_version"],
        "verification_model": args.verifier_model,
        "verification_raw_text": response["raw_text"],
        "verification_api_response_id": response["api_response_id"],
        "verification_parsed": parsed,
        "policy_rejection_flags": policy_failures,
        "axis_postcheck_reason": postcheck_reason,
        "accepted": accepted,
        "rejection_reason": "" if accepted else rejection_reason,
    }


def generate_bad_summary_candidate(
    *,
    seed_row: dict[str, Any],
    axis: str,
    generator_spec: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
) -> dict[str, Any]:
    if axis == "wrong_number_binding":
        structured_candidate = structured_wrong_number_binding_candidate(seed_row)
        if structured_candidate is not None:
            return {
                "paper_id": seed_row["paper_id"],
                "title": seed_row["title"],
                "abstract": seed_row["abstract"],
                "primary_category": seed_row.get("primary_category"),
                "categories": seed_row.get("categories", []),
                "created_date": seed_row.get("created_date"),
                "gold_summary": seed_row["gold_summary"],
                "bad_summary": structured_candidate["bad_summary"],
                "failure_axis": axis,
                "source_kind": "generated",
                "source_run": seed_row["source_run"],
                "candidate_origin": f"structured_wrong_number_binding:{seed_row.get('source_pool', 'unknown')}",
                "teacher_model": "deterministic_builder",
                "generator_prompt_version": "structured_wrong_number_binding_v1",
                "verifier_prompt_version": "",
                "gold_source": seed_row["gold_source"],
                "notes": normalize_text(str(seed_row.get("notes", ""))),
                "generator_raw_text": json.dumps(
                    {
                        "failure_axis": axis,
                        "bad_summary": structured_candidate["bad_summary"],
                        "generation_mode": "structured_wrong_number_binding_v1",
                    },
                    ensure_ascii=False,
                ),
                "generator_api_response_id": None,
                "generator_reported_axis": axis,
                "gold_teacher_model": seed_row.get("teacher_model", ""),
                "seed_source_pool": seed_row.get("source_pool", ""),
            }
    response = request_prompt_json(
        spec=generator_spec,
        model=args.generator_model,
        api_base=args.api_base,
        api_key=api_key,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        max_output_tokens=DEFAULT_GENERATION_MAX_OUTPUT_TOKENS,
        record=seed_row,
        extra_variables={
            "failure_axis": axis,
            "gold_summary": seed_row["gold_summary"],
        },
        label=f"negative-generate {seed_row['paper_id']} {axis}",
    )
    parsed = response["parsed"]
    bad_summary = normalize_text(str(parsed.get("bad_summary", "")))
    generated_axis = normalize_text(str(parsed.get("failure_axis", "")))
    return {
        "paper_id": seed_row["paper_id"],
        "title": seed_row["title"],
        "abstract": seed_row["abstract"],
        "primary_category": seed_row.get("primary_category"),
        "categories": seed_row.get("categories", []),
        "created_date": seed_row.get("created_date"),
        "gold_summary": seed_row["gold_summary"],
        "bad_summary": bad_summary,
        "failure_axis": axis,
        "source_kind": "generated",
        "source_run": seed_row["source_run"],
        "candidate_origin": f"hard_negative_generator:{seed_row.get('source_pool', 'unknown')}",
        "teacher_model": args.generator_model,
        "generator_prompt_version": generator_spec["prompt_version"],
        "verifier_prompt_version": "",
        "gold_source": seed_row["gold_source"],
        "notes": normalize_text(str(seed_row.get("notes", ""))),
        "generator_raw_text": response["raw_text"],
        "generator_api_response_id": response["api_response_id"],
        "generator_reported_axis": generated_axis,
        "gold_teacher_model": seed_row.get("teacher_model", ""),
        "seed_source_pool": seed_row.get("source_pool", ""),
    }


def build_prepared_generated_rows(
    *,
    pool_by_axis: dict[str, list[dict[str, Any]]],
    needed_by_axis: dict[str, int],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for axis, needed in needed_by_axis.items():
        for seed_row in pool_by_axis.get(axis, [])[:needed]:
            prepared.append(
                {
                    "paper_id": seed_row["paper_id"],
                    "title": seed_row["title"],
                    "abstract": seed_row["abstract"],
                    "gold_summary": normalize_text(str(seed_row.get("gold_summary", ""))),
                    "bad_summary": "",
                    "failure_axis": axis,
                    "source_kind": "generated",
                    "source_run": seed_row["source_run"],
                    "candidate_origin": f"prepared_seed:{seed_row.get('source_pool', 'unknown')}",
                    "teacher_model": "",
                    "generator_prompt_version": "",
                    "verifier_prompt_version": "",
                    "gold_source": str(seed_row.get("gold_source", "")),
                    "notes": normalize_text(str(seed_row.get("notes", ""))),
                    "status": "prepared_seed",
                }
            )
    return prepared


def canonicalize_verified_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": row["paper_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "gold_summary": row["gold_summary"],
        "bad_summary": row["bad_summary"],
        "failure_axis": row["failure_axis"],
        "source_kind": row["source_kind"],
        "source_run": row["source_run"],
        "candidate_origin": row["candidate_origin"],
        "teacher_model": row["teacher_model"],
        "generator_prompt_version": row["generator_prompt_version"],
        "verifier_prompt_version": row["verifier_prompt_version"],
        "gold_source": row["gold_source"],
        "notes": row.get("notes", ""),
    }


def generated_axis_row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    axis = str(row.get("failure_axis", ""))
    pool = str(row.get("seed_source_pool", ""))
    source_priority = GENERATED_AXIS_SOURCE_PRIORITY.get(axis, {}).get(pool, 99)
    return (
        source_priority,
        str(row.get("paper_id", "")),
        normalize_text(str(row.get("bad_summary", ""))).casefold(),
    )


def build_correction_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": row["paper_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "bad_summary": row["bad_summary"],
        "gold_summary": row["gold_summary"],
        "failure_axis": row["failure_axis"],
        "source_kind": row["source_kind"],
        "gold_source": row["gold_source"],
    }


def main() -> None:
    load_dotenv(Path(".env"))
    args = parse_args()
    validate_args(args)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mlx_output_dir = (
        output_dir if output_dir.name == "mlx_correction_chat" else output_dir / "mlx_correction_chat"
    )
    mlx_output_dir.mkdir(parents=True, exist_ok=True)

    release_dir = Path(args.release_dir)
    run_dir = Path(args.run_dir)
    holdout_review_path = Path(DEFAULT_HOLDOUT_REVIEW_JSONL)
    training_rows_path = release_dir / "training_rows.jsonl"
    review_outputs_path = run_dir / "review_outputs.jsonl"
    dropped_rows_path = run_dir / "dropped_rows.jsonl"

    print("[pack] loading source rows and prompt specs...", file=sys.stderr, flush=True)
    training_rows = load_jsonl(training_rows_path)
    training_by_id = {row["paper_id"]: row for row in training_rows}
    review_outputs = load_jsonl(review_outputs_path)
    dropped_rows = load_jsonl(dropped_rows_path)
    holdout_by_id = build_holdout_index(holdout_review_path)
    seed_entries = load_seed_entries(Path(args.seed_papers_json))

    negative_generator_spec = load_prompt_spec(Path(args.negative_generator_prompt))
    negative_verifier_spec = load_prompt_spec(Path(args.negative_verifier_prompt))
    gold_rewrite_spec = load_prompt_spec(Path(args.gold_rewrite_prompt))
    gold_verifier_spec = load_prompt_spec(Path(DEFAULT_GOLD_VERIFIER_PROMPT))
    correction_prompt_spec = load_correction_prompt_spec(Path(DEFAULT_CORRECTION_PROMPT))

    natural_pool, natural_stats = build_natural_candidate_pool(
        review_outputs=review_outputs,
        dropped_rows=dropped_rows,
        run_dir=run_dir,
    )
    derived_semantic_seed_pool, derived_semantic_seed_stats = build_derived_semantic_seed_pool(
        natural_pool
    )
    resolved_seed_pool = build_seed_pool(
        seed_entries=seed_entries,
        training_by_id=training_by_id,
        holdout_by_id=holdout_by_id,
        holdout_path=holdout_review_path,
        release_dir=release_dir,
    )
    selected_natural_candidates = select_mined_candidates(
        natural_pool=natural_pool,
        mined_axes=args.mined_axes,
        target_per_axis=args.target_per_axis,
    )
    regenerated_mined_seed_pool, regenerated_mined_audit_rows, regenerated_mined_seed_stats = (
        build_regenerated_mined_seed_pool(selected_natural_candidates)
    )
    direct_natural_candidates = [
        row
        for row in selected_natural_candidates
        if str(row.get("failure_axis", "")) not in REGENERATED_MINED_AXES
    ]
    generated_seed_pool_by_axis = build_generated_seed_pool_by_axis(
        training_rows=training_rows,
        resolved_seed_pool=resolved_seed_pool,
        regenerated_mined_seed_pool=regenerated_mined_seed_pool,
        derived_seed_pool=derived_semantic_seed_pool,
        requested_axes=args.generated_axes + args.mined_axes,
        release_dir=release_dir,
    )

    write_jsonl(output_dir / "semantic_seed_pool.jsonl", resolved_seed_pool)
    write_jsonl(output_dir / "derived_semantic_seed_pool.jsonl", derived_semantic_seed_pool)
    write_jsonl(output_dir / "regenerated_mined_seed_pool.jsonl", regenerated_mined_seed_pool)

    if args.prepare_only:
        print("[pack] prepare-only mode: writing pools without model calls.", file=sys.stderr, flush=True)
        write_jsonl(
            output_dir / "natural_negative_candidates.jsonl",
            direct_natural_candidates + regenerated_mined_audit_rows,
        )
        prepared_generated_rows = build_prepared_generated_rows(
            pool_by_axis=generated_seed_pool_by_axis,
            needed_by_axis={axis: args.target_per_axis for axis in args.generated_axes + args.mined_axes},
        )
        write_jsonl(output_dir / "generated_negative_candidates.jsonl", prepared_generated_rows)
        write_jsonl(output_dir / "verified_negative_rows.jsonl", [])
        write_jsonl(output_dir / "hard_negative_pack.jsonl", [])
        write_jsonl(output_dir / "correction_sft_rows.jsonl", [])
        write_jsonl(mlx_output_dir / "train.jsonl", [])
        write_jsonl(mlx_output_dir / "valid.jsonl", [])
        write_jsonl(mlx_output_dir / "test.jsonl", [])
        write_jsonl(output_dir / "spot_check_sample.jsonl", [])
        manifest = {
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "prepare_only": True,
            "release_dir": str(release_dir),
            "run_dir": str(run_dir),
            "holdout_review_jsonl": str(holdout_review_path),
            "seed_papers_json": args.seed_papers_json,
            "negative_generator_prompt_version": negative_generator_spec["prompt_version"],
            "negative_verifier_prompt_version": negative_verifier_spec["prompt_version"],
            "gold_rewrite_prompt_version": gold_rewrite_spec["prompt_version"],
            "gold_verifier_prompt_version": gold_verifier_spec["prompt_version"],
            "correction_prompt_version": correction_prompt_spec["prompt_version"],
            "generator_model": args.generator_model,
            "verifier_model": args.verifier_model,
            "gold_model": args.gold_model,
            "target_per_axis": args.target_per_axis,
            "generated_axes": args.generated_axes,
            "mined_axes": args.mined_axes,
            "natural_candidate_rows": len(direct_natural_candidates) + len(regenerated_mined_audit_rows),
            "prepared_generated_rows": len(prepared_generated_rows),
            "seed_pool_rows": len(resolved_seed_pool),
            **regenerated_mined_seed_stats,
            **derived_semantic_seed_stats,
            **natural_stats,
        }
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set {args.api_key_env} in the environment or .env before running."
        )

    gold_cache: dict[str, dict[str, Any]] = {}

    print("[pack] resolving gold summaries for mined candidates...", file=sys.stderr, flush=True)
    mined_candidates_with_gold: list[dict[str, Any]] = []
    for candidate in direct_natural_candidates:
        seed_like_row = {
            **candidate,
            "seed_candidate_summary": candidate["bad_summary"],
            "source_pool": "run_drop",
        }
        gold_result = ensure_seed_gold(
            seed_row=seed_like_row,
            gold_cache=gold_cache,
            gold_rewrite_spec=gold_rewrite_spec,
            gold_verifier_spec=gold_verifier_spec,
            args=args,
            api_key=api_key,
        )
        enriched = {
            **candidate,
            "gold_summary": gold_result["gold_summary"],
            "gold_source": gold_result["gold_source"],
            "gold_teacher_model": gold_result["teacher_model"],
            "gold_rewrite_prompt_version": gold_result["rewrite_prompt_version"],
            "gold_rewrite_model": gold_result["rewrite_model"],
            "gold_rewrite_raw_text": gold_result["rewrite_raw_text"],
            "gold_rewrite_api_response_id": gold_result["rewrite_api_response_id"],
            "gold_gate_prompt_version": gold_result["gate_prompt_version"],
            "gold_gate_model": gold_result["gate_model"],
            "gold_gate_raw_text": gold_result["gate_raw_text"],
            "gold_gate_api_response_id": gold_result["gate_api_response_id"],
            "gold_gate_reason_span": gold_result["gate_reason_span"],
            "teacher_model": candidate["teacher_model"] or gold_result["teacher_model"],
            "status": "gold_ready" if gold_result["accepted"] else "gold_rejected",
            "accepted": False,
            "rejection_reason": gold_result["gate_reason_span"],
        }
        if not enriched["gold_summary"]:
            mined_candidates_with_gold.append(enriched)
            continue
        if normalize_key(enriched["bad_summary"]) == normalize_key(enriched["gold_summary"]):
            enriched["status"] = "rejected_same_as_gold"
            enriched["rejection_reason"] = "bad summary matched gold summary"
        elif local_filter_flags(enriched["bad_summary"], enriched["title"]):
            enriched["status"] = "rejected_structural"
            enriched["rejection_reason"] = "bad summary failed local structural filter"
        else:
            mined_candidates_with_gold.append(enriched)

    natural_verification_inputs = [
        row for row in mined_candidates_with_gold if row.get("status") == "gold_ready"
    ]
    natural_candidates_verified = list(regenerated_mined_audit_rows) + [
        row for row in mined_candidates_with_gold if row.get("status") != "gold_ready"
    ]
    if natural_verification_inputs:
        print(
            f"[pack] verifying {len(natural_verification_inputs)} mined negatives...",
            file=sys.stderr,
            flush=True,
        )
        for _, row, future in iter_bounded_futures(
            natural_verification_inputs,
            args.verification_workers,
            lambda item: verify_negative_candidate(
                candidate_row=item,
                verifier_spec=negative_verifier_spec,
                args=args,
                api_key=api_key,
            ),
        ):
            verified = future.result()
            verified["status"] = "accepted" if verified["accepted"] else "rejected_verifier"
            natural_candidates_verified.append(verified)

    write_jsonl(output_dir / "natural_negative_candidates.jsonl", natural_candidates_verified)

    accepted_by_axis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dedupe_keys: set[str] = set()
    for row in natural_candidates_verified:
        if not row.get("accepted"):
            continue
        key = normalize_key(row["paper_id"], row["failure_axis"], row["bad_summary"])
        if key in dedupe_keys:
            continue
        dedupe_keys.add(key)
        accepted_by_axis[row["failure_axis"]].append(row)

    needed_by_axis: dict[str, int] = {}
    for axis in args.generated_axes:
        needed_by_axis[axis] = max(args.target_per_axis - len(accepted_by_axis.get(axis, [])), 0)
    for axis in args.mined_axes:
        needed_by_axis[axis] = max(args.target_per_axis - len(accepted_by_axis.get(axis, [])), 0)

    print("[pack] generating semantic negatives for remaining gaps...", file=sys.stderr, flush=True)
    generation_seed_jobs: list[tuple[str, dict[str, Any]]] = []
    for axis, needed in needed_by_axis.items():
        if needed <= 0:
            continue
        attempt_cap = generation_attempt_cap(axis, needed)
        for seed_row in generated_seed_pool_by_axis.get(axis, [])[:attempt_cap]:
            generation_seed_jobs.append((axis, seed_row))

    prepared_seed_rows: list[dict[str, Any]] = []
    ready_generation_jobs: list[tuple[str, dict[str, Any]]] = []
    for axis, seed_row in generation_seed_jobs:
        gold_result = ensure_seed_gold(
            seed_row=seed_row,
            gold_cache=gold_cache,
            gold_rewrite_spec=gold_rewrite_spec,
            gold_verifier_spec=gold_verifier_spec,
            args=args,
            api_key=api_key,
        )
        enriched_seed = {
            **seed_row,
            "gold_summary": gold_result["gold_summary"],
            "gold_source": gold_result["gold_source"],
            "teacher_model": seed_row.get("teacher_model", "") or gold_result["teacher_model"],
            "gold_gate_reason_span": gold_result["gate_reason_span"],
        }
        if not enriched_seed["gold_summary"]:
            prepared_seed_rows.append(
                {
                    "paper_id": enriched_seed["paper_id"],
                    "title": enriched_seed["title"],
                    "abstract": enriched_seed["abstract"],
                    "gold_summary": "",
                    "bad_summary": "",
                    "failure_axis": axis,
                    "source_kind": "generated",
                    "source_run": enriched_seed["source_run"],
                    "candidate_origin": f"prepared_seed:{enriched_seed.get('source_pool', 'unknown')}",
                    "teacher_model": "",
                    "generator_prompt_version": "",
                    "verifier_prompt_version": "",
                    "gold_source": "",
                    "notes": gold_result["gate_reason_span"] or "gold preparation failed",
                    "status": "seed_rejected",
                }
            )
            continue
        ready_generation_jobs.append((axis, enriched_seed))

    generated_candidates: list[dict[str, Any]] = list(prepared_seed_rows)
    generated_verification_inputs: list[dict[str, Any]] = []
    if ready_generation_jobs:
        for _, job, future in iter_bounded_futures(
            ready_generation_jobs,
            args.generation_workers,
            lambda item: generate_bad_summary_candidate(
                seed_row=item[1],
                axis=item[0],
                generator_spec=negative_generator_spec,
                args=args,
                api_key=api_key,
            ),
        ):
            generated = future.result()
            generated["status"] = "generated"
            if not generated["bad_summary"]:
                generated["status"] = "rejected_empty"
                generated["rejection_reason"] = "generator returned an empty bad summary"
                generated_candidates.append(generated)
                continue
            if normalize_key(generated["bad_summary"]) == normalize_key(generated["gold_summary"]):
                generated["status"] = "rejected_same_as_gold"
                generated["rejection_reason"] = "bad summary matched gold summary"
                generated_candidates.append(generated)
                continue
            if local_filter_flags(generated["bad_summary"], generated["title"]):
                generated["status"] = "rejected_structural"
                generated["rejection_reason"] = "bad summary failed local structural filter"
                generated_candidates.append(generated)
                continue
            generated_verification_inputs.append(generated)

    if generated_verification_inputs:
        for _, row, future in iter_bounded_futures(
            generated_verification_inputs,
            args.verification_workers,
            lambda item: verify_negative_candidate(
                candidate_row=item,
                verifier_spec=negative_verifier_spec,
                args=args,
                api_key=api_key,
            ),
        ):
            verified = future.result()
            verified["status"] = "accepted" if verified["accepted"] else "rejected_verifier"
            generated_candidates.append(verified)

    write_jsonl(output_dir / "generated_negative_candidates.jsonl", generated_candidates)

    for row in generated_candidates:
        if not row.get("accepted"):
            continue
        key = normalize_key(row["paper_id"], row["failure_axis"], row["bad_summary"])
        if key in dedupe_keys:
            continue
        dedupe_keys.add(key)
        accepted_by_axis[row["failure_axis"]].append(row)

    verified_rows: list[dict[str, Any]] = []
    for axis in args.mined_axes + args.generated_axes:
        axis_rows = accepted_by_axis.get(axis, [])
        if axis in args.generated_axes:
            axis_rows.sort(key=generated_axis_row_sort_key)
        else:
            axis_rows.sort(
                key=lambda row: (
                    str(row["paper_id"]),
                    normalize_text(str(row["bad_summary"])).casefold(),
                )
            )
        verified_rows.extend(axis_rows[: args.target_per_axis])

    verified_rows.sort(key=lambda row: (row["failure_axis"], str(row["paper_id"])))
    write_jsonl(output_dir / "verified_negative_rows.jsonl", verified_rows)

    hard_negative_pack = [canonicalize_verified_row(row) for row in verified_rows]
    write_jsonl(output_dir / "hard_negative_pack.jsonl", hard_negative_pack)

    correction_sft_rows = [build_correction_row(row) for row in verified_rows]
    write_jsonl(output_dir / "correction_sft_rows.jsonl", correction_sft_rows)

    correction_chat_rows = [
        build_correction_chat_example(row, correction_prompt_spec)
        for row in correction_sft_rows
    ]
    train_rows, valid_rows, test_rows = split_rows(
        correction_chat_rows,
        seed=args.seed,
    )
    write_jsonl(mlx_output_dir / "train.jsonl", train_rows)
    write_jsonl(mlx_output_dir / "valid.jsonl", valid_rows)
    write_jsonl(mlx_output_dir / "test.jsonl", test_rows)

    spot_check_rows: list[dict[str, Any]] = []
    sampled_ids_by_axis: dict[str, list[str]] = {}
    for axis in args.generated_axes:
        rows = [row for row in verified_rows if row["failure_axis"] == axis]
        chosen = sample_rows_for_axis(rows, 10, seed=args.seed + len(axis))
        spot_check_rows.extend(chosen)
        sampled_ids_by_axis[axis] = [str(row["paper_id"]) for row in chosen]
    for axis in args.mined_axes:
        rows = [row for row in verified_rows if row["failure_axis"] == axis]
        chosen = sample_rows_for_axis(rows, 5, seed=args.seed + len(axis))
        spot_check_rows.extend(chosen)
        sampled_ids_by_axis[axis] = [str(row["paper_id"]) for row in chosen]
    write_jsonl(output_dir / "spot_check_sample.jsonl", spot_check_rows)

    manifest = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "prepare_only": False,
        "release_dir": str(release_dir),
        "run_dir": str(run_dir),
        "holdout_review_jsonl": str(holdout_review_path),
        "seed_papers_json": args.seed_papers_json,
        "negative_generator_prompt_version": negative_generator_spec["prompt_version"],
        "negative_verifier_prompt_version": negative_verifier_spec["prompt_version"],
        "gold_rewrite_prompt_version": gold_rewrite_spec["prompt_version"],
        "gold_verifier_prompt_version": gold_verifier_spec["prompt_version"],
        "correction_prompt_version": correction_prompt_spec["prompt_version"],
        "generator_model": args.generator_model,
        "verifier_model": args.verifier_model,
        "gold_model": args.gold_model,
        "target_per_axis": args.target_per_axis,
        "generated_axes": args.generated_axes,
        "mined_axes": args.mined_axes,
        "seed_pool_rows": len(resolved_seed_pool),
        **regenerated_mined_seed_stats,
        **derived_semantic_seed_stats,
        "natural_candidate_rows": len(natural_candidates_verified),
        "generated_candidate_rows": len(generated_candidates),
        "verified_rows": len(verified_rows),
        "verified_axis_counts": dict(
            sorted(Counter(row["failure_axis"] for row in verified_rows).items())
        ),
        "source_kind_counts": dict(
            sorted(Counter(row["source_kind"] for row in verified_rows).items())
        ),
        "gold_source_counts": dict(
            sorted(Counter(row["gold_source"] for row in verified_rows).items())
        ),
        "spot_check_sampled_ids": sampled_ids_by_axis,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "test_rows": len(test_rows),
        **natural_stats,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
