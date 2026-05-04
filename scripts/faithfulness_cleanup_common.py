from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from scripts.synthetic_summary_runtime import (
        local_filter_flags,
        normalize_text,
        summary_flags,
        word_tokens,
    )
except ImportError:  # pragma: no cover - script execution from scripts/
    from synthetic_summary_runtime import (
        local_filter_flags,
        normalize_text,
        summary_flags,
        word_tokens,
    )

DEFAULT_SOURCE_RELEASE = "hf_dataset/releases/recent_3y_balanced_10746_v1"
DEFAULT_EXPERIMENT_ROOT = (
    "hf_bucket/experiments/faithfulness_cleanup/"
    "recent_3y_balanced_10746_faithfulness_overlay_800_v1"
)
DEFAULT_OVERLAY_RELEASE = (
    "hf_dataset/releases/recent_3y_balanced_10746_faithfulness_overlay_800_v1"
)
DEFAULT_CANDIDATE_TARGET = 1600
DEFAULT_FINAL_TARGET = 800
DEFAULT_QA_OVERLAP_RATIO = 0.2
DEFAULT_SEED = 20260427

NUMBER_WORDS = {
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
    "hundred": "100",
    "thousand": "1000",
    "million": "1000000",
    "billion": "1000000000",
}

STRONG_VERB_RE = re.compile(
    r"\b(?:"
    r"outperform\w*|surpass\w*|improv\w*|boost\w*|reduc\w*|cut\w*|"
    r"increas\w*|decreas\w*|accelerat\w*|stabili[sz]\w*|optimi[sz]\w*|"
    r"prevent\w*|mitigat\w*|detect\w*|predict\w*|identif\w*|reveal\w*|"
    r"explain\w*|demonstrat\w*|prov\w*|achiev\w*|enabl\w*|yield\w*"
    r")\b",
    re.IGNORECASE,
)
CAUSAL_RE = re.compile(
    r"\b(?:because|caus\w*|driv\w*|lead\w*|yield\w*|via|through|using|with|by)\b",
    re.IGNORECASE,
)
GENERIC_ENTITY_RE = re.compile(
    r"\b(?:method|framework|model|system|algorithm|approach|pipeline|"
    r"benchmark|dataset|controller|solver|optimizer|agent|policy|"
    r"architecture|mechanism)\b",
    re.IGNORECASE,
)
CLAIMY_ADJECTIVE_RE = re.compile(
    r"\b(?:new|novel|robust|efficient|scalable|accurate|simple|faster|"
    r"better|optimal|state-of-the-art|reliable|practical)\b",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(
    r"\b\d+(?:,\d{3})*(?:\.\d+)?(?:%|x|X|×|k|K|m|M|b|B)?\b|"
    r"\b(?:" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
LABEL_RE = re.compile(
    r"\b(?:[A-Z]{2,}[A-Za-z0-9-]*|[A-Za-z]+-[A-Za-z0-9-]+|"
    r"[A-Za-z]*\d+[A-Za-z0-9-]*)\b"
)
NUMERIC_LABEL_RE = re.compile(r"^\d+(?:\.\d+)?(?:%|x|X|×|k|K|m|M|b|B)?$")
YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
TOKEN_STRIP_CHARS = ".,;:!?()[]{}<>\"'`“”‘’"

RISK_FLAG_ORDER = (
    "unsupported_number",
    "unsupported_label",
    "strong_verb",
    "mechanism_or_causal",
    "generic_entity_noun",
    "claimy_adjective",
    "very_short",
)

REVIEW_DECISIONS = {"keep", "edit", "drop", "uncertain"}
CONFIDENCE_VALUES = {"high", "medium", "low"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def resolve_training_rows(source: Path) -> Path:
    if source.is_dir():
        source = source / "training_rows.jsonl"
    if not source.exists():
        raise FileNotFoundError(f"Could not find training rows at {source}")
    return source


def umbrella(row: dict[str, Any]) -> str:
    primary = str(row.get("primary_category", "")).strip()
    if not primary:
        return "unknown"
    return primary.split(".", 1)[0]


def normalized_signature(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def canonical_number(value: str) -> str:
    cleaned = value.casefold().replace(",", "").strip(TOKEN_STRIP_CHARS)
    if cleaned in NUMBER_WORDS:
        return NUMBER_WORDS[cleaned]
    if cleaned.endswith("%"):
        return cleaned
    if re.fullmatch(r"\d+(?:\.\d+)?(?:%|x|×|k|m|b)?", cleaned):
        return cleaned
    return cleaned


def numeric_mentions(value: str) -> list[dict[str, str]]:
    mentions: list[dict[str, str]] = []
    for match in NUMBER_RE.finditer(value):
        raw = match.group(0)
        canonical = canonical_number(raw)
        if YEAR_RE.fullmatch(canonical):
            continue
        mentions.append({"raw": raw, "canonical": canonical})
    return mentions


def source_supports_token(source_text: str, raw: str, canonical: str | None = None) -> bool:
    if numeric_mentions(raw):
        if canonical is None:
            canonical = canonical_number(raw)
        source_numbers = {item["canonical"] for item in numeric_mentions(source_text)}
        return canonical in source_numbers

    source_signature = normalized_signature(source_text)
    raw_signature = normalized_signature(raw)
    if raw_signature and raw_signature in source_signature:
        return True
    if canonical is not None:
        source_numbers = {item["canonical"] for item in numeric_mentions(source_text)}
        if canonical in source_numbers:
            return True
    return False


def unsupported_numbers(summary: str, source_text: str) -> list[str]:
    unsupported: list[str] = []
    seen: set[str] = set()
    for mention in numeric_mentions(summary):
        raw = mention["raw"]
        canonical = mention["canonical"]
        if source_supports_token(source_text, raw, canonical):
            continue
        key = canonical or raw.casefold()
        if key in seen:
            continue
        seen.add(key)
        unsupported.append(raw)
    return unsupported


def label_mentions(value: str) -> list[str]:
    mentions: list[str] = []
    seen: set[str] = set()
    for match in LABEL_RE.finditer(value):
        raw = match.group(0).strip(TOKEN_STRIP_CHARS)
        if not raw or raw.casefold() in seen:
            continue
        if not any(char.isalpha() for char in raw):
            continue
        if NUMBER_RE.fullmatch(raw):
            continue
        if NUMERIC_LABEL_RE.fullmatch(raw.replace(",", "")):
            continue
        seen.add(raw.casefold())
        mentions.append(raw)
    return mentions


def unsupported_labels(summary: str, source_text: str) -> list[str]:
    unsupported: list[str] = []
    for label in label_mentions(summary):
        if source_supports_token(source_text, label):
            continue
        unsupported.append(label)
    return unsupported


def risk_profile(row: dict[str, Any]) -> dict[str, Any]:
    summary = str(row.get("summary") or row.get("original_summary") or "").strip()
    source_text = " ".join([str(row.get("title", "")), str(row.get("abstract", ""))])
    unsup_numbers = unsupported_numbers(summary, source_text)
    unsup_labels = unsupported_labels(summary, source_text)

    flags: list[str] = []
    if unsup_numbers:
        flags.append("unsupported_number")
    if unsup_labels:
        flags.append("unsupported_label")
    if STRONG_VERB_RE.search(summary):
        flags.append("strong_verb")
    if CAUSAL_RE.search(summary):
        flags.append("mechanism_or_causal")
    if GENERIC_ENTITY_RE.search(summary):
        flags.append("generic_entity_noun")
    if CLAIMY_ADJECTIVE_RE.search(summary):
        flags.append("claimy_adjective")
    if len(word_tokens(summary)) <= 8:
        flags.append("very_short")

    score = 0
    score += 4 if "unsupported_number" in flags else 0
    score += 3 if "unsupported_label" in flags else 0
    score += 1 if "strong_verb" in flags else 0
    score += 1 if "mechanism_or_causal" in flags else 0
    score += 1 if "generic_entity_noun" in flags else 0
    score += 1 if "claimy_adjective" in flags else 0
    score += 1 if "very_short" in flags else 0

    stacked_claim = (
        "strong_verb" in flags
        and "mechanism_or_causal" in flags
        and ("generic_entity_noun" in flags or "claimy_adjective" in flags)
    )
    if score >= 4 or stacked_claim:
        tier = "high"
    elif score >= 2:
        tier = "medium"
    else:
        tier = "low"

    return {
        "risk_tier": tier,
        "risk_score": score,
        "risk_flags": [flag for flag in RISK_FLAG_ORDER if flag in flags],
        "unsupported_numbers": unsup_numbers,
        "unsupported_labels": unsup_labels,
        "summary_flags": summary_flags(summary, str(row.get("title", ""))),
        "local_filter_flags": local_filter_flags(summary, str(row.get("title", ""))),
    }


def risk_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    flags = set(row.get("risk_flags", []))
    tier_rank = {"high": 0, "medium": 1, "low": 2}.get(str(row.get("risk_tier", "")), 9)
    return (
        tier_rank,
        -int(row.get("risk_score", 0)),
        0 if "unsupported_number" in flags else 1,
        0 if "unsupported_label" in flags else 1,
        0 if {"strong_verb", "mechanism_or_causal"} <= flags else 1,
        str(row.get("primary_category", "")),
        str(row.get("paper_id", "")),
    )


def largest_remainder_allocations(
    counts: dict[str, int],
    total: int,
) -> dict[str, int]:
    if total <= 0 or not counts:
        return {key: 0 for key in counts}
    available = sum(counts.values())
    if available <= total:
        return dict(counts)

    raw_allocations = {
        key: (count / available) * total
        for key, count in counts.items()
    }
    allocations = {
        key: min(counts[key], int(math.floor(raw)))
        for key, raw in raw_allocations.items()
    }
    remaining = total - sum(allocations.values())
    remainders = sorted(
        (
            raw_allocations[key] - math.floor(raw_allocations[key]),
            key,
        )
        for key in counts
    )
    for _, key in reversed(remainders):
        if remaining <= 0:
            break
        if allocations[key] >= counts[key]:
            continue
        allocations[key] += 1
        remaining -= 1

    while remaining > 0:
        changed = False
        for key in sorted(counts):
            if remaining <= 0:
                break
            if allocations[key] >= counts[key]:
                continue
            allocations[key] += 1
            remaining -= 1
            changed = True
        if not changed:
            break
    return allocations


def counter_dict(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def stats_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"row_count": 0}
    word_counts = [
        len(word_tokens(str(row.get("summary") or row.get("original_summary") or "")))
        for row in rows
    ]
    return {
        "row_count": len(rows),
        "risk_tiers": counter_dict([str(row.get("risk_tier", "")) for row in rows]),
        "umbrellas": counter_dict([umbrella(row) for row in rows]),
        "primary_categories": dict(
            Counter(str(row.get("primary_category", "")) for row in rows).most_common(30)
        ),
        "summary_word_count": {
            "min": min(word_counts),
            "max": max(word_counts),
            "avg": round(mean(word_counts), 2),
        },
    }


def read_decision_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_decision_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "review_id",
        "candidate_id",
        "decision",
        "confidence",
        "final_summary",
        "issue_tags",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "review_id": row["review_id"],
                    "candidate_id": row["candidate_id"],
                    "decision": "",
                    "confidence": "",
                    "final_summary": "",
                    "issue_tags": "",
                    "note": "",
                }
            )


def grouped_by_umbrella(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[umbrella(row)].append(row)
    return dict(grouped)
