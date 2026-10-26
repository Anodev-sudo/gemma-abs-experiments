#!/usr/bin/env python3
"""Mine lower-weight silver natural-language summary rows from prior artifacts."""
from __future__ import annotations

import argparse
import json
import random
import re
import shlex
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import synthetic_summary_runtime as runtime
except ImportError:  # pragma: no cover - package-style execution
    from scripts import synthetic_summary_runtime as runtime


RUN_ROOT = Path("hf_bucket/experiments/natural_language_silver_mining_v1")
CATALOG_DIR = RUN_ROOT / "catalog"
FILTER_DIR = RUN_ROOT / "first_filter"
BRONZE_FILTER_DIR = RUN_ROOT / "bronze_filter"
SHARD_DIR = RUN_ROOT / "review_shards"
FINAL_DIR = RUN_ROOT / "final"
DEFAULT_RELEASE_DIR = Path("hf_dataset/releases/recent_3y_balanced_20000_natural_language_silver_v1")
DEFAULT_TRAIN_VIEW_DIR = Path(
    "hf_bucket/experiments/sft/natural_language_604_gold_silver_mix_v1/"
    "dataset_604gold_silver_w035"
)
DEFAULT_GOLD = Path(
    "hf_dataset/releases/"
    "recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_604_expansion_pool/"
    "training_rows.jsonl"
)
DEFAULT_PROMPT_SPEC = Path("prompts/mlx_lora/gemma_270m_it_abstract_to_natural_language_summary_v1.json")
DEFAULT_REMOTE_ROOT = "hf://buckets/peering-latent/gemma-abs-experiments"

KEEP_SOURCE_PATTERNS = (
    "natural_language_teacher",
    "q8_promoted_broad_reweighted_under14",
    "q5_step2400_broad_reweighted_basin_v2",
    "q5_step2400_broad_reweighted_basin_v1",
    "faithfulness_cleanup",
    "q5_kimi_qwen_correction_overlay",
    "q5_precision_kimi_qwen_overlay",
    "q5_new_source_mining",
)
DEFER_SOURCE_PATTERNS = (
    "/dpo/",
    "hard_negative",
    "review_comparisons",
    "safety_eval",
    "checkpoint-",
)
PRIMARY_PULL_ORDER = {
    "experiments/sft/q8_promoted_broad_reweighted_under14_v1/dataset_under14_kimi_qwen_v1/train.jsonl": 1,
    "experiments/sft/q5_step2400_broad_reweighted_basin_v2/dataset_broad_base_low_repair_high_v2/train.jsonl": 2,
    "experiments/sft/q5_step2400_broad_reweighted_basin_v1/dataset_broad_base_low_repair_high_v1/train.jsonl": 3,
}
SUMMARY_KEYS = (
    "summary",
    "correction_summary",
    "cleanup_draft_summary",
    "accepted_source_summary",
    "target",
    "candidate_summary",
    "rewrite_summary",
    "model_summary",
)
REVIEW_DECISIONS = {"silver_candidate", "bronze_only", "needs_rewrite", "reject"}
UNSAFE_REVIEW_TAGS = {
    "unsupported_claim",
    "unsupported_number",
    "unsupported_entity",
    "unsupported_metric",
    "wrong_object",
    "wrong_method",
    "wrong_mechanism",
    "wrong_polarity",
    "polarity_flip",
    "false_value_attribution",
    "scope_upgrade",
    "title_rehash",
    "too_technical",
    "not_plain_language",
    "jargon",
    "jargon_stack",
    "method_name_stack",
    "acronym_heavy",
    "uncommon_acronym",
    "too_vague",
    "too_generic",
    "dense_numbers",
    "symbolic_notation",
    "too_numerical",
    "number_forward",
    "metric_forward",
    "benchmarky",
    "benchmark_claim",
    "comparison_drift",
    "speculative",
    "too_claimy",
    "equation_heavy",
    "unnatural",
    "fragment",
    "malformed",
}
COMMON_ACRONYMS = {
    "AI",
    "CPU",
    "DNA",
    "ECG",
    "GPU",
    "GPS",
    "HTML",
    "JSON",
    "K12",
    "LLM",
    "LLMS",
    "ML",
    "MRI",
    "RNA",
    "ROS",
    "RGB",
    "SQL",
    "STEM",
    "VR",
    "XR",
}
BANNED_ACRONYMS = {
    "AUV",
    "CSI",
    "GCD",
    "CFT",
    "CDW",
    "DCMM",
    "INR",
    "MPC",
    "PHUMA",
    "PPO",
    "QFT",
    "RIS",
    "SSK",
    "SOTA",
    "SVM",
    "SYK",
}
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\\-]*")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*(?:\.\d+)?%?")
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9-]{1,}\b")
SYMBOLIC_NOTATION_RE = re.compile(r"[α-ωΑ-Ω₀-₉⁰-⁹]")
NUMBER_FORWARD_RE = re.compile(
    r"(?:\d+(?:\.\d+)?%|\d+(?:\.\d+)?\s?(?:x|×|dB|K|mm|cm|m|km|ms|s|GB|MB|GHz|MHz)\b)",
    re.IGNORECASE,
)
BENCHMARKY_RE = re.compile(
    r"\b(?:beats?|outperforms?|faster|better than|state-of-the-art|near-optimal|improves? over|"
    r"more accurate than|higher accuracy|lower error|cuts? .* by \d|boosts? .* by \d)\b",
    re.IGNORECASE,
)
SPECULATIVE_RE = re.compile(r"\b(?:could|may|might)\s+(?:improve|boost|enable|allow|help|let|lead)\b", re.IGNORECASE)
BOILERPLATE_PREFIXES = (
    "this paper",
    "we ",
    "the authors",
    "in this work",
    "in this paper",
    "a paper",
    "the paper",
    "the new",
)
TITLE_OVERLAP_BY_PROFILE = {
    "bare": 0.72,
    "bronze-loose": 0.55,
    "bronze-balanced": 0.55,
    "bronze-strict": 0.50,
    "bronze-ultra": 0.45,
}
BRONZE_SCORE_THRESHOLDS = {
    "bare": None,
    "bronze-loose": 4,
    "bronze-balanced": 5,
    "bronze-strict": 6,
    "bronze-ultra": 7,
}
TITLE_CONTENT_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "for",
    "to",
    "with",
    "and",
    "or",
    "by",
    "from",
    "as",
    "at",
    "via",
    "using",
    "based",
    "new",
    "novel",
    "toward",
    "towards",
    "approach",
    "method",
    "model",
    "models",
    "framework",
    "algorithm",
    "algorithms",
    "study",
    "analysis",
    "system",
    "systems",
    "data",
    "learning",
    "deep",
    "large",
    "paper",
}
WEAK_ACADEMIC_VERB_RE = re.compile(
    r"\b(presents?|proposes?|introduces?|explores?|studies|investigates?|discusses?|describes?|surveys?|reviews?)\b",
    re.IGNORECASE,
)
PAYOFF_VERB_RE = re.compile(
    r"\b("
    r"cuts?|reduces?|improves?|boosts?|speeds?|accelerates?|prevents?|enables?|lets?|"
    r"reveals?|shows?|finds?|predicts?|detects?|tracks?|matches?|outperforms?|beats?|"
    r"fixes?|explains?|links?|connects?|maps?|recovers?|estimates?|classifies?|identifies?|"
    r"proves?|gives?|yields?|constrains?|handles?|supports?|captures?|aligns?|stabilizes?|"
    r"denoises?|reconstructs?|removes?|preserves?|optimizes?|learns?|adapts?|balances?|"
    r"separates?|localizes?|quantifies?|uncovers?|tests?|validates?|extends?|generalizes?"
    r")\b",
    re.IGNORECASE,
)
CONCRETE_PAYOFF_RE = re.compile(
    r"\b(faster|lower|higher|better|less|more|robust|accurate|uncertain|reliable|safe|"
    r"real-world|without|while|despite|under|across|than|from|into|with)\b",
    re.IGNORECASE,
)
GENERIC_NOUN_RE = re.compile(
    r"\b(method|approach|framework|model|algorithm|system|tool|pipeline|technique|strategy|architecture)\b",
    re.IGNORECASE,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def norm(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def stable_source_id(path: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", path)
    return value.strip("._")[:180] or "source"


def local_path_for_remote(relative_path: str) -> Path:
    parts = Path(relative_path).parts
    if parts and parts[0] == "experiments":
        return Path("hf_bucket") / relative_path
    if parts and parts[0] == "releases":
        return Path("hf_dataset") / relative_path
    return Path(relative_path)


def restore_command(remote_uri: str, local_path: Path) -> str:
    return f"mkdir -p {shlex.quote(local_path.parent.as_posix())} && hf buckets cp {shlex.quote(remote_uri)} {shlex.quote(local_path.as_posix())}"


def temp_restore_path(source_id: str) -> Path:
    return RUN_ROOT / "tmp_raw" / f"{source_id}.jsonl"


def source_kind(path: str) -> str:
    lowered = path.casefold()
    if "natural_language_teacher" in lowered:
        return "natural_language_near_miss"
    if "under14" in lowered:
        return "under14_repair_dataset"
    if "broad_reweighted_basin" in lowered:
        return "broad_reweighted_dataset"
    if "faithfulness_cleanup" in lowered or "faithfulness_overlay" in lowered:
        return "faithfulness_cleanup"
    if "correction" in lowered or "precision" in lowered or "new_source_mining" in lowered:
        return "correction_overlay"
    if "dpo" in lowered:
        return "dpo_or_preference"
    return "unknown"


def source_score(path: str, name: str, metadata_text: str) -> tuple[int, list[str], str]:
    lowered = f"{path}\n{name}\n{metadata_text}".casefold()
    reasons: list[str] = []
    score = 0
    if any(pattern in lowered for pattern in KEEP_SOURCE_PATTERNS):
        score += 40
        reasons.append("known_summary_family")
    if any(pattern in lowered for pattern in DEFER_SOURCE_PATTERNS):
        score -= 50
        reasons.append("deferred_family")
    if path in PRIMARY_PULL_ORDER:
        score += 100 - PRIMARY_PULL_ORDER[path]
        reasons.append(f"planned_pull_order_{PRIMARY_PULL_ORDER[path]}")
    if name in {"training_rows.jsonl", "train.jsonl", "selected_review_rows.jsonl", "selected_candidates.jsonl"}:
        score += 25
        reasons.append("supervised_label_shape")
    if "reviewer" in lowered or "qwen" in lowered or "strict" in lowered:
        score += 12
        reasons.append("reviewed_or_audited")
    if "natural" in lowered or "under14" in lowered or "plain" in lowered:
        score += 12
        reasons.append("style_close")
    if "raw_response" in lowered or "generations" in lowered:
        score -= 15
        reasons.append("raw_or_runtime_output")
    tier = "defer"
    if score >= 110:
        tier = "pull_first"
    elif score >= 65:
        tier = "pull_later"
    elif score >= 35:
        tier = "local_or_probe"
    return score, reasons, tier


def manifest_text_near(path: Path) -> str:
    parts = [path.as_posix()]
    for candidate in (path.parent / "manifest.json", path.parent / "run_manifest.json", path.parent / "README.md"):
        if candidate.exists() and candidate.is_file():
            try:
                parts.append(candidate.read_text(encoding="utf-8")[:12000])
            except UnicodeDecodeError:
                continue
    return "\n".join(parts)


def collect_manifest_catalog(search_roots: list[Path]) -> list[dict[str, Any]]:
    by_relative: dict[str, dict[str, Any]] = {}

    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name == "REMOTE_ARTIFACTS_MANIFEST.json":
                data = read_json_if_exists(path)
                for item in data.get("removed_files", []) or []:
                    relative_path = str(item.get("relative_path", "")).strip()
                    name = str(item.get("name", "")).strip() or Path(relative_path).name
                    if not relative_path or not name.endswith((".jsonl", ".json")):
                        continue
                    if "tokenizer.json" in relative_path:
                        continue
                    local_path = local_path_for_remote(relative_path)
                    remote_uri = str(item.get("remote_uri", "")).strip()
                    text = manifest_text_near(path)
                    score, reasons, tier = source_score(relative_path, name, text)
                    source_id = stable_source_id(relative_path)
                    temp_path = temp_restore_path(source_id)
                    by_relative[relative_path] = {
                        "source_id": source_id,
                        "relative_path": relative_path,
                        "local_path": local_path.as_posix(),
                        "temp_local_path": temp_path.as_posix(),
                        "remote_uri": remote_uri,
                        "restore_command": str(item.get("restore_command") or restore_command(remote_uri, local_path)),
                        "temp_restore_command": restore_command(remote_uri, temp_path),
                        "name": name,
                        "size_bytes": int(item.get("size_bytes", 0) or 0),
                        "is_local": local_path.exists(),
                        "catalog_origin": path.as_posix(),
                        "source_kind": source_kind(relative_path),
                        "score": score,
                        "score_reasons": reasons,
                        "pull_tier": tier,
                    }
                continue

            if not path.name.endswith((".jsonl", ".json")):
                continue
            if path.name in {"tokenizer.json", "trainer_state.json", "adapter_config.json"}:
                continue
            relative = path.as_posix()
            if relative.startswith("hf_bucket/"):
                relative_path = relative.removeprefix("hf_bucket/")
                remote_uri = f"{DEFAULT_REMOTE_ROOT}/{relative_path}"
            elif relative.startswith("hf_dataset/"):
                relative_path = relative.removeprefix("hf_dataset/")
                remote_uri = ""
            else:
                relative_path = relative
                remote_uri = ""
            text = manifest_text_near(path)
            score, reasons, tier = source_score(relative_path, path.name, text)
            if score < 20:
                continue
            existing = by_relative.get(relative_path)
            if existing and existing.get("remote_uri"):
                continue
            source_id = stable_source_id(relative_path)
            temp_path = temp_restore_path(source_id)
            by_relative[relative_path] = {
                "source_id": source_id,
                "relative_path": relative_path,
                "local_path": path.as_posix(),
                "temp_local_path": temp_path.as_posix(),
                "remote_uri": remote_uri,
                "restore_command": restore_command(remote_uri, path) if remote_uri else "",
                "temp_restore_command": restore_command(remote_uri, temp_path) if remote_uri else "",
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "is_local": True,
                "catalog_origin": "local_file_scan",
                "source_kind": source_kind(relative_path),
                "score": score,
                "score_reasons": reasons,
                "pull_tier": tier,
            }
    return sorted(by_relative.values(), key=lambda row: (-int(row["score"]), row["relative_path"]))


def command_catalog(args: argparse.Namespace) -> None:
    sources = collect_manifest_catalog(args.search_root)
    if args.max_sources:
        sources = sources[: args.max_sources]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    catalog_json = args.output_dir / "source_catalog.json"
    catalog_md = args.output_dir / "source_catalog.md"
    write_json(
        catalog_json,
        {
            "created_at": utc_now_iso(),
            "created_by": "scripts/silver_summary_mining_workflow.py catalog",
            "search_roots": [path.as_posix() for path in args.search_root],
            "source_count": len(sources),
            "sources": sources,
        },
    )
    lines = [
        "# Silver Summary Source Catalog",
        "",
        "| rank | tier | score | kind | local | size MiB | source |",
        "| ---: | --- | ---: | --- | --- | ---: | --- |",
    ]
    for rank, source in enumerate(sources, start=1):
        size_mib = int(source.get("size_bytes", 0)) / 1024 / 1024
        lines.append(
            f"| {rank} | {source['pull_tier']} | {source['score']} | {source['source_kind']} | "
            f"{'yes' if source['is_local'] else 'no'} | {size_mib:.1f} | `{source['relative_path']}` |"
        )
    catalog_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[silver-catalog] wrote {catalog_json}")
    print(f"[silver-catalog] wrote {catalog_md}")
    print(f"[silver-catalog] sources={len(sources)}")


def last_assistant_message(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return norm(message.get("content"))
    if messages and isinstance(messages[-1], dict):
        return norm(messages[-1].get("content"))
    return ""


def pick_summary(row: dict[str, Any]) -> tuple[str, str]:
    for key in SUMMARY_KEYS:
        value = norm(row.get(key))
        if value:
            return value, key
    message_summary = last_assistant_message(row)
    if message_summary:
        return message_summary, "messages.assistant"
    return "", ""


def normalized_categories(value: Any) -> list[str]:
    if isinstance(value, list):
        return [norm(item) for item in value if norm(item)]
    if isinstance(value, str):
        return [part for part in re.split(r"[\s,]+", value.strip()) if part]
    return []


def primary_category(row: dict[str, Any]) -> str:
    primary = norm(row.get("primary_category"))
    if primary:
        return primary
    categories = normalized_categories(row.get("categories"))
    return categories[0] if categories else ""


def extract_numbers(value: str) -> set[str]:
    return set(NUMBER_RE.findall(value))


def title_token_overlap(summary: str, title: str) -> float:
    return runtime.title_overlap_ratio(summary, title)


def content_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    for token in runtime.word_tokens(value):
        cleaned = token.casefold().strip(".,;:!?()[]{}<>\"'`")
        if cleaned and cleaned not in TITLE_CONTENT_STOPWORDS and len(cleaned) > 2:
            tokens.append(cleaned)
    return tokens


def title_ngram_overlap(summary: str, title: str) -> int:
    summary_tokens = content_tokens(summary)
    title_tokens = content_tokens(title)
    for size in (5, 4, 3):
        title_ngrams = {
            tuple(title_tokens[index : index + size])
            for index in range(0, max(0, len(title_tokens) - size + 1))
        }
        if any(
            tuple(summary_tokens[index : index + size]) in title_ngrams
            for index in range(0, max(0, len(summary_tokens) - size + 1))
        ):
            return size
    return 0


def title_name_opening(summary: str, title: str) -> bool:
    tokens = runtime.word_tokens(summary)
    if not tokens:
        return False
    first = tokens[0]
    title_tokens = {token.casefold() for token in runtime.word_tokens(title)}
    if first.casefold() not in title_tokens:
        return False
    return bool(re.search(r"[A-Z].*[A-Z]|[A-Za-z]+-[A-Za-z]+", first))


def dense_number_cluster(summary: str) -> bool:
    numbers = extract_numbers(summary)
    if len(numbers) > 2:
        return True
    tokens = runtime.word_tokens(summary)
    return len(numbers) >= 2 and len(tokens) <= 10


def number_forward_summary(summary: str) -> bool:
    numbers = extract_numbers(summary)
    if len(numbers) > 1:
        return True
    return bool(NUMBER_FORWARD_RE.search(summary))


def unsupported_numbers(summary: str, source_text: str) -> list[str]:
    source_numbers = extract_numbers(source_text)
    return sorted(number for number in extract_numbers(summary) if number not in source_numbers)


def has_equation_heavy_text(summary: str) -> bool:
    if runtime.has_formula_shorthand(summary):
        return True
    return bool(re.search(r"(?:[=<>≤≥∑∫√]|\\[A-Za-z]+|\$)", summary))


def acronym_terms(summary: str) -> list[str]:
    terms: list[str] = []
    for match in ACRONYM_RE.findall(summary):
        cleaned = match.strip("-").replace("-", "")
        if len(cleaned) > 1:
            terms.append(cleaned)
    return terms


def uncommon_acronym_terms(summary: str) -> list[str]:
    return [term for term in acronym_terms(summary) if term not in COMMON_ACRONYMS]


def natural_language_surface_flags(summary: str) -> list[str]:
    flags: list[str] = []
    terms = acronym_terms(summary)
    uncommon = sorted(set(uncommon_acronym_terms(summary)))
    banned = sorted(set(terms) & BANNED_ACRONYMS)
    if banned:
        flags.append("jargon_acronym")
    elif len(uncommon) >= 2:
        flags.append("jargon_stack")
    if re.search(r"\bis given\b", summary, re.IGNORECASE):
        flags.append("academic_passive")
    if SYMBOLIC_NOTATION_RE.search(summary):
        flags.append("symbolic_notation")
    if BENCHMARKY_RE.search(summary):
        flags.append("benchmarky")
    if SPECULATIVE_RE.search(summary):
        flags.append("speculative")
    return flags


def bronze_score(summary: str, title: str) -> int:
    score = 0
    overlap = title_token_overlap(summary, title)
    words = runtime.word_count(summary)
    if overlap <= 0.18:
        score += 4
    elif overlap <= 0.28:
        score += 3
    elif overlap <= 0.38:
        score += 2
    elif overlap <= 0.45:
        score += 1
    else:
        score -= 1
    if PAYOFF_VERB_RE.search(summary):
        score += 2
    else:
        score -= 2
    if CONCRETE_PAYOFF_RE.search(summary):
        score += 1
    if GENERIC_NOUN_RE.search(summary):
        score -= 1
    if 9 <= words <= 12:
        score += 1
    tokens = runtime.word_tokens(summary)
    if tokens and tokens[0].casefold() in {token.casefold() for token in runtime.word_tokens(title)}:
        score -= 1
    return score


def bronze_profile_flags(
    summary: str,
    title: str,
    *,
    profile: str,
    score_threshold: int | None,
) -> list[str]:
    if profile == "bare":
        return []
    flags: list[str] = []
    if title_name_opening(summary, title):
        flags.append("title_name_opening")
    ngram = title_ngram_overlap(summary, title)
    if ngram >= 4:
        flags.append("title_4gram_overlap")
    elif ngram == 3:
        flags.append("title_3gram_overlap")
    if WEAK_ACADEMIC_VERB_RE.search(summary):
        flags.append("weak_academic_verb")
    if len([token for token in runtime.word_tokens(summary) if len(token) >= 13]) >= 3:
        flags.append("long_word_stack")
    if profile in {"bronze-strict", "bronze-ultra"} and uncommon_acronym_terms(summary):
        flags.append("uncommon_acronym")
    threshold = score_threshold if score_threshold is not None else BRONZE_SCORE_THRESHOLDS[profile]
    if threshold is not None:
        score = bronze_score(summary, title)
        if score < threshold:
            flags.append(f"bronze_score_below_{threshold}")
    return flags


def first_filter_flags(
    summary: str,
    title: str,
    abstract: str,
    primary_category_value: str = "",
    paper_id: str = "",
    *,
    min_words: int,
    max_words: int,
    profile: str,
    bronze_score_threshold: int | None,
) -> list[str]:
    flags: list[str] = []
    lowered = summary.casefold()
    words = runtime.word_count(summary)
    if not paper_id:
        flags.append("missing_paper_id")
    if not title:
        flags.append("missing_title")
    if not abstract:
        flags.append("missing_abstract")
    if profile != "bare" and not primary_category_value:
        flags.append("missing_primary_category")
    if not summary:
        flags.append("missing_summary")
        return flags
    if words < min_words:
        flags.append("too_short")
    if words > max_words:
        flags.append("too_long")
    if runtime.sentence_count(summary) != 1:
        flags.append("not_single_sentence")
    if any(lowered.startswith(prefix) for prefix in BOILERPLATE_PREFIXES):
        flags.append("boilerplate_opening")
    if title_token_overlap(summary, title) >= TITLE_OVERLAP_BY_PROFILE[profile]:
        flags.append("title_rehash")
    if has_equation_heavy_text(summary):
        flags.append("equation_heavy")
    if dense_number_cluster(summary):
        flags.append("dense_numbers")
    if unsupported_numbers(summary, f"{title}\n{abstract}"):
        flags.append("unsupported_number")
    if profile != "bare":
        flags.extend(natural_language_surface_flags(summary))
        flags.extend(
            bronze_profile_flags(
                summary,
                title,
                profile=profile,
                score_threshold=bronze_score_threshold,
            )
        )
    return flags


def normalize_candidate(row: dict[str, Any], *, source: dict[str, Any], ordinal: int) -> dict[str, Any]:
    summary, summary_key = pick_summary(row)
    paper_id = norm(row.get("paper_id"))
    title = norm(row.get("title"))
    abstract = norm(row.get("abstract"))
    categories = normalized_categories(row.get("categories"))
    source_id = str(source.get("source_id") or stable_source_id(str(source.get("relative_path", "source"))))
    return {
        "candidate_id": f"{source_id}:{ordinal:06d}:{paper_id or 'missing'}",
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "summary": summary,
        "primary_category": primary_category(row),
        "categories": categories,
        "created_date": norm(row.get("created_date")),
        "source_path": str(source.get("relative_path") or source.get("local_path") or ""),
        "source_id": source_id,
        "source_kind": str(source.get("source_kind", "")),
        "source_tier": str(source.get("pull_tier", "")),
        "source_component": norm(row.get("dataset_component") or row.get("overlay_component") or row.get("source_split")),
        "summary_source_key": summary_key,
        "word_count": runtime.word_count(summary),
        "sentence_count": runtime.sentence_count(summary),
        "title_overlap_ratio": round(title_token_overlap(summary, title), 4),
        "unsupported_numbers": unsupported_numbers(summary, f"{title}\n{abstract}"),
        "quality_stage": "first_filter_candidate",
        "provenance": {
            "original_source_split": row.get("source_split", ""),
            "original_teacher_model": row.get("teacher_model", ""),
            "original_generator_prompt_version": row.get("generator_prompt_version", ""),
            "original_review_prompt_version": row.get("review_prompt_version", ""),
        },
    }


def source_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.source_json:
        catalog = read_json_if_exists(args.source_json)
        sources = catalog.get("sources", []) if isinstance(catalog, dict) else []
        if args.source_id:
            for source in sources:
                if source.get("source_id") == args.source_id or source.get("relative_path") == args.source_id:
                    return source
            raise ValueError(f"Could not find source_id {args.source_id} in {args.source_json}")
        if len(sources) == 1:
            return sources[0]
    path = args.input_jsonl
    if path is None:
        raise ValueError("Pass --input-jsonl or --source-json plus --source-id")
    relative = path.as_posix()
    if relative.startswith("hf_bucket/"):
        relative = relative.removeprefix("hf_bucket/")
    return {
        "source_id": stable_source_id(relative),
        "relative_path": relative,
        "local_path": path.as_posix(),
        "source_kind": source_kind(relative),
        "pull_tier": "manual",
    }


def command_filter(args: argparse.Namespace) -> None:
    source = source_from_args(args)
    input_path = args.input_jsonl or Path(str(source.get("local_path", "")))
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input JSONL: {input_path}")
    source_id = str(source.get("source_id") or stable_source_id(input_path.as_posix()))
    output_dir = args.output_dir / source_id
    survivors_path = output_dir / "survivors.jsonl"
    reject_sample_path = output_dir / "reject_sample.jsonl"
    report_path = output_dir / "filter_report.json"

    seen_candidate_ids: set[str] = set()
    survivors: list[dict[str, Any]] = []
    reject_sample: list[dict[str, Any]] = []
    flag_counts: Counter[str] = Counter()
    source_component_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    input_rows = 0

    for ordinal, row in enumerate(iter_jsonl(input_path), start=1):
        input_rows += 1
        candidate = normalize_candidate(row, source=source, ordinal=ordinal)
        flags = first_filter_flags(
            candidate["summary"],
            candidate["title"],
            candidate["abstract"],
            candidate["primary_category"],
            candidate["paper_id"],
            min_words=args.min_words,
            max_words=args.max_words,
            profile=args.filter_profile,
            bronze_score_threshold=args.bronze_score_threshold,
        )
        source_component_counts[candidate.get("source_component", "")] += 1
        primary_counts[candidate.get("primary_category", "")] += 1
        if flags:
            flag_counts.update(flags)
            if len(reject_sample) < args.reject_sample_size:
                reject_sample.append({**candidate, "first_filter_flags": flags, "quality_stage": "first_filter_reject"})
            continue
        if candidate["candidate_id"] in seen_candidate_ids:
            flag_counts["duplicate_candidate_id"] += 1
            continue
        seen_candidate_ids.add(candidate["candidate_id"])
        survivors.append({**candidate, "first_filter_flags": [], "quality_stage": "first_filter_survivor"})

    write_jsonl(survivors_path, survivors)
    write_jsonl(reject_sample_path, reject_sample)
    report = {
        "created_at": utc_now_iso(),
        "created_by": "scripts/silver_summary_mining_workflow.py filter",
        "input_jsonl": input_path.as_posix(),
        "raw_deleted_after_filter": False,
        "source": source,
        "input_rows": input_rows,
        "survivor_rows": len(survivors),
        "rejected_rows": input_rows - len(survivors),
        "reject_sample_rows": len(reject_sample),
        "first_filter_policy": {
            "filter_profile": args.filter_profile,
            "min_words": args.min_words,
            "max_words": args.max_words,
            "single_sentence": True,
            "title_overlap_reject_at_or_above": TITLE_OVERLAP_BY_PROFILE[args.filter_profile],
            "bronze_score_threshold": args.bronze_score_threshold
            if args.bronze_score_threshold is not None
            else BRONZE_SCORE_THRESHOLDS[args.filter_profile],
            "unsupported_summary_numbers_rejected": True,
            "strict_bronze_surface_flags": args.filter_profile != "bare",
        },
        "flag_counts": dict(sorted(flag_counts.items())),
        "source_component_counts": dict(source_component_counts.most_common(30)),
        "primary_category_top30": dict(primary_counts.most_common(30)),
        "outputs": {
            "survivors": survivors_path.as_posix(),
            "reject_sample": reject_sample_path.as_posix(),
            "report": report_path.as_posix(),
        },
    }
    if args.delete_input_after_filter:
        input_path.unlink()
        report["raw_deleted_after_filter"] = True
    write_json(report_path, report)
    print(f"[silver-filter] source={source_id}")
    print(f"[silver-filter] input_rows={input_rows} survivors={len(survivors)} rejected={input_rows - len(survivors)}")
    print(f"[silver-filter] wrote {survivors_path}")
    if args.delete_input_after_filter:
        print(f"[silver-filter] deleted raw input {input_path}")


def command_refilter(args: argparse.Namespace) -> None:
    rows = list(iter_jsonl(args.input_jsonl))
    source_id = args.source_id or stable_source_id(args.input_jsonl.as_posix())
    profile_slug = args.filter_profile.replace("-", "_")
    output_dir = args.output_dir / source_id / profile_slug
    survivors_path = output_dir / "survivors.jsonl"
    reject_sample_path = output_dir / "reject_sample.jsonl"
    report_path = output_dir / "filter_report.json"

    survivors: list[dict[str, Any]] = []
    reject_sample: list[dict[str, Any]] = []
    flag_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    for row in rows:
        summary = norm(row.get("summary"))
        title = norm(row.get("title"))
        abstract = norm(row.get("abstract"))
        primary = norm(row.get("primary_category"))
        paper_id = norm(row.get("paper_id"))
        flags = first_filter_flags(
            summary,
            title,
            abstract,
            primary,
            paper_id,
            min_words=args.min_words,
            max_words=args.max_words,
            profile=args.filter_profile,
            bronze_score_threshold=args.bronze_score_threshold,
        )
        primary_counts[primary] += 1
        component_counts[norm(row.get("source_component"))] += 1
        if flags:
            flag_counts.update(flags)
            if len(reject_sample) < args.reject_sample_size:
                reject_sample.append({**row, "bronze_filter_flags": flags, "quality_stage": "bronze_filter_reject"})
            continue
        survivors.append(
            {
                **row,
                "bronze_filter_flags": [],
                "bronze_score": bronze_score(summary, title),
                "quality_stage": "bronze_filter_survivor",
            }
        )

    write_jsonl(survivors_path, survivors)
    write_jsonl(reject_sample_path, reject_sample)
    original_input_rows = args.original_input_rows or len(rows)
    report = {
        "created_at": utc_now_iso(),
        "created_by": "scripts/silver_summary_mining_workflow.py refilter",
        "input_jsonl": args.input_jsonl.as_posix(),
        "input_rows": len(rows),
        "original_input_rows": original_input_rows,
        "survivor_rows": len(survivors),
        "rejected_rows": len(rows) - len(survivors),
        "original_cut_rate": round((original_input_rows - len(survivors)) / original_input_rows, 6)
        if original_input_rows
        else 0.0,
        "filter_policy": {
            "filter_profile": args.filter_profile,
            "min_words": args.min_words,
            "max_words": args.max_words,
            "title_overlap_reject_at_or_above": TITLE_OVERLAP_BY_PROFILE[args.filter_profile],
            "bronze_score_threshold": args.bronze_score_threshold
            if args.bronze_score_threshold is not None
            else BRONZE_SCORE_THRESHOLDS[args.filter_profile],
            "unsupported_summary_numbers_rejected": True,
            "strict_bronze_surface_flags": args.filter_profile != "bare",
        },
        "flag_counts": dict(sorted(flag_counts.items())),
        "primary_category_top30": dict(primary_counts.most_common(30)),
        "source_component_counts": dict(component_counts.most_common(30)),
        "outputs": {
            "survivors": survivors_path.as_posix(),
            "reject_sample": reject_sample_path.as_posix(),
            "report": report_path.as_posix(),
        },
    }
    write_json(report_path, report)
    print(f"[silver-refilter] profile={args.filter_profile} input_rows={len(rows)} survivors={len(survivors)}")
    if original_input_rows != len(rows):
        print(f"[silver-refilter] original_input_rows={original_input_rows} cut_rate={report['original_cut_rate']:.3f}")
    print(f"[silver-refilter] wrote {survivors_path}")


def load_gold_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {norm(row.get("paper_id")) for row in iter_jsonl(path) if norm(row.get("paper_id"))}


def command_merge_survivors(args: argparse.Namespace) -> None:
    gold_ids = load_gold_ids(args.gold_jsonl)
    by_paper: dict[str, dict[str, Any]] = {}
    duplicate_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    input_rows = 0
    for source_path in args.survivors:
        for row in iter_jsonl(source_path):
            input_rows += 1
            paper_id = norm(row.get("paper_id"))
            if not paper_id:
                duplicate_counts["missing_paper_id"] += 1
                continue
            if paper_id in gold_ids:
                duplicate_counts["already_gold_604"] += 1
                continue
            if paper_id in by_paper:
                duplicate_counts["duplicate_survivor_paper_id"] += 1
                continue
            by_paper[paper_id] = row
            source_counts[str(row.get("source_id", ""))] += 1

    rows = [by_paper[paper_id] for paper_id in sorted(by_paper)]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_jsonl, rows)
    write_json(
        args.manifest,
        {
            "created_at": utc_now_iso(),
            "created_by": "scripts/silver_summary_mining_workflow.py merge-survivors",
            "gold_exclusion_jsonl": args.gold_jsonl.as_posix(),
            "input_survivor_files": [path.as_posix() for path in args.survivors],
            "input_rows": input_rows,
            "merged_rows": len(rows),
            "duplicate_counts": dict(sorted(duplicate_counts.items())),
            "source_counts": dict(source_counts.most_common()),
            "output_jsonl": args.output_jsonl.as_posix(),
        },
    )
    print(f"[silver-merge] input_rows={input_rows} merged_rows={len(rows)}")
    print(f"[silver-merge] wrote {args.output_jsonl}")


def command_shard(args: argparse.Namespace) -> None:
    rows = list(iter_jsonl(args.input_jsonl))
    if args.limit:
        if args.sample:
            rng = random.Random(args.seed)
            rows = rng.sample(rows, min(args.limit, len(rows)))
        else:
            rows = rows[: args.limit]
    elif args.sample:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
    if args.limit and not args.sample:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[str] = []
    for shard_index in range(0, len(rows), args.shard_size):
        shard_rows = rows[shard_index : shard_index + args.shard_size]
        path = args.output_dir / f"shard_{len(shard_paths):03d}.jsonl"
        write_jsonl(path, shard_rows)
        shard_paths.append(path.as_posix())
    rubric = {
        "task": "Review candidate one-sentence summaries for a mobile paper feed.",
        "decisions": sorted(REVIEW_DECISIONS),
        "core_target": "A short, plain-language reason to open the paper, not a compressed abstract.",
        "silver_candidate": (
            "Ready to train at low weight: 8-14 words, one sentence, faithful, natural, "
            "plain, useful, low title overlap, and no dense jargon, numbers, equations, "
            "benchmark claims, or unsupported value attribution."
        ),
        "bronze_only": (
            "Probably faithful but not trainable as silver because it is too technical, "
            "too number-forward, too benchmarky, too acronym-heavy, or too awkward."
        ),
        "needs_rewrite": "The source supports a good silver summary, but this wording should not train directly.",
        "reject": "Wrong, unsupported, title-rehashy, malformed, or not useful.",
        "must_pass": [
            "8 to 14 words",
            "one sentence",
            "natural plain English for a broad technical reader",
            "useful as a mobile-feed reason to open the paper",
            "adds value beyond the title",
            "faithful to title and abstract",
            "technical terms only when central and unavoidable",
        ],
        "hard_reject_or_not_silver": [
            "unsupported claim, number, comparison, mechanism, or value judgment",
            "polarity flip or false value attribution",
            "copied title chunks",
            "dense technical noun stack",
            "uncommon acronym stack",
            "equation-like notation",
            "percentages, decimals, units, or benchmark metrics as the main point",
            "speculative improvement wording such as could improve",
            "method inventory wording",
            "requires domain expertise to parse",
        ],
        "issue_tags": sorted(UNSAFE_REVIEW_TAGS | {"academic_passive", "jargon_acronym", "jargon_stack"}),
        "instruction": "When in doubt, choose bronze_only rather than silver_candidate.",
        "output_contract": {
            "candidate_id": "copy from input",
            "decision": "one of the allowed decisions",
            "issue_tags": "list of short snake_case tags",
            "rationale": "short reason, no long quotes",
        },
    }
    rubric_path = args.output_dir / "review_rubric.json"
    manifest_path = args.output_dir / "shard_manifest.json"
    write_json(rubric_path, rubric)
    write_json(
        manifest_path,
        {
            "created_at": utc_now_iso(),
            "created_by": "scripts/silver_summary_mining_workflow.py shard",
            "input_jsonl": args.input_jsonl.as_posix(),
            "input_rows": len(rows),
            "shard_size": args.shard_size,
            "shard_count": len(shard_paths),
            "sample": bool(args.sample),
            "seed": args.seed if args.sample else None,
            "model_instruction": "Use gpt-5.4-mini subagents, fork_context=false, no service_tier override, no fast/spark mode.",
            "rubric": rubric_path.as_posix(),
            "shards": shard_paths,
        },
    )
    print(f"[silver-shard] rows={len(rows)} shards={len(shard_paths)}")
    print(f"[silver-shard] wrote {manifest_path}")


def load_reviews(paths: list[Path]) -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in iter_jsonl(path):
            candidate_id = norm(row.get("candidate_id"))
            decision = norm(row.get("decision"))
            if not candidate_id:
                raise ValueError(f"Review row in {path} is missing candidate_id")
            if decision not in REVIEW_DECISIONS:
                raise ValueError(f"Review row {candidate_id} in {path} has invalid decision {decision!r}")
            reviews[candidate_id] = row
    return reviews


def review_tags(row: dict[str, Any]) -> list[str]:
    tags = row.get("issue_tags", [])
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        cleaned = norm(tag).casefold().replace("-", "_").replace(" ", "_")
        if cleaned and cleaned not in seen:
            out.append(cleaned)
            seen.add(cleaned)
    return out


def final_silver_flags(candidate: dict[str, Any], review: dict[str, Any], gold_ids: set[str]) -> list[str]:
    flags: list[str] = []
    paper_id = norm(candidate.get("paper_id"))
    summary = norm(candidate.get("summary"))
    if paper_id in gold_ids:
        flags.append("already_gold_604")
    if norm(review.get("decision")) != "silver_candidate":
        flags.append("not_silver_candidate")
    if runtime.word_count(summary) > 14:
        flags.append("over_14_words")
    if runtime.word_count(summary) < 8:
        flags.append("under_8_words")
    if runtime.sentence_count(summary) != 1:
        flags.append("not_single_sentence")
    if candidate.get("title_overlap_ratio", 0) >= 0.45 or title_ngram_overlap(summary, norm(candidate.get("title"))) >= 3:
        flags.append("title_rehash")
    if candidate.get("unsupported_numbers"):
        flags.append("unsupported_number")
    if has_equation_heavy_text(summary):
        flags.append("equation_heavy")
    if dense_number_cluster(summary):
        flags.append("dense_numbers")
    if number_forward_summary(summary):
        flags.append("number_forward")
    if uncommon_acronym_terms(summary):
        flags.append("uncommon_acronym")
    flags.extend(natural_language_surface_flags(summary))
    unsafe = sorted(set(review_tags(review)) & UNSAFE_REVIEW_TAGS)
    flags.extend(f"review_{tag}" for tag in unsafe)
    return flags


def command_finalize(args: argparse.Namespace) -> None:
    candidates = list(iter_jsonl(args.candidates_jsonl))
    by_id = {str(row["candidate_id"]): row for row in candidates}
    reviews = load_reviews(args.review_jsonl)
    gold_ids = load_gold_ids(args.gold_jsonl)
    silver_rows: list[dict[str, Any]] = []
    reviewed_metadata: list[dict[str, Any]] = []
    missing_review = 0
    flag_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        review = reviews.get(candidate_id)
        if review is None:
            missing_review += 1
            continue
        decision_counts[norm(review.get("decision"))] += 1
        flags = final_silver_flags(candidate, review, gold_ids)
        flag_counts.update(flags)
        metadata = {
            **candidate,
            "review_decision": norm(review.get("decision")),
            "review_issue_tags": review_tags(review),
            "review_rationale": norm(review.get("rationale")),
            "final_silver_flags": flags,
            "quality_stage": "silver" if not flags else "reviewed_bronze",
        }
        reviewed_metadata.append(metadata)
        if not flags:
            silver_rows.append(metadata)

    args.release_dir.mkdir(parents=True, exist_ok=True)
    training_rows_path = args.release_dir / "training_rows.jsonl"
    reviewed_path = args.release_dir / "reviewed_metadata.jsonl"
    manifest_path = args.release_dir / "run_manifest.json"
    write_jsonl(training_rows_path, silver_rows)
    write_jsonl(reviewed_path, reviewed_metadata)
    write_json(
        manifest_path,
        {
            "created_at": utc_now_iso(),
            "created_by": "scripts/silver_summary_mining_workflow.py finalize",
            "candidates_jsonl": args.candidates_jsonl.as_posix(),
            "review_jsonl": [path.as_posix() for path in args.review_jsonl],
            "gold_exclusion_jsonl": args.gold_jsonl.as_posix(),
            "candidate_rows": len(candidates),
            "review_rows": len(reviews),
            "missing_review_rows": missing_review,
            "reviewed_rows": len(reviewed_metadata),
            "silver_rows": len(silver_rows),
            "decision_counts": dict(sorted(decision_counts.items())),
            "final_flag_counts": dict(sorted(flag_counts.items())),
            "outputs": {
                "training_rows": training_rows_path.as_posix(),
                "reviewed_metadata": reviewed_path.as_posix(),
                "manifest": manifest_path.as_posix(),
            },
        },
    )
    unknown_reviews = sorted(candidate_id for candidate_id in reviews if candidate_id not in by_id)
    if unknown_reviews:
        print(f"[silver-finalize] warning: {len(unknown_reviews)} reviews did not match candidates")
    print(f"[silver-finalize] reviewed={len(reviewed_metadata)} silver={len(silver_rows)} missing_review={missing_review}")
    print(f"[silver-finalize] wrote {training_rows_path}")


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
    user = str(prompt_spec["user_prompt_templates"]["title_abstract"]).format(
        title=norm(row.get("title")),
        abstract=norm(row.get("abstract")),
    )
    return [
        {"role": "system", "content": str(prompt_spec["system_prompt"])},
        {"role": "user", "content": user},
        {"role": "assistant", "content": summary},
    ]


def train_view_row(row: dict[str, Any], *, component: str, weight: float, prompt_spec: dict[str, Any]) -> dict[str, Any]:
    summary = norm(row.get("summary"))
    out = {
        "paper_id": norm(row.get("paper_id")),
        "title": norm(row.get("title")),
        "abstract": norm(row.get("abstract")),
        "summary": summary,
        "primary_category": norm(row.get("primary_category")),
        "categories": row.get("categories", []),
        "created_date": norm(row.get("created_date")),
        "source_split": norm(row.get("source_split") or row.get("quality_stage") or component),
        "teacher_model": norm(row.get("teacher_model") or row.get("provenance", {}).get("original_teacher_model")),
        "generator_prompt_version": norm(
            row.get("generator_prompt_version") or row.get("provenance", {}).get("original_generator_prompt_version")
        ),
        "review_prompt_version": norm(
            row.get("review_prompt_version") or row.get("provenance", {}).get("original_review_prompt_version")
        ),
        "dataset_component": component,
        "prompt_version": prompt_spec["prompt_version"],
        "input_mode": "title_abstract",
        "messages": render_messages(row, summary, prompt_spec),
        "loss_weight_hints": {
            "enabled": True,
            "base_weight": weight,
            "spans": [],
            "policy": f"natural_language_gold_silver_mix_v1 {component} weight {weight:g}",
        },
        "provenance": row.get("provenance", {}) | {
            "silver_source_path": row.get("source_path", ""),
            "quality_stage": row.get("quality_stage", ""),
        },
    }
    missing = [key for key in ("paper_id", "title", "abstract", "summary") if not out[key]]
    if missing:
        raise ValueError(f"Cannot build train row for {row.get('candidate_id')}: missing {missing}")
    return out


def command_build_train_view(args: argparse.Namespace) -> None:
    prompt_spec = load_prompt_spec(args.prompt_spec)
    gold_rows = list(iter_jsonl(args.gold_jsonl))
    silver_rows = list(iter_jsonl(args.silver_jsonl))
    converted = [
        train_view_row(row, component="gold_604", weight=args.gold_weight, prompt_spec=prompt_spec)
        for row in gold_rows
    ]
    converted.extend(
        train_view_row(row, component="silver", weight=args.silver_weight, prompt_spec=prompt_spec)
        for row in silver_rows
    )
    converted.sort(key=lambda row: (row["dataset_component"], row["paper_id"]))
    output_path = args.output_dir / "train.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    write_jsonl(output_path, converted)
    component_counts = Counter(row["dataset_component"] for row in converted)
    primary_counts = Counter(row.get("primary_category", "") for row in converted)
    write_json(
        manifest_path,
        {
            "created_at": utc_now_iso(),
            "created_by": "scripts/silver_summary_mining_workflow.py build-train-view",
            "purpose": "Natural-language gold plus lower-weight silver SFT training view.",
            "prompt_spec": args.prompt_spec.as_posix(),
            "prompt_version": prompt_spec["prompt_version"],
            "gold_jsonl": args.gold_jsonl.as_posix(),
            "silver_jsonl": args.silver_jsonl.as_posix(),
            "output_path": output_path.as_posix(),
            "row_count": len(converted),
            "component_counts": dict(sorted(component_counts.items())),
            "weights": {"gold_604": args.gold_weight, "silver": args.silver_weight},
            "primary_category_top30": dict(primary_counts.most_common(30)),
        },
    )
    print(f"[silver-train-view] wrote {len(converted)} rows to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    catalog = sub.add_parser("catalog", help="Build a manifest-only ranked source catalog.")
    catalog.add_argument("--search-root", action="append", type=Path, default=[Path("hf_bucket/experiments"), Path("hf_dataset/releases")])
    catalog.add_argument("--output-dir", type=Path, default=CATALOG_DIR)
    catalog.add_argument("--max-sources", type=int, default=0)
    catalog.set_defaults(func=command_catalog)

    first_filter = sub.add_parser("filter", help="Normalize and first-filter one source JSONL.")
    first_filter.add_argument("--input-jsonl", type=Path)
    first_filter.add_argument("--source-json", type=Path)
    first_filter.add_argument("--source-id")
    first_filter.add_argument("--output-dir", type=Path, default=FILTER_DIR)
    first_filter.add_argument("--min-words", type=int, default=8)
    first_filter.add_argument("--max-words", type=int, default=14)
    first_filter.add_argument(
        "--filter-profile",
        choices=sorted(BRONZE_SCORE_THRESHOLDS),
        default="bronze-balanced",
        help="Use bare for the original permissive gate, or a bronze profile for review-volume reduction.",
    )
    first_filter.add_argument("--bronze-score-threshold", type=int)
    first_filter.add_argument("--reject-sample-size", type=int, default=120)
    first_filter.add_argument("--delete-input-after-filter", action="store_true")
    first_filter.set_defaults(func=command_filter)

    refilter = sub.add_parser("refilter", help="Apply a stricter bronze profile to normalized survivor rows.")
    refilter.add_argument("--input-jsonl", type=Path, required=True)
    refilter.add_argument("--source-id")
    refilter.add_argument("--output-dir", type=Path, default=BRONZE_FILTER_DIR)
    refilter.add_argument("--min-words", type=int, default=8)
    refilter.add_argument("--max-words", type=int, default=14)
    refilter.add_argument(
        "--filter-profile",
        choices=sorted(BRONZE_SCORE_THRESHOLDS),
        default="bronze-balanced",
    )
    refilter.add_argument("--bronze-score-threshold", type=int)
    refilter.add_argument("--reject-sample-size", type=int, default=120)
    refilter.add_argument("--original-input-rows", type=int, default=0)
    refilter.set_defaults(func=command_refilter)

    merge = sub.add_parser("merge-survivors", help="Merge survivor shards and remove gold duplicates.")
    merge.add_argument("--survivors", action="append", type=Path, required=True)
    merge.add_argument("--gold-jsonl", type=Path, default=DEFAULT_GOLD)
    merge.add_argument("--output-jsonl", type=Path, default=RUN_ROOT / "candidate_pool" / "merged_survivors.jsonl")
    merge.add_argument("--manifest", type=Path, default=RUN_ROOT / "candidate_pool" / "manifest.json")
    merge.set_defaults(func=command_merge_survivors)

    shard = sub.add_parser("shard", help="Create review shards for subagents.")
    shard.add_argument("--input-jsonl", type=Path, required=True)
    shard.add_argument("--output-dir", type=Path, default=SHARD_DIR)
    shard.add_argument("--shard-size", type=int, default=40)
    shard.add_argument("--limit", type=int, default=0)
    shard.add_argument("--sample", action="store_true")
    shard.add_argument("--seed", type=int, default=20260523)
    shard.set_defaults(func=command_shard)

    finalize = sub.add_parser("finalize", help="Apply final silver gate to subagent reviews.")
    finalize.add_argument("--candidates-jsonl", type=Path, required=True)
    finalize.add_argument("--review-jsonl", action="append", type=Path, required=True)
    finalize.add_argument("--gold-jsonl", type=Path, default=DEFAULT_GOLD)
    finalize.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    finalize.set_defaults(func=command_finalize)

    train = sub.add_parser("build-train-view", help="Build a gold plus lower-weight silver SFT view.")
    train.add_argument("--gold-jsonl", type=Path, default=DEFAULT_GOLD)
    train.add_argument("--silver-jsonl", type=Path, default=DEFAULT_RELEASE_DIR / "training_rows.jsonl")
    train.add_argument("--prompt-spec", type=Path, default=DEFAULT_PROMPT_SPEC)
    train.add_argument("--output-dir", type=Path, default=DEFAULT_TRAIN_VIEW_DIR)
    train.add_argument("--gold-weight", type=float, default=1.0)
    train.add_argument("--silver-weight", type=float, default=0.35)
    train.set_defaults(func=command_build_train_view)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
