#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_JSONL = Path("hf_dataset/releases/recent_3y_balanced_10746_v1/training_rows.jsonl")
DEFAULT_GOLD_DIR = Path(
    "hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/"
    "mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5"
)
DEFAULT_PROMPT_SPEC = Path("prompts/mlx_lora/gemma_270m_it_abstract_to_feed_blurb_v5.json")
DEFAULT_OUTPUT_DIR = Path(
    "hf_bucket/experiments/mlx_lora/"
    "gemma_3_270m_it_natural_language_10746_heuristic3000_goldfinish_v1/"
    "dataset_heuristic3000_title_abstract_feed_blurb_v5"
)
DEFAULT_REUSE_EVAL_JSONL = Path(
    "hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_994_lora_lr1e5_long_v1/"
    "review_samples/ckpt360_800_vs_ckpt500_994_fresh40/generations.jsonl"
)

TARGET_SHARES = {
    "cs": 0.37,
    "physics": 0.28,
    "math": 0.16,
    "other": 0.19,
}
PHYSICS_PREFIXES = (
    "astro-ph",
    "cond-mat",
    "gr-qc",
    "hep-",
    "math-ph",
    "nucl-",
    "physics.",
    "quant-ph",
)
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
    "are",
    "around",
    "because",
    "been",
    "being",
    "between",
    "both",
    "but",
    "can",
    "could",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "its",
    "may",
    "more",
    "new",
    "not",
    "off",
    "one",
    "only",
    "our",
    "over",
    "paper",
    "shows",
    "study",
    "than",
    "that",
    "the",
    "their",
    "them",
    "these",
    "this",
    "through",
    "under",
    "using",
    "was",
    "were",
    "when",
    "where",
    "which",
    "while",
    "with",
    "without",
}
GENERIC_TERMS = {
    "approach",
    "framework",
    "method",
    "model",
    "models",
    "novel",
    "paper",
    "presents",
    "propose",
    "proposes",
    "show",
    "shows",
    "significant",
    "state",
    "study",
    "system",
    "technique",
}
PAYOFF_WORDS = {
    "bounds",
    "classifies",
    "connects",
    "detects",
    "enables",
    "estimates",
    "explains",
    "finds",
    "helps",
    "improves",
    "links",
    "maps",
    "predicts",
    "proves",
    "reduces",
    "reveals",
    "shows",
    "tracks",
}
VAGUE_WORDS = {
    "addresses",
    "analyzes",
    "considers",
    "demonstrates",
    "discusses",
    "explores",
    "investigates",
    "presents",
    "proposes",
    "studies",
}
BOILERPLATE_OPENING_RE = re.compile(
    r"^(this|the)\s+(paper|study|work|article)\b|^we\s+(propose|present|show|study|investigate)\b",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'/-]*")
FORMULAISH_RE = re.compile(r"[$\\{}_=<>≤≥∑√∞]|\\b[A-Za-z]\^[0-9]|\\bO\(")
DIGIT_RE = re.compile(r"\d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select high-quality deterministic support rows from the old 10,746-row release."
    )
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--prompt-spec", type=Path, default=DEFAULT_PROMPT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--exclude-eval-jsonl", type=Path, default=DEFAULT_REUSE_EVAL_JSONL)
    parser.add_argument("--target-count", type=int, default=3000)
    parser.add_argument("--max-per-primary-category", type=int, default=175)
    parser.add_argument("--seed", type=int, default=20260527)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            row["_source_line_number"] = line_number
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def norm_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def tokens(value: str) -> list[str]:
    return [token.lower().strip("'") for token in TOKEN_RE.findall(value)]


def content_tokens(value: str) -> list[str]:
    return [
        token
        for token in tokens(value)
        if len(token) >= 3 and token not in STOPWORDS and not DIGIT_RE.search(token)
    ]


def word_count(value: str) -> int:
    return len([part for part in value.split() if part.strip()])


def collapsed_umbrella(row: dict[str, Any]) -> str:
    primary = str(row.get("primary_category") or "")
    if primary.startswith("cs."):
        return "cs"
    if primary.startswith("math."):
        return "math"
    if primary.startswith(PHYSICS_PREFIXES):
        return "physics"
    return "other"


def target_counts(target_count: int) -> dict[str, int]:
    raw = {bucket: TARGET_SHARES[bucket] * target_count for bucket in TARGET_SHARES}
    counts = {bucket: math.floor(value) for bucket, value in raw.items()}
    remainder = target_count - sum(counts.values())
    for bucket, _ in sorted(
        ((bucket, raw[bucket] - counts[bucket]) for bucket in TARGET_SHARES),
        key=lambda item: (-item[1], item[0]),
    )[:remainder]:
        counts[bucket] += 1
    return counts


def load_prompt_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = ("prompt_version", "system_prompt", "user_prompt_templates")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"Prompt spec {path} missing keys: {missing}")
    if "title_abstract" not in spec["user_prompt_templates"]:
        raise ValueError(f"Prompt spec {path} is missing title_abstract template")
    return spec


def assistant_text(row: dict[str, Any]) -> str:
    if row.get("summary"):
        return norm_text(row["summary"])
    messages = row.get("messages") or []
    if isinstance(messages, list):
        for message in reversed(messages):
            if message.get("role") == "assistant":
                return norm_text(message.get("content", ""))
    return ""


def title_trigram_hit(title: str, summary: str) -> bool:
    title_tokens = [token for token in content_tokens(title) if token not in STOPWORDS]
    summary_joined = " ".join(content_tokens(summary))
    if len(title_tokens) < 3:
        return False
    for index in range(len(title_tokens) - 2):
        trigram = " ".join(title_tokens[index : index + 3])
        if trigram and trigram in summary_joined:
            return True
    return False


def supported_digit_count(summary: str, source: str) -> tuple[int, int]:
    summary_digit_terms = [token for token in tokens(summary) if DIGIT_RE.search(token)]
    source_digits = set(token for token in tokens(source) if DIGIT_RE.search(token))
    unsupported = sum(1 for token in summary_digit_terms if token not in source_digits)
    return len(summary_digit_terms), unsupported


def acronym_count(summary: str) -> int:
    return sum(
        1
        for token in re.findall(r"\b[A-Z][A-Z0-9-]{1,}\b", summary)
        if token not in {"LLM", "AI", "DNA", "RNA"}
    )


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    title = norm_text(row.get("title"))
    abstract = norm_text(row.get("abstract"))
    summary = assistant_text(row)
    source_text = f"{title} {abstract}"
    flags: list[str] = []
    hard_rejects: list[str] = []
    score = 100.0

    if not row.get("paper_id") or not title or not abstract or not summary:
        hard_rejects.append("missing_required_field")

    words = word_count(summary)
    if words < 8:
        hard_rejects.append("too_short")
    elif words > 14:
        hard_rejects.append("too_long")
    else:
        score += max(0, 8 - abs(12 - words) * 2)

    if "\n" in summary or "\r" in summary:
        hard_rejects.append("multiline_summary")
    if re.search(r"[.!?]\s+[A-Z0-9]", summary.strip().rstrip(".")):
        hard_rejects.append("multi_sentence")
    if BOILERPLATE_OPENING_RE.search(summary):
        hard_rejects.append("boilerplate_opening")
    if FORMULAISH_RE.search(summary):
        hard_rejects.append("formulaish_summary")

    summary_tokens = content_tokens(summary)
    title_tokens = set(content_tokens(title))
    abstract_tokens = set(content_tokens(abstract))
    source_tokens = set(content_tokens(source_text))
    overlap_tokens = sorted(set(summary_tokens) & title_tokens)
    title_overlap_count = len(overlap_tokens)
    has_title_trigram = title_trigram_hit(title, summary)
    if has_title_trigram and title_overlap_count >= 3:
        flags.append("title_trigram")
        score -= 28
    if title_overlap_count >= 5:
        hard_rejects.append("high_title_overlap")
        score -= (title_overlap_count - 4) * 8
    elif title_overlap_count >= 3:
        score -= (title_overlap_count - 2) * 4

    summary_digit_terms, unsupported_digit_terms = supported_digit_count(summary, source_text)
    if summary_digit_terms:
        hard_rejects.append("numeric_summary")
        score -= 10
    if unsupported_digit_terms:
        hard_rejects.append("unsupported_number")

    acronyms = acronym_count(summary)
    if acronyms >= 3:
        hard_rejects.append("acronym_stack")
    elif acronyms == 2:
        flags.append("two_acronyms")
        score -= 8

    generic_hits = sorted(set(summary_tokens) & GENERIC_TERMS)
    if generic_hits:
        flags.append("generic_terms")
        score -= min(18, 4 * len(generic_hits))

    vague_hits = sorted(set(summary_tokens) & VAGUE_WORDS)
    if vague_hits:
        flags.append("vague_verbs")
        score -= min(18, 5 * len(vague_hits))

    payoff_hits = sorted(set(summary_tokens) & PAYOFF_WORDS)
    if payoff_hits:
        score += min(10, 4 * len(payoff_hits))

    if summary_tokens:
        source_support_fraction = len(set(summary_tokens) & source_tokens) / len(set(summary_tokens))
        abstract_non_title_hits = sorted((set(summary_tokens) & abstract_tokens) - title_tokens)
    else:
        source_support_fraction = 0.0
        abstract_non_title_hits = []
    if source_support_fraction < 0.45:
        flags.append("low_source_token_support")
        score -= 12
    elif source_support_fraction > 0.70:
        score += 5
    score += min(10, len(abstract_non_title_hits) * 2)

    average_token_length = (
        sum(len(token) for token in summary_tokens) / len(summary_tokens) if summary_tokens else 0.0
    )
    if average_token_length > 9.0:
        flags.append("dense_words")
        score -= 8
    elif average_token_length < 7.5:
        score += 4

    if summary.endswith(":") or summary.endswith(";"):
        hard_rejects.append("fragment_punctuation")

    return {
        "paper_id": str(row.get("paper_id") or ""),
        "primary_category": str(row.get("primary_category") or ""),
        "umbrella": collapsed_umbrella(row),
        "title": title,
        "summary": summary,
        "score": round(score, 3),
        "selected": False,
        "hard_rejects": hard_rejects,
        "flags": flags,
        "word_count": words,
        "title_overlap_count": title_overlap_count,
        "title_overlap_tokens": overlap_tokens,
        "title_trigram": has_title_trigram,
        "summary_digit_terms": summary_digit_terms,
        "unsupported_digit_terms": unsupported_digit_terms,
        "acronym_count": acronyms,
        "generic_hits": generic_hits,
        "vague_hits": vague_hits,
        "payoff_hits": payoff_hits,
        "source_support_fraction": round(source_support_fraction, 3),
        "abstract_non_title_hits": abstract_non_title_hits[:8],
        "average_content_token_length": round(average_token_length, 3),
        "source_line_number": row.get("_source_line_number"),
    }


def render_messages(row: dict[str, Any], summary: str, prompt_spec: dict[str, Any]) -> list[dict[str, str]]:
    title = norm_text(row.get("title"))
    abstract = norm_text(row.get("abstract"))
    user = str(prompt_spec["user_prompt_templates"]["title_abstract"]).format(
        title=title,
        abstract=abstract,
    )
    return [
        {"role": "system", "content": str(prompt_spec["system_prompt"])},
        {"role": "user", "content": user},
        {"role": "assistant", "content": summary},
    ]


def convert_selected_row(
    row: dict[str, Any],
    audit: dict[str, Any],
    *,
    prompt_spec: dict[str, Any],
    source_jsonl: Path,
) -> dict[str, Any]:
    summary = audit["summary"]
    return {
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
        "dataset_component": "heuristic_10746_support",
        "prompt_version": prompt_spec["prompt_version"],
        "input_mode": "title_abstract",
        "messages": render_messages(row, summary, prompt_spec),
        "heuristic_selection": {
            "score": audit["score"],
            "flags": audit["flags"],
            "word_count": audit["word_count"],
            "title_overlap_count": audit["title_overlap_count"],
            "source_support_fraction": audit["source_support_fraction"],
            "source_line_number": audit["source_line_number"],
        },
        "provenance": {
            "source_path": str(source_jsonl),
            "source_line_number": audit["source_line_number"],
            "original_prompt_version": row.get("prompt_version", ""),
            "original_generator_prompt_version": row.get("generator_prompt_version", ""),
            "original_review_prompt_version": row.get("review_prompt_version", ""),
        },
    }


def clone_gold_holdout(row: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    cloned = dict(row)
    cloned["dataset_component"] = "strict_gold_holdout"
    cloned["provenance"] = {
        **(cloned.get("provenance") if isinstance(cloned.get("provenance"), dict) else {}),
        "source_path": str(source_path),
    }
    return cloned


def select_rows(
    rows: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    *,
    target_count: int,
    max_per_primary_category: int,
) -> list[int]:
    targets = target_counts(target_count)
    eligible_by_bucket: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, audit in enumerate(audits):
        if audit["hard_rejects"]:
            continue
        eligible_by_bucket[audit["umbrella"]].append((index, audit))

    selected: list[int] = []
    primary_counts: Counter[str] = Counter()
    for bucket, quota in targets.items():
        ranked = sorted(
            eligible_by_bucket[bucket],
            key=lambda item: (
                -item[1]["score"],
                item[1]["title_overlap_count"],
                abs(12 - item[1]["word_count"]),
                item[1]["source_line_number"],
            ),
        )
        for index, audit in ranked:
            primary = audit["primary_category"] or "unknown"
            if primary_counts[primary] >= max_per_primary_category:
                continue
            selected.append(index)
            primary_counts[primary] += 1
            if sum(1 for selected_index in selected if audits[selected_index]["umbrella"] == bucket) >= quota:
                break

    if len(selected) < target_count:
        selected_set = set(selected)
        overflow = sorted(
            (
                (index, audit)
                for index, audit in enumerate(audits)
                if not audit["hard_rejects"] and index not in selected_set
            ),
            key=lambda item: (
                -item[1]["score"],
                item[1]["title_overlap_count"],
                abs(12 - item[1]["word_count"]),
                item[1]["source_line_number"],
            ),
        )
        for index, audit in overflow:
            primary = audit["primary_category"] or "unknown"
            if primary_counts[primary] >= max_per_primary_category:
                continue
            selected.append(index)
            primary_counts[primary] += 1
            if len(selected) >= target_count:
                break
    return selected


def main() -> None:
    args = parse_args()
    if args.target_count <= 0:
        raise ValueError("--target-count must be positive")
    if args.max_per_primary_category <= 0:
        raise ValueError("--max-per-primary-category must be positive")

    prompt_spec = load_prompt_spec(args.prompt_spec)
    source_rows = load_jsonl(args.source_jsonl)
    gold_train = load_jsonl(args.gold_dir / "train.jsonl")
    gold_valid = load_jsonl(args.gold_dir / "valid.jsonl")
    gold_test = load_jsonl(args.gold_dir / "test.jsonl")
    gold_ids = {str(row.get("paper_id") or "") for row in gold_train + gold_valid + gold_test}
    eval_ids: set[str] = set()
    if args.exclude_eval_jsonl.exists():
        eval_ids = {str(row.get("paper_id") or "") for row in load_jsonl(args.exclude_eval_jsonl)}

    filtered_rows: list[dict[str, Any]] = []
    duplicate_excluded = 0
    eval_excluded = 0
    for row in source_rows:
        paper_id = str(row.get("paper_id") or "")
        if paper_id in gold_ids:
            duplicate_excluded += 1
            continue
        if paper_id in eval_ids:
            eval_excluded += 1
            continue
        filtered_rows.append(row)

    audits = [score_row(row) for row in filtered_rows]
    selected_indexes = set(
        select_rows(
            filtered_rows,
            audits,
            target_count=args.target_count,
            max_per_primary_category=args.max_per_primary_category,
        )
    )
    for index in selected_indexes:
        audits[index]["selected"] = True

    selected_rows = [
        convert_selected_row(
            filtered_rows[index],
            audits[index],
            prompt_spec=prompt_spec,
            source_jsonl=args.source_jsonl,
        )
        for index in sorted(
            selected_indexes,
            key=lambda item: (
                audits[item]["umbrella"],
                -audits[item]["score"],
                audits[item]["source_line_number"],
            ),
        )
    ]

    valid_rows = [clone_gold_holdout(row, source_path=args.gold_dir / "valid.jsonl") for row in gold_valid]
    test_rows = [clone_gold_holdout(row, source_path=args.gold_dir / "test.jsonl") for row in gold_test]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", selected_rows)
    write_jsonl(args.output_dir / "valid.jsonl", valid_rows)
    write_jsonl(args.output_dir / "test.jsonl", test_rows)
    write_jsonl(args.output_dir / "selection_audit.jsonl", audits)

    selected_audits = [audit for audit in audits if audit["selected"]]
    rejected_audits = [audit for audit in audits if not audit["selected"]]
    write_jsonl(args.output_dir / "selected_audit.jsonl", selected_audits)
    write_jsonl(
        args.output_dir / "reject_sample.jsonl",
        sorted(
            rejected_audits,
            key=lambda audit: (
                bool(audit["hard_rejects"]),
                -audit["score"],
                audit["source_line_number"],
            ),
        )[:250],
    )

    hard_reject_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    for audit in audits:
        hard_reject_counts.update(audit["hard_rejects"])
        flag_counts.update(audit["flags"])

    selected_bucket_counts = Counter(audit["umbrella"] for audit in selected_audits)
    selected_primary_counts = Counter(audit["primary_category"] for audit in selected_audits)
    selected_score_values = [audit["score"] for audit in selected_audits]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "scripts/build_mlx_heuristic_10746_support_dataset.py",
        "purpose": "Deterministic best-row selection from the old 10,746-row release for a broad LoRA burn-in before strict gold finishing.",
        "source_jsonl": str(args.source_jsonl),
        "gold_dir": str(args.gold_dir),
        "prompt_spec": str(args.prompt_spec),
        "prompt_version": prompt_spec["prompt_version"],
        "input_mode": "title_abstract",
        "output_dir": str(args.output_dir),
        "target_count": args.target_count,
        "max_per_primary_category": args.max_per_primary_category,
        "source_rows": len(source_rows),
        "source_rows_after_gold_eval_exclusions": len(filtered_rows),
        "excluded_gold_ids": duplicate_excluded,
        "excluded_eval_ids": eval_excluded,
        "selected_rows": len(selected_rows),
        "valid_rows": len(valid_rows),
        "test_rows": len(test_rows),
        "target_bucket_counts": target_counts(args.target_count),
        "selected_bucket_counts": dict(sorted(selected_bucket_counts.items())),
        "selected_primary_category_top30": dict(selected_primary_counts.most_common(30)),
        "hard_reject_counts": dict(hard_reject_counts.most_common()),
        "flag_counts": dict(flag_counts.most_common()),
        "selected_score": {
            "min": round(min(selected_score_values), 3) if selected_score_values else None,
            "max": round(max(selected_score_values), 3) if selected_score_values else None,
            "avg": round(sum(selected_score_values) / len(selected_score_values), 3)
            if selected_score_values
            else None,
        },
        "selected_word_counts": dict(Counter(audit["word_count"] for audit in selected_audits).most_common()),
        "selected_title_overlap_counts": dict(
            Counter(audit["title_overlap_count"] for audit in selected_audits).most_common()
        ),
        "selected_flag_counts": dict(Counter(flag for audit in selected_audits for flag in audit["flags"]).most_common()),
    }
    write_json(args.output_dir / "manifest.json", manifest)

    readme = [
        "# Heuristic 10746 Support Dataset",
        "",
        f"- Rows: `{len(selected_rows)}` selected from `{len(source_rows)}` old 10,746-release rows.",
        f"- Prompt: `{prompt_spec['prompt_version']}` / `title_abstract`.",
        "- Labels are existing old-release summaries; no generated or manually written labels were added.",
        "- `valid.jsonl` and `test.jsonl` are strict-gold holdouts from the 994-row release.",
        "- Selection is deterministic and audited in `selection_audit.jsonl`.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(f"[heuristic-10746] wrote {len(selected_rows)} train rows to {args.output_dir / 'train.jsonl'}")
    print(f"[heuristic-10746] selected buckets={dict(sorted(selected_bucket_counts.items()))}")
    print(f"[heuristic-10746] hard rejects={dict(hard_reject_counts.most_common(10))}")
    print(f"[heuristic-10746] selected score={manifest['selected_score']}")


if __name__ == "__main__":
    main()
