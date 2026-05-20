#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

try:
    from scripts.faithfulness_cleanup_common import unsupported_labels, unsupported_numbers
except ImportError:  # pragma: no cover
    from faithfulness_cleanup_common import unsupported_labels, unsupported_numbers


DEFAULT_ROOT = Path(
    "hf_bucket/experiments/mlx_lora/runtime_shift/"
    "llama_cpp_q5_clean_slate_broad_v1_20260508/"
    "preference_mining/q5_step2400_2000_v1"
)
DEFAULT_INPUT = DEFAULT_ROOT / "eval_inputs_2000.jsonl"
DEFAULT_GENERATIONS = DEFAULT_ROOT / "q5_step2400_llama_generations_2000.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "heuristic_screen"

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|\d+(?:[./:-]\d+)*%?")
SENTENCE_END_RE = re.compile(r"""[.!?]+(?:['")\]]+)?(?=\s|$)""")
DOTTED_ABBREVIATION_RE = re.compile(
    r"""
    (?:
        \b(?:[A-Za-z]\.){2,} |
        \b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|fig|eq|cf|approx|resp|ref|sec|no|al|chap|appx)\. |
        \b(?:e\.g|i\.e|et\ al)\.
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
FORMULA_RE = re.compile(r"\\[$]|\\[a-zA-Z]+|\^|_\\{|\\frac|<sup>|</sup>|\\times")
BAD_REPEAT_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9-]{2,})\b(?:\s+\1\b){1,}", re.IGNORECASE)
POLARITY_SOURCE_RE = re.compile(
    r"\b(?:not|no |without|rather than|instead of|cannot|could not|"
    r"fails?|falls?\s+short|struggl\w*|lack(?:s|ing)?|limitation|limited|"
    r"incomplete|null|no significant|not intended|unclear|unknown|"
    r"challenge|barrier|impossible|hardness|complete|"
    r"shortcoming|drawback|downside|not\s+yet|"
    r"remains?\s+(?:a\s+)?(?:challenge|open|unclear|difficult))\b",
    re.IGNORECASE,
)
POSITIVE_VERB_RE = re.compile(
    r"\b(?:outperform\w*|surpass\w*|improv\w*|boost\w*|reduc\w*|cut\w*|"
    r"increas\w*|decreas\w*|accelerat\w*|stabili[sz]\w*|optimi[sz]\w*|"
    r"prevent\w*|mitigat\w*|detect\w*|predict\w*|reveal\w*|"
    r"explain\w*|demonstrat\w*|achiev\w*|yield\w*|"
    r"converg\w*|solv\w*|guarante\w*|generaliz\w*|eliminat\w*|"
    r"captur\w*|scal[ei]\w*|match\w*|exceed\w*|reach\w*|"
    r"attain\w*|obtain\w*|deliver\w*|produc\w*|identif\w*|enabl\w*)\b",
    re.IGNORECASE,
)
_NOUN_SUFFIX_RE = re.compile(
    r"(?:tion|sion|ence|ance|ment|ness|ity|ism|ology|graphy|metry)s?$",
    re.IGNORECASE,
)
SELF_REFERENCE_RE = re.compile(
    r"\b(?:our|we|this\s+paper|this\s+work|the\s+proposed|"
    r"we\s+find|we\s+observe|we\s+show|our\s+results?|our\s+method|"
    r"our\s+approach|our\s+model|our\s+framework|our\s+system|"
    r"in\s+this\s+(?:paper|work|study)|we\s+propose|we\s+present|"
    r"we\s+introduce|we\s+report|we\s+demonstrate|"
    r"the\s+present\s+(?:paper|work|study))\b",
    re.IGNORECASE,
)
PRIOR_WORK_RE = re.compile(
    r"\b(?:previous|existing|prior|traditional|"
    r"conventional|earlier|other\s+methods?|previous\s+work|"
    r"prior\s+work|past\s+work|"
    r"recent\s+work|baseline\s+methods?|"
    r"commonly\s+used|widely\s+used|"
    r"in\s+the\s+literature|in\s+previous\s+work|in\s+prior\s+work|"
    r"previously\s+(?:proposed|reported|published|known))\b",
    re.IGNORECASE,
)
NEGATION_CONTEXT_WINDOW = 120
NEGATION_FORWARD_WINDOW = 50
_MAX_NEGATION_GAP_WORDS = 4
_NEGATING_PREFIX_RE = re.compile(
    r"(?:fails?\s+to|does\s+not|cannot|could\s+not|unable\s+to|not|no)\b",
    re.IGNORECASE,
)

VALID_TAGS = {
    "invented_number",
    "wrong_number_binding",
    "numeric_error",
    "unsupported_number",
    "unsupported_entity",
    "unsupported_entity_label",
    "unsupported_mechanism",
    "unsupported_causal_claim",
    "overstrong_claim",
    "polarity_flip",
    "polarity_risk",
    "misses_point",
    "malformed_repetition",
    "non_answer",
    "title_rehash",
    "too_generic",
    "copies_q5_failure",
    "style_drift",
    "local_filter",
    "none",
}


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


def normalize(value: Any) -> str:
    return " ".join(str(value or "").split())


def word_count(value: str) -> int:
    return len(WORD_RE.findall(value or ""))


def title_overlap(title: str, summary: str) -> float:
    title_tokens = {token.casefold() for token in WORD_RE.findall(title or "") if len(token) > 3}
    summary_tokens = {token.casefold() for token in WORD_RE.findall(summary or "") if len(token) > 3}
    if not title_tokens or not summary_tokens:
        return 0.0
    return len(title_tokens & summary_tokens) / len(summary_tokens)


def _polarity_risk(source_text: str, summary: str) -> bool:
    """Check whether the summary uses positive claim language for a limitation
    that the source attributes to the paper itself (not to prior work).

    The original heuristic flagged any source negation + any summary positive
    verb.  This version suppresses the flag when every source-side negation
    sits in a prior-work context ("previous methods fail" -> summary says
    "our method succeeds" is correct).  It only fires when at least one
    negation appears alongside self-referential language ("our", "we",
    "this paper"), suggesting the paper acknowledges a limitation that the
    summary then overwrites with positive framing.

    Uses an asymmetric context window: looks backward generously (to catch
    "Our method" before "fails") but only a short distance forward (to
    avoid pulling in the contribution sentence after the limitation).

    Additionally filters out positive verb matches in the summary that are
    themselves negated (e.g. "fails to converge" should not match
    "converge" as a positive verb).
    """
    pos_matches = [m for m in POSITIVE_VERB_RE.finditer(summary)
                   if not _verb_is_negated(summary, m.start())
                   and not _NOUN_SUFFIX_RE.search(m.group())]
    if not pos_matches:
        return False

    negation_matches = list(POLARITY_SOURCE_RE.finditer(source_text))
    if not negation_matches:
        return False

    for match in negation_matches:
        start = max(0, match.start() - NEGATION_CONTEXT_WINDOW)
        end = min(len(source_text), match.end() + NEGATION_FORWARD_WINDOW)
        context = source_text[start:end]

        self_hits = len(SELF_REFERENCE_RE.findall(context))
        prior_hits = len(PRIOR_WORK_RE.findall(context))

        if self_hits > prior_hits:
            return True

    return False


def _verb_is_negated(text: str, verb_pos: int) -> bool:
    """True when the verb at *verb_pos* is preceded by a negation within a few words."""
    before = text[:verb_pos]
    neg_matches = list(_NEGATING_PREFIX_RE.finditer(before))
    if not neg_matches:
        return False
    last_neg = neg_matches[-1]
    gap_text = text[last_neg.end():verb_pos]
    gap_words = len([w for w in gap_text.split() if w.strip()])
    return gap_words <= _MAX_NEGATION_GAP_WORDS


def _protect_abbreviation_periods(value: str) -> str:
    def _replace_dot(match: re.Match[str]) -> str:
        return match.group(0).replace(".", "<DOT>")

    return DOTTED_ABBREVIATION_RE.sub(_replace_dot, value)


def sentence_count(value: str) -> int:
    text = value.strip()
    if not text:
        return 0
    protected = _protect_abbreviation_periods(text)
    endings = SENTENCE_END_RE.findall(protected)
    if endings:
        return len(endings)
    return 1


def classify_flags(source: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    title = normalize(source.get("title"))
    abstract = normalize(source.get("abstract"))
    source_text = f"{title}\n{abstract}"
    summary = normalize(generation.get("normalized_output"))
    wc = word_count(summary)
    title_ratio = title_overlap(title, summary)
    unsup_nums = unsupported_numbers(summary, source_text)
    unsup_labels = unsupported_labels(summary, source_text)

    flags: list[str] = []
    tags: list[str] = []
    if not generation.get("ok", True) or not summary:
        flags.append("empty_or_error")
        tags.append("non_answer")
    if wc < 10:
        flags.append("under10")
    if wc > 15:
        flags.append("over15")
    if wc > 20:
        flags.append("over20")
        tags.append("style_drift")
    if sentence_count(summary) != 1:
        flags.append("not_single_sentence")
        tags.append("style_drift")
    if title_ratio > 0.72:
        flags.append("high_title_overlap")
        tags.append("title_rehash")
    if unsup_nums:
        flags.append("unsupported_number")
        tags.extend(["unsupported_number", "wrong_number_binding"])
    if unsup_labels:
        flags.append("unsupported_label")
        tags.append("unsupported_entity_label")
    if FORMULA_RE.search(summary) or BAD_REPEAT_RE.search(summary):
        flags.append("malformed_or_notation")
        tags.append("malformed_repetition")
    if _polarity_risk(source_text, summary):
        flags.append("polarity_risk")
        tags.append("polarity_risk")
    if set(generation.get("format_flags", [])) & {"prompt_echo", "word_repeat3", "bigram_repeat3", "empty_output"}:
        flags.append("runtime_format_risk")
        tags.append("malformed_repetition")

    tags = [tag for tag in dict.fromkeys(tags) if tag in VALID_TAGS]
    high_signal = bool(
        set(flags)
        & {
            "empty_or_error",
            "over20",
            "unsupported_number",
            "unsupported_label",
            "malformed_or_notation",
            "polarity_risk",
            "runtime_format_risk",
        }
    )
    medium_signal = bool(flags) and not high_signal
    return {
        "summary": summary,
        "word_count": wc,
        "title_overlap": round(title_ratio, 4),
        "unsupported_numbers": unsup_nums,
        "unsupported_labels": unsup_labels,
        "flags": flags,
        "issue_tags": tags or ["style_drift"],
        "signal_tier": "high" if high_signal else "medium" if medium_signal else "none",
    }


def reviewer_note(screen: dict[str, Any]) -> str:
    parts = [f"heuristic flags: {', '.join(screen['flags'])}"]
    if screen["unsupported_numbers"]:
        parts.append(f"unsupported numbers: {', '.join(screen['unsupported_numbers'])}")
    if screen["unsupported_labels"]:
        parts.append(f"unsupported labels: {', '.join(screen['unsupported_labels'][:8])}")
    if screen["word_count"] < 10 or screen["word_count"] > 15:
        parts.append(f"word count: {screen['word_count']}")
    parts.append(f"title overlap: {screen['title_overlap']}")
    return " | ".join(parts)


def build_candidate(source: dict[str, Any], generation: dict[str, Any], screen: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": str(source["paper_id"]),
        "candidate_id": str(source.get("eval_id") or source["paper_id"]),
        "eval_id": str(source.get("eval_id") or ""),
        "title": normalize(source.get("title")),
        "abstract": normalize(source.get("abstract")),
        "primary_category": source.get("primary_category"),
        "categories": source.get("categories", []),
        "created_date": source.get("created_date"),
        "selection_bucket": source.get("selection_bucket", "q5_step2400_random_2000"),
        "q5_hardening_axes": screen["issue_tags"],
        "q5_issue_tags": screen["issue_tags"],
        "q5_severity_source": f"q5_step2400_2000_heuristic_{screen['signal_tier']}",
        "reviewer_note": reviewer_note(screen),
        "q5_summary": screen["summary"],
        "q8_summary": normalize(source.get("source_release_summary") or source.get("summary")),
        "q8_clean": None,
        "q8_severe": None,
        "q8_mangled": None,
        "source_split": "q5_step2400_2000_preference_mining_v1",
        "heuristic_flags": screen["flags"],
        "heuristic_signal_tier": screen["signal_tier"],
        "heuristic_unsupported_numbers": screen["unsupported_numbers"],
        "heuristic_unsupported_labels": screen["unsupported_labels"],
        "heuristic_word_count": screen["word_count"],
        "heuristic_title_overlap": screen["title_overlap"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen q5 step-2400 mining generations and emit repair candidates."
    )
    parser.add_argument("--input-jsonl", default=str(DEFAULT_INPUT))
    parser.add_argument("--generations-jsonl", default=str(DEFAULT_GENERATIONS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-candidates", type=int, default=800)
    parser.add_argument("--include-medium", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_rows = load_jsonl(Path(args.input_jsonl))
    generation_rows = load_jsonl(Path(args.generations_jsonl))
    source_by_eval = {str(row.get("eval_id")): row for row in input_rows}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    screened: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for generation in generation_rows:
        eval_id = str(generation.get("eval_id"))
        source = source_by_eval.get(eval_id)
        if source is None:
            raise ValueError(f"Generation eval_id={eval_id} not found in inputs")
        screen = classify_flags(source, generation)
        detail = {
            "eval_id": eval_id,
            "paper_id": str(source["paper_id"]),
            "title": source.get("title"),
            "summary": screen["summary"],
            **screen,
        }
        screened.append(detail)
        if screen["signal_tier"] == "high" or (args.include_medium and screen["signal_tier"] == "medium"):
            candidates.append(build_candidate(source, generation, screen))

    def candidate_sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
        flags = set(row.get("heuristic_flags", []))
        priority = 0 if row.get("heuristic_signal_tier") == "high" else 1
        risk = (
            8 * int("unsupported_number" in flags)
            + 7 * int("unsupported_label" in flags)
            + 6 * int("polarity_risk" in flags)
            + 5 * int("malformed_or_notation" in flags)
            + 4 * int("over20" in flags)
            + 2 * int("high_title_overlap" in flags)
        )
        return (priority, -risk, -int(row.get("heuristic_word_count", 0) or 0), str(row["paper_id"]))

    candidates.sort(key=candidate_sort_key)
    selected_candidates = candidates[: args.max_candidates]

    write_jsonl(output_dir / "screened_rows.jsonl", screened)
    write_jsonl(output_dir / "repair_candidates.jsonl", selected_candidates)
    write_jsonl(output_dir / "repair_candidates_all.jsonl", candidates)

    word_counts = [row["word_count"] for row in screened if row["summary"]]
    flag_counts = Counter(flag for row in screened for flag in row["flags"])
    tag_counts = Counter(tag for row in selected_candidates for tag in row["q5_issue_tags"])
    report = {
        "input_jsonl": args.input_jsonl,
        "generations_jsonl": args.generations_jsonl,
        "rows": len(screened),
        "ok_generation_rows": sum(1 for row in generation_rows if row.get("ok")),
        "flagged_rows": sum(1 for row in screened if row["flags"]),
        "high_signal_rows": sum(1 for row in screened if row["signal_tier"] == "high"),
        "medium_signal_rows": sum(1 for row in screened if row["signal_tier"] == "medium"),
        "selected_repair_candidates": len(selected_candidates),
        "all_repair_candidates": len(candidates),
        "flag_counts": dict(sorted(flag_counts.items())),
        "selected_issue_tag_counts": dict(sorted(tag_counts.items())),
        "word_count": {
            "mean": round(mean(word_counts), 2) if word_counts else None,
            "median": round(median(word_counts), 2) if word_counts else None,
            "min": min(word_counts) if word_counts else None,
            "max": max(word_counts) if word_counts else None,
        },
    }
    (output_dir / "heuristic_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# q5 step-2400 2000-row heuristic screen",
        "",
        f"- Rows: {report['rows']}",
        f"- OK generation rows: {report['ok_generation_rows']}",
        f"- Flagged rows: {report['flagged_rows']}",
        f"- High-signal rows: {report['high_signal_rows']}",
        f"- Medium-signal rows: {report['medium_signal_rows']}",
        f"- Selected repair candidates: {report['selected_repair_candidates']}",
        "",
        "## Flag Counts",
        "",
    ]
    for flag, count in sorted(flag_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{flag}`: {count}")
    lines.extend(["", "## Selected Issue Tags", ""])
    for tag, count in sorted(tag_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{tag}`: {count}")
    (output_dir / "heuristic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[q5-mining-screen] screened {len(screened)} rows")
    print(f"[q5-mining-screen] selected {len(selected_candidates)} repair candidates")
    print(f"[q5-mining-screen] report: {output_dir / 'heuristic_report.md'}")


if __name__ == "__main__":
    main()
