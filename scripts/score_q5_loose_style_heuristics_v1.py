#!/usr/bin/env python3
"""Loose style heuristics for q5 burn-in output triage.

These flags are candidate selectors, not pass/fail judgments. They are meant to
surface rows for human or teacher-model review and to find low-risk self-anchor
rows from a post-burn-in model.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from faithfulness_cleanup_common import unsupported_numbers
except Exception:
    unsupported_numbers = None


OUT_ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/eval/burnin150_style_probe_5000_v1/heuristic_screen")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")
DIGIT_RE = re.compile(r"\d")
NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?(?:\s?[%x×]|e[-+]?\d+)?|\b\d+[A-Za-z]+|\b[A-Za-z]+\d+[A-Za-z0-9-]*")
FORMULA_RE = re.compile(r"<[^>]+>|\$|\\[A-Za-z]+|\bO\([^)]*\)|\^|_|[≤≥≈]|[α-ωΑ-Ω]")
BANNED_OPENING_RE = re.compile(r"^\s*(this paper|the paper|the authors|we propose|in this work|in this paper)\b", re.I)
WEAK_VERB_RE = re.compile(r"\b(presents?|explores?|studies|investigates|discusses|introduces|proposes|describes)\b", re.I)
GENERIC_PHRASE_RE = re.compile(
    r"\b(improves performance|shows effectiveness|provides insights|enhances understanding|"
    r"various tasks|different domains|multiple settings|better results|high performance|new method|novel approach)\b",
    re.I,
)
BENCHMARK_RE = re.compile(r"\b(outperform|outperforms|state-of-the-art|best|superior|beats?|benchmark|leaderboard|achieves?)\b", re.I)
CAVEAT_RE = re.compile(r"\b(but|however|although|despite|except|fail|fails|failed|failure|limited|limitation|cannot|can not|does not|do not|without|lack|lacks|tradeoff|trade-off|risk|uncertain)\b", re.I)
NEGATIVE_RE = re.compile(r"\b(fail|fails|failed|failure|cannot|can not|does not|do not|without|lack|lacks|risk|uncertain|incorrect|wrong|limited)\b", re.I)
POSITIVE_VERB_RE = re.compile(r"\b(enables?|improves?|reduces?|increases?|boosts?|supports?|reveals?|solves?|proves?|guarantees?|eliminates?|confirms?)\b", re.I)
STRONG_CLAIM_RE = re.compile(r"\b(proves?|guarantees?|eliminates?|confirms?|solves?|robust|accurate|efficient|scalable|low-cost)\b", re.I)
WEIRD_MORPH_RE = re.compile(r"\b[A-Za-z]{8,}(?:fulness|istic|ova|ens|izingness|ationness)\b")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def norm(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def word_count(text: str) -> int:
    return len(words(text))


def word_set(text: str, min_len: int = 3) -> set[str]:
    return {w.lower() for w in words(text) if len(w) >= min_len}


def sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(re.findall(r"[.!?]+(?:['\")\]]+)?(?=\s|$)", stripped))


def title_overlap(summary: str, title: str) -> tuple[int, float]:
    summary_words = word_set(summary)
    title_words = word_set(title)
    if not summary_words:
        return 0, 0.0
    overlap = len(summary_words & title_words)
    return overlap, overlap / len(summary_words)


def title_phrase_overlap(summary: str, title: str, n: int = 4) -> bool:
    def grams(text: str) -> set[tuple[str, ...]]:
        toks = [w.lower() for w in words(text)]
        return {tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))}

    return bool(grams(summary) & grams(title))


def adjacent_duplicate(text: str) -> bool:
    prev = ""
    for token in (w.lower() for w in words(text)):
        if token == prev and len(token) > 2:
            return True
        prev = token
    return False


def repeated_stem_window(text: str, window: int = 6) -> bool:
    toks = [w.lower() for w in words(text) if len(w) >= 5]
    stems = [token[:6] for token in toks]
    for i, stem in enumerate(stems):
        if stem in stems[i + 1 : i + window]:
            return True
    return False


def unsupported_number_list(summary: str, row: dict[str, Any]) -> list[str]:
    if unsupported_numbers is None:
        return []
    try:
        return unsupported_numbers(summary, f"{row.get('title', '')}\n{row.get('abstract', '')}")
    except Exception:
        return []


def rough_fragment(summary: str) -> bool:
    if not summary:
        return True
    finite_like = re.search(r"\b(is|are|was|were|be|being|been|has|have|had|can|could|may|might|must|will|would|shows?|finds?|reveals?|enables?|supports?|uses?|generates?|improves?|reduces?|links?|maps?|detects?|predicts?|classifies?|estimates?|models?)\b", summary, re.I)
    return not bool(finite_like)


def style_flags(summary: str, row: dict[str, Any]) -> dict[str, Any]:
    summary = norm(summary)
    title = norm(row.get("title", ""))
    source = f"{title}\n{row.get('abstract', '')}"
    wc = word_count(summary)
    flags: list[str] = []
    categories: dict[str, list[str]] = defaultdict(list)

    def add(category: str, flag: str) -> None:
        flags.append(flag)
        categories[category].append(flag)

    if wc < 9:
        add("length_shape", "under_9_words")
    if wc > 16:
        add("length_shape", "over_16_words")
    if sentence_count(summary) > 1:
        add("length_shape", "multi_sentence")
    if ";" in summary or ":" in summary:
        add("length_shape", "colon_or_semicolon")
    if rough_fragment(summary):
        add("length_shape", "fragment_like")

    nums = NUM_RE.findall(summary)
    if DIGIT_RE.search(summary):
        add("digit_symbol_residue", "digit_or_numberlike_token")
    if len(nums) >= 2:
        add("digit_symbol_residue", "dense_numberlike_tokens")
    if FORMULA_RE.search(summary):
        add("digit_symbol_residue", "formula_or_symbol_residue")
    unsup = unsupported_number_list(summary, row)
    if unsup:
        add("digit_symbol_residue", "unsupported_number")

    if adjacent_duplicate(summary):
        add("awkward_malformed", "adjacent_duplicate_word")
    if repeated_stem_window(summary):
        add("awkward_malformed", "repeated_stem_window")
    if WEIRD_MORPH_RE.search(summary):
        add("awkward_malformed", "weird_morphology")
    if len([w for w in words(summary) if "-" in w]) >= 4:
        add("awkward_malformed", "hyphen_dense")
    if len([w for w in words(summary) if len(w) >= 12]) >= 5:
        add("awkward_malformed", "long_token_dense")

    if BANNED_OPENING_RE.search(summary):
        add("generic_low_value", "boilerplate_opening")
    if WEAK_VERB_RE.search(summary):
        add("generic_low_value", "weak_academic_verb")
    if GENERIC_PHRASE_RE.search(summary):
        add("generic_low_value", "generic_benefit_phrase")
    source_overlap = len(word_set(summary, 5) & word_set(source, 5))
    if source_overlap <= 2 and wc >= 9:
        add("generic_low_value", "low_source_specificity")

    overlap_count, overlap_ratio = title_overlap(summary, title)
    if overlap_ratio >= 0.55:
        add("title_rehash", "soft_title_overlap")
    if overlap_ratio >= 0.75:
        add("title_rehash", "strong_title_overlap")
    if title_phrase_overlap(summary, title):
        add("title_rehash", "copied_title_phrase")

    if BENCHMARK_RE.search(summary):
        add("benchmark_winner_shape", "benchmark_or_winner_wording")

    source_has_caveat = bool(CAVEAT_RE.search(source))
    summary_has_caveat = bool(CAVEAT_RE.search(summary))
    if source_has_caveat and not summary_has_caveat and POSITIVE_VERB_RE.search(summary):
        add("value_polarity_risk", "source_caveat_absent_with_positive_output")
    if summary_has_caveat and not source_has_caveat:
        add("value_polarity_risk", "output_adds_caveat_not_in_source")
    if NEGATIVE_RE.search(source) and POSITIVE_VERB_RE.search(summary):
        add("value_polarity_risk", "negative_source_positive_output")

    source_terms = word_set(source, 5)
    introduced = sorted(w for w in word_set(summary, 6) if w not in source_terms)
    introduced_strong = [w for w in introduced if STRONG_CLAIM_RE.search(w)]
    if introduced_strong:
        add("unsupported_specificity", "introduced_strong_claim_word")
    if STRONG_CLAIM_RE.search(summary) and not STRONG_CLAIM_RE.search(source):
        add("unsupported_specificity", "strong_claim_not_in_source")

    repair_weights = {
        "digit_symbol_residue": 100,
        "awkward_malformed": 85,
        "value_polarity_risk": 75,
        "benchmark_winner_shape": 60,
        "title_rehash": 45,
        "generic_low_value": 35,
        "length_shape": 25,
        "unsupported_specificity": 55,
    }
    score = sum(repair_weights[category] * len(items) for category, items in categories.items())
    anchor_blocking_categories = {
        "digit_symbol_residue",
        "awkward_malformed",
        "benchmark_winner_shape",
        "value_polarity_risk",
        "unsupported_specificity",
    }
    self_anchor = (
        10 <= wc <= 15
        and sentence_count(summary) == 1
        and not any(category in categories for category in anchor_blocking_categories)
        and "boilerplate_opening" not in flags
        and "strong_title_overlap" not in flags
        and overlap_ratio < 0.55
        and source_overlap >= 3
    )

    return {
        "summary": summary,
        "word_count": wc,
        "sentence_count": sentence_count(summary),
        "title_overlap_count": overlap_count,
        "title_overlap_ratio": round(overlap_ratio, 4),
        "source_overlap_terms": source_overlap,
        "numbers": nums,
        "unsupported_numbers": unsup,
        "introduced_terms_sample": introduced[:12],
        "flags": flags,
        "flag_categories": {key: value for key, value in sorted(categories.items())},
        "repair_score": score,
        "is_repair_candidate": bool(score >= 75 or any(category in categories for category in ("digit_symbol_residue", "awkward_malformed", "value_polarity_risk"))),
        "is_self_anchor_candidate": self_anchor,
    }


def output_by_pid(path: Path) -> dict[str, str]:
    outputs = {}
    for row in load_jsonl(path):
        pid = str(row.get("paper_id", ""))
        if not pid:
            continue
        outputs[pid] = norm(row.get("normalized_output") or row.get("raw_output") or row.get("summary", ""))
    return outputs


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"rows": 0}
    flag_counts = Counter(flag for row in records for flag in row["style_heuristics"]["flags"])
    category_counts = Counter(cat for row in records for cat in row["style_heuristics"]["flag_categories"])
    wc_values = [row["style_heuristics"]["word_count"] for row in records]
    return {
        "rows": len(records),
        "avg_word_count": round(mean(wc_values), 2),
        "repair_candidates": sum(1 for row in records if row["style_heuristics"]["is_repair_candidate"]),
        "self_anchor_candidates": sum(1 for row in records if row["style_heuristics"]["is_self_anchor_candidate"]),
        "flag_counts": dict(flag_counts.most_common()),
        "category_counts": dict(category_counts.most_common()),
    }


def write_report(path: Path, records: list[dict[str, Any]], by_split: dict[str, list[dict[str, Any]]]) -> None:
    summary = summarize(records)
    lines = [
        "# q5 loose style heuristic report",
        "",
        "These are loose candidate-selection heuristics, not final quality judgments.",
        "",
        f"Rows: `{summary['rows']}`",
        f"Repair candidates: `{summary['repair_candidates']}`",
        f"Self-anchor candidates: `{summary['self_anchor_candidates']}`",
        f"Average word count: `{summary['avg_word_count']}`",
        "",
        "## By Split",
        "",
        "| split | rows | repair | self-anchor | avg wc |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, rows in sorted(by_split.items()):
        s = summarize(rows)
        lines.append(f"| {split} | {s['rows']} | {s['repair_candidates']} | {s['self_anchor_candidates']} | {s.get('avg_word_count', 0)} |")

    lines.extend(["", "## Flag Categories", "", "| category | rows |", "|---|---:|"])
    for category, count in summary["category_counts"].items():
        lines.append(f"| {category} | {count} |")

    lines.extend(["", "## Top Repair Candidates", ""])
    repair_rows = sorted(
        [row for row in records if row["style_heuristics"]["is_repair_candidate"]],
        key=lambda row: (-row["style_heuristics"]["repair_score"], row["paper_id"]),
    )[:40]
    for row in repair_rows:
        h = row["style_heuristics"]
        lines.append(f"- `{row['paper_id']}` `{row.get('split', '')}` score={h['repair_score']} flags={h['flags']}: {h['summary']}")

    lines.extend(["", "## Self-Anchor Samples", ""])
    anchor_rows = [row for row in records if row["style_heuristics"]["is_self_anchor_candidate"]][:40]
    for row in anchor_rows:
        h = row["style_heuristics"]
        lines.append(f"- `{row['paper_id']}` `{row.get('split', '')}` {h['word_count']}w: {h['summary']}")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score loose q5 style heuristics for generated outputs.")
    parser.add_argument("--input", type=Path, required=True, help="Source generation input JSONL.")
    parser.add_argument("--outputs", type=Path, required=True, help="llama.cpp generation output JSONL.")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_rows = {str(row["paper_id"]): row for row in load_jsonl(args.input)}
    outputs = output_by_pid(args.outputs)
    missing = sorted(pid for pid in source_rows if pid not in outputs)
    if missing:
        raise SystemExit(f"Missing outputs for {len(missing)} source rows; first missing: {missing[:10]}")

    records: list[dict[str, Any]] = []
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pid, source in source_rows.items():
        h = style_flags(outputs[pid], source)
        row = {
            "paper_id": pid,
            "eval_id": source.get("eval_id", ""),
            "split": source.get("split", ""),
            "title": source.get("title", ""),
            "abstract": source.get("abstract", ""),
            "primary_category": source.get("primary_category", ""),
            "generated_summary": h["summary"],
            "style_heuristics": h,
        }
        records.append(row)
        by_split[str(row.get("split", ""))].append(row)

    repair = sorted(
        [row for row in records if row["style_heuristics"]["is_repair_candidate"]],
        key=lambda row: (-row["style_heuristics"]["repair_score"], row["paper_id"]),
    )
    anchors = sorted(
        [row for row in records if row["style_heuristics"]["is_self_anchor_candidate"]],
        key=lambda row: (row.get("split", ""), row["paper_id"]),
    )

    args.out_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_root / "style_heuristic_rows.jsonl", records)
    write_jsonl(args.out_root / "repair_candidates.jsonl", repair)
    write_jsonl(args.out_root / "self_anchor_candidates.jsonl", anchors)
    manifest = {
        "created_by": "scripts/score_q5_loose_style_heuristics_v1.py",
        "input": str(args.input),
        "outputs": str(args.outputs),
        "rows": len(records),
        "repair_candidates": len(repair),
        "self_anchor_candidates": len(anchors),
        "summary": summarize(records),
        "by_split": {split: summarize(rows) for split, rows in sorted(by_split.items())},
        "heuristic_policy": "loose candidate selectors for review and self-anchor mining, not pass/fail gates",
    }
    (args.out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.out_root / "STYLE_HEURISTIC_REPORT.md", records, by_split)
    print(f"[loose-style-heuristics] rows={len(records)} repair={len(repair)} self_anchor={len(anchors)} out={args.out_root}")


if __name__ == "__main__":
    main()
