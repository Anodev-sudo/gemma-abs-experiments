#!/usr/bin/env python3
"""Compare checkpoint summaries against pre-DPO summaries with heuristic flags."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def word_count(text: str) -> int:
    return len(text.split())


def check_heuristics(text: str) -> dict:
    """Run heuristic checks on a summary and return flags and counts."""
    flags = []
    wc = word_count(text) if text else 0

    if wc < 10:
        flags.append("under10")
    if wc > 15:
        flags.append("over15")

    # Check for title-like phrasing
    boring_openers = [
        r"^this paper ", r"^we propose ", r"^we present ", r"^we introduce ",
        r"^we study ", r"^we show ", r"^we demonstrate ", r"^we develop ",
        r"^we investigate ", r"^in this paper ", r"^this work ",
        r"^the paper ", r"^this article ",
    ]
    for pat in boring_openers:
        if re.search(pat, text.lower()):
            flags.append("filler_opener")
            break

    # Check for invented numbers (digits that could be hallucinated)
    numbers = re.findall(r'\b\d+\b', text)
    # Common safe numbers: years, percentages that don't look suspicious
    # This is a crude check - flag any number >= 10 (likely invented) 
    # unless it looks like a year (19xx/20xx)
    suspicious_numbers = [n for n in numbers 
                          if int(n) >= 10 and not (1900 <= int(n) <= 2030)]
    if suspicious_numbers:
        flags.append("numbers_present")

    # Check for unsupported labels/entities
    # Math notation, chemical formulas, etc.
    notation_patterns = [
        r'\bO\(\w', r'\\\w+', r'\$\w', r'\b\w+\^\{', r'\b\w+_\{\w+\}',
        r'<\w+>', r'\\mathbb', r'\\mathcal',
    ]
    for pat in notation_patterns:
        if re.search(pat, text):
            flags.append("notation_entity")
            break

    # Polarity check: "not" / "no" / "without" / "never" could indicate 
    # negative-summary risk (describing what something is NOT rather than IS)
    polarity_words = [r'\bnot\b', r'\bno\b', r'\bwithout\b', r'\bnever\b',
                      r'\bdoes not\b', r'\bcannot\b', r'\bunable\b']
    polarity_count = sum(1 for pat in polarity_words if re.search(pat, text.lower()))

    # Check for malformed/truncated
    if text and not text.rstrip().endswith(('.', '?', '!')):
        flags.append("no_period")
    if text and len(text) < 20:
        flags.append("too_short")

    # Check for title text in summary (basic overlap)
    # This would need the title; we'll do it in the paired comparison

    return {
        "word_count": wc,
        "flags": flags,
        "polarity_count": polarity_count,
        "suspicious_numbers": suspicious_numbers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-outputs", required=True, help="Checkpoint generation outputs JSONL")
    parser.add_argument("--pre-dpo-pairs", required=True, help="Holdout pair rows with pre-DPO summaries")
    parser.add_argument("--output", required=True, help="Output comparison JSONL")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    # Load checkpoint outputs
    ckpt_by_pid = {}
    with open(args.ckpt_outputs) as f:
        for line in f:
            d = json.loads(line)
            ckpt_by_pid[d["paper_id"]] = d

    # Load pre-DPO pairs
    pre_by_pid = {}
    with open(args.pre_dpo_pairs) as f:
        for line in f:
            d = json.loads(line)
            if len(pre_by_pid) >= args.limit:
                break
            pre_by_pid[d["paper_id"]] = d

    # Build comparison rows
    comparisons = []
    stats = {
        "total": 0,
        "ckpt_under10": 0,
        "ckpt_over15": 0,
        "ckpt_not_numbers": 0,
        "ckpt_notation": 0,
        "ckpt_polarity_words": 0,
        "ckpt_filler": 0,
        "pre_under10": 0,
        "pre_over15": 0,
        "pre_not_numbers": 0,
        "pre_notation": 0,
        "pre_polarity_words": 0,
        "pre_filler": 0,
        "exact_match": 0,
        "ckpt_flags_counter": Counter(),
        "pre_flags_counter": Counter(),
        "ckpt_avg_wc": 0,
        "pre_avg_wc": 0,
        "ckpt_total_wc": 0,
        "pre_total_wc": 0,
    }

    for pid in pre_by_pid:
        if pid not in ckpt_by_pid:
            print(f"WARNING: {pid} in holdout but not in ckpt outputs", file=sys.stderr)
            continue

        pre_row = pre_by_pid[pid]
        ckpt_row = ckpt_by_pid[pid]
        
        pre_summary = pre_row["left_summary"]
        ckpt_summary = ckpt_row.get("normalized_output", "")
        
        pre_h = check_heuristics(pre_summary)
        ckpt_h = check_heuristics(ckpt_summary)

        # Title overlap check
        title_words = set(pre_row["title"].lower().split())
        title_words -= {"the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "is", "be", "at", "by", "with", "from"}
        
        pre_summ_words = set(pre_summary.lower().split())
        ckpt_summ_words = set(ckpt_summary.lower().split()) if ckpt_summary else set()
        
        pre_title_overlap = len(title_words & pre_summ_words) if pre_summary else 0
        ckpt_title_overlap = len(title_words & ckpt_summ_words) if ckpt_summary else 0

        exact = (pre_summary.strip() == ckpt_summary.strip())

        row = {
            "paper_id": pid,
            "title": pre_row["title"],
            "pre_summary": pre_summary,
            "ckpt_summary": ckpt_summary,
            "pre_heuristics": pre_h,
            "ckpt_heuristics": ckpt_h,
            "pre_title_overlap": pre_title_overlap,
            "ckpt_title_overlap": ckpt_title_overlap,
            "exact_match": exact,
        }
        comparisons.append(row)

        # Update stats
        stats["total"] += 1
        stats["pre_total_wc"] += pre_h["word_count"]
        stats["ckpt_total_wc"] += ckpt_h["word_count"]
        if pre_h["word_count"] < 10: stats["pre_under10"] += 1
        if pre_h["word_count"] > 15: stats["pre_over15"] += 1
        if ckpt_h["word_count"] < 10: stats["ckpt_under10"] += 1
        if ckpt_h["word_count"] > 15: stats["ckpt_over15"] += 1
        if "numbers_present" in pre_h["flags"]: stats["pre_not_numbers"] += 1
        if "numbers_present" in ckpt_h["flags"]: stats["ckpt_not_numbers"] += 1
        if "notation_entity" in pre_h["flags"]: stats["pre_notation"] += 1
        if "notation_entity" in ckpt_h["flags"]: stats["ckpt_notation"] += 1
        if pre_h["polarity_count"] > 0: stats["pre_polarity_words"] += 1
        if ckpt_h["polarity_count"] > 0: stats["ckpt_polarity_words"] += 1
        if "filler_opener" in pre_h["flags"]: stats["pre_filler"] += 1
        if "filler_opener" in ckpt_h["flags"]: stats["ckpt_filler"] += 1
        if exact: stats["exact_match"] += 1
        
        for f in pre_h["flags"]: stats["pre_flags_counter"][f] += 1
        for f in ckpt_h["flags"]: stats["ckpt_flags_counter"][f] += 1

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in comparisons:
            f.write(json.dumps(row) + "\n")

    # Print summary
    n = stats["total"]
    print(f"\n=== Heuristic Comparison ({n} papers) ===")
    print(f"{'':<30} {'Pre-DPO':>10} {'Ckpt-100':>10}")
    print(f"{'Word count avg':<30} {stats['pre_total_wc']/n:>10.1f} {stats['ckpt_total_wc']/n:>10.1f}")
    print(f"{'Under 10 words':<30} {stats['pre_under10']:>10} {stats['ckpt_under10']:>10}")
    print(f"{'Over 15 words':<30} {stats['pre_over15']:>10} {stats['ckpt_over15']:>10}")
    print(f"{'Has numbers':<30} {stats['pre_not_numbers']:>10} {stats['ckpt_not_numbers']:>10}")
    print(f"{'Notation/entity':<30} {stats['pre_notation']:>10} {stats['ckpt_notation']:>10}")
    print(f"{'Polarity words':<30} {stats['pre_polarity_words']:>10} {stats['ckpt_polarity_words']:>10}")
    print(f"{'Filler opener':<30} {stats['pre_filler']:>10} {stats['ckpt_filler']:>10}")
    print(f"{'Exact matches':<30} {stats['exact_match']:>10}")

    print(f"\nPre-DPO flag distribution:")
    for flag, count in stats["pre_flags_counter"].most_common():
        print(f"  {flag}: {count}")
    print(f"\nCkpt-100 flag distribution:")
    for flag, count in stats["ckpt_flags_counter"].most_common():
        print(f"  {flag}: {count}")

    # Show changed rows
    changed = [r for r in comparisons if not r["exact_match"]]
    print(f"\nChanged rows: {len(changed)}/{n}")


if __name__ == "__main__":
    main()