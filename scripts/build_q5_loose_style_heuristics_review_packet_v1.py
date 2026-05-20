#!/usr/bin/env python3
"""Write a review packet for the q5 loose style heuristics only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OUT_ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/eval/burnin150_style_probe_5000_v1/heuristics_review_v1")

HEURISTIC_SPEC: dict[str, Any] = {
    "spec_version": "q5_loose_style_heuristics_v1",
    "policy": {
        "purpose": "Candidate selection for repair and self-anchor mining, not pass/fail judging.",
        "scope": "One-sentence mobile-feed paper summaries generated from title plus abstract.",
        "review_posture": "Prefer high recall. A noisy flag is acceptable if it catches failures worth reviewing.",
        "non_goals": [
            "Do not enforce product gating.",
            "Do not replace human or teacher-model semantic review.",
            "Do not decide whether a summary is globally good.",
        ],
    },
    "repair_score_weights": {
        "digit_symbol_residue": 100,
        "awkward_malformed": 85,
        "value_polarity_risk": 75,
        "benchmark_winner_shape": 60,
        "unsupported_specificity": 55,
        "title_rehash": 45,
        "generic_low_value": 35,
        "length_shape": 25,
    },
    "self_anchor_policy": {
        "positive_requirements": [
            "10 to 15 words",
            "single sentence",
            "at least three long-token overlaps with title plus abstract",
            "title-overlap ratio below 0.55",
        ],
        "blocking_categories": [
            "digit_symbol_residue",
            "awkward_malformed",
            "benchmark_winner_shape",
            "value_polarity_risk",
            "unsupported_specificity",
        ],
        "extra_blockers": [
            "boilerplate_opening",
            "strong_title_overlap",
        ],
    },
    "categories": [
        {
            "id": "length_shape",
            "intent": "Find outputs that may be too thin, too dense, fragmented, or over-compressed for the feed.",
            "looseness_note": "Length alone should rarely imply a bad summary. It mostly prioritizes review when paired with other flags.",
            "flags": [
                {
                    "id": "under_9_words",
                    "trigger": "Word count below 9.",
                    "why": "Very short summaries often lose the object, result, or reason to open.",
                    "false_positive_note": "Some clean feed summaries are short, especially when the result is simple.",
                },
                {
                    "id": "over_16_words",
                    "trigger": "Word count above 16.",
                    "why": "Overlength outputs often become technical noun chains or carry too many claims.",
                    "false_positive_note": "A 17-word summary can still be excellent.",
                },
                {
                    "id": "multi_sentence",
                    "trigger": "More than one sentence terminator.",
                    "why": "The target format is one concise feed sentence.",
                    "false_positive_note": "Abbreviations and initials may confuse simple sentence counting.",
                },
                {
                    "id": "colon_or_semicolon",
                    "trigger": "Contains a colon or semicolon.",
                    "why": "These often mark academic compression or two-part headline phrasing.",
                    "false_positive_note": "Some math or dataset names include punctuation harmlessly.",
                },
                {
                    "id": "fragment_like",
                    "trigger": "No rough finite-verb pattern is detected.",
                    "why": "Fragments often read like labels rather than useful feed summaries.",
                    "false_positive_note": "The finite-verb detector is intentionally crude and can miss valid verbs.",
                },
            ],
        },
        {
            "id": "digit_symbol_residue",
            "intent": "Catch outputs that still surface digits, formula fragments, or brittle exact-number bindings.",
            "looseness_note": "Digits are not always semantically bad, but this model line is trying to learn no-digit paraphrases.",
            "flags": [
                {
                    "id": "digit_or_numberlike_token",
                    "trigger": "Any digit or alphanumeric number-like token appears.",
                    "why": "The q5 failure mode often copies or corrupts exact metrics.",
                    "false_positive_note": "Terms like K3, 5G, 3D, or L4S may be good summaries that need no-digit paraphrases.",
                },
                {
                    "id": "dense_numberlike_tokens",
                    "trigger": "Two or more number-like tokens appear.",
                    "why": "Dense numeric outputs are often attractor failures.",
                    "false_positive_note": "Some source domains naturally have multiple numbered entities.",
                },
                {
                    "id": "formula_or_symbol_residue",
                    "trigger": "Formula, LaTeX, inequality, caret, underscore, HTML, or Greek-like symbols appear.",
                    "why": "These rarely fit the mobile-feed target style.",
                    "false_positive_note": "A symbol can be part of a proper object name, but it should still be reviewed.",
                },
                {
                    "id": "unsupported_number",
                    "trigger": "Output number is not matched by title plus abstract using the existing number-support helper.",
                    "why": "Unsupported numbers are high-value repair candidates.",
                    "false_positive_note": "The number-support helper is brittle around transformed or rounded quantities.",
                },
            ],
        },
        {
            "id": "awkward_malformed",
            "intent": "Find outputs with visible language damage from burn-in or numeric abstraction pressure.",
            "looseness_note": "This category should stay high-priority because malformed labels poison self-anchors.",
            "flags": [
                {
                    "id": "adjacent_duplicate_word",
                    "trigger": "Same nontrivial word appears twice in a row.",
                    "why": "This is almost always visible model damage.",
                    "false_positive_note": "Rare named phrases can repeat intentionally, but review cost is low.",
                },
                {
                    "id": "repeated_stem_window",
                    "trigger": "A six-character stem repeats within a short word window.",
                    "why": "Catches patterns like classification of classification or structure structure variants.",
                    "false_positive_note": "This is noisy for technical prose with legitimate repeated roots.",
                },
                {
                    "id": "weird_morphology",
                    "trigger": "Long words ending in suspicious generated suffixes.",
                    "why": "Catches outputs like structurefulness or malformed blends.",
                    "false_positive_note": "Some real technical terms have unusual morphology.",
                },
                {
                    "id": "hyphen_dense",
                    "trigger": "Four or more hyphenated tokens.",
                    "why": "Often marks over-compressed technical noun chains.",
                    "false_positive_note": "Some titles and domains are naturally hyphen-heavy.",
                },
                {
                    "id": "long_token_dense",
                    "trigger": "Five or more tokens with length at least 12.",
                    "why": "Long-token density can indicate unreadable technical packing.",
                    "false_positive_note": "Biomedical and astronomy terms can be long but acceptable.",
                },
            ],
        },
        {
            "id": "generic_low_value",
            "intent": "Find summaries that may be bland, boilerplate, or insufficiently source-specific.",
            "looseness_note": "This category should be reviewed gently. Generic wording is sometimes acceptable if the summary is accurate.",
            "flags": [
                {
                    "id": "boilerplate_opening",
                    "trigger": "Starts with this paper, the paper, the authors, we propose, in this work, or in this paper.",
                    "why": "Violates the feed style and wastes words.",
                    "false_positive_note": "Mostly safe, but should still be treated as a style repair, not a semantic failure.",
                },
                {
                    "id": "weak_academic_verb",
                    "trigger": "Uses presents, explores, studies, investigates, discusses, introduces, proposes, or describes.",
                    "why": "Often summarizes author action instead of why to open.",
                    "false_positive_note": "Some of these are fine when the payoff is concrete.",
                },
                {
                    "id": "generic_benefit_phrase",
                    "trigger": "Contains phrases like improves performance or provides insights.",
                    "why": "These are low-information feed phrases.",
                    "false_positive_note": "May be okay when paired with a concrete object and result.",
                },
                {
                    "id": "low_source_specificity",
                    "trigger": "At most two long-token overlaps with title plus abstract in an otherwise normal-length output.",
                    "why": "May indicate generic abstraction or invented wording.",
                    "false_positive_note": "Paraphrases can be excellent while sharing few exact long tokens.",
                },
            ],
        },
        {
            "id": "title_rehash",
            "intent": "Find outputs that mostly rearrange the title rather than adding value beyond it.",
            "looseness_note": "Title overlap is a style risk, not a semantic verdict.",
            "flags": [
                {
                    "id": "soft_title_overlap",
                    "trigger": "At least 55 percent of summary content words overlap title words.",
                    "why": "The app already shows the title.",
                    "false_positive_note": "Short technical titles can force overlap with unavoidable object names.",
                },
                {
                    "id": "strong_title_overlap",
                    "trigger": "At least 75 percent of summary content words overlap title words.",
                    "why": "Strong sign of title rehash.",
                    "false_positive_note": "Some very specific paper objects have few usable synonyms.",
                },
                {
                    "id": "copied_title_phrase",
                    "trigger": "A contiguous four-word phrase is shared with the title.",
                    "why": "Catches copy-like behavior even when token ratio is lower.",
                    "false_positive_note": "Proper names and fixed dataset names may be valid copied phrases.",
                },
            ],
        },
        {
            "id": "benchmark_winner_shape",
            "intent": "Find leaderboard or winner phrasing that may revive checkpoint-700 style failures.",
            "looseness_note": "Benchmark phrasing is not always wrong; it is risky when it replaces the actual contribution.",
            "flags": [
                {
                    "id": "benchmark_or_winner_wording",
                    "trigger": "Contains outperform, state-of-the-art, best, superior, beat, benchmark, leaderboard, or achieve.",
                    "why": "The model often collapses nuanced results into winner claims.",
                    "false_positive_note": "Some papers really are benchmark papers; review for grounding.",
                },
            ],
        },
        {
            "id": "value_polarity_risk",
            "intent": "Find summaries that may flip caveats, limitations, or directionality.",
            "looseness_note": "This is the noisiest source-aware category and should feed review, not automatic rejection.",
            "flags": [
                {
                    "id": "source_caveat_absent_with_positive_output",
                    "trigger": "Source has a caveat term, output lacks a caveat and uses a positive-result verb.",
                    "why": "Burn-in can turn caveated findings into clean positive claims.",
                    "false_positive_note": "Many abstracts contain irrelevant caveat words; this flag is intentionally noisy.",
                },
                {
                    "id": "output_adds_caveat_not_in_source",
                    "trigger": "Output adds a caveat term absent from source.",
                    "why": "Can invent caution or reverse the paper's value.",
                    "false_positive_note": "Paraphrased caution may be valid even without exact caveat words.",
                },
                {
                    "id": "negative_source_positive_output",
                    "trigger": "Source has negative or limitation language, output uses positive-result verbs.",
                    "why": "Flags possible polarity flips.",
                    "false_positive_note": "Negative source language may describe prior work rather than the paper's result.",
                },
            ],
        },
        {
            "id": "unsupported_specificity",
            "intent": "Find outputs that introduce strong claims or qualities not grounded in the source wording.",
            "looseness_note": "This catches suspicious wording, not all unsupported claims.",
            "flags": [
                {
                    "id": "introduced_strong_claim_word",
                    "trigger": "A strong claim word appears in output but not among long source terms.",
                    "why": "Strong words like proves or guarantees can overstate results.",
                    "false_positive_note": "Exact-token matching misses legitimate paraphrases.",
                },
                {
                    "id": "strong_claim_not_in_source",
                    "trigger": "Output contains strong claim vocabulary absent from the source.",
                    "why": "Helps catch unsupported robustness, accuracy, scalability, or solving claims.",
                    "false_positive_note": "Some strong claims are semantically supported without exact wording.",
                },
            ],
        },
    ],
}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def review_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for category in HEURISTIC_SPEC["categories"]:
        items.append(
            {
                "item_type": "category",
                "category_id": category["id"],
                "intent": category["intent"],
                "looseness_note": category["looseness_note"],
                "flags": category["flags"],
                "review_questions": [
                    "Is this category useful for selecting rows for human or teacher-model review?",
                    "Which flags are too noisy, too strict, or missing?",
                    "Should this category block self-anchor rows?",
                    "What concrete examples should reviewers inspect after generation?",
                ],
            }
        )
    return items


def write_markdown(path: Path) -> None:
    lines = [
        "# q5 loose style heuristics review packet",
        "",
        "This packet describes the heuristic design only. It does not score any model outputs.",
        "",
        "The heuristics are loose candidate selectors for two later uses:",
        "",
        "- find repair candidates for Kimi/Qwen or human review",
        "- find low-risk post-burn-in self-anchor rows",
        "",
        "They are not product gates, and they are not final quality judgments.",
        "",
        "## Review Brief",
        "",
        "For each category, judge whether the heuristic is useful for high-recall triage.",
        "Prefer comments like:",
        "",
        "- too noisy, but acceptable for repair mining",
        "- too noisy and likely harmful",
        "- too strict, would miss important failures",
        "- should block self-anchor rows",
        "- should not block self-anchor rows",
        "- missing nearby failure pattern",
        "",
        "## Self-Anchor Policy",
        "",
    ]
    for item in HEURISTIC_SPEC["self_anchor_policy"]["positive_requirements"]:
        lines.append(f"- Require: {item}")
    for item in HEURISTIC_SPEC["self_anchor_policy"]["blocking_categories"]:
        lines.append(f"- Block category: `{item}`")
    for item in HEURISTIC_SPEC["self_anchor_policy"]["extra_blockers"]:
        lines.append(f"- Block flag: `{item}`")

    lines.extend(["", "## Categories", ""])
    weights = HEURISTIC_SPEC["repair_score_weights"]
    for category in HEURISTIC_SPEC["categories"]:
        lines.extend(
            [
                f"### `{category['id']}`",
                "",
                f"Intent: {category['intent']}",
                "",
                f"Loose-use note: {category['looseness_note']}",
                "",
                f"Repair weight: `{weights.get(category['id'], 0)}`",
                "",
                "| flag | trigger | why | false-positive note |",
                "|---|---|---|---|",
            ]
        )
        for flag in category["flags"]:
            lines.append(f"| `{flag['id']}` | {flag['trigger']} | {flag['why']} | {flag['false_positive_note']} |")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_subagent_prompt(path: Path, shards: int) -> None:
    text = f"""# Subagent Review Prompt

You are reviewing q5 loose style heuristics, not model outputs.

Read your assigned JSONL shard from this packet. For each category:

- say whether the category is useful for loose candidate selection;
- identify flags that are too noisy, too strict, or missing;
- say whether the category should block self-anchor rows;
- suggest concrete wording or threshold changes;
- do not propose runtime guards, post-processing, or DPO.

Important context:

- The target output is one concise mobile-feed summary sentence.
- The model line is SFT-only.
- The heuristics should be high-recall review selectors, not pass/fail tests.
- Good summaries may trip a heuristic. That is acceptable if the flag is useful for review.

Return a compact report with sections:

1. Keep unchanged
2. Modify
3. Remove or demote
4. Missing heuristics
5. Self-anchor blocker recommendations

Shard count: {shards}
"""
    path.write_text(text, encoding="utf-8")


def write_shards(out_root: Path, items: list[dict[str, Any]], shards: int) -> list[str]:
    shard_paths: list[str] = []
    for shard_index in range(shards):
        shard = [item for idx, item in enumerate(items) if idx % shards == shard_index]
        path = out_root / f"review_items_shard_{shard_index + 1:02d}.jsonl"
        write_jsonl(path, shard)
        shard_paths.append(str(path))
    return shard_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a heuristics-only review packet for q5 loose style heuristics.")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--shards", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shards < 1:
        raise SystemExit("--shards must be at least 1")
    args.out_root.mkdir(parents=True, exist_ok=True)
    items = review_items()
    shard_paths = write_shards(args.out_root, items, args.shards)
    write_json(args.out_root / "heuristics_spec.json", HEURISTIC_SPEC)
    write_jsonl(args.out_root / "review_items.jsonl", items)
    write_markdown(args.out_root / "HEURISTICS_REVIEW_PACKET.md")
    write_subagent_prompt(args.out_root / "SUBAGENT_REVIEW_PROMPT.md", args.shards)
    manifest = {
        "created_by": "scripts/build_q5_loose_style_heuristics_review_packet_v1.py",
        "spec_version": HEURISTIC_SPEC["spec_version"],
        "category_count": len(HEURISTIC_SPEC["categories"]),
        "flag_count": sum(len(category["flags"]) for category in HEURISTIC_SPEC["categories"]),
        "shards": shard_paths,
        "outputs": {
            "heuristics_spec": str(args.out_root / "heuristics_spec.json"),
            "review_packet": str(args.out_root / "HEURISTICS_REVIEW_PACKET.md"),
            "subagent_prompt": str(args.out_root / "SUBAGENT_REVIEW_PROMPT.md"),
            "review_items": str(args.out_root / "review_items.jsonl"),
        },
    }
    write_json(args.out_root / "manifest.json", manifest)
    print(f"[heuristics-review-packet] wrote {len(items)} category items and {manifest['flag_count']} flags to {args.out_root}")


if __name__ == "__main__":
    main()
