#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")
DIGIT_RE = re.compile(r"\d|[⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉]")
FORMULAISH_RE = re.compile(r"[$^_=<>]|\\\\|<sup>|</sup>|[{}]|\bO\s*\(")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "their",
    "this",
    "to",
    "with",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def words(text: str) -> list[str]:
    return WORD_RE.findall(text or "")


def content_words(text: str) -> set[str]:
    return {
        word.lower()
        for word in words(text)
        if len(word) > 2 and word.lower() not in STOPWORDS
    }


def title_overlap_count(summary: str, title: str) -> int:
    return len(content_words(summary) & content_words(title))


def summarize_rows(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    label_rows = [row for row in rows if row["label"] == label]
    word_counts = [int(row["words"]) for row in label_rows]
    return {
        "label": label,
        "rows": len(label_rows),
        "avg_words": round(mean(word_counts), 2) if word_counts else 0,
        "median_words": float(median(word_counts)) if word_counts else 0,
        "max_words": max(word_counts) if word_counts else 0,
        "under10": sum(1 for count in word_counts if count < 10),
        "le14": sum(1 for count in word_counts if count <= 14),
        "over14": sum(1 for count in word_counts if count > 14),
        "over15": sum(1 for count in word_counts if count > 15),
        "digit_rows": sum(1 for row in label_rows if row["digit"]),
        "formulaish_rows": sum(1 for row in label_rows if row["formulaish"]),
        "title_overlap_ge3": sum(1 for row in label_rows if row["title_overlap_count"] >= 3),
        "title_overlap_ge4": sum(1 for row in label_rows if row["title_overlap_count"] >= 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MLX paired generation mechanics.")
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--variant-outputs", type=Path, default=None)
    args = parser.parse_args()

    generation_rows = load_jsonl(args.generations)
    labels = []
    variant_rows: list[dict[str, Any]] = []
    for row in generation_rows:
        for side in ("left", "right"):
            label = str(row[f"{side}_label"])
            if label not in labels:
                labels.append(label)
            summary = str(row[f"{side}_summary"])
            variant_rows.append(
                {
                    "label": label,
                    "paper_id": row["paper_id"],
                    "title": row["title"],
                    "primary_category": row.get("primary_category"),
                    "summary": summary,
                    "words": len(words(summary)),
                    "digit": bool(DIGIT_RE.search(summary)),
                    "formulaish": bool(FORMULAISH_RE.search(summary)),
                    "title_overlap_count": title_overlap_count(summary, str(row.get("title", ""))),
                }
            )

    summary = [summarize_rows(variant_rows, label) for label in labels]
    payload = {
        "generations": str(args.generations),
        "summary": summary,
        "duplicate_generation_mismatches": [
            row["paper_id"] for row in generation_rows if not row.get("outputs_match")
        ],
    }
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.variant_outputs is not None:
        with args.variant_outputs.open("w", encoding="utf-8") as handle:
            for row in variant_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    lines = [
        "# MLX Generation Mechanics",
        "",
        "| label | rows | avg | median | max | <=14 | >14 | >15 | <10 | digits | formula | title >=3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['label']} | {item['rows']} | {item['avg_words']:.2f} | "
            f"{item['median_words']:.1f} | {item['max_words']} | {item['le14']} | "
            f"{item['over14']} | {item['over15']} | {item['under10']} | "
            f"{item['digit_rows']} | {item['formulaish_rows']} | {item['title_overlap_ge3']} |"
        )
    lines.append("")
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[mlx-mechanics] wrote {args.output_json}")
    print(f"[mlx-mechanics] wrote {args.output_md}")


if __name__ == "__main__":
    main()
