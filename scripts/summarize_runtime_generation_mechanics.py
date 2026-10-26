#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
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


def parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Variant must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("Variant label cannot be empty")
    return label, Path(path)


def variant_summary(row: dict[str, Any]) -> str:
    return str(
        row.get("normalized_output")
        or row.get("normalizedOutput")
        or row.get("summary")
        or row.get("raw_output")
        or row.get("rawOutput")
        or ""
    )


def summarize(rows: list[dict[str, Any]], label: str, baseline_by_id: dict[str, str] | None) -> dict[str, Any]:
    label_rows = [row for row in rows if row["label"] == label]
    counts = [int(row["words"]) for row in label_rows]
    return {
        "label": label,
        "rows": len(label_rows),
        "ok_rows": sum(1 for row in label_rows if row.get("ok", True)),
        "avg_words": round(mean(counts), 2) if counts else 0,
        "median_words": float(median(counts)) if counts else 0,
        "max_words": max(counts) if counts else 0,
        "under10": sum(1 for count in counts if count < 10),
        "le14": sum(1 for count in counts if count <= 14),
        "over14": sum(1 for count in counts if count > 14),
        "over15": sum(1 for count in counts if count > 15),
        "digit_rows": sum(1 for row in label_rows if row["digit"]),
        "formulaish_rows": sum(1 for row in label_rows if row["formulaish"]),
        "title_overlap_ge3": sum(1 for row in label_rows if row["title_overlap_count"] >= 3),
        "title_overlap_ge4": sum(1 for row in label_rows if row["title_overlap_count"] >= 4),
        "changed_vs_baseline": (
            sum(
                1
                for row in label_rows
                if baseline_by_id is not None
                and row["summary"] != baseline_by_id.get(row["paper_id"], "")
            )
            if baseline_by_id is not None
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-generations", type=Path, required=True)
    parser.add_argument("--baseline-side", choices=("left", "right"), default="right")
    parser.add_argument("--baseline-label", default=None)
    parser.add_argument("--variant", action="append", type=parse_variant, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples-md", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = load_jsonl(args.source_generations)
    side = args.baseline_side
    baseline_label = args.baseline_label or str(source_rows[0][f"{side}_label"])

    source_by_id = {str(row["paper_id"]): row for row in source_rows}
    variant_rows: list[dict[str, Any]] = []
    baseline_by_id: dict[str, str] = {}
    for row in source_rows:
        summary = str(row[f"{side}_summary"])
        paper_id = str(row["paper_id"])
        baseline_by_id[paper_id] = summary
        variant_rows.append(
            {
                "label": baseline_label,
                "paper_id": paper_id,
                "title": row["title"],
                "primary_category": row.get("primary_category"),
                "summary": summary,
                "ok": True,
                "words": len(words(summary)),
                "digit": bool(DIGIT_RE.search(summary)),
                "formulaish": bool(FORMULAISH_RE.search(summary)),
                "title_overlap_count": title_overlap_count(summary, str(row.get("title", ""))),
            }
        )

    labels = [baseline_label]
    for label, path in args.variant:
        labels.append(label)
        for row in load_jsonl(path):
            paper_id = str(row["paper_id"])
            source = source_by_id[paper_id]
            summary = variant_summary(row)
            variant_rows.append(
                {
                    "label": label,
                    "paper_id": paper_id,
                    "title": source["title"],
                    "primary_category": source.get("primary_category"),
                    "summary": summary,
                    "ok": bool(row.get("ok")),
                    "words": len(words(summary)),
                    "digit": bool(DIGIT_RE.search(summary)),
                    "formulaish": bool(FORMULAISH_RE.search(summary)),
                    "title_overlap_count": title_overlap_count(summary, str(source.get("title", ""))),
                }
            )

    summary = [summarize(variant_rows, label, baseline_by_id if label != baseline_label else None) for label in labels]
    payload = {
        "source_generations": str(args.source_generations),
        "baseline_label": baseline_label,
        "summary": summary,
    }
    (args.output_dir / "mechanics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "variant_outputs.jsonl").open("w", encoding="utf-8") as handle:
        for row in variant_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    lines = [
        "# Runtime Generation Mechanics",
        "",
        "| label | rows | avg | median | max | <=14 | >14 | >15 | <10 | digits | formula | title >=3 | changed vs baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        changed = item["changed_vs_baseline"]
        lines.append(
            f"| {item['label']} | {item['rows']} | {item['avg_words']:.2f} | "
            f"{item['median_words']:.1f} | {item['max_words']} | {item['le14']} | "
            f"{item['over14']} | {item['over15']} | {item['under10']} | "
            f"{item['digit_rows']} | {item['formulaish_rows']} | {item['title_overlap_ge3']} | "
            f"{'' if changed is None else changed} |"
        )
    lines.append("")
    (args.output_dir / "MECHANICS.md").write_text("\n".join(lines), encoding="utf-8")

    if args.examples_md:
        example_lines = ["# Changed Runtime Examples", ""]
        for paper_id, baseline in list(baseline_by_id.items())[:40]:
            rows = [row for row in variant_rows if row["paper_id"] == paper_id and row["label"] != baseline_label]
            if not rows:
                continue
            if all(row["summary"] == baseline for row in rows):
                continue
            source = source_by_id[paper_id]
            example_lines.extend(
                [
                    f"## {paper_id}",
                    "",
                    f"Title: {source['title']}",
                    "",
                    f"- {baseline_label}: {baseline}",
                ]
            )
            for row in rows:
                example_lines.append(f"- {row['label']}: {row['summary']}")
            example_lines.append("")
        args.examples_md.write_text("\n".join(example_lines), encoding="utf-8")

    print(f"[runtime-mechanics] wrote {args.output_dir / 'MECHANICS.md'}")


if __name__ == "__main__":
    main()
