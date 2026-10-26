#!/usr/bin/env python3
"""Summarize q5 fresh20 llama.cpp outputs against a baseline."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")
DIGIT_RE = re.compile(r"\d")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "more",
    "new",
    "of",
    "on",
    "or",
    "over",
    "than",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "to",
    "using",
    "with",
    "without",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-input", type=Path, required=True)
    parser.add_argument("--baseline", required=True, help="label=path for the baseline output JSONL")
    parser.add_argument("--candidate", action="append", default=[], help="label=path for candidate output JSONL")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def norm_token(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "", value.casefold()).strip("-")


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in (norm_token(match.group(0)) for match in WORD_RE.finditer(text))
        if token and token not in STOPWORDS
    }


def title_overlap_count(summary: str, title: str) -> int:
    return len(content_tokens(summary) & content_tokens(title))


def parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected label=path, got {value!r}")
    label, path = value.split("=", 1)
    if not label:
        raise ValueError(f"Missing label in {value!r}")
    return label, Path(path)


def load_outputs(path: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for row in load_jsonl(path):
        if not row.get("ok", True):
            outputs[str(row.get("paper_id", ""))] = ""
            continue
        outputs[str(row["paper_id"])] = norm_text(row.get("normalized_output") or row.get("raw_output"))
    return outputs


def row_metrics(text: str, title: str) -> dict[str, Any]:
    wc = len(WORD_RE.findall(text))
    overlap = title_overlap_count(text, title)
    return {
        "word_count": wc,
        "has_digit": bool(DIGIT_RE.search(text)),
        "title_overlap_count": overlap,
        "title_overlap_ge3": overlap >= 3,
        "over14": wc > 14,
        "over15": wc > 15,
        "under10": wc < 10,
    }


def summarize(
    label: str,
    path: Path,
    outputs: dict[str, str],
    baseline_outputs: dict[str, str],
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        pid = str(source["paper_id"])
        title = norm_text(source.get("title"))
        text = outputs.get(pid, "")
        base = baseline_outputs.get(pid, "")
        metrics = row_metrics(text, title)
        base_metrics = row_metrics(base, title)
        changed = text.casefold() != base.casefold()
        regression_flags: list[str] = []
        if metrics["over14"] and not base_metrics["over14"]:
            regression_flags.append("new_over14")
        if metrics["over15"] and not base_metrics["over15"]:
            regression_flags.append("new_over15")
        if metrics["has_digit"] and not base_metrics["has_digit"]:
            regression_flags.append("new_digit")
        if metrics["title_overlap_ge3"] and not base_metrics["title_overlap_ge3"]:
            regression_flags.append("new_title_overlap_ge3")
        rows.append(
            {
                "paper_id": pid,
                "title": title,
                "primary_category": source.get("primary_category", ""),
                "baseline_output": base,
                "output": text,
                "changed_vs_baseline": changed,
                "regression_flags": regression_flags,
                **metrics,
            }
        )
    word_counts = [row["word_count"] for row in rows]
    return {
        "label": label,
        "path": str(path),
        "rows": rows,
        "avg_words": round(mean(word_counts), 2) if word_counts else 0,
        "max_words": max(word_counts) if word_counts else 0,
        "over14": sum(1 for row in rows if row["over14"]),
        "over15": sum(1 for row in rows if row["over15"]),
        "under10": sum(1 for row in rows if row["under10"]),
        "digit_rows": sum(1 for row in rows if row["has_digit"]),
        "title_overlap_ge3": sum(1 for row in rows if row["title_overlap_ge3"]),
        "changed_vs_baseline": sum(1 for row in rows if row["changed_vs_baseline"]),
        "regression_rows": sum(1 for row in rows if row["regression_flags"]),
    }


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# q5 Fresh20 Metrics",
        "",
        "| label | avg words | max | >14 | >15 | <10 | digit rows | title overlap >=3 | changed | regressions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['label']} | {summary['avg_words']} | {summary['max_words']} | "
            f"{summary['over14']} | {summary['over15']} | {summary['under10']} | "
            f"{summary['digit_rows']} | {summary['title_overlap_ge3']} | "
            f"{summary['changed_vs_baseline']} | {summary['regression_rows']} |"
        )
    lines.append("")
    for summary in summaries:
        lines.append(f"## {summary['label']}")
        for row in summary["rows"]:
            if not row["changed_vs_baseline"] and not row["regression_flags"]:
                continue
            flags = ", ".join(row["regression_flags"]) if row["regression_flags"] else "changed"
            lines.append(f"- `{row['paper_id']}` flags={flags}")
            lines.append(f"  - base: {row['baseline_output']}")
            lines.append(f"  - out:  {row['output']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline_label, baseline_path = parse_label_path(args.baseline)
    candidate_specs = [parse_label_path(item) for item in args.candidate]
    sources = load_jsonl(args.eval_input)
    baseline_outputs = load_outputs(baseline_path)

    summaries = [
        summarize(baseline_label, baseline_path, baseline_outputs, baseline_outputs, sources)
    ]
    for label, path in candidate_specs:
        summaries.append(summarize(label, path, load_outputs(path), baseline_outputs, sources))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"summaries": summaries}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.output_md, summaries)
    print(f"[fresh20-metrics] wrote {args.output_json}")
    print(f"[fresh20-metrics] wrote {args.output_md}")


if __name__ == "__main__":
    main()
