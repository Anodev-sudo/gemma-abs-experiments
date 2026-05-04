#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

DEFAULT_ROOT = (
    "hf_bucket/experiments/mlx_lora/safety_eval_generations/"
    "q8_realdb_safety_10000_v1"
)
DEFAULT_REVIEWS_DIR = f"{DEFAULT_ROOT}/allowed_review_subagent_reviews"
DEFAULT_PACKETS_MANIFEST = f"{DEFAULT_ROOT}/allowed_review_packets/manifest.json"
DEFAULT_OUTPUT_JSON = f"{DEFAULT_ROOT}/allowed_review_summary.json"
DEFAULT_OUTPUT_MD = f"{DEFAULT_ROOT}/allowed_review_summary.md"
MISS_PATTERN = re.compile(
    r"eval_id:\s*(?P<eval_id>[^\n]+)\n\s*risk:\s*(?P<risk>[^\n]+)\n\s*why:\s*(?P<why>[^\n]+)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate gpt-5.4-mini allowed-packet safety review files."
    )
    parser.add_argument("--reviews-dir", default=DEFAULT_REVIEWS_DIR)
    parser.add_argument("--packets-manifest", default=DEFAULT_PACKETS_MANIFEST)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def expected_review_files(manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        f"{Path(str(packet['jsonl'])).stem}_review.md"
        for packet in manifest.get("packets", [])
    ]


def parse_review(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    misses = [
        {
            "eval_id": match.group("eval_id").strip().strip("`"),
            "risk": match.group("risk").strip().strip("`"),
            "why": match.group("why").strip(),
        }
        for match in MISS_PATTERN.finditer(text)
    ]
    return {
        "file": path.name,
        "reviewed": path.exists(),
        "miss_count": len(misses),
        "misses": misses,
        "no_misses_statement": "No suspected launch-risk misses." in text,
    }


def aggregate_reviews(reviews_dir: Path, manifest_path: Path) -> dict[str, Any]:
    expected_files = expected_review_files(manifest_path)
    reviews = []
    missing_files = []
    for filename in expected_files:
        path = reviews_dir / filename
        if not path.exists():
            missing_files.append(filename)
            continue
        reviews.append(parse_review(path))

    misses = [miss for review in reviews for miss in review["misses"]]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reviews_dir": str(reviews_dir),
        "packets_manifest": str(manifest_path),
        "expected_review_files": len(expected_files),
        "review_files_found": len(reviews),
        "missing_review_files": missing_files,
        "suspected_miss_count": len(misses),
        "suspected_misses": misses,
        "reviews": reviews,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Allowed Review Summary",
        "",
        f"- Review files found: {summary['review_files_found']}/{summary['expected_review_files']}",
        f"- Missing review files: {len(summary['missing_review_files'])}",
        f"- Suspected launch-risk misses: {summary['suspected_miss_count']}",
        "",
    ]
    if summary["missing_review_files"]:
        lines.append("## Missing Review Files")
        lines.append("")
        lines.extend(f"- `{filename}`" for filename in summary["missing_review_files"])
        lines.append("")

    lines.append("## Suspected Misses")
    lines.append("")
    if not summary["suspected_misses"]:
        lines.append("No suspected launch-risk misses.")
    else:
        for miss in summary["suspected_misses"]:
            lines.extend(
                [
                    f"- `{miss['eval_id']}`",
                    f"  - Risk: `{miss['risk']}`",
                    f"  - Why: {miss['why']}",
                ]
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    summary = aggregate_reviews(Path(args.reviews_dir), Path(args.packets_manifest))
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(
        f"Aggregated {summary['review_files_found']}/{summary['expected_review_files']} "
        f"review files with {summary['suspected_miss_count']} suspected misses"
    )


if __name__ == "__main__":
    main()
