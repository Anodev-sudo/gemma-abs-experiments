#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SYSTEM = (
    "You write one-sentence paper summaries for a mobile research feed. "
    "Keep them concrete, readable, and useful. Output only the summary sentence."
)
USER_TMPL = (
    "The app already shows the title, author list, and date. "
    "Write one sentence that tells a reader why they might open this paper. "
    "Aim for 10 to 15 words. Avoid repeating the title. "
    'Avoid filler like "This paper" or "We propose". '
    "Do not mention authors or dates. "
    "Keep the wording natural and specific.\n\n"
    "Title:\n{title}\n\n"
    "Abstract:\n{abstract}"
)
DIGIT_RE = re.compile(r"\d")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["patch_component"] = "targeted_sft700_repair_v1_only"
    user = SYSTEM + "\n\n" + USER_TMPL.format(title=row["title"], abstract=row["abstract"])
    out["messages"] = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": row["summary"]},
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targeted-repairs", default="hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/failure_mining/checkpoint700_10000_v1/repairs_targeted_v1/materialized/training_rows.jsonl")
    ap.add_argument("--output-dir", default="hf_bucket/experiments/sft/q5_step2400_sft700_targeted_only_v2/dataset")
    args = ap.parse_args()

    rows = [normalize_row(r) for r in load_jsonl(Path(args.targeted_repairs))]
    out_dir = Path(args.output_dir)
    write_jsonl(out_dir / "train.jsonl", rows)

    wcs = [len(r["summary"].split()) for r in rows]
    manifest = {
        "dataset_id": "q5_step2400_sft700_targeted_only_v2",
        "purpose": "Assistant-only SFT patch using only accepted checkpoint-700 targeted repairs; no stabilizers.",
        "source_files": {"targeted_repairs": args.targeted_repairs},
        "row_count": len(rows),
        "component_counts": dict(Counter(r["patch_component"] for r in rows)),
        "failure_tag_counts": dict(Counter(t for r in rows for t in r.get("failure_tags", []))),
        "number_policy_counts": dict(Counter(r.get("number_policy", "") for r in rows)),
        "digit_summary_count": sum(bool(DIGIT_RE.search(r["summary"])) for r in rows),
        "word_count": {
            "avg": sum(wcs) / len(wcs),
            "min": min(wcs),
            "max": max(wcs),
            "under10": sum(w < 10 for w in wcs),
            "over15": sum(w > 15 for w in wcs),
            "in_10_15": sum(10 <= w <= 15 for w in wcs),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# q5 Step-2400 SFT700 Targeted-only v2 Dataset\n\n"
        "Assistant-only patch dataset using only accepted checkpoint-700 targeted repairs.\n\n"
        "No stabilizer rows are included. This is intentionally sharper and riskier than patch v1.\n\n"
        f"Rows: {len(rows)}\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
