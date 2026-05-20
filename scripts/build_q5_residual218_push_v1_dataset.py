#!/usr/bin/env python3
"""Build the weighted SFT dataset for the checkpoint-218 q5 push pass."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_REPAIR_ROWS = Path(
    "hf_dataset/releases/recent_3y_balanced_10746_q5_residual218_push_285_v1/"
    "training_rows.jsonl"
)
DEFAULT_PRESERVE_DATASET = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7/"
    "dataset_193repair_600selfanchor_77hard_v1/train.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "hf_bucket/experiments/sft/q5_residual218_push_285_v1/"
    "dataset_repair_600selfanchor_77hard_v1"
)

SYSTEM_PROMPT = (
    "You write one-sentence paper summaries for a mobile research feed. "
    "Keep them concrete, readable, and useful. Output only the summary sentence."
)
USER_TEMPLATE = (
    "The app already shows the title, author list, and date. "
    "Write one sentence that tells a reader why they might open this paper. "
    "Aim for 10 to 15 words. Avoid repeating the title. "
    'Avoid filler like "This paper" or "We propose". '
    "Do not mention authors or dates. "
    "Keep the wording natural and specific.\n\n"
    "Title:\n{title}\n\n"
    "Abstract:\n{abstract}"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def norm(text: Any) -> str:
    return " ".join(str(text or "").split())


def messages(title: str, abstract: str, summary: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(title=title, abstract=abstract)},
        {"role": "assistant", "content": summary},
    ]


def loss_weight(weight: float, policy: str) -> dict[str, Any]:
    return {
        "enabled": weight != 1.0,
        "base_weight": weight,
        "spans": [],
        "policy": policy,
    }


def base_paper_id(value: Any) -> str:
    return str(value or "").split("::", 1)[0]


def normalize_repair_row(row: dict[str, Any], weight: float) -> dict[str, Any]:
    out = dict(row)
    summary = norm(out.get("summary", ""))
    out["summary"] = summary
    out["dataset_component"] = "repair"
    out["weighted_numeric_patch_row_type"] = "q5_residual218_push_kimi_qwen"
    out["source_split"] = "q5_residual218_push_285_v1"
    out["loss_weight_hints"] = loss_weight(
        weight,
        f"q5_residual218_push_v1 full assistant target weight {weight:g}",
    )
    out["messages"] = messages(out.get("title", ""), out.get("abstract", ""), summary)
    return out


def is_preserve_row(row: dict[str, Any]) -> bool:
    component = str(row.get("dataset_component", ""))
    row_type = str(row.get("weighted_numeric_patch_row_type", ""))
    return component in {"self_anchor", "hard77"} or row_type.startswith(
        ("burnin150_self_anchor", "hard77_preserve")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine q5 push repairs with checkpoint-218 preservation rows."
    )
    parser.add_argument("--repair-rows", type=Path, default=DEFAULT_REPAIR_ROWS)
    parser.add_argument("--preserve-dataset", type=Path, default=DEFAULT_PRESERVE_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repair-weight", type=float, default=2.0)
    args = parser.parse_args()

    if not args.repair_rows.exists():
        raise SystemExit(
            f"Missing repair rows: {args.repair_rows}. "
            "Run the Kimi/Qwen summary-generation pipeline through materialize first."
        )

    repair_rows = [normalize_repair_row(row, args.repair_weight) for row in load_jsonl(args.repair_rows)]
    repair_ids = {base_paper_id(row.get("paper_id")) for row in repair_rows}
    preserve_rows = [
        row
        for row in load_jsonl(args.preserve_dataset)
        if is_preserve_row(row) and base_paper_id(row.get("paper_id")) not in repair_ids
    ]
    rows = repair_rows + preserve_rows

    output_dir = args.output_dir
    write_jsonl(output_dir / "train.jsonl", rows)
    write_jsonl(output_dir / "repair_rows.jsonl", repair_rows)
    write_jsonl(output_dir / "preserve_rows.jsonl", preserve_rows)

    manifest = {
        "created_by": "scripts/build_q5_residual218_push_v1_dataset.py",
        "dataset_path": str(output_dir / "train.jsonl"),
        "repair_rows_source": str(args.repair_rows),
        "preserve_dataset_source": str(args.preserve_dataset),
        "rows": len(rows),
        "repair_rows": len(repair_rows),
        "preserve_rows": len(preserve_rows),
        "component_counts": dict(Counter(row.get("dataset_component", "") for row in rows)),
        "row_type_counts": dict(
            Counter(row.get("weighted_numeric_patch_row_type", "") for row in rows)
        ),
        "repair_weight": args.repair_weight,
        "notes": [
            "Repairs come from the q5 residual-218 push Kimi/Qwen release.",
            "Preserve rows are checkpoint-218 self-anchor and hard77 rows only.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
