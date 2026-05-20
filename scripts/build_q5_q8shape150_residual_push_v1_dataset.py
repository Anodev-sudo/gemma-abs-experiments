#!/usr/bin/env python3
"""Build a q8-shape checkpoint-150 continuation dataset with residual-push repairs."""
from __future__ import annotations

import collections
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path("hf_bucket/experiments/sft/q5_q8shape150_residual_push_v1")
OUT_DIR = ROOT / "dataset_51push_518q8shape_193residual_77hard_v1"

PUSH_ROWS = Path(
    "hf_bucket/experiments/sft/q5_residual218_push_285_v1/"
    "dataset_repair_600selfanchor_77hard_v1/repair_rows.jsonl"
)
Q8_SHAPE_ROWS = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_residual218_q8_shape_distill_v1_lr2e7/"
    "dataset_259q8shape_x2_193residual_77hard_200cleanself_v1/q8_shape_rows_repeated.jsonl"
)
RESIDUAL_ROWS = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_residual218_q8_shape_distill_v1_lr2e7/"
    "dataset_259q8shape_x2_193residual_77hard_200cleanself_v1/residual_repair_preserve_rows.jsonl"
)
HARD77_ROWS = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_residual218_q8_shape_distill_v1_lr2e7/"
    "dataset_259q8shape_x2_193residual_77hard_200cleanself_v1/hard77_rows.jsonl"
)

WEIGHTS = {
    "push": 2.5,
    "q8_shape": 0.75,
    "residual_preserve": 0.5,
    "hard77": 1.0,
    "hard77_stress": 1.5,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def set_weight(row: dict[str, Any], weight: float, policy: str) -> dict[str, Any]:
    out = deepcopy(row)
    hints = dict(out.get("loss_weight_hints") or {})
    hints.update({"enabled": True, "base_weight": weight, "policy": policy, "spans": []})
    out["loss_weight_hints"] = hints
    return out


def prepare_push_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        new_row = set_weight(
            row,
            WEIGHTS["push"],
            f"q8shape150_residual_push_v1 full assistant target weight {WEIGHTS['push']:g}",
        )
        new_row["dataset_component"] = "residual_push"
        new_row["source_split"] = "q5_q8shape150_residual_push_v1"
        new_row["weighted_numeric_patch_row_type"] = "q8shape150_residual_push_kimi_qwen"
        out.append(new_row)
    return out


def prepare_q8_shape_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        set_weight(
            row,
            WEIGHTS["q8_shape"],
            f"q8shape150_residual_push_v1 q8-shape preserve weight {WEIGHTS['q8_shape']:g}",
        )
        for row in rows
    ]


def prepare_residual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        set_weight(
            row,
            WEIGHTS["residual_preserve"],
            f"q8shape150_residual_push_v1 residual preserve weight {WEIGHTS['residual_preserve']:g}",
        )
        for row in rows
    ]


def prepare_hard77_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        row_type = str(row.get("weighted_numeric_patch_row_type", ""))
        weight = WEIGHTS["hard77_stress"] if row_type == "hard77_preserve_stress" else WEIGHTS["hard77"]
        out.append(
            set_weight(
                row,
                weight,
                f"q8shape150_residual_push_v1 hard77 preserve weight {weight:g}",
            )
        )
    return out


def source_paper_id(row: dict[str, Any]) -> str:
    return str(row.get("source_paper_id") or row.get("paper_id") or "").split("::")[0]


def main() -> None:
    push = prepare_push_rows(load_jsonl(PUSH_ROWS))
    q8_shape = prepare_q8_shape_rows(load_jsonl(Q8_SHAPE_ROWS))
    residual = prepare_residual_rows(load_jsonl(RESIDUAL_ROWS))
    hard77 = prepare_hard77_rows(load_jsonl(HARD77_ROWS))
    rows = [*push, *q8_shape, *residual, *hard77]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "push_rows.jsonl", push)
    write_jsonl(OUT_DIR / "q8_shape_preserve_rows.jsonl", q8_shape)
    write_jsonl(OUT_DIR / "residual_preserve_rows.jsonl", residual)
    write_jsonl(OUT_DIR / "hard77_rows.jsonl", hard77)
    write_jsonl(OUT_DIR / "train.jsonl", rows)

    component_counts = collections.Counter(str(row.get("dataset_component", "")) for row in rows)
    row_type_counts = collections.Counter(str(row.get("weighted_numeric_patch_row_type", "")) for row in rows)
    source_counts = collections.Counter(source_paper_id(row) for row in rows)
    duplicate_sources = sorted(pid for pid, count in source_counts.items() if pid and count > 1)
    manifest = {
        "created_by": "scripts/build_q5_q8shape150_residual_push_v1_dataset.py",
        "dataset_path": str(OUT_DIR / "train.jsonl"),
        "rows": len(rows),
        "component_counts": dict(sorted(component_counts.items())),
        "row_type_counts": dict(sorted(row_type_counts.items())),
        "weights": WEIGHTS,
        "source_paths": {
            "push_rows": str(PUSH_ROWS),
            "q8_shape_rows": str(Q8_SHAPE_ROWS),
            "residual_rows": str(RESIDUAL_ROWS),
            "hard77_rows": str(HARD77_ROWS),
        },
        "duplicate_source_paper_ids": len(duplicate_sources),
        "duplicate_source_paper_id_examples": duplicate_sources[:20],
        "notes": [
            "Continuation starts from q8-shape checkpoint 150, not checkpoint 218.",
            "Checkpoint-218 self anchors are intentionally omitted.",
            "All new push labels were written by Kimi and verified by Qwen.",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[q8shape150-push-dataset] rows={len(rows)} output={OUT_DIR / 'train.jsonl'}")
    print(f"[q8shape150-push-dataset] components={dict(sorted(component_counts.items()))}")


if __name__ == "__main__":
    main()
