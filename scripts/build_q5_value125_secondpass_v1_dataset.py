#!/usr/bin/env python3
"""Build a second-pass checkpoint-125 continuation dataset."""
from __future__ import annotations

import collections
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path("hf_bucket/experiments/sft/q5_q8shape150_value_aware_v1/lr1e6/secondpass_after_ckpt125")
OUT_DIR = ROOT / "dataset_54second_x5_87first_518q8shape_193residual_77hard_v1"

SECOND_PASS_ROWS = Path(
    "hf_dataset/releases/recent_3y_balanced_10746_q5_value125_q8win_value_aware_v1/"
    "training_rows.jsonl"
)
FIRST_PASS_ROWS = Path(
    "hf_dataset/releases/recent_3y_balanced_10746_q5_q8shape150_q8win_value_aware_v1/"
    "training_rows.jsonl"
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

WEIGHTS = {
    "second_pass": 3.0,
    "second_pass_repeat": 5,
    "first_pass": 0.7,
    "q8_shape": 0.2,
    "residual_preserve": 0.1,
    "hard77": 0.8,
    "hard77_stress": 1.2,
}


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


def set_weight(row: dict[str, Any], weight: float, policy: str) -> dict[str, Any]:
    out = deepcopy(row)
    hints = dict(out.get("loss_weight_hints") or {})
    hints.update({"enabled": True, "base_weight": weight, "policy": policy, "spans": []})
    out["loss_weight_hints"] = hints
    return out


def base_paper_id(row: dict[str, Any]) -> str:
    return str(row.get("source_paper_id") or row.get("paper_id") or "").split("::", 1)[0]


def prepare_generated_rows(
    rows: list[dict[str, Any]],
    *,
    repeat: int,
    weight: float,
    component: str,
    row_type: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        source_pid = base_paper_id(row)
        for repeat_index in range(repeat):
            new_row = set_weight(
                row,
                weight,
                f"q5_value125_secondpass_v1 {component} weight {weight:g}",
            )
            summary = norm(new_row.get("summary", ""))
            new_row["summary"] = summary
            new_row["source_paper_id"] = source_pid
            new_row["paper_id"] = f"{source_pid}::{component}_{repeat_index:02d}"
            new_row["dataset_component"] = component
            new_row["source_split"] = "q5_value125_secondpass_v1"
            new_row["weighted_numeric_patch_row_type"] = row_type
            new_row[f"{component}_repeat_index"] = repeat_index
            new_row["messages"] = messages(
                str(new_row.get("title", "")),
                str(new_row.get("abstract", "")),
                summary,
            )
            out.append(new_row)
    return out


def prepare_preserve_rows(rows: list[dict[str, Any]], weight: float, component: str) -> list[dict[str, Any]]:
    return [
        set_weight(
            row,
            weight,
            f"q5_value125_secondpass_v1 {component} preserve weight {weight:g}",
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
                f"q5_value125_secondpass_v1 hard77 preserve weight {weight:g}",
            )
        )
    return out


def main() -> None:
    second_pass = prepare_generated_rows(
        load_jsonl(SECOND_PASS_ROWS),
        repeat=int(WEIGHTS["second_pass_repeat"]),
        weight=float(WEIGHTS["second_pass"]),
        component="second_pass_q8win_value_aware",
        row_type="q5_value125_q8win_value_aware_kimi_qwen",
    )
    first_pass = prepare_generated_rows(
        load_jsonl(FIRST_PASS_ROWS),
        repeat=1,
        weight=float(WEIGHTS["first_pass"]),
        component="first_pass_q8win_value_aware",
        row_type="q5_q8shape150_q8win_value_aware_kimi_qwen",
    )
    q8_shape = prepare_preserve_rows(load_jsonl(Q8_SHAPE_ROWS), WEIGHTS["q8_shape"], "q8_shape_distill")
    residual = prepare_preserve_rows(load_jsonl(RESIDUAL_ROWS), WEIGHTS["residual_preserve"], "residual_repair")
    hard77 = prepare_hard77_rows(load_jsonl(HARD77_ROWS))
    rows = [*second_pass, *first_pass, *q8_shape, *residual, *hard77]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_DIR / "second_pass_rows_repeated.jsonl", second_pass)
    write_jsonl(OUT_DIR / "first_pass_rows.jsonl", first_pass)
    write_jsonl(OUT_DIR / "q8_shape_preserve_rows.jsonl", q8_shape)
    write_jsonl(OUT_DIR / "residual_preserve_rows.jsonl", residual)
    write_jsonl(OUT_DIR / "hard77_rows.jsonl", hard77)
    write_jsonl(OUT_DIR / "train.jsonl", rows)

    component_counts = collections.Counter(str(row.get("dataset_component", "")) for row in rows)
    row_type_counts = collections.Counter(str(row.get("weighted_numeric_patch_row_type", "")) for row in rows)
    manifest = {
        "created_by": "scripts/build_q5_value125_secondpass_v1_dataset.py",
        "dataset_path": str(OUT_DIR / "train.jsonl"),
        "rows": len(rows),
        "component_counts": dict(sorted(component_counts.items())),
        "row_type_counts": dict(sorted(row_type_counts.items())),
        "weights": WEIGHTS,
        "source_paths": {
            "second_pass_rows": str(SECOND_PASS_ROWS),
            "first_pass_rows": str(FIRST_PASS_ROWS),
            "q8_shape_rows": str(Q8_SHAPE_ROWS),
            "residual_rows": str(RESIDUAL_ROWS),
            "hard77_rows": str(HARD77_ROWS),
        },
        "notes": [
            "Continuation starts from value-aware lr1e6 checkpoint 125.",
            "Second-pass labels were written by Kimi and verified by Qwen from checkpoint-125 blind-review q8 wins.",
            "Preservation rows are kept low-weight so the new targeted repairs can move the runtime surface.",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[q5-value125-secondpass-dataset] rows={len(rows)} output={OUT_DIR / 'train.jsonl'}")
    print(f"[q5-value125-secondpass-dataset] components={dict(sorted(component_counts.items()))}")


if __name__ == "__main__":
    main()
