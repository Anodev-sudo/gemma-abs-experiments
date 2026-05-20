#!/usr/bin/env python3
"""Build weighted numeric patch v3 dataset with targeted stress boosters."""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

BASE_BUILDER_PATH = Path("scripts/build_sft700_weighted_numeric_patch_v1_dataset.py")


def load_base_builder():
    spec = importlib.util.spec_from_file_location("weighted_numeric_v1_builder", BASE_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {BASE_BUILDER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def apply_manual_weight_overrides(row: dict[str, Any], builder: Any) -> None:
    """Give sticky stress rows stronger direct pressure on the desired binding."""
    source_id = str(row.get("source_paper_id") or row.get("paper_id", "")).split("#")[0]
    summary = row["summary"]
    hints = row.get("loss_weight_hints") or {"enabled": True, "base_weight": 1.0, "spans": []}
    hints["enabled"] = True
    hints["base_weight"] = 1.0
    spans = list(hints.get("spans", []))

    if source_id == "2111.10635":
        targets = {
            "HeterPS": 3.0,
            "improves": 6.0,
            "DNN": 3.0,
            "training": 3.0,
            "throughput": 8.0,
            "14.5x": 8.0,
            "cutting": 6.0,
            "costs": 8.0,
            "312.3%": 8.0,
        }
    elif source_id == "1901.06548":
        targets = {
            "Height-optimal": 8.0,
            "tangles": 8.0,
            "NP-hard": 8.0,
            "faster": 6.0,
            "algorithms": 4.0,
            "structured": 6.0,
            "swap": 6.0,
            "lists": 6.0,
        }
    else:
        return

    for needle, weight in targets.items():
        start = summary.find(needle)
        if start < 0:
            continue
        new = {"start": start, "end": start + len(needle), "text": needle, "weight": weight, "kind": "manual_stress_booster"}
        builder.add_span(spans, new)

    spans.sort(key=lambda x: (int(x["start"]), int(x["end"])))
    hints["spans"] = spans
    hints["stress_booster_manual_override"] = True
    row["loss_weight_hints"] = hints


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal-source", default="hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/train.jsonl")
    ap.add_argument("--numeric-source", default="hf_bucket/experiments/sft/q5_step2400_sft700_numeric_role_curriculum_v1/dataset_majority_synthetic_v1/train.jsonl")
    ap.add_argument("--stress-source", default="hf_bucket/experiments/sft/q5_step2400_sft700_numeric_binding_poc_exact_prompt_v1/dataset/train.jsonl")
    ap.add_argument("--output-dir", default="hf_bucket/experiments/sft/q5_step2400_sft700_weighted_numeric_patch_v3_stress_boost/dataset_75normal_20general_numeric_5stress_v1")
    ap.add_argument("--seed", type=int, default=20260517)
    args = ap.parse_args()

    builder = load_base_builder()
    normal_source = Path(args.normal_source)
    numeric_source = Path(args.numeric_source)
    stress_source = Path(args.stress_source)
    out_dir = Path(args.output_dir)

    normal_raw = load_jsonl(normal_source)
    numeric_raw = load_jsonl(numeric_source)
    stress_raw = load_jsonl(stress_source)

    normal_candidates = []
    seen_pids: set[str] = set()
    for row in normal_raw:
        pid = str(row.get("paper_id", ""))
        if not pid or pid in seen_pids:
            continue
        if builder.good_summary(row, allow_digits=False):
            normal_candidates.append(row)
            seen_pids.add(pid)
    normal_selected = builder.select_by_umbrella(normal_candidates, 600, args.seed)
    normal_rows = [
        builder.normalize_row(row, "normal_style_anchor", str(normal_source), "normal_style_anchor")
        for row in normal_selected
    ]

    real_numeric = [
        row for row in numeric_raw
        if row.get("patch_component") == "real_mined_numeric_repair_subagent_accepted"
        and builder.good_summary(row, allow_digits=True)
        and builder.DIGIT_RE.search(str(row.get("summary", "")))
    ]
    real_numeric.sort(key=lambda r: (str(r.get("number_policy", "")), str(r.get("paper_id", ""))))

    synthetic_candidates = [
        row for row in numeric_raw
        if row.get("patch_component") == "synthetic_numeric_role_current_failed_v5"
        and row.get("current_sft700_failed_numeric_policy") is True
        and builder.good_summary(row, allow_digits=True)
        and builder.DIGIT_RE.search(str(row.get("summary", "")))
    ]
    synthetic_selected = builder.round_robin_by_key(synthetic_candidates, "numeric_pattern", 160 - len(real_numeric), args.seed + 1)

    general_numeric_rows = []
    for row in real_numeric:
        general_numeric_rows.append(builder.normalize_row(row, "numeric_challenge_real_mined_current_failed", str(numeric_source), "numeric_challenge"))
    for row in synthetic_selected:
        general_numeric_rows.append(builder.normalize_row(row, "numeric_challenge_synthetic_current_failed", str(numeric_source), "numeric_challenge"))

    stress_by_source: dict[str, list[dict[str, Any]]] = {"1901.06548": [], "2111.10635": []}
    for row in stress_raw:
        sid = str(row.get("source_paper_id") or row.get("paper_id", "")).split("#")[0]
        if sid in stress_by_source and builder.good_summary(row, allow_digits=True):
            stress_by_source[sid].append(row)
    for sid in stress_by_source:
        stress_by_source[sid].sort(key=lambda r: str(r.get("paper_id", "")))
        if len(stress_by_source[sid]) < 20:
            raise ValueError(f"Need 20 stress rows for {sid}, have {len(stress_by_source[sid])}")

    stress_rows = []
    for sid in ["1901.06548", "2111.10635"]:
        for i, row in enumerate(stress_by_source[sid][:20]):
            copied = dict(row)
            copied["paper_id"] = f"{sid}#stressboost{i:02d}"
            copied["source_paper_id"] = sid
            out = builder.normalize_row(copied, "numeric_challenge_known_stress_booster", str(stress_source), "numeric_challenge_stress_booster")
            out["stress_booster_index"] = i
            out["stress_booster_target"] = sid
            apply_manual_weight_overrides(out, builder)
            stress_rows.append(out)

    all_rows = normal_rows + general_numeric_rows + stress_rows
    if (len(normal_rows), len(general_numeric_rows), len(stress_rows), len(all_rows)) != (600, 160, 40, 800):
        raise AssertionError((len(normal_rows), len(general_numeric_rows), len(stress_rows), len(all_rows)))

    rnd = random.Random(args.seed + 2)
    rnd.shuffle(all_rows)
    for i, row in enumerate(all_rows):
        row["weighted_numeric_patch_dataset_id"] = "q5_step2400_sft700_weighted_numeric_patch_v3_stress_boost"
        row["weighted_numeric_patch_mix_index"] = i

    write_jsonl(out_dir / "train.jsonl", all_rows)
    write_jsonl(out_dir / "normal_style_anchor_rows.jsonl", normal_rows)
    write_jsonl(out_dir / "general_numeric_challenge_rows.jsonl", general_numeric_rows)
    write_jsonl(out_dir / "stress_booster_rows.jsonl", stress_rows)

    def avg(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    manifest = {
        "dataset_id": "q5_step2400_sft700_weighted_numeric_patch_v3_stress_boost",
        "purpose": "Keep v1/v2 style-stable weighted setup but reserve 5% of rows for exact sticky stress boosters.",
        "base_checkpoint_intent": "hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/checkpoint-700",
        "output_train_jsonl": str(out_dir / "train.jsonl"),
        "seed": args.seed,
        "row_count": len(all_rows),
        "normal_rows": len(normal_rows),
        "general_numeric_rows": len(general_numeric_rows),
        "stress_booster_rows": len(stress_rows),
        "mix": {"normal_style_anchor": 0.75, "general_numeric_challenge": 0.20, "stress_booster": 0.05},
        "component_counts": dict(Counter(row["patch_component"] for row in all_rows)),
        "row_type_counts": dict(Counter(row["weighted_numeric_patch_row_type"] for row in all_rows)),
        "stress_booster_target_counts": dict(Counter(row.get("stress_booster_target", "") for row in stress_rows)),
        "normal_umbrella_counts": dict(Counter(builder.umbrella(str(row.get("primary_category", ""))) for row in normal_rows)),
        "numeric_pattern_counts": dict(Counter(str(row.get("numeric_pattern") or "real_or_stress") for row in general_numeric_rows + stress_rows)),
        "summary_word_counts": dict(sorted(Counter(row["summary_word_count"] for row in all_rows).items())),
        "avg_summary_words": avg([row["summary_word_count"] for row in all_rows]),
        "avg_normal_summary_words": avg([row["summary_word_count"] for row in normal_rows]),
        "avg_general_numeric_summary_words": avg([row["summary_word_count"] for row in general_numeric_rows]),
        "avg_stress_booster_summary_words": avg([row["summary_word_count"] for row in stress_rows]),
        "loss_weighting": {
            "normal_rows": "base assistant-token weight 1.0 only",
            "general_numeric_rows": "same v1 span hints: numbers/symbols 6, direction/polarity 5, concepts/near-number words 3",
            "stress_booster_rows": "manual override up to 8x on sticky target concepts and number-role bindings",
            "important": "weights apply only to assistant target tokens; source/prompt tokens remain masked",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Weighted numeric patch v3 stress boost dataset\n\n",
        "800 rows: 600 normal style anchors, 160 general numeric challenges, 40 exact stress boosters.\n\n",
        "Stress boosters use 20 rows each for `1901.06548` and `2111.10635`, with manual weight overrides.\n\n",
        "## Stress booster examples\n\n",
    ]
    for row in stress_rows[:3] + stress_rows[20:23]:
        hints = ", ".join(f"{s['text']}:{s['weight']}" for s in row["loss_weight_hints"]["spans"][:14])
        lines.append(f"- `{row['paper_id']}` ({row['summary_word_count']}w): {row['summary']}\n  - hints: {hints}\n")
    (out_dir / "README.md").write_text("".join(lines), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
