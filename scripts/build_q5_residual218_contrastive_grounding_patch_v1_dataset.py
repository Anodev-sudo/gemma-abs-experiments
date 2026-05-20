#!/usr/bin/env python3
"""Build a checkpoint-218 contrastive grounding patch SFT dataset."""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path
from typing import Any

from generate_llama_stream import SYSTEM_PROMPT, USER_TEMPLATE


ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_residual218_contrastive_grounding_patch_v1_lr2e7")
DEFAULT_DATASET_NAME = "dataset_272contrastive_77hard_200ideal_600normal_v1"
DEFAULT_SOURCE_SPLIT = "q5_residual218_contrastive_grounding_patch_v1"
CONTRASTIVE_REPAIRS = Path(
    "hf_bucket/experiments/sft/q5_promoted218_vs_q8_contrastive_mine_10000_v1/"
    "deepseek_v4_qwen_repair_v3/materialized/training_rows.jsonl"
)
NORMAL_ANCHORS = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_numeric_hard_attractor_booster_v1_lr1e6/"
    "dataset_600normal_200ideal_77booster_v1/normal_selected_600.jsonl"
)
NORMAL_FILLER = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin150_style_replay_v1_lr5e7_steps225/"
    "dataset_1200oldstyle_200ideal_77hard_v1/old_style_selected_1200.jsonl"
)
IDEAL_NUMERIC = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_numeric_ideal200_regen_v1/"
    "clean_200_after_filters_v1/training_rows_200.jsonl"
)
HARD77 = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7/"
    "dataset_193repair_600selfanchor_77hard_v1/hard77_rows_77.jsonl"
)

STRESS_IDS = {"1901.06548", "2111.10635", "2303.09722", "2406.04373"}
PRIORITY_GROUNDING_FAILURES = {"invented_or_wrong_entity", "lost_main_result"}
MALFORMED_OR_POLARITY = {"malformed_terms", "value_or_polarity_error"}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")
SENTENCE_END_RE = re.compile(r"""[.!?]+(?:['")\]]+)?(?=\s|$)""")
FORMULA_RE = re.compile(r"<[^>]+>|\$|\\[A-Za-z]+|\bO\([^)]*\)|\^|_|[{}]|[≤≥≈]")
BANNED_OPENINGS = (
    "this paper",
    "the paper",
    "the authors",
    "the work",
    "we propose",
    "in this work",
    "in this paper",
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
    return " ".join(str(text or "").strip().split())


def has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def word_count(text: str) -> int:
    return len(norm(text).split())


def sentence_count(text: str) -> int:
    stripped = norm(text)
    if not stripped:
        return 0
    endings = SENTENCE_END_RE.findall(stripped)
    return len(endings) if endings else 1


def title_overlap_ratio(summary: str, title: str) -> float:
    summary_tokens = {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z-]{2,}", summary)}
    title_tokens = {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z-]{2,}", title)}
    if not summary_tokens:
        return 0.0
    return len(summary_tokens & title_tokens) / len(summary_tokens)


def adjacent_duplicate(text: str) -> bool:
    previous = ""
    for token in (token.lower() for token in WORD_RE.findall(text)):
        if len(token) > 2 and token == previous:
            return True
        previous = token
    return False


def label_reject_reasons(summary: str, title: str, *, reject_high_title_overlap: bool = True) -> list[str]:
    text = norm(summary)
    low = text.lower()
    reasons: list[str] = []
    wc = word_count(text)
    if not text:
        reasons.append("empty")
    if wc < 10:
        reasons.append("under_10_words")
    if wc > 15:
        reasons.append("over_15_words")
    if sentence_count(text) != 1:
        reasons.append("not_single_sentence")
    if has_digit(text):
        reasons.append("digit")
    if FORMULA_RE.search(text):
        reasons.append("formula_or_symbol")
    if any(low.startswith(prefix) for prefix in BANNED_OPENINGS):
        reasons.append("banned_opening")
    if reject_high_title_overlap and title_overlap_ratio(text, title) >= 0.8:
        reasons.append("high_title_overlap")
    if adjacent_duplicate(text):
        reasons.append("adjacent_duplicate")
    return reasons


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


def normalize_row(
    row: dict[str, Any],
    *,
    component: str,
    row_type: str,
    weight: float,
    source_split: str,
    summary_key: str = "summary",
) -> dict[str, Any]:
    summary = norm(row.get(summary_key, ""))
    out = dict(row)
    out["summary"] = summary
    out["dataset_component"] = component
    out["weighted_numeric_patch_row_type"] = row_type
    out["source_split"] = source_split
    out["loss_weight_hints"] = loss_weight(
        weight,
        f"contrastive_grounding_patch_v1 full assistant target weight {weight:g}",
    )
    out["messages"] = messages(out.get("title", ""), out.get("abstract", ""), summary)
    return out


def contrastive_weight(row: dict[str, Any], args: argparse.Namespace) -> tuple[float, str]:
    bucket = str(row.get("source_bucket", ""))
    family = str(row.get("manual_failure_type", ""))
    if bucket == "both":
        weight = args.contrastive_both_weight
    elif bucket == "high_precision_only":
        weight = args.contrastive_high_precision_weight
    elif bucket == "semantic_fog_only":
        weight = args.contrastive_semantic_fog_weight
    else:
        weight = args.contrastive_default_weight
    if family in PRIORITY_GROUNDING_FAILURES:
        weight += args.grounding_failure_bonus
    if family in MALFORMED_OR_POLARITY:
        weight += args.malformed_or_polarity_bonus
    return weight, f"contrastive_repair_{bucket or 'unknown'}_{family or 'unknown'}"


def normalize_contrastive(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item["paper_id"])):
        reasons = label_reject_reasons(row.get("summary", ""), row.get("title", ""), reject_high_title_overlap=False)
        if reasons:
            raise SystemExit(f"Contrastive repair failed label filter: {row.get('paper_id')} {reasons}")
        weight, row_type = contrastive_weight(row, args)
        out.append(
            normalize_row(
                row,
                component="contrastive_deepseek_qwen_repair",
                row_type=row_type,
                weight=weight,
                source_split=args.source_split,
            )
        )
    return out


def normalize_ideal(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item["paper_id"])):
        reasons = label_reject_reasons(row.get("summary", ""), row.get("title", ""), reject_high_title_overlap=False)
        if reasons:
            raise SystemExit(f"Ideal numeric row failed label filter: {row.get('paper_id')} {reasons}")
        out.append(
            normalize_row(
                row,
                component="ideal_numeric_anchor",
                row_type="ideal_numeric_low_weight",
                weight=args.ideal_numeric_weight,
                source_split=args.source_split,
            )
        )
    return out


def normalize_hard77(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item["paper_id"])):
        pid = str(row["paper_id"])
        reasons = label_reject_reasons(row.get("summary", ""), row.get("title", ""), reject_high_title_overlap=False)
        if reasons:
            raise SystemExit(f"Hard77 row failed label filter: {pid} {reasons}")
        out.append(
            normalize_row(
                row,
                component="hard77_preserve",
                row_type="hard77_preserve_stress" if pid in STRESS_IDS else "hard77_preserve",
                weight=args.hard77_stress_weight if pid in STRESS_IDS else args.hard77_weight,
                source_split=args.source_split,
            )
        )
    return out


def normalize_normal_anchors(
    primary_rows: list[dict[str, Any]],
    filler_rows: list[dict[str, Any]],
    *,
    exclude_ids: set[str],
    count: int,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def consider(row: dict[str, Any], origin: str) -> None:
        if len(accepted) >= count:
            return
        pid = str(row.get("paper_id", ""))
        if not pid or pid in seen or pid in exclude_ids:
            return
        seen.add(pid)
        reasons = label_reject_reasons(row.get("summary", ""), row.get("title", ""))
        if reasons:
            rejected.append({**row, "normal_anchor_origin": origin, "normal_anchor_reject_reasons": reasons})
            return
        normalized = normalize_row(
            {**row, "normal_anchor_origin": origin},
            component="normal_style_anchor",
            row_type="normal_style_anchor_very_low_weight",
            weight=args.normal_anchor_weight,
            source_split=args.source_split,
        )
        accepted.append(normalized)

    for row in sorted(primary_rows, key=lambda item: str(item["paper_id"])):
        consider(row, "trusted_600normal_pool")

    rng = random.Random(seed)
    shuffled_fillers = sorted(filler_rows, key=lambda item: str(item["paper_id"]))
    rng.shuffle(shuffled_fillers)
    for row in shuffled_fillers:
        consider(row, "old_style_clean_filler")

    if len(accepted) != count:
        raise SystemExit(f"Expected {count} normal anchors after fill; got {len(accepted)}")
    return accepted, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build q5 residual218 contrastive grounding patch dataset.")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--contrastive-repairs", type=Path, default=CONTRASTIVE_REPAIRS)
    parser.add_argument("--normal-anchors", type=Path, default=NORMAL_ANCHORS)
    parser.add_argument("--normal-filler", type=Path, default=NORMAL_FILLER)
    parser.add_argument("--ideal-numeric", type=Path, default=IDEAL_NUMERIC)
    parser.add_argument("--hard77", type=Path, default=HARD77)
    parser.add_argument("--normal-anchor-count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--source-split", default=DEFAULT_SOURCE_SPLIT)
    parser.add_argument("--contrastive-both-weight", type=float, default=2.5)
    parser.add_argument("--contrastive-high-precision-weight", type=float, default=2.0)
    parser.add_argument("--contrastive-semantic-fog-weight", type=float, default=1.5)
    parser.add_argument("--contrastive-default-weight", type=float, default=1.75)
    parser.add_argument("--grounding-failure-bonus", type=float, default=0.0)
    parser.add_argument("--malformed-or-polarity-bonus", type=float, default=0.25)
    parser.add_argument("--hard77-weight", type=float, default=1.0)
    parser.add_argument("--hard77-stress-weight", type=float, default=1.5)
    parser.add_argument("--ideal-numeric-weight", type=float, default=0.75)
    parser.add_argument("--normal-anchor-weight", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contrastive = normalize_contrastive(load_jsonl(args.contrastive_repairs), args)
    hard77 = normalize_hard77(load_jsonl(args.hard77), args)
    ideal = normalize_ideal(load_jsonl(args.ideal_numeric), args)

    higher_priority_ids = {str(row["paper_id"]) for row in contrastive}
    higher_priority_ids.update(str(row["paper_id"]) for row in hard77)
    higher_priority_ids.update(str(row["paper_id"]) for row in ideal)
    normal, normal_rejected = normalize_normal_anchors(
        load_jsonl(args.normal_anchors),
        load_jsonl(args.normal_filler),
        exclude_ids=higher_priority_ids,
        count=args.normal_anchor_count,
        seed=args.seed,
        args=args,
    )

    final = [*contrastive, *hard77, *ideal, *normal]
    if len(contrastive) != 272:
        raise SystemExit(f"Expected 272 contrastive rows; got {len(contrastive)}")
    if len(hard77) != 77:
        raise SystemExit(f"Expected 77 hard77 rows; got {len(hard77)}")
    if len(ideal) != 200:
        raise SystemExit(f"Expected 200 ideal numeric rows; got {len(ideal)}")
    if len(final) != 1149:
        raise SystemExit(f"Expected 1149 final rows; got {len(final)}")

    digit_rows = [str(row["paper_id"]) for row in final if has_digit(row.get("summary", ""))]
    if digit_rows:
        raise SystemExit(f"Digit labels in final dataset: {digit_rows[:20]}")

    bad_rows: list[dict[str, Any]] = []
    for row in final:
        reasons = label_reject_reasons(
            row.get("summary", ""),
            row.get("title", ""),
            reject_high_title_overlap=row.get("dataset_component") == "normal_style_anchor",
        )
        if reasons:
            bad_rows.append({"paper_id": row.get("paper_id"), "component": row.get("dataset_component"), "reasons": reasons})
    if bad_rows:
        raise SystemExit(f"Final dataset failed label screen: {bad_rows[:20]}")

    dataset_dir = args.output_root / args.dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(dataset_dir / "train.jsonl", final)
    write_jsonl(dataset_dir / "contrastive_repair_rows_272.jsonl", contrastive)
    write_jsonl(dataset_dir / "hard77_rows_77.jsonl", hard77)
    write_jsonl(dataset_dir / "ideal_numeric_rows_200.jsonl", ideal)
    write_jsonl(dataset_dir / "normal_style_anchor_rows_600.jsonl", normal)
    write_jsonl(dataset_dir / "normal_style_anchor_rejected_or_skipped.jsonl", normal_rejected)

    duplicate_source_pids = {
        pid: count
        for pid, count in collections.Counter(str(row["paper_id"]) for row in final).items()
        if count > 1
    }
    row_weights = [float(row["loss_weight_hints"]["base_weight"]) for row in final]
    component_weight_sums: dict[str, float] = {}
    row_type_weight_sums: dict[str, float] = {}
    for row, weight in zip(final, row_weights, strict=True):
        component = str(row["dataset_component"])
        row_type = str(row["weighted_numeric_patch_row_type"])
        component_weight_sums[component] = component_weight_sums.get(component, 0.0) + weight
        row_type_weight_sums[row_type] = row_type_weight_sums.get(row_type, 0.0) + weight
    manifest = {
        "created_by": "scripts/build_q5_residual218_contrastive_grounding_patch_v1_dataset.py",
        "dataset_path": str(dataset_dir / "train.jsonl"),
        "source_split": args.source_split,
        "start_adapter_intended": (
            "hf_bucket/experiments/sft/"
            "q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7/checkpoint-218"
        ),
        "rows": len(final),
        "contrastive_repair_rows": len(contrastive),
        "hard77_rows": len(hard77),
        "ideal_numeric_rows": len(ideal),
        "normal_style_anchor_rows": len(normal),
        "normal_style_anchor_rejected_or_skipped": len(normal_rejected),
        "digit_label_rows": 0,
        "component_counts": dict(collections.Counter(row["dataset_component"] for row in final)),
        "row_type_counts": dict(collections.Counter(row["weighted_numeric_patch_row_type"] for row in final)),
        "weight_counts": dict(collections.Counter(f"{weight:g}" for weight in row_weights)),
        "row_weight_sum": round(sum(row_weights), 6),
        "component_weight_sums": {key: round(value, 6) for key, value in sorted(component_weight_sums.items())},
        "row_type_weight_sums": {key: round(value, 6) for key, value in sorted(row_type_weight_sums.items())},
        "teacher_model_counts": dict(collections.Counter(row.get("teacher_model", "") for row in final)),
        "contrastive_source_bucket_counts": dict(collections.Counter(row.get("source_bucket", "") for row in contrastive)),
        "contrastive_failure_type_counts": dict(collections.Counter(row.get("manual_failure_type", "") for row in contrastive)),
        "normal_anchor_origin_counts": dict(collections.Counter(row.get("normal_anchor_origin", "") for row in normal)),
        "duplicate_source_paper_ids": duplicate_source_pids,
        "duplicate_source_paper_id_count": len(duplicate_source_pids),
        "weight_policy": {
            "contrastive_both": args.contrastive_both_weight,
            "contrastive_high_precision_only": args.contrastive_high_precision_weight,
            "contrastive_semantic_fog_only": args.contrastive_semantic_fog_weight,
            "contrastive_default": args.contrastive_default_weight,
            "contrastive_grounding_failure_bonus": args.grounding_failure_bonus,
            "contrastive_malformed_or_polarity_bonus": args.malformed_or_polarity_bonus,
            "hard77": args.hard77_weight,
            "hard77_stress": args.hard77_stress_weight,
            "ideal_numeric": args.ideal_numeric_weight,
            "normal_style_anchor": args.normal_anchor_weight,
        },
        "policy_notes": [
            "No q5 self-anchor rows are included.",
            "Q8 outputs are not used as labels.",
            "Rejected DeepSeek rows and Qwen rejects are excluded.",
            "Normal/style anchors are screened for digit residue using Python isdigit, including superscripts.",
            "All rows use full assistant-token loss only, without prompt/source loss or span boosts.",
        ],
        "source_paths": {
            "contrastive_repair_rows": str(args.contrastive_repairs),
            "hard77_rows": str(args.hard77),
            "ideal_numeric_rows": str(args.ideal_numeric),
            "normal_primary_rows": str(args.normal_anchors),
            "normal_filler_rows": str(args.normal_filler),
        },
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[contrastive-grounding-dataset] wrote {len(final)} rows to {dataset_dir / 'train.jsonl'}")
    print(f"[contrastive-grounding-dataset] normal anchors: {manifest['normal_anchor_origin_counts']}")


if __name__ == "__main__":
    main()
