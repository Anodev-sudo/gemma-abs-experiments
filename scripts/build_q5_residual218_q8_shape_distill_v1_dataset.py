#!/usr/bin/env python3
"""Build a checkpoint-218 q8-shape distillation SFT dataset."""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path
from typing import Any

from generate_llama_stream import SYSTEM_PROMPT, USER_TEMPLATE


ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_residual218_q8_shape_distill_v1_lr2e7")
DEFAULT_Q8_SHAPE_ROWS = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_residual218_q8_shape_distill_v1/"
    "kimi_qwen_repair_v1/materialized/training_rows.jsonl"
)
DEFAULT_RESIDUAL_REPAIR_ROWS = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7/"
    "dataset_193repair_600selfanchor_77hard_v1/repair_rows_193.jsonl"
)
DEFAULT_HARD77_ROWS = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7/"
    "dataset_193repair_600selfanchor_77hard_v1/hard77_rows_77.jsonl"
)
DEFAULT_SELF_ANCHOR_ROWS = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7/"
    "dataset_193repair_600selfanchor_77hard_v1/self_anchor_rows_600.jsonl"
)

STRESS_IDS = {"1901.06548", "2111.10635", "2303.09722", "2406.04373"}
DIGIT_RE = re.compile(r"\d")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")
SENTENCE_END_RE = re.compile(r"""[.!?]+(?:['")\]]+)?(?=\s|$)""")
FORMULA_RE = re.compile(r"[$<>≤≥≈∑∫√∞πμσλβγθφψωΔΩαηχτ]|\\[A-Za-z]+|\^|_|[{}]|<[^>]+>")
EXACT_QUANTITY_WORD_RE = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
    r"thousand|million|billion|trillion|first|second|third|fourth|fifth|"
    r"sixth|seventh|eighth|ninth|tenth)\b",
    re.IGNORECASE,
)
BENCHMARK_RE = re.compile(
    r"\b(?:outperform|outperforms|outperformed|state-of-the-art|best|highest|"
    r"largest|superior|dramatic|major|significant)\b",
    re.IGNORECASE,
)
FOG_RE = re.compile(
    r"\b(?:frameworks?|structures?|patterns?|constraints?|resource allocation|"
    r"domain-specific|mapping|validation|deployment|model spaces|inheritance maps)\b",
    re.IGNORECASE,
)
MALFORMED_RE = re.compile(
    r"\b(?:Bayron|Curravariance|Ather|Sterochastic|Parametriding|Hermetic|"
    r"Coordined|pierded|weakary|Arenormalizable|Thermoregulatory|Noyehoteles|"
    r"chimerical|indoped|ansatzs|tired)\b",
    re.IGNORECASE,
)
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


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def sentence_count(text: str) -> int:
    text = norm(text)
    if not text:
        return 0
    endings = SENTENCE_END_RE.findall(text)
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


def hard_label_reject_reasons(
    summary: str,
    title: str,
    *,
    reject_fog: bool,
    allow_exact_quantity_words: bool = False,
) -> list[str]:
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
    if DIGIT_RE.search(text):
        reasons.append("digit")
    if FORMULA_RE.search(text):
        reasons.append("formula_or_symbol")
    if not allow_exact_quantity_words and EXACT_QUANTITY_WORD_RE.search(text):
        reasons.append("exact_quantity_word")
    if BENCHMARK_RE.search(text):
        reasons.append("benchmark_or_superlative")
    if any(low.startswith(prefix) for prefix in BANNED_OPENINGS):
        reasons.append("banned_opening")
    if title_overlap_ratio(text, title) >= 0.8:
        reasons.append("high_title_overlap")
    if adjacent_duplicate(text):
        reasons.append("adjacent_duplicate")
    if MALFORMED_RE.search(text):
        reasons.append("malformed_term")
    if reject_fog and FOG_RE.search(text):
        reasons.append("semantic_fog_term")
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
    row_type: str,
    component: str,
    weight: float,
    source_split: str,
) -> dict[str, Any]:
    summary = norm(row.get("summary", ""))
    out = dict(row)
    out["summary"] = summary
    out["weighted_numeric_patch_row_type"] = row_type
    out["dataset_component"] = component
    out["source_split"] = source_split
    out["loss_weight_hints"] = loss_weight(
        weight,
        f"q8_shape_distill_v1 full assistant target weight {weight:g}",
    )
    out["messages"] = messages(out.get("title", ""), out.get("abstract", ""), summary)
    return out


def normalize_q8_shape_row(row: dict[str, Any], *, repeat_index: int, weight: float) -> dict[str, Any]:
    source_pid = str(row["paper_id"])
    out = normalize_row(
        row,
        row_type="q8_shape_distill_teacher_qwen",
        component="q8_shape_distill",
        weight=weight,
        source_split="q5_residual218_q8_shape_distill_v1",
    )
    out["source_paper_id"] = source_pid
    out["paper_id"] = f"{source_pid}::q8_shape_distill_{repeat_index:02d}"
    out["q8_shape_distill_repeat_index"] = repeat_index
    return out


def normalize_preserve_row(row: dict[str, Any], *, weight: float, row_type: str, component: str) -> dict[str, Any]:
    reasons = hard_label_reject_reasons(
        row.get("summary", ""),
        row.get("title", ""),
        reject_fog=False,
        allow_exact_quantity_words=component == "hard77",
    )
    if reasons:
        raise SystemExit(f"Preserve row failed label filter: {row.get('paper_id')} {reasons}")
    return normalize_row(
        row,
        row_type=row_type,
        component=component,
        weight=weight,
        source_split="q5_residual218_q8_shape_distill_v1",
    )


def select_self_anchors(
    rows: list[dict[str, Any]],
    *,
    exclude_ids: set[str],
    count: int,
    seed: int,
    weight: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        pid = str(row.get("paper_id", ""))
        if not pid or pid in seen or pid in exclude_ids:
            continue
        seen.add(pid)
        reasons = hard_label_reject_reasons(row.get("summary", ""), row.get("title", ""), reject_fog=True)
        if reasons:
            rejected.append({**row, "anchor_reject_reasons": reasons})
            continue
        clean.append(row)
    rng = random.Random(seed)
    rng.shuffle(clean)
    selected = clean[:count]
    normalized = [
        normalize_row(
            row,
            row_type="checkpoint218_clean_self_anchor_very_low_weight",
            component="clean_self_anchor",
            weight=weight,
            source_split="q5_residual218_q8_shape_distill_v1",
        )
        for row in selected
    ]
    return normalized, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build q5 residual218 q8-shape distillation dataset.")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--q8-shape-rows", type=Path, default=DEFAULT_Q8_SHAPE_ROWS)
    parser.add_argument("--residual-repair-rows", type=Path, default=DEFAULT_RESIDUAL_REPAIR_ROWS)
    parser.add_argument("--hard77-rows", type=Path, default=DEFAULT_HARD77_ROWS)
    parser.add_argument("--self-anchor-rows", type=Path, default=DEFAULT_SELF_ANCHOR_ROWS)
    parser.add_argument("--q8-shape-repeat", type=int, default=2)
    parser.add_argument("--q8-shape-weight", type=float, default=1.5)
    parser.add_argument("--residual-repair-weight", type=float, default=0.75)
    parser.add_argument("--hard77-weight", type=float, default=1.0)
    parser.add_argument("--hard77-stress-weight", type=float, default=1.5)
    parser.add_argument("--self-anchor-count", type=int, default=200)
    parser.add_argument("--self-anchor-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260516)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.q8_shape_repeat < 1:
        raise SystemExit("--q8-shape-repeat must be >= 1")

    q8_unique = load_jsonl(args.q8_shape_rows)
    residual_raw = load_jsonl(args.residual_repair_rows)
    hard77_raw = load_jsonl(args.hard77_rows)
    self_anchor_raw = load_jsonl(args.self_anchor_rows)

    q8_clean_unique: list[dict[str, Any]] = []
    q8_rejected_by_dataset_filter: list[dict[str, Any]] = []
    for row in sorted(q8_unique, key=lambda item: str(item["paper_id"])):
        reasons = hard_label_reject_reasons(row.get("summary", ""), row.get("title", ""), reject_fog=True)
        if reasons:
            q8_rejected_by_dataset_filter.append({**row, "dataset_filter_reject_reasons": reasons})
            continue
        q8_clean_unique.append(row)

    q8_shape_rows: list[dict[str, Any]] = []
    for row in q8_clean_unique:
        for repeat_index in range(args.q8_shape_repeat):
            q8_shape_rows.append(
                normalize_q8_shape_row(row, repeat_index=repeat_index, weight=args.q8_shape_weight)
            )

    residual_rows = [
        normalize_preserve_row(
            row,
            weight=args.residual_repair_weight,
            row_type="residual_repair_preserve_low_weight",
            component="residual_repair_preserve",
        )
        for row in residual_raw
    ]
    hard77_rows = []
    for row in hard77_raw:
        pid = str(row["paper_id"])
        hard77_rows.append(
            normalize_preserve_row(
                row,
                weight=args.hard77_stress_weight if pid in STRESS_IDS else args.hard77_weight,
                row_type="hard77_preserve_stress" if pid in STRESS_IDS else "hard77_preserve",
                component="hard77",
            )
        )

    exclude_ids = {str(row.get("paper_id", "")) for row in residual_rows}
    exclude_ids.update(str(row.get("paper_id", "")) for row in hard77_rows)
    exclude_ids.update(str(row.get("source_paper_id", row.get("paper_id", ""))).split("::")[0] for row in q8_shape_rows)
    self_anchors, rejected_anchors = select_self_anchors(
        self_anchor_raw,
        exclude_ids=exclude_ids,
        count=args.self_anchor_count,
        seed=args.seed,
        weight=args.self_anchor_weight,
    )

    final = [*residual_rows, *hard77_rows, *self_anchors, *q8_shape_rows]
    duplicate_ids = [
        pid for pid, count in collections.Counter(str(row["paper_id"]) for row in final).items() if count > 1
    ]
    if duplicate_ids:
        raise SystemExit(f"Duplicate training paper IDs: {duplicate_ids[:20]}")
    digit_rows = [str(row["paper_id"]) for row in final if DIGIT_RE.search(str(row.get("summary", "")))]
    if digit_rows:
        raise SystemExit(f"Digit labels in final dataset: {digit_rows[:20]}")

    dataset_dir = (
        args.output_root
        / f"dataset_{len(q8_clean_unique)}q8shape_x{args.q8_shape_repeat}_"
        f"193residual_77hard_{len(self_anchors)}cleanself_v1"
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(dataset_dir / "train.jsonl", final)
    write_jsonl(dataset_dir / "unique_q8_shape_rows.jsonl", q8_clean_unique)
    write_jsonl(dataset_dir / "q8_shape_rows_rejected_by_dataset_filter.jsonl", q8_rejected_by_dataset_filter)
    write_jsonl(dataset_dir / "q8_shape_rows_repeated.jsonl", q8_shape_rows)
    write_jsonl(dataset_dir / "residual_repair_preserve_rows.jsonl", residual_rows)
    write_jsonl(dataset_dir / "hard77_rows.jsonl", hard77_rows)
    write_jsonl(dataset_dir / "clean_self_anchor_rows.jsonl", self_anchors)
    write_jsonl(dataset_dir / "self_anchor_rejected_by_grounding_filter.jsonl", rejected_anchors)

    manifest = {
        "created_by": "scripts/build_q5_residual218_q8_shape_distill_v1_dataset.py",
        "dataset_path": str(dataset_dir / "train.jsonl"),
        "start_adapter_intended": (
            "hf_bucket/experiments/sft/"
            "q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7/checkpoint-218"
        ),
        "rows": len(final),
        "unique_q8_shape_rows": len(q8_clean_unique),
        "q8_shape_rows_rejected_by_dataset_filter": len(q8_rejected_by_dataset_filter),
        "q8_shape_rows_repeated": len(q8_shape_rows),
        "q8_shape_repeat": args.q8_shape_repeat,
        "residual_repair_rows": len(residual_rows),
        "hard77_rows": len(hard77_rows),
        "clean_self_anchor_rows": len(self_anchors),
        "self_anchor_rejected_by_grounding_filter": len(rejected_anchors),
        "digit_label_rows": 0,
        "duplicate_paper_ids": 0,
        "row_type_counts": dict(collections.Counter(row["weighted_numeric_patch_row_type"] for row in final)),
        "component_counts": dict(collections.Counter(row["dataset_component"] for row in final)),
        "q8_shape_failure_family_counts": dict(
            collections.Counter(row.get("failure_family", "") for row in q8_clean_unique)
        ),
        "q8_shape_dataset_filter_reject_counts": dict(
            collections.Counter(
                reason
                for row in q8_rejected_by_dataset_filter
                for reason in row.get("dataset_filter_reject_reasons", [])
            )
        ),
        "weight_policy": {
            "q8_shape": args.q8_shape_weight,
            "q8_shape_repeat": args.q8_shape_repeat,
            "residual_repairs": args.residual_repair_weight,
            "hard77": args.hard77_weight,
            "hard77_stress": args.hard77_stress_weight,
            "clean_self_anchors": args.self_anchor_weight,
        },
        "policy_notes": [
            "Q8-shape rows are teacher-authored and Qwen-verified.",
            "Q8 outputs are clue sources, not copied labels.",
            "Self-anchors are filtered for semantic-fog terms and kept at very low weight.",
            "No prompt/source loss is used; rows rely on full assistant-token labels.",
        ],
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[q8-shape-dataset] wrote {len(final)} rows to {dataset_dir / 'train.jsonl'}")
    print(f"[q8-shape-dataset] unique q8-shape rows={len(q8_clean_unique)} repeated={len(q8_shape_rows)}")


if __name__ == "__main__":
    main()
