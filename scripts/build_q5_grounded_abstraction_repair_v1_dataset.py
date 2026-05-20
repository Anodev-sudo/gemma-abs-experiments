#!/usr/bin/env python3
"""Build a tiny checkpoint-218 grounded-abstraction continuation dataset."""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin150_residual218_grounded_abstraction_v1_lr1e7")
REPAIR_SRCS = [
    Path(
        "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin150_residual218_grounded_abstraction_v1/"
        "kimi_qwen_repair_v1/materialized/training_rows.jsonl"
    ),
    Path(
        "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin150_residual218_grounded_abstraction_v1/"
        "kimi_qwen_repair_retry_v1/materialized/training_rows.jsonl"
    ),
]
HARD77_SRC = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/"
    "dataset_hard77_repeats20_v1/eval_hard77.jsonl"
)
Q8_REVIEW_SRC = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7/"
    "eval/blind_ab_q8_mlx_python_vs_residual218_100/blind_review_merged_unblinded.jsonl"
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
BANNED_OPENINGS = (
    "this paper",
    "the paper",
    "the authors",
    "the work",
    "we propose",
    "in this work",
    "in this paper",
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
    return " ".join(str(text or "").strip().split())


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


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


def hard_label_reject_reasons(summary: str, title: str) -> list[str]:
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
    if EXACT_QUANTITY_WORD_RE.search(text):
        reasons.append("exact_quantity_word")
    if BENCHMARK_RE.search(text):
        reasons.append("benchmark_or_superlative")
    if any(low.startswith(prefix) for prefix in BANNED_OPENINGS):
        reasons.append("banned_opening")
    if title_overlap_ratio(text, title) >= 0.8:
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


def full_target_weight(weight: float, policy: str) -> dict[str, Any]:
    if weight == 1.0:
        return {"enabled": False, "base_weight": 1.0, "spans": [], "policy": policy}
    return {"enabled": True, "base_weight": weight, "spans": [], "policy": policy}


def normalize_train_row(row: dict[str, Any], row_type: str, weight: float) -> dict[str, Any]:
    out = dict(row)
    out["summary"] = norm(out["summary"])
    out["weighted_numeric_patch_row_type"] = row_type
    out["loss_weight_hints"] = full_target_weight(weight, f"grounded_abstraction_v1 full assistant target weight {weight}")
    out["messages"] = messages(out.get("title", ""), out.get("abstract", ""), out["summary"])
    return out


def load_repair_rows(paths: list[Path], repeat: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in load_jsonl(path):
            pid = str(row["paper_id"])
            if pid in seen:
                continue
            seen.add(pid)
            reasons = hard_label_reject_reasons(row["summary"], row.get("title", ""))
            if reasons:
                raise SystemExit(f"Accepted repair failed hard label filter: {pid} {reasons}")
            for replica in range(repeat):
                train_row = dict(row)
                train_row["source_paper_id"] = pid
                train_row["repair_repeat_index"] = replica
                if repeat > 1:
                    train_row["paper_id"] = f"{pid}::grounded_repair_repeat_{replica:02d}"
                normalized = normalize_train_row(train_row, "grounded_abstraction_repair", 2.0)
                normalized["dataset_component"] = "grounded_abstraction_repair"
                rows.append(normalized)
    return rows


def hard77_row(row: dict[str, Any]) -> dict[str, Any]:
    pid = str(row["paper_id"])
    out = {
        "paper_id": pid,
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "summary": row.get("target_summary", ""),
        "primary_category": row.get("primary_category", ""),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date", ""),
        "teacher_model": row.get("teacher_model", "moonshotai/kimi-k2.6"),
        "generator_prompt_version": row.get("generator_prompt_version", ""),
        "review_prompt_version": row.get("review_prompt_version", ""),
        "hard77_source_eval_id": row.get("eval_id", ""),
    }
    reasons = [
        reason
        for reason in hard_label_reject_reasons(out["summary"], out["title"])
        if reason != "exact_quantity_word"
    ]
    if reasons:
        raise SystemExit(f"Hard77 row failed hard label filter: {pid} {reasons}")
    weight = 1.5 if pid in STRESS_IDS else 1.0
    normalized = normalize_train_row(out, "hard77_preserve_stress" if pid in STRESS_IDS else "hard77_preserve", weight)
    normalized["dataset_component"] = "hard77"
    return normalized


def q8_anchor_rows(path: Path, exclude_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    anchors: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in load_jsonl(path):
        if str(row.get("preferred_model", "")).startswith("candidate_") is False:
            continue
        if row.get("more_accurate_model") not in {"candidate_residual_checkpoint_218", "tie"}:
            continue
        pid = str(row["paper_id"])
        if pid in exclude_ids:
            continue
        summary = row["summary_a"] if str(row.get("summary_a_model", "")).startswith("candidate_") else row["summary_b"]
        out = {
            "paper_id": pid,
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "summary": summary,
            "primary_category": row.get("primary_category", ""),
            "categories": [],
            "created_date": "",
            "teacher_model": "checkpoint_218_self_anchor",
            "generator_prompt_version": "q5_grounded_abstraction_anchor_from_blind_review_v1",
            "review_prompt_version": "blind_q8_vs_checkpoint218_candidate_win",
            "anchor_source_split": row.get("split", ""),
            "anchor_review_id": row.get("review_id", ""),
            "anchor_review_notes": row.get("review_notes", ""),
        }
        reasons = hard_label_reject_reasons(summary, out["title"])
        if reasons:
            rejected.append({**out, "anchor_reject_reasons": reasons})
            continue
        normalized = normalize_train_row(out, "checkpoint218_q8_win_self_anchor", 0.25)
        normalized["dataset_component"] = "q8_win_self_anchor"
        anchors.append(normalized)
    return anchors, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build q5 grounded abstraction repair SFT dataset.")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--repair-src", type=Path, action="append", default=None)
    parser.add_argument("--hard77-src", type=Path, default=HARD77_SRC)
    parser.add_argument("--q8-review-src", type=Path, default=Q8_REVIEW_SRC)
    parser.add_argument("--repair-repeat", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repair_srcs = args.repair_src or REPAIR_SRCS
    if args.repair_repeat < 1:
        raise SystemExit("--repair-repeat must be >= 1")
    repairs = load_repair_rows(repair_srcs, args.repair_repeat)
    repair_source_ids = {str(row.get("source_paper_id") or row["paper_id"]).split("::", 1)[0] for row in repairs}
    hard77 = [hard77_row(row) for row in load_jsonl(args.hard77_src) if str(row["paper_id"]) not in repair_source_ids]
    exclude_ids = repair_source_ids | {row["paper_id"] for row in hard77}
    anchors, rejected_anchors = q8_anchor_rows(args.q8_review_src, exclude_ids)

    final = [*repairs, *hard77, *anchors]
    duplicate_ids = [pid for pid, count in collections.Counter(row["paper_id"] for row in final).items() if count > 1]
    if duplicate_ids:
        raise SystemExit(f"Duplicate paper IDs in final dataset: {duplicate_ids[:20]}")
    digit_rows = [row["paper_id"] for row in final if DIGIT_RE.search(row["summary"])]
    if digit_rows:
        raise SystemExit(f"Digit labels in final dataset: {digit_rows[:20]}")

    unique_repairs = len(repair_source_ids)
    repair_slug = f"{unique_repairs}grounded" if args.repair_repeat == 1 else f"{unique_repairs}groundedx{args.repair_repeat}"
    out = args.output_root / f"dataset_{repair_slug}_{len(anchors)}anchor_{len(hard77)}hard_v1"
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", final)
    write_jsonl(out / f"grounded_repair_rows_{len(repairs)}.jsonl", repairs)
    write_jsonl(out / f"q8_win_self_anchor_rows_{len(anchors)}.jsonl", anchors)
    write_jsonl(out / f"hard77_rows_{len(hard77)}.jsonl", hard77)
    write_jsonl(out / "q8_win_self_anchor_rejected_by_hard_filter.jsonl", rejected_anchors)
    manifest = {
        "created_by": "scripts/build_q5_grounded_abstraction_repair_v1_dataset.py",
        "dataset_path": str(out / "train.jsonl"),
        "rows": len(final),
        "grounded_repair_rows": len(repairs),
        "grounded_repair_unique_papers": unique_repairs,
        "grounded_repair_repeat": args.repair_repeat,
        "q8_win_self_anchor_rows": len(anchors),
        "hard77_rows": len(hard77),
        "hard77_rows_replaced_by_grounded_repairs": 77 - len(hard77),
        "digit_label_rows": 0,
        "duplicate_paper_ids": 0,
        "row_type_counts": dict(collections.Counter(row["weighted_numeric_patch_row_type"] for row in final)),
        "component_counts": dict(collections.Counter(row["dataset_component"] for row in final)),
        "repair_failure_family_counts": dict(collections.Counter(row.get("failure_family", "") for row in repairs)),
        "anchor_source_split_counts": dict(collections.Counter(row.get("anchor_source_split", "") for row in anchors)),
        "self_anchor_rejected_by_hard_filter": len(rejected_anchors),
        "weight_policy": {
            "grounded_repairs": 2.0,
            "q8_win_self_anchors": 0.25,
            "hard77": 1.0,
            "hard77_stress": 1.5,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[grounded-abstraction-dataset] wrote {len(final)} rows to {out / 'train.jsonl'}")


if __name__ == "__main__":
    main()
