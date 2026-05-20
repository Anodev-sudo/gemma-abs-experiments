#!/usr/bin/env python3
"""Build a conservative style-replay dataset from old clean target rows."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin200_style_replay_v1_lr2e7")
STYLE_SRC = Path("hf_dataset/releases/recent_3y_balanced_10746_v1/training_rows.jsonl")
IDEAL_SRC = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_ideal200_regen_v1/clean_200_after_filters_v1/training_rows_200.jsonl")
HARD77_SRC = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/dataset_hard77_repeats20_v1/eval_hard77.jsonl")
COMBINED_EVAL = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_hard_attractor_booster_v1_lr1e6/eval/heuristic_inputs/combined_454.jsonl")
MANUAL_REVIEW_INPUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/eval/manual_review_100/review_input_100.jsonl")
STRESS_IDS = {"1901.06548", "2111.10635", "2303.09722", "2406.04373"}

DIGIT_RE = re.compile(r"\d")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")
FORMULA_RE = re.compile(r"[$<>≤≥≈]|\\[A-Za-z]+|\bO\(|\^|_|𝔽|[α-ωΑ-Ω]")
BANNED_OPENINGS = ("this paper", "the paper", "the authors", "we propose", "in this work", "in this paper")

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
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def norm(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def title_overlap_ratio(summary: str, title: str) -> float:
    tokens = lambda value: {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z-]{2,}", value)}
    summary_tokens = tokens(summary)
    title_tokens = tokens(title)
    if not summary_tokens:
        return 0.0
    return len(summary_tokens & title_tokens) / len(summary_tokens)


def sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(re.findall(r"[.!?]+(?:['\")\]]+)?(?=\s|$)", stripped))


def repeated_token(text: str) -> bool:
    previous = ""
    for token in (w.lower() for w in WORD_RE.findall(text)):
        if token == previous and len(token) > 2:
            return True
        previous = token
    return False


def clean_label_reasons(row: dict[str, Any]) -> list[str]:
    summary = norm(row.get("summary", ""))
    low = summary.lower()
    reasons: list[str] = []
    wc = word_count(summary)
    if not summary:
        reasons.append("empty")
    if wc < 10:
        reasons.append("under_10_words")
    if wc > 15:
        reasons.append("over_15_words")
    if sentence_count(summary) != 1:
        reasons.append("not_single_sentence")
    if DIGIT_RE.search(summary):
        reasons.append("digit")
    if FORMULA_RE.search(summary):
        reasons.append("formula_or_symbol")
    if any(low.startswith(prefix) for prefix in BANNED_OPENINGS):
        reasons.append("banned_opening")
    if title_overlap_ratio(summary, row.get("title", "")) >= 0.75:
        reasons.append("high_title_overlap")
    if repeated_token(summary):
        reasons.append("repeated_token")
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


def normalize_row(row: dict[str, Any], row_type: str, weight: float) -> dict[str, Any]:
    summary = norm(row.get("summary", ""))
    out = dict(row)
    out["summary"] = summary
    out["weighted_numeric_patch_row_type"] = row_type
    out["loss_weight_hints"] = full_target_weight(weight, f"style_replay_v1 full assistant target weight {weight}")
    out["messages"] = messages(out.get("title", ""), out.get("abstract", ""), summary)
    out["source_split"] = "q5_burnin200_style_replay_v1"
    return out


def hard77_row(row: dict[str, Any]) -> dict[str, Any]:
    pid = str(row["paper_id"])
    weight = 2.0 if pid in STRESS_IDS else 1.25
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
    return normalize_row(out, "hard77_preserve_stress" if pid in STRESS_IDS else "hard77_preserve", weight)


def excluded_paper_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        ids.update(str(row["paper_id"]) for row in load_jsonl(path) if row.get("paper_id"))
    return ids


def select_style_rows(rows: list[dict[str, Any]], exclude_ids: set[str], count: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        pid = str(row.get("paper_id", ""))
        if not pid or pid in exclude_ids or pid in seen:
            continue
        seen.add(pid)
        reasons = clean_label_reasons(row)
        if reasons:
            rejected.append({**row, "style_replay_reject_reasons": reasons})
            continue
        clean.append(row)

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean:
        by_category[str(row.get("primary_category", ""))].append(row)
    rng = random.Random(seed)
    for rows_for_category in by_category.values():
        rng.shuffle(rows_for_category)

    selected: list[dict[str, Any]] = []
    category_cycle = sorted(by_category)
    while len(selected) < count and category_cycle:
        next_cycle: list[str] = []
        for category in category_cycle:
            bucket = by_category[category]
            if not bucket:
                continue
            selected.append(bucket.pop())
            if bucket:
                next_cycle.append(category)
            if len(selected) == count:
                break
        category_cycle = next_cycle

    if len(selected) != count:
        raise SystemExit(f"Only selected {len(selected)} style rows; expected {count}")
    return selected, rejected


def assert_zero_digits(rows: list[dict[str, Any]], label: str) -> None:
    bad = [row["paper_id"] for row in rows if DIGIT_RE.search(row.get("summary", ""))]
    if bad:
        raise SystemExit(f"{label} has digit summaries: {bad[:20]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build q5 burn-in checkpoint-200 style replay rows.")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--style-src", type=Path, default=STYLE_SRC)
    parser.add_argument("--ideal-src", type=Path, default=IDEAL_SRC)
    parser.add_argument("--hard77-src", type=Path, default=HARD77_SRC)
    parser.add_argument("--combined-eval", type=Path, default=COMBINED_EVAL)
    parser.add_argument("--manual-review-input", type=Path, default=MANUAL_REVIEW_INPUT)
    parser.add_argument("--style-count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260515)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hard77_raw = load_jsonl(args.hard77_src)
    ideal_raw = load_jsonl(args.ideal_src)
    style_raw = load_jsonl(args.style_src)
    if len(hard77_raw) != 77:
        raise SystemExit(f"Expected 77 hard77 rows, got {len(hard77_raw)}")
    if len(ideal_raw) != 200:
        raise SystemExit(f"Expected 200 ideal numeric rows, got {len(ideal_raw)}")

    exclude_ids = excluded_paper_ids([args.hard77_src, args.combined_eval, args.manual_review_input])
    style_selected_raw, style_rejected = select_style_rows(style_raw, exclude_ids, args.style_count, args.seed)

    style = [normalize_row(row, "old_style_replay_clean", 1.0) for row in style_selected_raw]
    ideal = [normalize_row(row, "numeric_ideal200_anchor", 1.0) for row in ideal_raw]
    hard77 = [hard77_row(row) for row in hard77_raw]

    assert_zero_digits(style, "old style replay")
    assert_zero_digits(ideal, "ideal numeric")
    assert_zero_digits(hard77, "hard77")

    final = [*style, *ideal, *hard77]
    expected = args.style_count + 200 + 77
    if len(final) != expected:
        raise SystemExit(f"Expected {expected} final rows, got {len(final)}")

    out = args.output_root / f"dataset_{args.style_count}oldstyle_200ideal_77hard_v1"
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", final)
    write_jsonl(out / f"old_style_selected_{args.style_count}.jsonl", style)
    write_jsonl(out / "ideal_selected_200.jsonl", ideal)
    write_jsonl(out / "hard77_selected_77.jsonl", hard77)
    write_jsonl(out / "old_style_rejected.jsonl", style_rejected)
    manifest = {
        "created_by": "scripts/build_q5_burnin200_style_replay_v1_dataset.py",
        "rows": len(final),
        "dataset_path": str(out / "train.jsonl"),
        "style_source": str(args.style_src),
        "style_source_rows": len(style_raw),
        "style_selected_rows": len(style),
        "style_rejected_rows": len(style_rejected),
        "ideal_rows": len(ideal),
        "hard77_rows": len(hard77),
        "excluded_paper_ids": len(exclude_ids),
        "row_type_counts": dict(Counter(row["weighted_numeric_patch_row_type"] for row in final)),
        "style_primary_category_counts": dict(Counter(row.get("primary_category", "") for row in style)),
        "digit_rows_checked": {
            "style": 0,
            "ideal": 0,
            "hard77": 0,
        },
        "weight_policy": {
            "old_style_replay_clean": 1.0,
            "ideal": 1.0,
            "hard77": 1.25,
            "stress": 2.0,
        },
        "selection_policy": {
            "style_filters": [
                "10_to_15_words",
                "digit_free",
                "single_sentence",
                "no_formula_symbols",
                "no_banned_opening",
                "title_overlap_ratio_lt_0.75",
                "no_repeated_adjacent_token",
            ],
            "excluded_sources": [str(args.hard77_src), str(args.combined_eval), str(args.manual_review_input)],
            "category_balanced_round_robin": True,
            "seed": args.seed,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[style-replay-dataset] wrote {len(final)} rows to {out / 'train.jsonl'}")


if __name__ == "__main__":
    main()
