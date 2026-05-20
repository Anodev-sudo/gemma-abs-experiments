#!/usr/bin/env python3
"""Build a residual-repair SFT set from burn-in repairs and self-anchors."""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path
from typing import Any


ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin150_residual_repair_v1_lr2e7")
REPAIRS_SRC = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/"
    "eval/burnin150_style_probe_5000_v1/residual_repair_kimi_qwen_v1/materialized/training_rows.jsonl"
)
SELF_ANCHOR_SRC = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/"
    "eval/burnin150_style_probe_5000_v1/heuristic_screen_checkpoint150/self_anchor_candidates.jsonl"
)
REVIEWED_HITS_SRC = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/"
    "eval/burnin150_style_probe_5000_v1/heuristic_hit_review_v1/binary_review_merged.jsonl"
)
HARD77_SRC = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/"
    "dataset_hard77_repeats20_v1/eval_hard77.jsonl"
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
UNIT_RE = re.compile(
    r"\b(?:percent|percentage|nm|mm|cm|km|kg|hz|khz|mhz|ghz|ev|kev|mev|gev|"
    r"kelvin|tesla|joule|watt|volt|ampere|meter|metre|meters|metres|"
    r"second|seconds|minute|minutes|hour|hours)\b",
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
    if UNIT_RE.search(text):
        reasons.append("unit_or_percentage_word")
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


def normalize_train_row(row: dict[str, Any], row_type: str, weight: float, summary_key: str = "summary") -> dict[str, Any]:
    summary = norm(row.get(summary_key, ""))
    out = dict(row)
    out["summary"] = summary
    out["weighted_numeric_patch_row_type"] = row_type
    out["loss_weight_hints"] = full_target_weight(weight, f"residual_repair_v1 full assistant target weight {weight}")
    out["messages"] = messages(out.get("title", ""), out.get("abstract", ""), summary)
    return out


def repair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        reasons = hard_label_reject_reasons(row.get("summary", ""), row.get("title", ""))
        if reasons:
            raise SystemExit(f"Accepted repair failed hard label filter: {row.get('paper_id')} {reasons}")
        normalized = normalize_train_row(row, "residual_repair_kimi_qwen", 2.0)
        normalized["dataset_component"] = "repair"
        normalized["source_split"] = "q5_burnin150_residual_repair_v1"
        out.append(normalized)
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
    reasons = [
        reason
        for reason in hard_label_reject_reasons(out["summary"], out["title"])
        if reason != "exact_quantity_word"
    ]
    if reasons:
        raise SystemExit(f"Hard77 row failed hard label filter: {pid} {reasons}")
    normalized = normalize_train_row(out, "hard77_preserve_stress" if pid in STRESS_IDS else "hard77_preserve", weight)
    normalized["dataset_component"] = "hard77"
    normalized["source_split"] = "q5_burnin150_residual_repair_v1"
    return normalized


def raw_anchor_rows(self_anchor_rows: list[dict[str, Any]], reviewed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    anchors: dict[str, dict[str, Any]] = {}
    for row in self_anchor_rows:
        pid = str(row.get("paper_id", ""))
        if pid:
            anchors[pid] = {
                "paper_id": pid,
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
                "summary": row.get("generated_summary", ""),
                "primary_category": row.get("primary_category", ""),
                "categories": row.get("categories", []),
                "created_date": row.get("created_date", ""),
                "anchor_origin": "heuristic_self_anchor",
                "anchor_source_split": row.get("split", ""),
                "anchor_eval_id": row.get("eval_id", ""),
            }
    for row in reviewed_rows:
        if row.get("actual_failure"):
            continue
        pid = str(row.get("paper_id", ""))
        if pid and pid not in anchors:
            anchors[pid] = {
                "paper_id": pid,
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
                "summary": row.get("generated_summary", ""),
                "primary_category": row.get("primary_category", ""),
                "categories": [],
                "created_date": "",
                "anchor_origin": "reviewed_false_positive_anchor",
                "anchor_source_split": row.get("split", ""),
                "anchor_review_id": row.get("review_id", ""),
                "anchor_review_notes": row.get("review_notes", ""),
            }
    return list(anchors.values())


def select_anchors(
    rows: list[dict[str, Any]],
    *,
    exclude_ids: set[str],
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        pid = str(row.get("paper_id", ""))
        if not pid or pid in seen or pid in exclude_ids:
            continue
        seen.add(pid)
        reasons = hard_label_reject_reasons(row.get("summary", ""), row.get("title", ""))
        if reasons:
            rejected.append({**row, "anchor_reject_reasons": reasons})
            continue
        clean.append(row)

    rng = random.Random(seed)
    by_split: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in clean:
        by_split[str(row.get("anchor_source_split", ""))].append(row)
    for bucket in by_split.values():
        rng.shuffle(bucket)

    split_targets = [
        ("release_source_normal", 160),
        ("release_source_risk", 150),
        ("release_source_risk_backfill", 140),
        ("numeric_ckpt700_risk_pool350", 90),
        ("fresh_recent_normal_100", 60),
    ]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for split, target in split_targets:
        bucket = by_split.get(split, [])
        for row in bucket[:target]:
            if row["paper_id"] in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row["paper_id"])
    if len(selected) < count:
        remaining = [row for row in clean if row["paper_id"] not in selected_ids]
        rng.shuffle(remaining)
        for row in remaining:
            selected.append(row)
            selected_ids.add(row["paper_id"])
            if len(selected) == count:
                break
    if len(selected) != count:
        raise SystemExit(f"Selected {len(selected)} anchors; expected {count}")

    normalized = []
    for row in selected:
        out = normalize_train_row(row, "burnin150_self_anchor_low_weight", 0.25)
        out["dataset_component"] = "self_anchor"
        out["source_split"] = "q5_burnin150_residual_repair_v1"
        normalized.append(out)
    return normalized, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build burn-in 150 residual-repair SFT dataset.")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--repairs-src", type=Path, default=REPAIRS_SRC)
    parser.add_argument("--self-anchor-src", type=Path, default=SELF_ANCHOR_SRC)
    parser.add_argument("--reviewed-hits-src", type=Path, default=REVIEWED_HITS_SRC)
    parser.add_argument("--hard77-src", type=Path, default=HARD77_SRC)
    parser.add_argument("--anchor-count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260516)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repairs_raw = load_jsonl(args.repairs_src)
    self_anchor_raw = load_jsonl(args.self_anchor_src)
    reviewed_raw = load_jsonl(args.reviewed_hits_src)
    hard77_raw = load_jsonl(args.hard77_src)
    if len(repairs_raw) != 193:
        raise SystemExit(f"Expected 193 accepted repairs, got {len(repairs_raw)}")
    if len(hard77_raw) != 77:
        raise SystemExit(f"Expected 77 hard77 rows, got {len(hard77_raw)}")

    repairs = repair_rows(repairs_raw)
    hard77 = [hard77_row(row) for row in hard77_raw]
    exclude_ids = {row["paper_id"] for row in repairs} | {row["paper_id"] for row in hard77}
    anchor_raw = raw_anchor_rows(self_anchor_raw, reviewed_raw)
    anchors, rejected_anchors = select_anchors(
        anchor_raw,
        exclude_ids=exclude_ids,
        count=args.anchor_count,
        seed=args.seed,
    )

    final = [*repairs, *hard77, *anchors]
    expected = len(repairs) + len(hard77) + args.anchor_count
    if len(final) != expected:
        raise SystemExit(f"Expected {expected} final rows, got {len(final)}")
    duplicate_ids = [pid for pid, count in collections.Counter(row["paper_id"] for row in final).items() if count > 1]
    if duplicate_ids:
        raise SystemExit(f"Duplicate paper IDs in final dataset: {duplicate_ids[:20]}")
    digit_rows = [row["paper_id"] for row in final if DIGIT_RE.search(row["summary"])]
    if digit_rows:
        raise SystemExit(f"Digit labels in final dataset: {digit_rows[:20]}")

    out = args.output_root / f"dataset_{len(repairs)}repair_{args.anchor_count}selfanchor_77hard_v1"
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", final)
    write_jsonl(out / "repair_rows_193.jsonl", repairs)
    write_jsonl(out / f"self_anchor_rows_{args.anchor_count}.jsonl", anchors)
    write_jsonl(out / "hard77_rows_77.jsonl", hard77)
    write_jsonl(out / "self_anchor_rejected_by_hard_filter.jsonl", rejected_anchors)
    manifest = {
        "created_by": "scripts/build_q5_burnin150_residual_repair_v1_dataset.py",
        "dataset_path": str(out / "train.jsonl"),
        "rows": len(final),
        "repair_rows": len(repairs),
        "self_anchor_rows": len(anchors),
        "hard77_rows": len(hard77),
        "digit_label_rows": 0,
        "duplicate_paper_ids": 0,
        "row_type_counts": dict(collections.Counter(row["weighted_numeric_patch_row_type"] for row in final)),
        "component_counts": dict(collections.Counter(row["dataset_component"] for row in final)),
        "anchor_source_split_counts": dict(collections.Counter(row.get("anchor_source_split", "") for row in anchors)),
        "anchor_origin_counts": dict(collections.Counter(row.get("anchor_origin", "") for row in anchors)),
        "repair_failure_family_counts": dict(collections.Counter(row.get("failure_family", "") for row in repairs)),
        "hard77_stress_rows": sum(1 for row in hard77 if row["paper_id"] in STRESS_IDS),
        "self_anchor_rejected_by_hard_filter": len(rejected_anchors),
        "weight_policy": {
            "repairs": 2.0,
            "self_anchors": 0.25,
            "hard77": 1.25,
            "hard77_stress": 2.0,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[residual-repair-dataset] wrote {len(final)} rows to {out / 'train.jsonl'}")


if __name__ == "__main__":
    main()
