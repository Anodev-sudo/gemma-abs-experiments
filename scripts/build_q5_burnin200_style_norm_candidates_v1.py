#!/usr/bin/env python3
"""Build candidate rows for q5 hard77 burn-in checkpoint-200 style normalization."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin200_style_norm_v1_lr5e7")
BURNIN_ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5")
COMBINED_INPUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_hard_attractor_booster_v1_lr1e6/eval/heuristic_inputs/combined_454.jsonl")
HARD77_INPUT = BURNIN_ROOT / "dataset_hard77_repeats20_v1/eval_hard77.jsonl"
MANUAL_PAIRED = BURNIN_ROOT / "eval/manual_review_100/paired_review_100.jsonl"
BASELINE_BAD = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_hard_attractor_booster_v1_lr1e6/eval/baseline_ckpt700_bad_summaries_pool350.jsonl")

CATEGORIES = (
    "generic_title_drift",
    "value_attachment_or_polarity",
    "awkward_malformed_technical",
    "result_preserving_no_digit_numeric",
)
STRESS_IDS = {"1901.06548", "2111.10635", "2303.09722", "2406.04373"}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")
DIGIT_RE = re.compile(r"\d")
NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?(?:\s?[%x×]|e[-+]?\d+)?|\b\d+[A-Za-z]+|\b[A-Za-z]+\d+[A-Za-z0-9-]*")
GENERIC_RE = re.compile(r"\b(method|model|system|approach|framework|data|task|various|different|multiple|supports|enables|improves|analysis|features|parameters|classification|prediction)\b", re.I)
AWKWARD_RE = re.compile(
    r"\b("
    r"hybridova|hybridistic|sphere-space|smoothful|disregularizing|poster-based|frombroader|"
    r"noyehot|blackbody gravitational|fuzzy collider|collider parametrizations|"
    r"structure structure|local local|wind-stream|minimum model fidelity|restricted airspace|soft learning"
    r")\b",
    re.I,
)
BENCHMARK_RE = re.compile(r"\b(outperform|outperforms|state-of-the-art|best|benchmark|superior|highest|largest|significant)\b", re.I)
POLARITY_RE = re.compile(r"\b(enhance|enhances|reduce|reduces|cut|cuts|increase|increases|decrease|decreases|prevent|prevents|rule out|rules out|support|supports|against|not|no)\b", re.I)
ROLE_RE = re.compile(r"\b(accuracy|cost|speed|efficiency|throughput|latency|energy|power|mass|distance|rate|bound|limit|error|quality|coverage|emission|signal|resource|carbon|forecast)\b", re.I)

KNOWN_CATEGORY_HINTS: dict[str, set[str]] = {
    "generic_title_drift": {
        "2202.11323", "2210.05440", "2304.10778", "2311.09464", "2312.13485",
        "2307.06293", "2411.00902", "2508.14890", "2601.16680", "2305.11935",
        "2305.12830", "2404.01457", "2208.14276", "2310.07469", "2309.10750",
        "2403.18968", "2501.05739", "2506.20088",
    },
    "value_attachment_or_polarity": {
        "2201.12981", "2210.05440", "2307.02365", "2309.17223", "2310.10100",
        "2501.09812", "2601.16670", "2306.10876", "astro-ph/0607643", "2305.12830",
        "2302.08162", "2311.12340", "2309.16761", "astro-ph/9907384", "2309.11603",
        "2505.03183", "2408.16646",
    },
    "awkward_malformed_technical": {
        "2004.08807", "2307.13873", "alg-geom/9506022", "alg-geom/9305001",
        "2509.11474", "2302.02028", "2306.10876", "2302.08162", "2208.06302",
        "0804.3390", "2404.06312", "2411.13315", "2309.16761",
    },
    "result_preserving_no_digit_numeric": {
        "2403.02693", "2407.20852", "2411.13315", "2411.17627", "2501.05739",
        "2501.09812", "2506.08863", "2405.01210", "2408.16646", "2307.06293",
        "2601.16670", "0704.2253", "2307.02365", "2503.17116", "2405.10684",
    },
}


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


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def word_count(text: str) -> int:
    return len(words(text))


def output_by_pid(path: Path) -> dict[str, str]:
    return {
        str(row["paper_id"]): norm(row.get("normalized_output") or row.get("raw_output") or row.get("summary"))
        for row in load_jsonl(path)
        if row.get("paper_id")
    }


def baseline_bad_by_pid(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        str(row["paper_id"]): norm(row.get("summary", ""))
        for row in load_jsonl(path)
        if row.get("paper_id") and row.get("summary")
    }


def title_overlap(summary: str, title: str) -> int:
    toks = lambda s: {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z-]{2,}", s)}
    return len(toks(summary) & toks(title))


def repeated_word(text: str) -> bool:
    prev = ""
    for token in (w.lower() for w in words(text)):
        if token == prev and len(token) > 2:
            return True
        prev = token
    return False


def source_terms(row: dict[str, Any]) -> set[str]:
    return {w.lower() for w in words(f"{row.get('title','')} {row.get('abstract','')}") if len(w) >= 4}


def candidate_record(
    *,
    row: dict[str, Any],
    category: str,
    score: int,
    reasons: list[str],
    baseline_summary: str,
    burnin_summary: str,
    source_kind: str,
    rank: int,
    candidate_prefix: str,
) -> dict[str, Any]:
    pid = str(row["paper_id"])
    reason_text = (
        f"Normalize checkpoint-200 output for category {category}. "
        f"Reasons: {', '.join(reasons[:6])}. Preserve source specificity while keeping no digits."
    )
    tags = [category, *reasons[:8]]
    return {
        "candidate_id": f"{candidate_prefix}:{category}:{rank:04d}",
        "paper_id": pid,
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "primary_category": row.get("primary_category", ""),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date", ""),
        "bad_summary": burnin_summary,
        "pre_dpo_summary": baseline_summary,
        "triage_decision": "style_normalization",
        "triage_severity": "high" if score >= 250 else "medium",
        "failure_tags": tags,
        "short_reason": reason_text,
        "heuristic_numeric_tags": [],
        "heuristic_semantic_tags": reasons,
        "source_split": "q5_burnin200_style_norm_v1",
        "normalization_category": category,
        "style_norm_candidate_score": score,
        "style_norm_source_kind": source_kind,
        "current_burnin_summary": burnin_summary,
        "checkpoint700_summary": baseline_summary,
    }


def category_scores(row: dict[str, Any], baseline: str, burnin: str, source_kind: str) -> dict[str, tuple[int, list[str]]]:
    pid = str(row["paper_id"])
    title = row.get("title", "")
    split = str(row.get("split", ""))
    wc = word_count(burnin)
    base_wc = word_count(baseline)
    overlap = title_overlap(burnin, title)
    src = source_terms(row)
    burn_words = {w.lower() for w in words(burnin)}
    grounded_nonstop = {w for w in burn_words & src if len(w) >= 5}
    base_has_digit = bool(DIGIT_RE.search(baseline))
    burn_has_digit = bool(DIGIT_RE.search(burnin))
    base_num_count = len(NUM_RE.findall(baseline))
    burn_num_count = len(NUM_RE.findall(burnin))
    scores: dict[str, tuple[int, list[str]]] = {}

    def add(category: str, points: int, reason: str) -> None:
        score, reasons = scores.get(category, (0, []))
        scores[category] = (score + points, [*reasons, reason])

    for category, pids in KNOWN_CATEGORY_HINTS.items():
        if pid in pids:
            add(category, 180, "manual_review_flag")

    if pid in STRESS_IDS:
        add("value_attachment_or_polarity", 90, "remembered_stress_preservation")
        add("result_preserving_no_digit_numeric", 60, "remembered_stress_numeric_role")

    if split == "fresh_recent_normal_100":
        add("generic_title_drift", 35, "fresh_normal_style_drift_risk")
    if split == "numeric_ckpt700_risk_pool350" or source_kind == "manual_numeric":
        add("result_preserving_no_digit_numeric", 70, "numeric_risk_row")

    if base_has_digit and not burn_has_digit:
        add("result_preserving_no_digit_numeric", 90, "digits_removed_from_baseline")
    if burn_has_digit:
        add("result_preserving_no_digit_numeric", 140, "digit_residue_remains")
    if base_num_count >= 2 and burn_num_count < base_num_count:
        add("result_preserving_no_digit_numeric", 45, "dense_metric_reduced")
    if ROLE_RE.search(baseline) and not ROLE_RE.search(burnin):
        add("result_preserving_no_digit_numeric", 50, "metric_role_lost")

    if wc < 10:
        add("awkward_malformed_technical", 90, "under_ten_words")
        add("generic_title_drift", 45, "underinformative")
    if wc > 15:
        add("generic_title_drift", 35, "overlong")
    if overlap >= 4 or overlap >= title_overlap(baseline, title) + 2:
        add("generic_title_drift", 75, "title_overlap_risk")
    if GENERIC_RE.search(burnin) and len(grounded_nonstop) <= 4:
        add("generic_title_drift", 75, "generic_low_specificity")
    if len(grounded_nonstop) <= 2:
        add("generic_title_drift", 45, "few_source_anchors")

    if AWKWARD_RE.search(burnin):
        add("awkward_malformed_technical", 170, "known_awkward_phrase")
    if repeated_word(burnin):
        add("awkward_malformed_technical", 110, "repeated_word")
    if "  " in burnin or not burnin.endswith("."):
        add("awkward_malformed_technical", 35, "surface_form_issue")
    if wc < 10 or any(len(w) > 18 for w in words(burnin)):
        add("awkward_malformed_technical", 35, "brittle_token_shape")

    if POLARITY_RE.search(baseline) and not POLARITY_RE.search(burnin):
        add("value_attachment_or_polarity", 60, "polarity_marker_lost")
    if BENCHMARK_RE.search(baseline) or BENCHMARK_RE.search(burnin):
        add("value_attachment_or_polarity", 60, "benchmark_or_superlative_risk")
    if len(grounded_nonstop) <= 3 and split == "fresh_recent_normal_100":
        add("value_attachment_or_polarity", 35, "wrong_object_risk")

    return {cat: value for cat, value in scores.items() if value[0] > 0}


def collect_sources(args: argparse.Namespace) -> list[dict[str, Any]]:
    combined = load_jsonl(args.combined_input)
    hard77 = load_jsonl(args.hard77_input)
    manual = load_jsonl(args.manual_paired)
    baseline_bad = baseline_bad_by_pid(args.baseline_bad)
    baseline_combined = output_by_pid(args.baseline_combined_output) if args.baseline_combined_output else {}
    burnin_combined = output_by_pid(args.burnin_combined_output)
    hard77_burnin = output_by_pid(args.hard77_burnin_output) if args.hard77_burnin_output else {}

    sources: dict[tuple[str, str], dict[str, Any]] = {}
    for row in combined:
        pid = str(row["paper_id"])
        burnin = burnin_combined.get(pid, "")
        if not burnin:
            continue
        baseline = baseline_combined.get(pid) or baseline_bad.get(pid) or row.get("summary", "")
        source_kind = "combined_numeric" if row.get("split") == "numeric_ckpt700_risk_pool350" else "combined_fresh"
        sources[(pid, source_kind)] = {**row, "_baseline_summary": norm(baseline), "_burnin_summary": burnin, "_source_kind": source_kind}

    for row in hard77:
        pid = str(row["paper_id"])
        burnin = hard77_burnin.get(pid, row.get("target_summary", ""))
        baseline = row.get("bad_summary_current_q5") or row.get("target_summary", "")
        sources[(pid, "hard77")] = {**row, "split": "hard77_burnin", "_baseline_summary": norm(baseline), "_burnin_summary": burnin, "_source_kind": "hard77"}

    for row in manual:
        pid = str(row["paper_id"])
        kind = f"manual_{row.get('source_bucket', 'review')}"
        sources[(pid, kind)] = {
            **row,
            "_baseline_summary": norm(row.get("baseline_output", "")),
            "_burnin_summary": norm(row.get("burnin_output", "")),
            "_source_kind": kind,
        }
    return list(sources.values())


def build_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    scored: dict[str, list[tuple[int, dict[str, Any], list[str], str, str, str]]] = defaultdict(list)
    sources = collect_sources(args)
    excluded_pairs = load_excluded_pairs(args.exclude_reviewer_outputs)
    for row in sources:
        baseline = row["_baseline_summary"]
        burnin = row["_burnin_summary"]
        for category, (score, reasons) in category_scores(row, baseline, burnin, row["_source_kind"]).items():
            if (str(row["paper_id"]), category) in excluded_pairs:
                continue
            scored[category].append((score, row, reasons, baseline, burnin, row["_source_kind"]))

    candidates: list[dict[str, Any]] = []
    selected_categories = tuple(args.categories or CATEGORIES)
    for category in selected_categories:
        ranked = sorted(
            scored.get(category, []),
            key=lambda item: (-item[0], item[1]["paper_id"], item[5]),
        )
        if len(ranked) < args.per_category_candidates and args.allow_fallback_fill:
            seen = {(item[1]["paper_id"], item[5]) for item in ranked}
            for row in sorted(sources, key=lambda r: (str(r["paper_id"]), str(r["_source_kind"]))):
                if len(ranked) >= args.per_category_candidates:
                    break
                if (str(row["paper_id"]), category) in excluded_pairs:
                    continue
                key = (row["paper_id"], row["_source_kind"])
                if key in seen:
                    continue
                seen.add(key)
                ranked.append(
                    (
                        1,
                        row,
                        [f"{category}_coverage_topup"],
                        row["_baseline_summary"],
                        row["_burnin_summary"],
                        row["_source_kind"],
                    )
                )
        selected = ranked[: args.per_category_candidates]
        if len(selected) < args.min_per_category:
            raise SystemExit(f"Category {category} has only {len(selected)} candidates before labeling")
        for idx, (score, row, reasons, baseline, burnin, source_kind) in enumerate(selected, start=1):
            candidates.append(
                candidate_record(
                    row=row,
                    category=category,
                    score=score,
                    reasons=reasons,
                    baseline_summary=baseline,
                    burnin_summary=burnin,
                    source_kind=source_kind,
                    rank=idx,
                    candidate_prefix=args.candidate_prefix,
                )
            )
    return candidates


def load_excluded_pairs(paths: list[Path] | None) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for path in paths or []:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            pid = str(row.get("paper_id", ""))
            category = str(row.get("normalization_category", ""))
            if pid and category:
                pairs.add((pid, category))
    return pairs


def write_markdown(path: Path, candidates: list[dict[str, Any]]) -> None:
    lines = ["# q5 burn-in checkpoint-200 style-normalization candidates", ""]
    for category in CATEGORIES:
        rows = [r for r in candidates if r["normalization_category"] == category]
        lines.extend(["", f"## {category}", ""])
        for row in rows[:20]:
            lines.append(f"- `{row['paper_id']}` score={row['style_norm_candidate_score']} source={row['style_norm_source_kind']}")
            lines.append(f"  - title: {row['title']}")
            lines.append(f"  - checkpoint-200: {row['bad_summary']}")
            lines.append(f"  - reasons: {', '.join(row['heuristic_semantic_tags'])}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build q5 burn-in checkpoint-200 style-normalization candidates.")
    ap.add_argument("--output-root", type=Path, default=ROOT)
    ap.add_argument("--combined-input", type=Path, default=COMBINED_INPUT)
    ap.add_argument("--hard77-input", type=Path, default=HARD77_INPUT)
    ap.add_argument("--manual-paired", type=Path, default=MANUAL_PAIRED)
    ap.add_argument("--baseline-bad", type=Path, default=BASELINE_BAD)
    ap.add_argument("--baseline-combined-output", type=Path, default=None)
    ap.add_argument("--burnin-combined-output", type=Path, required=True)
    ap.add_argument("--hard77-burnin-output", type=Path, default=None)
    ap.add_argument("--per-category-candidates", type=int, default=100)
    ap.add_argument("--min-per-category", type=int, default=40)
    ap.add_argument("--categories", nargs="+", choices=CATEGORIES, default=None)
    ap.add_argument("--candidate-prefix", default="style_norm_v1")
    ap.add_argument("--allow-fallback-fill", action="store_true")
    ap.add_argument("--exclude-reviewer-outputs", type=Path, nargs="*", default=[])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    candidates = build_candidates(args)
    out = args.output_root / "candidate_pool"
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "candidates.jsonl", candidates)
    for category in CATEGORIES:
        write_jsonl(out / f"candidates_{category}.jsonl", [r for r in candidates if r["normalization_category"] == category])
    write_markdown(out / "CANDIDATE_POOL.md", candidates)
    manifest = {
        "created_by": "scripts/build_q5_burnin200_style_norm_candidates_v1.py",
        "rows": len(candidates),
        "categories": tuple(args.categories or CATEGORIES),
        "category_counts": dict(Counter(r["normalization_category"] for r in candidates)),
        "per_category_candidates": args.per_category_candidates,
        "candidate_prefix": args.candidate_prefix,
        "allow_fallback_fill": bool(args.allow_fallback_fill),
        "inputs": {
            "combined_input": str(args.combined_input),
            "hard77_input": str(args.hard77_input),
            "manual_paired": str(args.manual_paired),
            "baseline_combined_output": str(args.baseline_combined_output) if args.baseline_combined_output else "",
            "burnin_combined_output": str(args.burnin_combined_output),
            "hard77_burnin_output": str(args.hard77_burnin_output) if args.hard77_burnin_output else "",
        },
        "source_kind_counts": dict(Counter(r["style_norm_source_kind"] for r in candidates)),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[style-norm-candidates] wrote {len(candidates)} candidates to {out / 'candidates.jsonl'}")


if __name__ == "__main__":
    main()
