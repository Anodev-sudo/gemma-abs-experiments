#!/usr/bin/env python3
"""Analyze the repair-only overfit diagnostic trajectory.

The goal is not to grade a promotion candidate. This script turns the small
overfit run into a failure-shape map: which targets were learnable, which bad
attractors persisted, and where the label itself looks too weak or too brittle.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "hf_bucket/experiments/sft/"
    "q5_step2400_sft700_hard77_burnin150_residual218_grounded_abstraction_"
    "overfit14_diag_v1_lr1e5"
)
DEFAULT_EVAL_INPUT = DEFAULT_ROOT / "dataset_14repair_repeats100_fulltarget8_v1/eval_repair14.jsonl"
DEFAULT_SUMMARY = DEFAULT_ROOT / "eval/repair14_overfit_diag_summary.json"
DEFAULT_OUT_JSONL = DEFAULT_ROOT / "eval/repair14_failure_shape_diagnostic.jsonl"
DEFAULT_OUT_MD = DEFAULT_ROOT / "eval/REPAIR14_FAILURE_SHAPE_DIAGNOSTIC.md"

DIGIT_RE = re.compile(r"\d")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "into",
    "is",
    "it",
    "near",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "through",
    "to",
    "with",
}


MANUAL_ROW_DIAGNOSTICS: dict[str, dict[str, Any]] = {
    "1912.07751": {
        "bucket": "partial_grounding_success",
        "manual_verdict": "Moved off exact numeric/unit attractor, but retained low-energy modulation instead of frequency-search framing.",
        "likely_issue": "Target is supported, but the abstract's modulation/low-mass wording remains a strong competing attractor.",
        "next_label_guidance": "Keep cavity plus interferometric frequency search; avoid exact limits and avoid low-energy modulation wording.",
    },
    "2410.06812": {
        "bucket": "near_success_role_verb_drift",
        "manual_verdict": "Mostly learned the target; verb drift changes permits to finds but preserves the main result.",
        "likely_issue": "Small model substitutes a familiar result verb for a precise math relation.",
        "next_label_guidance": "Treat as low-risk; do not overweight unless exact math role matters.",
    },
    "2507.01963": {
        "bucket": "near_success_title_object_substitution",
        "manual_verdict": "Essentially repaired; tokenomics replaces analysis but the meaning is acceptable.",
        "likely_issue": "Title and abstract make tokenomics a stronger anchor than analysis.",
        "next_label_guidance": "Keep as a success-style exemplar if included again.",
    },
    "2405.10684": {
        "bucket": "title_name_digit_leakage",
        "manual_verdict": "Moves toward changing-look blazar, then reintroduces the exact source name with digits.",
        "likely_issue": "A digit-heavy source name in the title is a sticky identifier attractor even when target avoids it.",
        "next_label_guidance": "Use this as a no-name object-label case: changing-look MeV blazar, heightened jet activity, no PKS or catalog name.",
    },
    "2404.18022": {
        "bucket": "nearby_object_caveat_leakage",
        "manual_verdict": "Improves early, but later snaps back to neutron-star/remnant context not supported by the paper.",
        "likely_issue": "The abstract's nearby supernova-remnant context acts as a competing object, and positive labels alone are weak against it.",
        "next_label_guidance": "Needs family examples for nearby-but-not-associated objects; target should anchor CI Cam, oxygen shell, optical filaments.",
    },
    "2506.14395": {
        "bucket": "technical_method_drift",
        "manual_verdict": "Fixes pulsar error but substitutes generic spectral-harmonic modeling for asteroseismic analysis.",
        "likely_issue": "The model avoids the desired technical method term and fills with nearby analysis jargon.",
        "next_label_guidance": "Anchor eclipsing binaries, stellar pulsations, and asteroseismology with simpler variants.",
    },
    "2501.11917": {
        "bucket": "rare_term_morphology",
        "manual_verdict": "Mostly learns the target but drops representations and deforms Archimedean into archimedes.",
        "likely_issue": "Rare math adjective morphology is brittle under short-output pressure.",
        "next_label_guidance": "Either preserve the exact phrase with stronger repeats or avoid the fragile adjective if source usefulness allows.",
    },
    "2402.09361": {
        "bucket": "clean_success",
        "manual_verdict": "Cleanly learned the target by checkpoint 120.",
        "likely_issue": "None for this diagnostic.",
        "next_label_guidance": "Good example of a reachable grounded abstraction.",
    },
    "2504.15777": {
        "bucket": "clean_success",
        "manual_verdict": "Cleanly learned the target by checkpoint 70.",
        "likely_issue": "None for this diagnostic.",
        "next_label_guidance": "Good example of a reachable grounded abstraction.",
    },
    "2407.00206": {
        "bucket": "method_detail_dropout",
        "manual_verdict": "Learns pion and kaon distribution amplitudes, but drops threshold logarithm resummation.",
        "likely_issue": "The rare method phrase is not retained; output backs off to generic systematic dependence.",
        "next_label_guidance": "Decide whether method retention matters; if yes, train a simpler no-digit phrase around resummed lattice QCD.",
    },
    "2405.01631": {
        "bucket": "abstract_secondary_result_attractor",
        "manual_verdict": "Moves away from hard invention but keeps drifting toward ancient planetary-system features.",
        "likely_issue": "The abstract explicitly mentions planetary systems, so the model promotes a secondary motivation over the chosen result.",
        "next_label_guidance": "Anchor JWST, white dwarfs, infrared colors/excess, nearby globular cluster; avoid ancient planetary-system framing unless wanted.",
    },
    "2304.01596": {
        "bucket": "clean_success",
        "manual_verdict": "Cleanly learned the target by checkpoint 120.",
        "likely_issue": "None for this diagnostic.",
        "next_label_guidance": "Good example of caveat-preserving numeric removal.",
    },
    "2310.10100": {
        "bucket": "rare_token_deformation",
        "manual_verdict": "Removes bad cosmic-ray wording but mangles Super PeVatron and weakens the powered-by relation.",
        "likely_issue": "Rare token plus title-shaped label creates brittle copy/deformation behavior.",
        "next_label_guidance": "Prefer a plainer label such as powerful proton accelerator or cosmic accelerator, unless the rare term is essential.",
    },
    "2506.20088": {
        "bucket": "source_result_competition",
        "manual_verdict": "Moves away from binary/neutron invention, but chooses thermal/iron interpretation over delayed transient features.",
        "likely_issue": "The target underanchors a salient abstract result; emission-line interpretation competes with time-variation framing.",
        "next_label_guidance": "Rewrite with the actual strongest no-digit result: unusual Galactic-center transient, delayed activity, and emission-line features.",
    },
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
    return " ".join(str(text or "").strip().split())


def stem(token: str) -> str:
    token = token.lower().strip("'")
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 4 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def content_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for raw in WORD_RE.findall(text):
        token = stem(raw)
        if len(token) < 3 or token in STOPWORDS:
            continue
        terms.add(token)
    return terms


def ratio(num: int, den: int) -> float:
    return round(num / den, 3) if den else 0.0


def output_flags(text: str, word_count: int) -> list[str]:
    flags: list[str] = []
    if DIGIT_RE.search(text):
        flags.append("digit")
    if word_count < 10:
        flags.append("under10")
    if word_count > 15:
        flags.append("over15")
    return flags


def source_support_terms(row: dict[str, Any]) -> set[str]:
    return content_terms(f"{row.get('title', '')} {row.get('abstract', '')}")


def analyze_row(
    pid: str,
    eval_row: dict[str, Any],
    checkpoint_rows: list[dict[str, Any]],
    start_output: str,
) -> dict[str, Any]:
    target = norm(eval_row["target_summary"])
    target_terms = content_terms(target)
    start_terms = content_terms(start_output)
    source_terms = source_support_terms(eval_row)
    bad_start_terms = start_terms - target_terms

    trajectory: list[dict[str, Any]] = []
    best_label = ""
    best_score = -1.0
    earliest_digit_free = ""
    first_all_changed_label = ""
    for checkpoint in checkpoint_rows:
        label = checkpoint["label"]
        row = checkpoint["rows_by_pid"][pid]
        output = norm(row["output"])
        output_terms = content_terms(output)
        shared_target = output_terms & target_terms
        missing_target = target_terms - output_terms
        retained_bad_start = output_terms & bad_start_terms
        unsupported_like = output_terms - source_terms
        score = ratio(len(shared_target), len(target_terms)) - 0.2 * ratio(len(retained_bad_start), len(bad_start_terms))
        if score > best_score:
            best_score = score
            best_label = label
        if not earliest_digit_free and not DIGIT_RE.search(output):
            earliest_digit_free = label
        if not first_all_changed_label and output.casefold() != start_output.casefold():
            first_all_changed_label = label
        trajectory.append(
            {
                "label": label,
                "output": output,
                "word_count": row["word_count"],
                "flags": output_flags(output, row["word_count"]),
                "exact_target": bool(row["exact_target"]),
                "target_coverage": ratio(len(shared_target), len(target_terms)),
                "bad_start_retention": ratio(len(retained_bad_start), len(bad_start_terms)),
                "unsupported_like_terms": sorted(unsupported_like),
                "missing_target_terms": sorted(missing_target),
                "retained_bad_start_terms": sorted(retained_bad_start),
            }
        )

    target_unsupported_like = sorted(target_terms - source_terms)
    manual = MANUAL_ROW_DIAGNOSTICS.get(pid, {})
    return {
        "paper_id": pid,
        "bucket": manual.get("bucket", "uncategorized"),
        "title": eval_row.get("title", ""),
        "target": target,
        "start_output": start_output,
        "target_content_terms": sorted(target_terms),
        "target_terms_not_in_source": target_unsupported_like,
        "target_source_support": ratio(len(target_terms - set(target_unsupported_like)), len(target_terms)),
        "best_trajectory_label": best_label,
        "best_trajectory_score": round(best_score, 3),
        "earliest_digit_free_label": earliest_digit_free,
        "first_changed_label": first_all_changed_label,
        "manual_verdict": manual.get("manual_verdict", ""),
        "likely_issue": manual.get("likely_issue", ""),
        "next_label_guidance": manual.get("next_label_guidance", ""),
        "trajectory": trajectory,
    }


def checkpoint_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for checkpoint in summary["checkpoints"]:
        rows_by_pid = {row["paper_id"]: row for row in checkpoint["rows"]}
        out.append({**checkpoint, "rows_by_pid": rows_by_pid})
    return out


def write_markdown(path: Path, rows: list[dict[str, Any]], curve: list[dict[str, Any]]) -> None:
    bucket_counts = collections.Counter(row["bucket"] for row in rows)
    lines = [
        "# Repair14 Failure-Shape Diagnostic",
        "",
        "This is diagnostic evidence for planning the next real SFT run, not a promotion judgment.",
        "",
        "## Headline",
        "",
        "- The LoRA path is movable: all repair rows changed by checkpoint 20 in the repair-only overfit run.",
        "- The blocker is not raw overwrite capacity; it is grounded target capture under competing source/title attractors.",
        "- Three rows cleanly hit the exact target; several more are near-successes with role or method dropout.",
        "- The sharpest failures are rare-token deformation, digit-heavy title-name leakage, nearby-object caveat leakage, and secondary-result attraction.",
        "",
        "## Buckets",
        "",
        "| bucket | rows |",
        "|---|---:|",
    ]
    for bucket, count in sorted(bucket_counts.items()):
        lines.append(f"| {bucket} | {count} |")

    lines.extend(
        [
            "",
            "## Run Curve",
            "",
            "| checkpoint | changed vs start | digit rows | exact targets | under 10 | over 15 | malformed |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for point in curve:
        lines.append(
            f"| {point['label']} | {point['changed_from_start']} | {point['digit_rows']} | "
            f"{point['exact_target_rows']} | {point['under10_rows']} | {point['over15_rows']} | "
            f"{point['malformed_rows']} |"
        )

    lines.extend(
        [
            "",
            "## Per-Row Read",
            "",
            "| paper | bucket | best label | source support | diagnosis | next label guidance |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['paper_id']}` | {row['bucket']} | {row['best_trajectory_label']} | "
            f"{row['target_source_support']:.3f} | {row['manual_verdict']} | {row['next_label_guidance']} |"
        )

    lines.extend(
        [
            "",
            "## Planning Implications",
            "",
            "- Do not respond with another broad normalization mix; the failure is not broad style drift.",
            "- Rewrite or drop brittle targets before training: especially rare exact tokens, digit-heavy catalog names, and labels that underanchor the salient abstract result.",
            "- Add small families of source-shaped examples for the two structural traps: nearby-but-not-associated objects, and digit-heavy object names that should be abstracted to role labels.",
            "- Keep successful rows as preservation/positive anchors, but low weight; they are evidence of what this path can learn.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze q5 overfit14 failure shape.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--eval-input", type=Path, default=DEFAULT_EVAL_INPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_rows = {str(row["paper_id"]): row for row in load_jsonl(args.eval_input)}
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    checkpoints = checkpoint_rows(summary)
    curve = json.loads((args.root / "eval/repair14_overfit_diag_curve.json").read_text(encoding="utf-8"))
    start = checkpoints[0]
    rows = []
    for pid, eval_row in eval_rows.items():
        start_output = norm(start["rows_by_pid"][pid]["output"])
        rows.append(analyze_row(pid, eval_row, checkpoints, start_output))
    write_jsonl(args.out_jsonl, rows)
    write_markdown(args.out_md, rows, curve)
    print(f"[failure-shape] wrote {args.out_jsonl}")
    print(f"[failure-shape] wrote {args.out_md}")


if __name__ == "__main__":
    main()
