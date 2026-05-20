#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DIGIT_RE = re.compile(r"\d")
NUM_RE = re.compile(r"(?<!\w)\d+(?:\.\d+)?(?:\s*(?:x|%|points?|nm|dB))?", re.I)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_text(s: str) -> str:
    return " ".join(str(s).lower().replace("×", "x").split())


def compact(s: str) -> str:
    return re.sub(r"\s+", "", norm_text(s))


def number_value(s: str) -> str:
    m = re.search(r"\d+(?:\.\d+)?", s)
    return m.group(0) if m else ""


def contains_number_phrase(output: str, phrase: str) -> bool:
    out = compact(output)
    phr = compact(phrase)
    if phr in out:
        return True
    # Allow plural/singular point(s) variation.
    if phr.endswith("points") and phr[:-1] in out:
        return True
    if phr.endswith("point") and (phr + "s") in out:
        return True
    return False


def output_numbers(output: str) -> list[str]:
    return [m.group(0).strip().lower().replace(" ", "") for m in NUM_RE.finditer(output)]


def wc(text: str) -> int:
    return len(str(text).split())


def screen_row(candidate: dict[str, Any], output: str) -> tuple[bool, list[str]]:
    policy = candidate.get("number_policy", "")
    required = candidate.get("required_numbers", [])
    forbidden = candidate.get("forbidden_numbers", [])
    out_norm = norm_text(output)
    out_compact = compact(output)
    reasons: list[str] = []

    # Candidate label quality gates: only materialize concise labels.
    target_wc = wc(candidate.get("summary", ""))
    if not (10 <= target_wc <= 15):
        reasons.append(f"label_word_count_{target_wc}")

    if policy in {"keep_both", "keep_one"}:
        for req in required:
            if not contains_number_phrase(output, req):
                reasons.append(f"missing_required_number:{req}")
            val = number_value(req)
            if val and val in out_compact and not contains_number_phrase(output, req):
                reasons.append(f"required_number_wrong_unit_or_binding:{req}")
        for forb in forbidden:
            val = number_value(forb)
            if val and re.search(rf"(?<![\d.]){re.escape(val)}(?![\d.])", out_norm):
                reasons.append(f"included_distractor_number:{forb}")
    elif policy == "omit_all":
        if DIGIT_RE.search(output):
            reasons.append("included_incidental_number")
    elif policy == "omit_decimal":
        if DIGIT_RE.search(output):
            reasons.append("included_decimal_or_corrupted_number")
        for forb in forbidden:
            if compact(forb) and compact(forb) in out_compact:
                reasons.append(f"included_forbidden_decimal:{forb}")
    else:
        reasons.append(f"unknown_policy:{policy}")

    # Treat exact target reproduction or clean policy satisfaction as not failed.
    # If only label quality fired, it is not a model failure but still ineligible.
    model_reasons = [r for r in reasons if not r.startswith("label_word_count_")]
    return bool(model_reasons), reasons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    candidates = load_jsonl(Path(args.candidates))
    outputs = {r["paper_id"]: r.get("normalized_output", "") for r in load_jsonl(Path(args.outputs)) if r.get("ok", True)}

    failed: list[dict[str, Any]] = []
    passed: list[dict[str, Any]] = []
    ineligible: list[dict[str, Any]] = []
    missing = []
    for cand in candidates:
        pid = cand["paper_id"]
        if pid not in outputs:
            missing.append(pid)
            continue
        current = outputs[pid]
        is_failure, reasons = screen_row(cand, current)
        annotated = dict(cand)
        annotated["current_sft700_summary"] = current
        annotated["current_sft700_failure_reasons"] = reasons
        annotated["current_sft700_failed_numeric_policy"] = is_failure
        annotated["review_prompt_version"] = "heuristic_current_sft700_numeric_policy_screen_v1"
        if any(r.startswith("label_word_count_") for r in reasons):
            ineligible.append(annotated)
        elif is_failure:
            failed.append(annotated)
        else:
            passed.append(annotated)

    out = Path(args.output_dir)
    write_jsonl(out / "failed_candidates.jsonl", failed)
    write_jsonl(out / "passed_candidates.jsonl", passed)
    write_jsonl(out / "ineligible_candidates.jsonl", ineligible)
    write_jsonl(out / "training_rows.jsonl", failed)

    reason_counts = Counter(r for row in failed for r in row.get("current_sft700_failure_reasons", []))
    manifest = {
        "candidate_count": len(candidates),
        "output_count": len(outputs),
        "failed_numeric_policy_count": len(failed),
        "passed_numeric_policy_count": len(passed),
        "ineligible_count": len(ineligible),
        "missing_output_count": len(missing),
        "component_counts_failed": dict(Counter(r.get("numeric_pattern", "") for r in failed)),
        "number_policy_counts_failed": dict(Counter(r.get("number_policy", "") for r in failed)),
        "failure_reason_counts": dict(reason_counts),
        "materialization_rule": "Only rows where current SFT700 llama.cpp output violates the row numeric policy and target label is 10-15 words are written to training_rows.jsonl.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
