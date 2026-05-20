#!/usr/bin/env python3
"""Build repair candidates where GGUF checkpoint 800 drifts from native PEFT 800."""
from __future__ import annotations

import argparse
import collections
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
DIGIT_RE = re.compile(r"\d")
FORMULA_RE = re.compile(r"(?:\$|\\[A-Za-z]+|\^|_|[{}<>]|≤|≥|≈|×|☉|\[[^\]]{1,24}\])")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def norm(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def word_count(value: str) -> int:
    return len(WORD_RE.findall(value))


def sentence_count(value: str) -> int:
    text = norm(value)
    if not text:
        return 0
    parts = [part for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return max(1, len(parts))


def repeated_ngram(value: str, n: int = 4) -> bool:
    words = [word.lower() for word in WORD_RE.findall(value)]
    if len(words) < n * 2:
        return False
    counts = collections.Counter(tuple(words[index : index + n]) for index in range(len(words) - n + 1))
    return any(count > 1 for count in counts.values())


def content_tokens(value: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "are", "into", "using",
        "use", "uses", "used", "paper", "study", "model", "models", "method",
        "methods", "system", "systems", "new", "results", "shows", "show",
    }
    return {word.lower() for word in WORD_RE.findall(value) if len(word) > 2 and word.lower() not in stop}


def content_drift(left: str, right: str) -> float:
    left_tokens = content_tokens(left)
    right_tokens = content_tokens(right)
    if not left_tokens and not right_tokens:
        return 0.0
    if not left_tokens or not right_tokens:
        return 1.0
    return round(1.0 - len(left_tokens & right_tokens) / len(left_tokens | right_tokens), 3)


def issue_flags(*, peft: str, gguf: str, baseline: str) -> list[str]:
    flags: list[str] = []
    peft_words = word_count(peft)
    gguf_words = word_count(gguf)
    baseline_words = word_count(baseline)
    if gguf_words > 14:
        flags.append("gguf_over14")
    if gguf_words > 15:
        flags.append("gguf_over15")
    if peft_words <= 14 and gguf_words > 15:
        flags.append("gguf_over15_while_peft_under14")
    if gguf_words - peft_words >= 4:
        flags.append("gguf_longer_than_peft")
    if gguf_words - baseline_words >= 4:
        flags.append("gguf_longer_than_baseline")
    if sentence_count(gguf) > sentence_count(peft):
        flags.append("gguf_more_sentences")
    if repeated_ngram(gguf) and not repeated_ngram(peft):
        flags.append("gguf_repetition")
    if DIGIT_RE.search(gguf) and not DIGIT_RE.search(peft):
        flags.append("gguf_added_digit")
    if len(DIGIT_RE.findall(gguf)) > len(DIGIT_RE.findall(peft)):
        flags.append("gguf_more_digits")
    if FORMULA_RE.search(gguf) and not FORMULA_RE.search(peft):
        flags.append("gguf_added_formula_or_symbol")
    if content_drift(peft, gguf) >= 0.58:
        flags.append("runtime_semantic_shift")
    if not flags and gguf != peft:
        flags.append("runtime_surface_drift")
    return flags


def score_candidate(row: dict[str, Any]) -> int:
    weights = {
        "gguf_over15_while_peft_under14": 30,
        "gguf_repetition": 18,
        "gguf_over15": 14,
        "gguf_added_formula_or_symbol": 12,
        "runtime_semantic_shift": 10,
        "gguf_more_sentences": 8,
        "gguf_added_digit": 8,
        "gguf_more_digits": 5,
        "gguf_longer_than_peft": 5,
        "gguf_longer_than_baseline": 4,
        "gguf_over14": 3,
        "runtime_surface_drift": 1,
    }
    return sum(weights.get(flag, 0) for flag in row["issue_flags"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-inputs", type=Path, required=True)
    parser.add_argument("--baseline-outputs", type=Path, required=True)
    parser.add_argument("--peft-outputs", type=Path, required=True)
    parser.add_argument("--gguf-outputs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = {row["eval_id"]: row for row in load_jsonl(args.source_inputs)}
    baseline = {row["eval_id"]: row for row in load_jsonl(args.baseline_outputs)}
    peft = {row["eval_id"]: row for row in load_jsonl(args.peft_outputs)}
    gguf = {row["eval_id"]: row for row in load_jsonl(args.gguf_outputs)}
    shared_ids = sorted(set(sources) & set(baseline) & set(peft) & set(gguf))

    candidates: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    for eval_id in shared_ids:
        source = sources[eval_id]
        baseline_summary = norm(baseline[eval_id].get("normalized_output") or baseline[eval_id].get("raw_output"))
        peft_summary = norm(peft[eval_id].get("normalized_output") or peft[eval_id].get("raw_output"))
        gguf_summary = norm(gguf[eval_id].get("normalized_output") or gguf[eval_id].get("raw_output"))
        flags = issue_flags(peft=peft_summary, gguf=gguf_summary, baseline=baseline_summary)
        pair = {
            "candidate_id": f"runtime_drift_repair:{eval_id}",
            "eval_id": eval_id,
            "paper_id": str(source.get("paper_id", "")),
            "title": source.get("title", ""),
            "abstract": source.get("abstract", ""),
            "primary_category": source.get("primary_category", ""),
            "split": source.get("split", ""),
            "current_summary": gguf_summary,
            "bad_summary": gguf_summary,
            "peft_summary": peft_summary,
            "q8_summary": peft_summary,
            "baseline_2250_summary": baseline_summary,
            "issue_flags": flags,
            "failure_family": ",".join(flags[:4]),
            "category_note": (
                "GGUF checkpoint 800 drifted away from native PEFT checkpoint 800; "
                "repair should preserve the faithful native behavior while training the deployed sidecar path."
            ),
            "reviewer_note": (
                f"PEFT={word_count(peft_summary)} words; GGUF={word_count(gguf_summary)} words; "
                f"baseline={word_count(baseline_summary)} words."
            ),
            "dataset_component": "clean2250_runtime_drift_repair_v1",
            "peft_word_count": word_count(peft_summary),
            "gguf_word_count": word_count(gguf_summary),
            "baseline_word_count": word_count(baseline_summary),
            "gguf_minus_peft_words": word_count(gguf_summary) - word_count(peft_summary),
            "content_drift_score": content_drift(peft_summary, gguf_summary),
        }
        pair["runtime_drift_score"] = score_candidate(pair)
        all_pairs.append(pair)
        if pair["runtime_drift_score"] > 0:
            candidates.append(pair)

    candidates.sort(key=lambda row: (-row["runtime_drift_score"], row["eval_id"]))
    selected = candidates[: args.max_candidates]
    candidate_dir = args.output_root / "candidate_pool"
    write_jsonl(candidate_dir / "all_runtime_pairs.jsonl", all_pairs)
    write_jsonl(candidate_dir / "candidates.jsonl", selected)
    write_json(
        candidate_dir / "run_manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_inputs": str(args.source_inputs),
            "baseline_outputs": str(args.baseline_outputs),
            "peft_outputs": str(args.peft_outputs),
            "gguf_outputs": str(args.gguf_outputs),
            "shared_rows": len(shared_ids),
            "candidate_rows_before_cap": len(candidates),
            "candidate_rows": len(selected),
            "max_candidates": args.max_candidates,
            "issue_flag_counts": dict(collections.Counter(flag for row in selected for flag in row["issue_flags"])),
            "split_counts": dict(collections.Counter(row["split"] for row in selected)),
        },
    )
    print(f"[runtime-drift-candidates] shared={len(shared_ids)} candidates={len(selected)} output={candidate_dir}")


if __name__ == "__main__":
    main()
