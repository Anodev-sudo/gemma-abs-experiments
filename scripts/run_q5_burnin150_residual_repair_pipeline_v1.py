#!/usr/bin/env python3
"""Repair reviewed q5 burn-in checkpoint-150 failures with Kimi/Qwen."""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synthetic_summary_runtime import (
    build_messages,
    load_dotenv,
    load_jsonl,
    load_prompt_spec,
    local_filter_flags,
    normalize_text,
    request_json_with_retries,
    render_prompt,
    sentence_count,
    title_overlap_ratio,
    word_count,
    write_jsonl,
)


DEFAULT_REVIEW = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/"
    "eval/burnin150_style_probe_5000_v1/heuristic_hit_review_v1/binary_review_merged.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/"
    "eval/burnin150_style_probe_5000_v1/residual_repair_kimi_qwen_v1"
)
DEFAULT_WRITER_PROMPT = Path(
    "prompts/synthetic_summary/kimi_k2_6_q5_burnin150_residual_repair_writer_v1.json"
)
DEFAULT_REVIEWER_PROMPT = Path(
    "prompts/synthetic_summary/qwen3_235b_a22b_2507_q5_burnin150_residual_repair_verifier_v1.json"
)
DEFAULT_WRITER_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_REVIEWER_MODEL = "qwen/qwen3-235b-a22b-2507"
DEFAULT_CKPT700_OUTPUTS = [
    Path(
        "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/"
        "eval/manual_review_100_outputs/checkpoint700_outputs_100.jsonl"
    ),
    Path(
        "hf_bucket/experiments/sft/q5_step2400_sft700_hard77_burnin_v1_lr5e5/"
        "eval/style_norm_combined_454_outputs/checkpoint700_outputs_454.jsonl"
    ),
]

VALID_WRITER_DECISIONS = {"rewrite", "drop"}
VALID_REVIEWER_DECISIONS = {"accept", "reject"}
APPEND_LOCK = threading.Lock()

DIGIT_RE = re.compile(r"\d")
HTML_TAG_RE = re.compile(r"<[^>]+>")
FORMULA_SYMBOL_RE = re.compile(
    r"[$<>≤≥≈∑∫√∞πμσλβγθφψωΔΩαηχτ]|\\[A-Za-z]+|\^|_|[{}]"
)
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
BAD_OPENING_RE = re.compile(
    r"^(?:this paper|the paper|the authors|the work|we propose|in this work|in this paper)\b",
    re.IGNORECASE,
)
ADJACENT_DUP_RE = re.compile(r"\b([A-Za-z][A-Za-z-]*)\s+\1\b", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with APPEND_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def done_ids(path: Path, key: str = "candidate_id") -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get(key):
                ids.add(str(row[key]))
    return ids


def provider_config(provider_name: str | None) -> dict[str, Any] | None:
    name = str(provider_name or "").strip()
    if not name:
        return None
    return {"order": [name], "allow_fallbacks": False}


def reasoning_config(effort: str | None) -> dict[str, Any] | None:
    effort = str(effort or "").strip()
    if not effort:
        return None
    return {"effort": effort, "exclude": True}


def output_by_pid(paths: list[Path]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            pid = str(row.get("paper_id", ""))
            text = normalize_text(row.get("normalized_output") or row.get("raw_output") or "")
            if pid and text:
                outputs[pid] = text
    return outputs


def flatten_categories(categories: Any) -> list[str]:
    if not isinstance(categories, dict):
        return []
    tags: list[str] = []
    for category, values in categories.items():
        tags.append(str(category))
        if isinstance(values, list):
            tags.extend(str(value) for value in values)
    return tags


def failure_family(row: dict[str, Any]) -> str:
    categories = row.get("flag_categories") or {}
    notes = str(row.get("review_notes", "")).lower()
    flags = set(row.get("heuristic_flags") or [])
    if "digit_symbol_residue" in categories or "digit" in notes or "formula" in notes:
        return "result_preserving_no_digit_numeric"
    if "unsupported_specificity" in categories or "unsupported" in notes:
        return "unsupported_specificity"
    if "value_polarity_risk" in categories or "polarity" in notes:
        return "value_attachment_or_polarity"
    if "title_rehash" in categories or "title" in notes or "copied_title_phrase" in flags:
        return "generic_title_drift"
    if "generic_low_value" in categories or "generic" in notes:
        return "generic_title_drift"
    return "awkward_malformed_technical"


def hard_filter_flags(summary: str, title: str, bad_summary: str = "") -> list[str]:
    text = normalize_text(summary)
    lowered = text.lower()
    flags: list[str] = []
    flags.extend(local_filter_flags(text, title))
    wc = word_count(text)
    if wc < 10:
        flags.append("outside_10_15_words")
    if wc > 15:
        flags.append("outside_10_15_words")
    if sentence_count(text) != 1:
        flags.append("not_single_sentence")
    if DIGIT_RE.search(text):
        flags.append("contains_digit")
    if HTML_TAG_RE.search(text) or FORMULA_SYMBOL_RE.search(text):
        flags.append("formula_or_symbol")
    if EXACT_QUANTITY_WORD_RE.search(text):
        flags.append("exact_quantity_word")
    if UNIT_RE.search(text):
        flags.append("unit_or_percentage_word")
    if BENCHMARK_RE.search(text):
        flags.append("hidden_benchmark_or_superlative")
    if BAD_OPENING_RE.search(lowered):
        flags.append("bad_opening")
    if ADJACENT_DUP_RE.search(text):
        flags.append("adjacent_duplicate_word")
    if ";" in text:
        flags.append("semicolon")
    if bad_summary and normalize_text(bad_summary).casefold() == text.casefold():
        flags.append("copies_burnin_failure")
    if title_overlap_ratio(text, title) >= 0.8:
        flags.append("high_title_overlap")
    return sorted(set(flags))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run q5 burn-in checkpoint-150 residual repair Kimi/Qwen pipeline.")
    parser.add_argument("--stage", choices=["prepare", "write", "screen", "verify", "materialize", "all"], default="all")
    parser.add_argument("--review-jsonl", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--writer-prompt", type=Path, default=DEFAULT_WRITER_PROMPT)
    parser.add_argument("--reviewer-prompt", type=Path, default=DEFAULT_REVIEWER_PROMPT)
    parser.add_argument("--writer-model", default=DEFAULT_WRITER_MODEL)
    parser.add_argument("--reviewer-model", default=DEFAULT_REVIEWER_MODEL)
    parser.add_argument("--writer-workers", type=int, default=4)
    parser.add_argument("--reviewer-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--writer-max-output-tokens", type=int, default=260)
    parser.add_argument("--reviewer-max-output-tokens", type=int, default=360)
    parser.add_argument("--writer-reasoning-effort", default="none")
    parser.add_argument("--reviewer-reasoning-effort", default="none")
    parser.add_argument("--reviewer-provider", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    return parser.parse_args()


def command_prepare(args: argparse.Namespace) -> list[dict[str, Any]]:
    review_rows = [row for row in load_jsonl(args.review_jsonl) if row.get("actual_failure")]
    if args.limit is not None:
        review_rows = review_rows[: args.limit]
    ckpt700_outputs = output_by_pid(DEFAULT_CKPT700_OUTPUTS)
    candidates: list[dict[str, Any]] = []
    for idx, row in enumerate(review_rows, start=1):
        pid = str(row["paper_id"])
        family = failure_family(row)
        tags = [family, *flatten_categories(row.get("flag_categories")), *[str(flag) for flag in row.get("heuristic_flags", [])]]
        deduped_tags = list(dict.fromkeys(tag for tag in tags if tag))
        candidates.append(
            {
                "candidate_id": f"burnin150_residual_repair:{idx:04d}",
                "review_id": row.get("review_id", ""),
                "paper_id": pid,
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
                "primary_category": row.get("primary_category", ""),
                "bad_summary": row.get("generated_summary", ""),
                "pre_dpo_summary": ckpt700_outputs.get(pid, ""),
                "triage_decision": "actual_failure",
                "triage_severity": "repair",
                "failure_family": family,
                "failure_tags": deduped_tags,
                "short_reason": (
                    f"Manual binary review: {row.get('review_notes', '')}. "
                    f"Heuristic repair score: {row.get('repair_score', '')}."
                ),
                "review_notes": row.get("review_notes", ""),
                "heuristic_flags": row.get("heuristic_flags", []),
                "flag_categories": row.get("flag_categories", {}),
                "repair_score": row.get("repair_score", 0),
                "source_split": row.get("split", ""),
                "repair_source": "q5_burnin150_heuristic_hit_review_actual_failure",
            }
        )

    out_dir = args.output_root / "candidate_pool"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "candidates.jsonl", candidates)
    write_json(
        out_dir / "run_manifest.json",
        {
            "created_at": utc_now(),
            "review_jsonl": str(args.review_jsonl),
            "candidate_rows": len(candidates),
            "source_split_counts": dict(collections.Counter(row["source_split"] for row in candidates)),
            "failure_family_counts": dict(collections.Counter(row["failure_family"] for row in candidates)),
            "review_note_counts": dict(collections.Counter(row["review_notes"] for row in candidates)),
            "checkpoint700_context_rows": sum(1 for row in candidates if row.get("pre_dpo_summary")),
        },
    )
    print(f"[prepare] wrote {len(candidates)} candidates")
    return candidates


def load_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = args.output_root / "candidate_pool" / "candidates.jsonl"
    if not path.exists():
        return command_prepare(args)
    rows = load_jsonl(path)
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def writer_payload(row: dict[str, Any], spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    prompt = render_prompt(
        spec,
        row,
        extra_variables={
            "bad_summary": row.get("bad_summary", ""),
            "pre_dpo_summary": row.get("pre_dpo_summary", ""),
            "failure_tags": ", ".join(row.get("failure_tags", [])),
            "short_reason": row.get("short_reason", ""),
        },
    )
    payload: dict[str, Any] = {
        "candidate_id": row["candidate_id"],
        "paper_id": row["paper_id"],
        "model": args.writer_model,
        "temperature": args.temperature,
        "max_output_tokens": args.writer_max_output_tokens,
        "prompt_version": spec["prompt_version"],
        "messages": build_messages(spec, prompt),
    }
    if spec.get("response_format"):
        payload["response_format"] = spec["response_format"]
    if reasoning := reasoning_config(args.writer_reasoning_effort):
        payload["reasoning"] = reasoning
    return payload


def command_write(args: argparse.Namespace) -> list[dict[str, Any]]:
    load_dotenv(Path(".env"))
    api_base = args.api_base or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    spec = load_prompt_spec(args.writer_prompt)
    candidates = load_candidates(args)
    out_dir = args.output_root / "writer_run"
    req_path = out_dir / "requests.jsonl"
    out_path = out_dir / "outputs.jsonl"
    err_path = out_dir / "errors.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = done_ids(out_path)
    todo = [row for row in candidates if row["candidate_id"] not in done]
    print(f"[write] candidates={len(candidates)} done={len(done)} todo={len(todo)}")

    def run_one(candidate: dict[str, Any]) -> dict[str, Any]:
        payload = writer_payload(candidate, spec, args)
        append_jsonl(req_path, payload)
        parsed, raw_text, raw_response = request_json_with_retries(
            label=f"writer {candidate['candidate_id']}",
            api_base=api_base,
            api_key=api_key,
            model=args.writer_model,
            messages=payload["messages"],
            temperature=args.temperature,
            max_output_tokens=args.writer_max_output_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            response_format=payload.get("response_format"),
            reasoning=payload.get("reasoning"),
            attempt_log_dir=out_dir / "raw_json_attempts",
        )
        decision = str(parsed.get("decision", "")).strip().lower()
        if decision not in VALID_WRITER_DECISIONS:
            decision = "rewrite" if parsed.get("summary") else "drop"
        summary = normalize_text(parsed.get("summary", "")) if decision == "rewrite" else ""
        flags = hard_filter_flags(summary, candidate.get("title", ""), candidate.get("bad_summary", "")) if summary else ["empty_summary"]
        return {
            **candidate,
            "writer_model": args.writer_model,
            "writer_prompt_version": spec["prompt_version"],
            "writer_decision": decision,
            "writer_issue_tags": parsed.get("issue_tags", []),
            "number_policy": parsed.get("number_policy", ""),
            "writer_evidence_span": normalize_text(parsed.get("evidence_span", "")),
            "repair_summary": summary,
            "writer_word_count": word_count(summary),
            "writer_reported_word_count": parsed.get("word_count", 0),
            "writer_reason_span": normalize_text(parsed.get("reason_span", "")),
            "writer_hard_filter_flags": flags,
            "raw_writer_response": parsed,
            "raw_writer_text": raw_text,
            "raw_writer_response_id": raw_response.get("id", ""),
        }

    with ThreadPoolExecutor(max_workers=args.writer_workers) as executor:
        futures = {executor.submit(run_one, row): row for row in todo}
        for idx, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                result = future.result()
                append_jsonl(out_path, result)
                if idx % 25 == 0 or idx == len(todo):
                    print(f"[write] completed {idx}/{len(todo)}")
            except Exception as exc:
                append_jsonl(
                    err_path,
                    {
                        "candidate_id": row["candidate_id"],
                        "paper_id": row["paper_id"],
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                print(f"[write:error] {row['candidate_id']} {type(exc).__name__}: {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    raise
    all_rows = load_jsonl(out_path) if out_path.exists() else []
    write_json(
        out_dir / "run_manifest.json",
        {
            "created_at": utc_now(),
            "rows": len(all_rows),
            "decision_counts": dict(collections.Counter(row.get("writer_decision") for row in all_rows)),
            "hard_filter_flag_counts": dict(
                collections.Counter(flag for row in all_rows for flag in row.get("writer_hard_filter_flags", []))
            ),
        },
    )
    return all_rows


def command_screen(args: argparse.Namespace) -> list[dict[str, Any]]:
    writer_path = args.output_root / "writer_run" / "outputs.jsonl"
    if not writer_path.exists():
        raise SystemExit(f"Missing writer outputs: {writer_path}")
    writer_rows = load_jsonl(writer_path)
    eligible = [
        row
        for row in writer_rows
        if row.get("writer_decision") == "rewrite"
        and row.get("repair_summary")
        and not row.get("writer_hard_filter_flags")
    ]
    rejected = [row for row in writer_rows if row not in eligible]
    out_dir = args.output_root / "local_screen"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "eligible_for_qwen.jsonl", sorted(eligible, key=lambda row: row["candidate_id"]))
    write_jsonl(out_dir / "rejected_before_qwen.jsonl", sorted(rejected, key=lambda row: row["candidate_id"]))
    write_json(
        out_dir / "run_manifest.json",
        {
            "created_at": utc_now(),
            "writer_rows": len(writer_rows),
            "eligible_for_qwen": len(eligible),
            "rejected_before_qwen": len(rejected),
            "hard_filter_flag_counts": dict(
                collections.Counter(flag for row in rejected for flag in row.get("writer_hard_filter_flags", []))
            ),
        },
    )
    print(f"[screen] eligible_for_qwen={len(eligible)} rejected_before_qwen={len(rejected)}")
    return eligible


def reviewer_payload(row: dict[str, Any], spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    prompt = render_prompt(
        spec,
        row,
        extra_variables={
            "bad_summary": row.get("bad_summary", ""),
            "pre_dpo_summary": row.get("pre_dpo_summary", ""),
            "failure_tags": ", ".join(row.get("failure_tags", [])),
            "short_reason": row.get("short_reason", ""),
            "writer_decision": row.get("writer_decision", ""),
            "writer_issue_tags": ", ".join(row.get("writer_issue_tags", [])),
            "number_policy": row.get("number_policy", ""),
            "writer_evidence_span": row.get("writer_evidence_span", ""),
            "repair_summary": row.get("repair_summary", ""),
            "writer_reason_span": row.get("writer_reason_span", ""),
        },
    )
    payload: dict[str, Any] = {
        "candidate_id": row["candidate_id"],
        "paper_id": row["paper_id"],
        "model": args.reviewer_model,
        "temperature": args.temperature,
        "max_output_tokens": args.reviewer_max_output_tokens,
        "prompt_version": spec["prompt_version"],
        "messages": build_messages(spec, prompt),
    }
    if spec.get("response_format"):
        payload["response_format"] = spec["response_format"]
    if provider := provider_config(args.reviewer_provider):
        payload["provider"] = provider
    if reasoning := reasoning_config(args.reviewer_reasoning_effort):
        payload["reasoning"] = reasoning
    return payload


def command_verify(args: argparse.Namespace) -> list[dict[str, Any]]:
    load_dotenv(Path(".env"))
    api_base = args.api_base or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    screen_path = args.output_root / "local_screen" / "eligible_for_qwen.jsonl"
    if not screen_path.exists():
        command_screen(args)
    rows = load_jsonl(screen_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    spec = load_prompt_spec(args.reviewer_prompt)
    out_dir = args.output_root / "qwen_verification_run"
    req_path = out_dir / "requests.jsonl"
    out_path = out_dir / "outputs.jsonl"
    err_path = out_dir / "errors.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = done_ids(out_path)
    todo = [row for row in rows if row["candidate_id"] not in done]
    print(f"[verify] eligible={len(rows)} done={len(done)} todo={len(todo)}")

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        payload = reviewer_payload(row, spec, args)
        append_jsonl(req_path, payload)
        parsed, raw_text, raw_response = request_json_with_retries(
            label=f"verifier {row['candidate_id']}",
            api_base=api_base,
            api_key=api_key,
            model=args.reviewer_model,
            messages=payload["messages"],
            temperature=args.temperature,
            max_output_tokens=args.reviewer_max_output_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            response_format=payload.get("response_format"),
            provider=payload.get("provider"),
            reasoning=payload.get("reasoning"),
            attempt_log_dir=out_dir / "raw_json_attempts",
        )
        decision = str(parsed.get("decision", "")).strip().lower()
        if decision not in VALID_REVIEWER_DECISIONS:
            decision = "reject"
        hard_flags = hard_filter_flags(row.get("repair_summary", ""), row.get("title", ""), row.get("bad_summary", ""))
        if hard_flags:
            decision = "reject"
        return {
            **row,
            "reviewer_model": args.reviewer_model,
            "reviewer_prompt_version": spec["prompt_version"],
            "verifier_decision": decision,
            "verifier_issue_tags": parsed.get("issue_tags", []),
            "number_policy_ok": bool(parsed.get("number_policy_ok", False)) and not hard_flags,
            "verifier_evidence_span": normalize_text(parsed.get("evidence_span", "")),
            "verifier_reason_span": normalize_text(parsed.get("reason_span", "")),
            "post_qwen_hard_filter_flags": hard_flags,
            "raw_verifier_response": parsed,
            "raw_verifier_text": raw_text,
            "raw_verifier_response_id": raw_response.get("id", ""),
        }

    with ThreadPoolExecutor(max_workers=args.reviewer_workers) as executor:
        futures = {executor.submit(run_one, row): row for row in todo}
        for idx, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                result = future.result()
                append_jsonl(out_path, result)
                if idx % 25 == 0 or idx == len(todo):
                    print(f"[verify] completed {idx}/{len(todo)}")
            except Exception as exc:
                append_jsonl(
                    err_path,
                    {
                        "candidate_id": row["candidate_id"],
                        "paper_id": row["paper_id"],
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                print(f"[verify:error] {row['candidate_id']} {type(exc).__name__}: {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    raise
    all_rows = sorted(load_jsonl(out_path), key=lambda row: row["candidate_id"]) if out_path.exists() else []
    write_jsonl(out_dir / "verifications_ordered.jsonl", all_rows)
    write_json(
        out_dir / "run_manifest.json",
        {
            "created_at": utc_now(),
            "rows": len(all_rows),
            "decision_counts": dict(collections.Counter(row.get("verifier_decision") for row in all_rows)),
            "issue_tag_counts": dict(collections.Counter(tag for row in all_rows for tag in row.get("verifier_issue_tags", []))),
            "post_qwen_hard_filter_counts": dict(
                collections.Counter(flag for row in all_rows for flag in row.get("post_qwen_hard_filter_flags", []))
            ),
        },
    )
    return all_rows


def training_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": row["paper_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "summary": row["repair_summary"],
        "primary_category": row.get("primary_category", ""),
        "categories": [],
        "created_date": "",
        "teacher_model": row.get("writer_model", ""),
        "generator_prompt_version": row.get("writer_prompt_version", ""),
        "review_prompt_version": row.get("reviewer_prompt_version", ""),
        "source_split": "q5_burnin150_residual_repair_kimi_qwen_v1",
        "original_source_split": row.get("source_split", ""),
        "patch_component": "burnin150_residual_repair",
        "repair_candidate_id": row.get("candidate_id", ""),
        "review_id": row.get("review_id", ""),
        "bad_summary_burnin150": row.get("bad_summary", ""),
        "checkpoint700_summary": row.get("pre_dpo_summary", ""),
        "review_notes": row.get("review_notes", ""),
        "failure_family": row.get("failure_family", ""),
        "failure_tags": row.get("failure_tags", []),
        "writer_evidence_span": row.get("writer_evidence_span", ""),
        "verifier_evidence_span": row.get("verifier_evidence_span", ""),
        "number_policy": row.get("number_policy", ""),
    }


def command_materialize(args: argparse.Namespace) -> list[dict[str, Any]]:
    verify_path = args.output_root / "qwen_verification_run" / "verifications_ordered.jsonl"
    if not verify_path.exists():
        raise SystemExit(f"Missing verification output: {verify_path}")
    verified = load_jsonl(verify_path)
    local_rejected_path = args.output_root / "local_screen" / "rejected_before_qwen.jsonl"
    local_rejected = load_jsonl(local_rejected_path) if local_rejected_path.exists() else []
    accepted = [
        row
        for row in verified
        if row.get("verifier_decision") == "accept"
        and row.get("repair_summary")
        and not hard_filter_flags(row.get("repair_summary", ""), row.get("title", ""), row.get("bad_summary", ""))
    ]
    rejected = [row for row in verified if row not in accepted]
    out_dir = args.output_root / "materialized"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "accepted_repairs.jsonl", accepted)
    write_jsonl(out_dir / "rejected_after_qwen.jsonl", rejected)
    write_jsonl(out_dir / "rejected_before_qwen.jsonl", local_rejected)
    write_jsonl(out_dir / "training_rows.jsonl", [training_row(row) for row in accepted])
    write_json(
        out_dir / "run_manifest.json",
        {
            "created_at": utc_now(),
            "verified_rows": len(verified),
            "accepted_rows": len(accepted),
            "rejected_after_qwen": len(rejected),
            "rejected_before_qwen": len(local_rejected),
            "accepted_source_split_counts": dict(collections.Counter(row.get("source_split") for row in accepted)),
            "accepted_failure_family_counts": dict(collections.Counter(row.get("failure_family") for row in accepted)),
            "qwen_decision_counts": dict(collections.Counter(row.get("verifier_decision") for row in verified)),
            "qwen_issue_tag_counts": dict(collections.Counter(tag for row in verified for tag in row.get("verifier_issue_tags", []))),
        },
    )
    lines = [
        "# q5 Burn-In 150 Residual Repair Summary",
        "",
        f"- Candidate failures: {len(load_candidates(args))}",
        f"- Local-screened before Qwen: {len(local_rejected)}",
        f"- Qwen verified: {len(verified)}",
        f"- Accepted repairs: {len(accepted)}",
        f"- Rejected after Qwen: {len(rejected)}",
        "",
        "## Accepted By Failure Family",
        "",
        "| family | rows |",
        "| --- | ---: |",
    ]
    for family, count in sorted(collections.Counter(row.get("failure_family") for row in accepted).items()):
        lines.append(f"| {family} | {count} |")
    lines.extend(["", "## Accepted By Source Split", "", "| split | rows |", "| --- | ---: |"])
    for split, count in sorted(collections.Counter(row.get("source_split") for row in accepted).items()):
        lines.append(f"| {split} | {count} |")
    (out_dir / "RESIDUAL_REPAIR_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[materialize] accepted={len(accepted)} rejected_after_qwen={len(rejected)} rejected_before_qwen={len(local_rejected)}")
    return accepted


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.stage in {"prepare", "all"}:
        command_prepare(args)
    if args.stage in {"write", "all"}:
        command_write(args)
    if args.stage in {"screen", "all"}:
        command_screen(args)
    if args.stage in {"verify", "all"}:
        command_verify(args)
    if args.stage in {"materialize", "all"}:
        command_materialize(args)


if __name__ == "__main__":
    main()
