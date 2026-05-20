#!/usr/bin/env python3
"""Repair manually reviewed q8-vs-q5 contrastive failures with DeepSeek/Qwen."""
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
    "hf_bucket/experiments/sft/q5_promoted218_vs_q8_contrastive_mine_10000_v1/"
    "manual_review_hp_and_fog_union_gpt54mini/manual_review_merged.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "hf_bucket/experiments/sft/q5_promoted218_vs_q8_contrastive_mine_10000_v1/"
    "deepseek_v4_qwen_repair_v3"
)
DEFAULT_WRITER_PROMPT = Path(
    "prompts/synthetic_summary/deepseek_v4_pro_q5_q8_contrastive_repair_writer_v3.json"
)
DEFAULT_REVIEWER_PROMPT = Path(
    "prompts/synthetic_summary/qwen3_235b_a22b_2507_q5_q8_contrastive_repair_verifier_v2.json"
)
DEFAULT_WRITER_MODEL = "deepseek/deepseek-v4-pro"
DEFAULT_REVIEWER_MODEL = "qwen/qwen3-235b-a22b-2507"
VALID_WRITER_DECISIONS = {"rewrite", "drop"}
VALID_REVIEWER_DECISIONS = {"accept", "reject"}
APPEND_LOCK = threading.Lock()

DIGIT_RE = re.compile(r"\d")
HTML_TAG_RE = re.compile(r"<[^>]+>")
FORMULA_SYMBOL_RE = re.compile(
    r"[$<>≤≥≈∑∫√∞πμσλβγθφψωΔΩαηχτ]|\\[A-Za-z]+|\^|_|[{}]"
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
KNOWN_BAD_RE = re.compile(
    r"\b(?:bayron|curravariance|ather-proof|sterochastic|parametriding|nevyzed)\b",
    re.IGNORECASE,
)


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
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def norm_list(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value or "")


def done_ids(path: Path, key: str = "candidate_id") -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get(key):
            ids.add(str(row[key]))
    return ids


def reasoning_config(effort: str | None) -> dict[str, Any] | None:
    effort = str(effort or "").strip()
    if not effort:
        return None
    return {"effort": effort, "exclude": True}


def provider_config(provider_name: str | None) -> dict[str, Any] | None:
    name = str(provider_name or "").strip()
    if not name:
        return None
    return {"order": [name], "allow_fallbacks": False}


def hard_filter_flags(
    summary: str,
    title: str,
    bad_summary: str = "",
    q8_summary: str = "",
) -> list[str]:
    text = normalize_text(summary)
    lowered = text.lower()
    flags: list[str] = []
    flags.extend(local_filter_flags(text, title))
    wc = word_count(text)
    if wc < 10 or wc > 15:
        flags.append("outside_10_15_words")
    if sentence_count(text) != 1:
        flags.append("not_single_sentence")
    if has_digit(text):
        flags.append("contains_digit")
    if HTML_TAG_RE.search(text) or FORMULA_SYMBOL_RE.search(text):
        flags.append("formula_or_symbol")
    if BENCHMARK_RE.search(text):
        flags.append("hidden_benchmark_or_superlative")
    if BAD_OPENING_RE.search(lowered):
        flags.append("bad_opening")
    if ADJACENT_DUP_RE.search(text):
        flags.append("adjacent_duplicate_word")
    if KNOWN_BAD_RE.search(text):
        flags.append("known_malformed_term")
    if ";" in text:
        flags.append("semicolon")
    if bad_summary and normalize_text(bad_summary).casefold() == text.casefold():
        flags.append("copies_q5_failure")
    if q8_summary and normalize_text(q8_summary).casefold() == text.casefold():
        flags.append("copies_q8_summary")
    if title_overlap_ratio(text, title) >= 0.8:
        flags.append("high_title_overlap")
    return sorted(set(flags))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run q8-vs-q5 contrastive DeepSeek/Qwen repair pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "write", "screen", "verify", "materialize", "all"],
        default="all",
    )
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
    parser.add_argument("--writer-max-output-tokens", type=int, default=280)
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
    review_rows.sort(
        key=lambda row: (
            str(row.get("source_bucket", "")),
            str(row.get("failure_type", "")),
            str(row.get("review_id", "")),
        )
    )
    if args.limit is not None:
        review_rows = review_rows[: args.limit]
    candidates: list[dict[str, Any]] = []
    for idx, row in enumerate(review_rows, start=1):
        candidates.append(
            {
                "candidate_id": f"q5_q8_contrastive_repair:{idx:04d}",
                "paper_id": str(row["paper_id"]),
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
                "primary_category": row.get("primary_category", ""),
                "q8_summary": row.get("q8_summary", ""),
                "bad_summary": row.get("q5_summary", ""),
                "failure_family": row.get("failure_type", ""),
                "review_severity": row.get("severity", ""),
                "reviewer_note": row.get("note", ""),
                "source_bucket": row.get("source_bucket", ""),
                "source_review_id": row.get("review_id", ""),
                "heuristic_score": row.get("score"),
                "heuristic_flags": norm_list(row.get("flags", [])),
                "missing_q8_source_terms": norm_list(row.get("missing_q8_source_terms", [])),
                "q5_unanchored_terms": norm_list(row.get("q5_unanchored_terms", [])),
                "manual_preferred": row.get("preferred", ""),
                "repair_source": "q5_q8_contrastive_manual_review_actual_failure_v1",
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
            "source_bucket_counts": dict(collections.Counter(row["source_bucket"] for row in candidates)),
            "failure_family_counts": dict(collections.Counter(row["failure_family"] for row in candidates)),
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
            "q8_summary": row.get("q8_summary", ""),
            "bad_summary": row.get("bad_summary", ""),
            "failure_family": row.get("failure_family", ""),
            "review_severity": row.get("review_severity", ""),
            "reviewer_note": row.get("reviewer_note", ""),
            "source_bucket": row.get("source_bucket", ""),
            "heuristic_flags": row.get("heuristic_flags", ""),
            "missing_q8_source_terms": row.get("missing_q8_source_terms", ""),
            "q5_unanchored_terms": row.get("q5_unanchored_terms", ""),
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
        flags = (
            hard_filter_flags(
                summary,
                candidate.get("title", ""),
                candidate.get("bad_summary", ""),
                candidate.get("q8_summary", ""),
            )
            if summary
            else ["empty_summary"]
        )
        return {
            **candidate,
            "writer_model": args.writer_model,
            "writer_prompt_version": spec["prompt_version"],
            "writer_decision": decision,
            "writer_issue_tags": parsed.get("issue_tags", []),
            "grounding_policy": parsed.get("grounding_policy", ""),
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
                if idx % 10 == 0 or idx == len(todo):
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
            "hard_filter_flag_counts": dict(collections.Counter(flag for row in all_rows for flag in row.get("writer_hard_filter_flags", []))),
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
    rejected_ids = {row["candidate_id"] for row in eligible}
    rejected = [row for row in writer_rows if row["candidate_id"] not in rejected_ids]
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
            "hard_filter_flag_counts": dict(collections.Counter(flag for row in rejected for flag in row.get("writer_hard_filter_flags", []))),
        },
    )
    print(f"[screen] eligible_for_qwen={len(eligible)} rejected_before_qwen={len(rejected)}")
    return eligible


def reviewer_payload(row: dict[str, Any], spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    prompt = render_prompt(
        spec,
        row,
        extra_variables={
            "q8_summary": row.get("q8_summary", ""),
            "bad_summary": row.get("bad_summary", ""),
            "failure_family": row.get("failure_family", ""),
            "review_severity": row.get("review_severity", ""),
            "reviewer_note": row.get("reviewer_note", ""),
            "source_bucket": row.get("source_bucket", ""),
            "writer_decision": row.get("writer_decision", ""),
            "writer_issue_tags": ", ".join(row.get("writer_issue_tags", [])),
            "grounding_policy": row.get("grounding_policy", ""),
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
        hard_flags = hard_filter_flags(
            row.get("repair_summary", ""),
            row.get("title", ""),
            row.get("bad_summary", ""),
            row.get("q8_summary", ""),
        )
        if hard_flags:
            decision = "reject"
        return {
            **row,
            "reviewer_model": args.reviewer_model,
            "reviewer_prompt_version": spec["prompt_version"],
            "verifier_decision": decision,
            "verifier_issue_tags": parsed.get("issue_tags", []),
            "grounding_ok": bool(parsed.get("grounding_ok", False)) and not hard_flags,
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
                if idx % 10 == 0 or idx == len(todo):
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
            "post_qwen_hard_filter_counts": dict(collections.Counter(flag for row in all_rows for flag in row.get("post_qwen_hard_filter_flags", []))),
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
        "source_split": "q5_q8_contrastive_deepseek_qwen_repair_v3",
        "patch_component": "q8_contrastive_grounding_repair",
        "repair_candidate_id": row.get("candidate_id", ""),
        "manual_failure_type": row.get("failure_family", ""),
        "manual_review_severity": row.get("review_severity", ""),
        "manual_review_note": row.get("reviewer_note", ""),
        "source_bucket": row.get("source_bucket", ""),
        "source_review_id": row.get("source_review_id", ""),
        "q8_summary": row.get("q8_summary", ""),
        "q5_promoted218_summary": row.get("bad_summary", ""),
        "writer_evidence_span": row.get("writer_evidence_span", ""),
        "verifier_evidence_span": row.get("verifier_evidence_span", ""),
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
        and not hard_filter_flags(
            row.get("repair_summary", ""),
            row.get("title", ""),
            row.get("bad_summary", ""),
            row.get("q8_summary", ""),
        )
    ]
    accepted_ids = {row["candidate_id"] for row in accepted}
    rejected = [row for row in verified if row["candidate_id"] not in accepted_ids]
    out_dir = args.output_root / "materialized"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "accepted_repairs.jsonl", accepted)
    write_jsonl(out_dir / "rejected_after_qwen.jsonl", rejected)
    write_jsonl(out_dir / "rejected_before_qwen.jsonl", local_rejected)
    write_jsonl(out_dir / "training_rows.jsonl", [training_row(row) for row in accepted])
    manifest = {
        "created_at": utc_now(),
        "candidate_failures": len(load_candidates(args)),
        "verified_rows": len(verified),
        "accepted_rows": len(accepted),
        "rejected_after_qwen": len(rejected),
        "rejected_before_qwen": len(local_rejected),
        "accepted_source_bucket_counts": dict(collections.Counter(row.get("source_bucket") for row in accepted)),
        "accepted_failure_family_counts": dict(collections.Counter(row.get("failure_family") for row in accepted)),
        "qwen_decision_counts": dict(collections.Counter(row.get("verifier_decision") for row in verified)),
        "qwen_issue_tag_counts": dict(collections.Counter(tag for row in verified for tag in row.get("verifier_issue_tags", []))),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    lines = [
        "# q5/q8 Contrastive DeepSeek Repair Summary",
        "",
        f"- Candidate failures: {manifest['candidate_failures']}",
        f"- Local-screened before Qwen: {manifest['rejected_before_qwen']}",
        f"- Qwen verified: {manifest['verified_rows']}",
        f"- Accepted repairs: {manifest['accepted_rows']}",
        f"- Rejected after Qwen: {manifest['rejected_after_qwen']}",
        "",
        "## Accepted By Source Bucket",
        "",
        "| bucket | rows |",
        "| --- | ---: |",
    ]
    for bucket, count in sorted(collections.Counter(row.get("source_bucket") for row in accepted).items()):
        lines.append(f"| {bucket} | {count} |")
    lines.extend(["", "## Accepted By Failure Type", "", "| type | rows |", "| --- | ---: |"])
    for family, count in sorted(collections.Counter(row.get("failure_family") for row in accepted).items()):
        lines.append(f"| {family} | {count} |")
    (out_dir / "CONTRASTIVE_REPAIR_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[materialize] accepted={len(accepted)} "
        f"rejected_after_qwen={len(rejected)} rejected_before_qwen={len(local_rejected)}"
    )
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
