#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_q5_correction_overlay_pipeline import (
    DEFAULT_REVIEWER_MODEL,
    DEFAULT_REVIEWER_PROMPT,
    DEFAULT_WRITER_MODEL,
    VALID_DECISIONS,
    VALID_VERIFIER_DECISIONS,
    VALID_WRITER_TAGS,
    append_jsonl,
    normalize_list,
    provider_config,
    reasoning_config,
    same_summary,
    training_row,
    whitespace_word_count,
    write_json,
)
from synthetic_summary_runtime import (
    build_messages,
    iter_bounded_futures,
    load_dotenv,
    load_jsonl,
    load_prompt_spec,
    local_filter_flags,
    normalize_text,
    request_json_with_retries,
    render_prompt,
    write_jsonl,
)

DEFAULT_PARENT_ROOT = (
    "hf_bucket/experiments/mlx_lora/q5_kimi_qwen_correction_overlay_v1"
)
DEFAULT_OUTPUT_ROOT = (
    "hf_bucket/experiments/mlx_lora/q5_kimi_qwen_correction_overlay_v1/"
    "rejection_repair_v1"
)
DEFAULT_RELEASE_DIR = (
    "hf_dataset/releases/"
    "recent_3y_balanced_10746_q5_kimi_qwen_correction_overlay_repair_v1"
)
DEFAULT_REPAIR_PROMPT = (
    "prompts/synthetic_summary/kimi_k2_6_q5_rejection_repair_v1.json"
)
REPAIR_SOURCE_SPLIT = "q5_kimi_qwen_correction_overlay_repair_v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair Qwen-rejected q5 correction rows and re-verify them."
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "write", "verify", "materialize", "all"),
        default="all",
    )
    parser.add_argument("--parent-root", default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--release-dir", default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--writer-prompt", default=DEFAULT_REPAIR_PROMPT)
    parser.add_argument("--reviewer-prompt", default=DEFAULT_REVIEWER_PROMPT)
    parser.add_argument("--writer-model", default=DEFAULT_WRITER_MODEL)
    parser.add_argument("--reviewer-model", default=DEFAULT_REVIEWER_MODEL)
    parser.add_argument("--writer-workers", type=int, default=4)
    parser.add_argument("--reviewer-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--writer-max-output-tokens", type=int, default=192)
    parser.add_argument("--reviewer-max-output-tokens", type=int, default=512)
    parser.add_argument("--writer-reasoning-effort", default="none")
    parser.add_argument("--reviewer-reasoning-effort", default="none")
    parser.add_argument("--reviewer-provider", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    return parser.parse_args()


def parent_verification_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = Path(args.parent_root) / "qwen_verification_run" / "verifications_ordered.jsonl"
    return load_jsonl(path)


def build_repair_candidate(row: dict[str, Any]) -> dict[str, Any]:
    candidate = {
        **row,
        "parent_candidate_id": row["candidate_id"],
        "candidate_id": f"{row['candidate_id']}:repair1",
        "repair_round": 1,
        "first_correction_writer": row.get("correction_writer", ""),
        "first_correction_model": row.get("correction_model", ""),
        "first_correction_prompt_version": row.get("correction_prompt_version", ""),
        "first_correction_issue_tags": row.get("correction_issue_tags", []),
        "first_correction_evidence_span": row.get("correction_evidence_span", ""),
        "first_correction_summary": row.get("correction_summary", ""),
        "first_correction_reason_span": row.get("correction_reason_span", ""),
        "first_verifier_model": row.get("verifier_model", ""),
        "first_verifier_prompt_version": row.get("verifier_prompt_version", ""),
        "first_verifier_issue_tags": row.get("verifier_issue_tags", []),
        "first_verifier_evidence_span": row.get("verifier_evidence_span", ""),
        "first_verifier_reason_span": row.get("verifier_reason_span", ""),
        "source_split": REPAIR_SOURCE_SPLIT,
    }
    for key in list(candidate):
        if key.startswith("raw_"):
            candidate.pop(key, None)
    return candidate


def command_prepare(args: argparse.Namespace) -> list[dict[str, Any]]:
    rejected = [
        row
        for row in parent_verification_rows(args)
        if row.get("verifier_decision") == "reject"
    ]
    if args.limit is not None:
        rejected = rejected[: args.limit]
    candidates = [build_repair_candidate(row) for row in rejected]
    output_dir = Path(args.output_root) / "repair_candidate_pool"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "candidates.jsonl", candidates)
    write_json(
        output_dir / "run_manifest.json",
        {
            "created_at": utc_now_iso(),
            "parent_root": args.parent_root,
            "candidate_rows": len(candidates),
            "parent_rejection_tag_counts": dict(
                Counter(tag for row in candidates for tag in row.get("first_verifier_issue_tags", []))
            ),
            "outputs": {"candidates": str(output_dir / "candidates.jsonl")},
        },
    )
    print(f"[q5-repair:prepare] wrote {len(candidates)} rejected-row candidates")
    return candidates


def load_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = Path(args.output_root) / "repair_candidate_pool" / "candidates.jsonl"
    if not path.exists():
        return command_prepare(args)
    rows = load_jsonl(path)
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def build_writer_payload(
    candidate: dict[str, Any],
    *,
    prompt_spec: dict[str, Any],
    model: str,
    temperature: float,
    max_output_tokens: int,
    reasoning: dict[str, Any] | None,
) -> dict[str, Any]:
    prompt = render_prompt(
        prompt_spec,
        candidate,
        extra_variables={
            "q5_summary": candidate.get("q5_summary", ""),
            "q5_issue_tags": ", ".join(candidate.get("q5_issue_tags", [])),
            "reviewer_note": candidate.get("reviewer_note", ""),
            "first_correction_summary": candidate.get("first_correction_summary", ""),
            "first_correction_issue_tags": ", ".join(candidate.get("first_correction_issue_tags", [])),
            "first_correction_evidence_span": candidate.get("first_correction_evidence_span", ""),
            "qwen_rejection_tags": ", ".join(candidate.get("first_verifier_issue_tags", [])),
            "qwen_rejection_reason_span": candidate.get("first_verifier_reason_span", ""),
            "qwen_rejection_evidence_span": candidate.get("first_verifier_evidence_span", ""),
        },
    )
    payload = {
        "candidate_id": candidate["candidate_id"],
        "paper_id": candidate["paper_id"],
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "prompt_version": prompt_spec["prompt_version"],
        "messages": build_messages(prompt_spec, prompt),
    }
    if prompt_spec.get("response_format"):
        payload["response_format"] = prompt_spec["response_format"]
    if reasoning:
        payload["reasoning"] = reasoning
    return payload


def writer_result(
    candidate: dict[str, Any],
    parsed: dict[str, Any],
    *,
    prompt_spec: dict[str, Any],
    model: str,
    raw_text: str,
    raw_response: dict[str, Any],
) -> dict[str, Any]:
    decision = str(parsed.get("decision", "")).strip().lower()
    if decision not in VALID_DECISIONS:
        decision = "drop"
    summary = normalize_text(str(parsed.get("summary", "")))
    if decision == "drop":
        summary = ""
    reported_word_count = int(parsed.get("word_count", 0) or 0)
    actual_word_count = whitespace_word_count(summary)
    issue_tags = normalize_list(parsed.get("issue_tags", []), allowed=VALID_WRITER_TAGS)
    if not issue_tags:
        issue_tags = ["none"] if decision == "rewrite" else []
    local_flags = local_filter_flags(summary, str(candidate.get("title", ""))) if summary else []
    format_audit_flags: list[str] = []
    if decision == "rewrite":
        if actual_word_count < 10 or actual_word_count > 15:
            local_flags.append("outside_10_15_words")
        if reported_word_count != actual_word_count:
            format_audit_flags.append("word_count_mismatch")
        if same_summary(summary, str(candidate.get("q5_summary", ""))):
            local_flags.append("copies_q5_failure")
        if same_summary(summary, str(candidate.get("first_correction_summary", ""))):
            local_flags.append("copies_rejected_correction")
    return {
        **candidate,
        "correction_writer": "kimi_repair",
        "correction_model": model,
        "correction_prompt_version": prompt_spec["prompt_version"],
        "correction_decision": decision,
        "correction_issue_tags": issue_tags,
        "correction_evidence_span": normalize_text(str(parsed.get("evidence_span", ""))),
        "correction_summary": summary,
        "correction_reported_word_count": reported_word_count,
        "correction_actual_word_count": actual_word_count,
        "correction_reason_span": normalize_text(str(parsed.get("reason_span", ""))),
        "correction_format_audit_flags": format_audit_flags,
        "correction_local_filter_flags": local_flags,
        "raw_correction_response": parsed,
        "raw_correction_text": raw_text,
        "raw_correction_response_id": raw_response.get("id", ""),
    }


def run_writer_one(
    candidate: dict[str, Any],
    *,
    prompt_spec: dict[str, Any],
    model: str,
    api_base: str,
    api_key: str,
    temperature: float,
    max_output_tokens: int,
    timeout: int,
    max_retries: int,
    retry_backoff_seconds: float,
    attempt_log_dir: Path,
    reasoning: dict[str, Any] | None,
) -> dict[str, Any]:
    parsed, raw_text, raw_response = request_json_with_retries(
        label=f"repair-writer {candidate['candidate_id']}",
        api_base=api_base,
        api_key=api_key,
        model=model,
        messages=build_writer_payload(
            candidate,
            prompt_spec=prompt_spec,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning=reasoning,
        )["messages"],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        response_format=prompt_spec.get("response_format"),
        plugins=prompt_spec.get("plugins"),
        assistant_prefill=prompt_spec.get("assistant_prefill"),
        reasoning=reasoning,
        attempt_log_dir=attempt_log_dir,
    )
    return writer_result(
        candidate,
        parsed,
        prompt_spec=prompt_spec,
        model=model,
        raw_text=raw_text,
        raw_response=raw_response,
    )


def run_writer(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = load_candidates(args)
    output_dir = Path(args.output_root) / "kimi_repair_writer_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_spec = load_prompt_spec(Path(args.writer_prompt))
    reasoning = reasoning_config(args.writer_reasoning_effort)
    payloads = [
        build_writer_payload(
            row,
            prompt_spec=prompt_spec,
            model=args.writer_model,
            temperature=args.temperature,
            max_output_tokens=args.writer_max_output_tokens,
            reasoning=reasoning,
        )
        for row in candidates
    ]
    write_jsonl(output_dir / "request_payloads.jsonl", payloads)
    write_json(
        output_dir / "run_manifest.json",
        {
            "created_at": utc_now_iso(),
            "mode": "prepare_only" if args.prepare_only else "live_run",
            "prompt_path": args.writer_prompt,
            "prompt_version": prompt_spec["prompt_version"],
            "model": args.writer_model,
            "reasoning": reasoning,
            "requested_candidates": len(candidates),
            "workers": args.writer_workers,
        },
    )
    if args.prepare_only:
        print(f"[q5-repair:writer] prepare-only payloads: {len(payloads)}")
        return []

    load_dotenv(Path(".env"))
    api_base = args.api_base or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    output_path = output_dir / "corrections.jsonl"
    existing = load_jsonl(output_path) if output_path.exists() else []
    completed = {row["candidate_id"]: row for row in existing}
    pending = [row for row in candidates if row["candidate_id"] not in completed]
    print(
        f"[q5-repair:writer] writing {len(pending)} pending ({len(existing)} existing) "
        f"with {args.writer_workers} worker(s)...",
        file=sys.stderr,
        flush=True,
    )

    def submit(candidate: dict[str, Any]) -> dict[str, Any]:
        return run_writer_one(
            candidate,
            prompt_spec=prompt_spec,
            model=args.writer_model,
            api_base=api_base,
            api_key=api_key,
            temperature=args.temperature,
            max_output_tokens=args.writer_max_output_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            attempt_log_dir=output_dir / "raw_json_attempts",
            reasoning=reasoning,
        )

    failed_rows: list[dict[str, Any]] = []
    for _, candidate, future in iter_bounded_futures(pending, args.writer_workers, submit):
        try:
            result = future.result()
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failed = {
                "candidate_id": candidate["candidate_id"],
                "paper_id": candidate["paper_id"],
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
            failed_rows.append(failed)
            append_jsonl(output_dir / "failed_corrections.jsonl", failed)
            print(
                f"[q5-repair:writer] failed {candidate['candidate_id']}: {type(exc).__name__}",
                file=sys.stderr,
                flush=True,
            )
            continue
        completed[result["candidate_id"]] = result
        append_jsonl(output_path, result)
        print(
            f"[q5-repair:writer] {len(completed)}/{len(candidates)} "
            f"{result['candidate_id']} {result['correction_decision']} "
            f"flags={','.join(result['correction_local_filter_flags']) or '-'}",
            file=sys.stderr,
            flush=True,
        )
    ordered = [completed[row["candidate_id"]] for row in candidates if row["candidate_id"] in completed]
    write_jsonl(output_dir / "corrections_ordered.jsonl", ordered)
    write_json(
        output_dir / "run_manifest.json",
        {
            "created_at": utc_now_iso(),
            "prompt_path": args.writer_prompt,
            "prompt_version": prompt_spec["prompt_version"],
            "model": args.writer_model,
            "reasoning": reasoning,
            "requested_candidates": len(candidates),
            "completed_corrections": len(ordered),
            "failed_corrections": len(failed_rows),
            "pending_corrections": len(candidates) - len(ordered) - len(failed_rows),
            "decision_counts": dict(Counter(row["correction_decision"] for row in ordered)),
            "local_filter_counts": dict(
                Counter(flag for row in ordered for flag in row["correction_local_filter_flags"])
            ),
        },
    )
    return ordered


def collect_local_eligible(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_dir = Path(args.output_root) / "local_eligible_for_qwen_repair_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    corrections = load_jsonl(
        Path(args.output_root) / "kimi_repair_writer_run" / "corrections_ordered.jsonl"
    )
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in corrections:
        if row.get("correction_decision") == "rewrite" and not row.get("correction_local_filter_flags"):
            eligible.append(row)
        else:
            excluded.append(row)
    write_jsonl(output_dir / "corrections_ordered.jsonl", eligible)
    write_jsonl(output_dir / "excluded.jsonl", excluded)
    write_json(
        output_dir / "run_manifest.json",
        {
            "created_at": utc_now_iso(),
            "eligible_rows": len(eligible),
            "excluded_rows": len(excluded),
            "excluded_local_filter_counts": dict(
                Counter(flag for row in excluded for flag in row.get("correction_local_filter_flags", []))
            ),
        },
    )
    return eligible


def build_reviewer_payload(
    correction: dict[str, Any],
    *,
    prompt_spec: dict[str, Any],
    model: str,
    temperature: float,
    max_output_tokens: int,
    reasoning: dict[str, Any] | None,
) -> dict[str, Any]:
    prompt = render_prompt(
        prompt_spec,
        correction,
        extra_variables={
            "q5_summary": correction.get("q5_summary", ""),
            "q5_issue_tags": ", ".join(correction.get("q5_issue_tags", [])),
            "reviewer_note": correction.get("reviewer_note", ""),
            "correction_writer": correction.get("correction_writer", ""),
            "correction_summary": correction.get("correction_summary", ""),
            "correction_issue_tags": ", ".join(correction.get("correction_issue_tags", [])),
            "correction_evidence_span": correction.get("correction_evidence_span", ""),
            "correction_reason_span": correction.get("correction_reason_span", ""),
        },
    )
    payload = {
        "candidate_id": correction["candidate_id"],
        "paper_id": correction["paper_id"],
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "prompt_version": prompt_spec["prompt_version"],
        "messages": build_messages(prompt_spec, prompt),
    }
    if prompt_spec.get("response_format"):
        payload["response_format"] = prompt_spec["response_format"]
    if reasoning:
        payload["reasoning"] = reasoning
    return payload


def reviewer_result(
    correction: dict[str, Any],
    parsed: dict[str, Any],
    *,
    prompt_spec: dict[str, Any],
    model: str,
    raw_text: str,
    raw_response: dict[str, Any],
) -> dict[str, Any]:
    decision = str(parsed.get("decision", "")).strip().lower()
    if decision not in VALID_VERIFIER_DECISIONS:
        decision = "reject"
    issue_tags = normalize_list(parsed.get("issue_tags", []), allowed=VALID_WRITER_TAGS)
    if not issue_tags:
        issue_tags = ["none"] if decision == "accept" else []
    return {
        **correction,
        "verifier_model": model,
        "verifier_prompt_version": prompt_spec["prompt_version"],
        "verifier_decision": decision,
        "verifier_issue_tags": issue_tags,
        "verifier_evidence_span": normalize_text(str(parsed.get("evidence_span", ""))),
        "verifier_reason_span": normalize_text(str(parsed.get("reason_span", ""))),
        "raw_verifier_response": parsed,
        "raw_verifier_text": raw_text,
        "raw_verifier_response_id": raw_response.get("id", ""),
    }


def run_reviewer_one(
    correction: dict[str, Any],
    *,
    prompt_spec: dict[str, Any],
    model: str,
    api_base: str,
    api_key: str,
    temperature: float,
    max_output_tokens: int,
    timeout: int,
    max_retries: int,
    retry_backoff_seconds: float,
    attempt_log_dir: Path,
    reasoning: dict[str, Any] | None,
    provider: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = build_reviewer_payload(
        correction,
        prompt_spec=prompt_spec,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
    )
    parsed, raw_text, raw_response = request_json_with_retries(
        label=f"repair-verifier {correction['candidate_id']}",
        api_base=api_base,
        api_key=api_key,
        model=model,
        messages=payload["messages"],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        response_format=prompt_spec.get("response_format"),
        plugins=prompt_spec.get("plugins"),
        assistant_prefill=prompt_spec.get("assistant_prefill"),
        reasoning=reasoning,
        provider=provider,
        attempt_log_dir=attempt_log_dir,
    )
    return reviewer_result(
        correction,
        parsed,
        prompt_spec=prompt_spec,
        model=model,
        raw_text=raw_text,
        raw_response=raw_response,
    )


def run_reviewer(args: argparse.Namespace) -> list[dict[str, Any]]:
    corrections = collect_local_eligible(args)
    output_dir = Path(args.output_root) / "qwen_repair_verification_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_spec = load_prompt_spec(Path(args.reviewer_prompt))
    reasoning = reasoning_config(args.reviewer_reasoning_effort)
    provider = provider_config(args.reviewer_provider)
    payloads = [
        build_reviewer_payload(
            row,
            prompt_spec=prompt_spec,
            model=args.reviewer_model,
            temperature=args.temperature,
            max_output_tokens=args.reviewer_max_output_tokens,
            reasoning=reasoning,
        )
        for row in corrections
    ]
    write_jsonl(output_dir / "request_payloads.jsonl", payloads)
    write_json(
        output_dir / "run_manifest.json",
        {
            "created_at": utc_now_iso(),
            "mode": "prepare_only" if args.prepare_only else "live_run",
            "prompt_path": args.reviewer_prompt,
            "prompt_version": prompt_spec["prompt_version"],
            "model": args.reviewer_model,
            "reasoning": reasoning,
            "provider": provider,
            "requested_verifications": len(corrections),
            "workers": args.reviewer_workers,
        },
    )
    if args.prepare_only:
        print(f"[q5-repair:reviewer] prepare-only payloads: {len(payloads)}")
        return []

    load_dotenv(Path(".env"))
    api_base = args.api_base or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    output_path = output_dir / "verifications.jsonl"
    existing = load_jsonl(output_path) if output_path.exists() else []
    completed = {row["candidate_id"]: row for row in existing}
    pending = [row for row in corrections if row["candidate_id"] not in completed]
    print(
        f"[q5-repair:reviewer] verifying {len(pending)} pending ({len(existing)} existing) "
        f"with {args.reviewer_workers} worker(s)...",
        file=sys.stderr,
        flush=True,
    )

    def submit(correction: dict[str, Any]) -> dict[str, Any]:
        return run_reviewer_one(
            correction,
            prompt_spec=prompt_spec,
            model=args.reviewer_model,
            api_base=api_base,
            api_key=api_key,
            temperature=args.temperature,
            max_output_tokens=args.reviewer_max_output_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            attempt_log_dir=output_dir / "raw_json_attempts",
            reasoning=reasoning,
            provider=provider,
        )

    failed_rows: list[dict[str, Any]] = []
    for _, correction, future in iter_bounded_futures(pending, args.reviewer_workers, submit):
        try:
            result = future.result()
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failed = {
                "candidate_id": correction["candidate_id"],
                "paper_id": correction["paper_id"],
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
            failed_rows.append(failed)
            append_jsonl(output_dir / "failed_verifications.jsonl", failed)
            print(
                f"[q5-repair:reviewer] failed {correction['candidate_id']}: {type(exc).__name__}",
                file=sys.stderr,
                flush=True,
            )
            continue
        completed[result["candidate_id"]] = result
        append_jsonl(output_path, result)
        print(
            f"[q5-repair:reviewer] {len(completed)}/{len(corrections)} "
            f"{result['candidate_id']} {result['verifier_decision']}",
            file=sys.stderr,
            flush=True,
        )
    ordered = [completed[row["candidate_id"]] for row in corrections if row["candidate_id"] in completed]
    write_jsonl(output_dir / "verifications_ordered.jsonl", ordered)
    write_json(
        output_dir / "run_manifest.json",
        {
            "created_at": utc_now_iso(),
            "prompt_path": args.reviewer_prompt,
            "prompt_version": prompt_spec["prompt_version"],
            "model": args.reviewer_model,
            "reasoning": reasoning,
            "provider": provider,
            "requested_verifications": len(corrections),
            "completed_verifications": len(ordered),
            "failed_verifications": len(failed_rows),
            "pending_verifications": len(corrections) - len(ordered) - len(failed_rows),
            "decision_counts": dict(Counter(row["verifier_decision"] for row in ordered)),
            "issue_tag_counts": dict(
                Counter(tag for row in ordered for tag in row.get("verifier_issue_tags", []))
            ),
        },
    )
    return ordered


def make_training_row(row: dict[str, Any], *, source_split: str) -> dict[str, Any]:
    result = training_row(row)
    result["source_split"] = source_split
    if row.get("parent_candidate_id"):
        result["q5_correction_parent_candidate_id"] = row.get("parent_candidate_id")
        result["repair_round"] = row.get("repair_round")
        result["first_correction_summary"] = row.get("first_correction_summary")
        result["first_verifier_issue_tags"] = row.get("first_verifier_issue_tags", [])
        result["first_verifier_reason_span"] = row.get("first_verifier_reason_span", "")
    return result


def command_materialize(args: argparse.Namespace) -> None:
    parent_rows = parent_verification_rows(args)
    original_accepted = [row for row in parent_rows if row.get("verifier_decision") == "accept"]
    repair_verification_path = (
        Path(args.output_root) / "qwen_repair_verification_run" / "verifications_ordered.jsonl"
    )
    repair_rows = load_jsonl(repair_verification_path)
    repair_accepted = [row for row in repair_rows if row.get("verifier_decision") == "accept"]
    repair_rejected = [row for row in repair_rows if row.get("verifier_decision") != "accept"]
    release_dir = Path(args.release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)
    materialized_dir = Path(args.output_root) / "materialized_release"
    materialized_dir.mkdir(parents=True, exist_ok=True)
    training_rows = [
        *[
            make_training_row(row, source_split="q5_kimi_qwen_correction_overlay_v1")
            for row in original_accepted
        ],
        *[make_training_row(row, source_split=REPAIR_SOURCE_SPLIT) for row in repair_accepted],
    ]
    write_jsonl(release_dir / "training_rows.jsonl", training_rows)
    write_jsonl(materialized_dir / "original_selected_review_rows.jsonl", original_accepted)
    write_jsonl(materialized_dir / "repaired_selected_review_rows.jsonl", repair_accepted)
    write_jsonl(materialized_dir / "repaired_excluded_review_rows.jsonl", repair_rejected)
    write_jsonl(materialized_dir / "combined_selected_review_rows.jsonl", [*original_accepted, *repair_accepted])
    manifest = {
        "created_at": utc_now_iso(),
        "parent_root": args.parent_root,
        "release_dir": str(release_dir),
        "original_accepted_rows": len(original_accepted),
        "repair_reviewed_rows": len(repair_rows),
        "repair_accepted_rows": len(repair_accepted),
        "repair_rejected_rows": len(repair_rejected),
        "combined_training_rows": len(training_rows),
        "repair_acceptance_rate": round(len(repair_accepted) / len(repair_rows), 4)
        if repair_rows
        else 0,
        "outputs": {
            "training_rows": str(release_dir / "training_rows.jsonl"),
            "original_selected_review_rows": str(
                materialized_dir / "original_selected_review_rows.jsonl"
            ),
            "repaired_selected_review_rows": str(
                materialized_dir / "repaired_selected_review_rows.jsonl"
            ),
            "repaired_excluded_review_rows": str(
                materialized_dir / "repaired_excluded_review_rows.jsonl"
            ),
            "combined_selected_review_rows": str(
                materialized_dir / "combined_selected_review_rows.jsonl"
            ),
        },
    }
    write_json(release_dir / "run_manifest.json", manifest)
    write_json(materialized_dir / "run_manifest.json", manifest)
    print(
        f"[q5-repair:materialize] original={len(original_accepted)} "
        f"repair_accepted={len(repair_accepted)} combined={len(training_rows)} "
        f"release={release_dir}"
    )


def main() -> None:
    args = parse_args()
    if args.stage in {"prepare", "all"}:
        command_prepare(args)
    if args.stage in {"write", "all"}:
        run_writer(args)
    if args.stage in {"verify", "all"}:
        run_reviewer(args)
    if args.stage in {"materialize", "all"} and not args.prepare_only:
        command_materialize(args)


if __name__ == "__main__":
    main()
