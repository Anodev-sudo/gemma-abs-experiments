#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    word_count,
    write_jsonl,
)

DEFAULT_INPUT = (
    "hf_bucket/experiments/mlx_lora/q5_hardening_distill_v1/"
    "identified_rows/q5_severe.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    "hf_bucket/experiments/mlx_lora/q5_kimi_qwen_correction_overlay_v1"
)
DEFAULT_RELEASE_DIR = (
    "hf_dataset/releases/recent_3y_balanced_10746_q5_kimi_qwen_correction_overlay_v1"
)
DEFAULT_WRITER_PROMPT = (
    "prompts/synthetic_summary/kimi_k2_6_q5_correction_rewriter_v1.json"
)
DEFAULT_REVIEWER_PROMPT = (
    "prompts/synthetic_summary/qwen3_235b_a22b_2507_q5_correction_verifier_v1.json"
)
DEFAULT_WRITER_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_REVIEWER_MODEL = "qwen/qwen3-235b-a22b-2507"
DEFAULT_SEED = 20260507
DEFAULT_SOURCE_SPLIT = "q5_kimi_qwen_correction_overlay_v1"
DEFAULT_CANDIDATE_PREFIX = "q5_kimi_qwen"
VALID_WRITER_TAGS = {
    "invented_number",
    "wrong_number_binding",
    "numeric_error",
    "unsupported_number",
    "unsupported_entity",
    "unsupported_entity_label",
    "unsupported_mechanism",
    "unsupported_causal_claim",
    "overstrong_claim",
    "polarity_flip",
    "polarity_risk",
    "misses_point",
    "malformed_repetition",
    "non_answer",
    "title_rehash",
    "too_generic",
    "copies_q5_failure",
    "style_drift",
    "local_filter",
    "none",
}
VALID_DECISIONS = {"rewrite", "drop"}
VALID_VERIFIER_DECISIONS = {"accept", "reject"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def normalize_list(value: Any, *, allowed: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        tag = str(item).strip()
        if not tag or tag not in allowed or tag in seen:
            continue
        seen.add(tag)
        result.append(tag)
    return result


def whitespace_word_count(value: str) -> int:
    return len([part for part in normalize_text(value).split(" ") if part])


def same_summary(left: str, right: str) -> bool:
    return normalize_text(left).casefold() == normalize_text(right).casefold()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Kimi writer and Qwen verifier for q5 correction overlay rows."
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "write", "verify", "materialize", "all"),
        default="all",
    )
    parser.add_argument("--input-jsonl", default=DEFAULT_INPUT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--release-dir", default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--writer-prompt", default=DEFAULT_WRITER_PROMPT)
    parser.add_argument("--reviewer-prompt", default=DEFAULT_REVIEWER_PROMPT)
    parser.add_argument("--writer-model", default=DEFAULT_WRITER_MODEL)
    parser.add_argument("--reviewer-model", default=DEFAULT_REVIEWER_MODEL)
    parser.add_argument("--writer-workers", type=int, default=4)
    parser.add_argument("--reviewer-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--writer-max-output-tokens", type=int, default=192)
    parser.add_argument("--reviewer-max-output-tokens", type=int, default=192)
    parser.add_argument("--writer-reasoning-effort", default="none")
    parser.add_argument("--reviewer-reasoning-effort", default="none")
    parser.add_argument("--reviewer-provider", default=None)
    parser.add_argument("--reviewer-disable-response-format", action="store_true")
    parser.add_argument("--source-split", default=DEFAULT_SOURCE_SPLIT)
    parser.add_argument("--candidate-prefix", default=DEFAULT_CANDIDATE_PREFIX)
    parser.add_argument("--q5-severity-source", default="blind_review:q5_severe")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def reasoning_config(effort: str | None) -> dict[str, Any] | None:
    effort = str(effort or "").strip()
    if not effort:
        return None
    return {"effort": effort, "exclude": True}


def provider_config(provider_name: str | None) -> dict[str, Any] | None:
    provider_name = str(provider_name or "").strip()
    if not provider_name:
        return None
    return {"order": [provider_name], "allow_fallbacks": False}


def build_candidate(row: dict[str, Any], index: int, args: argparse.Namespace) -> dict[str, Any]:
    issue_tags = [str(tag) for tag in row.get("q5_issue_tags", []) if str(tag)]
    return {
        "candidate_id": f"{args.candidate_prefix}:{index:04d}",
        "paper_id": row["paper_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "primary_category": row.get("primary_category"),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date"),
        "selection_bucket": row.get("selection_bucket"),
        "q5_hardening_axes": row.get("q5_hardening_axes", []),
        "q5_hardening_review_id": row.get("review_id"),
        "q5_summary": row.get("q5_summary", ""),
        "q8_summary": row.get("q8_summary", ""),
        "q8_clean": row.get("q8_clean"),
        "q8_severe": row.get("q8_severe"),
        "q8_mangled": row.get("q8_mangled"),
        "q5_issue_tags": issue_tags,
        "q5_severity_source": row.get("q5_severity_source") or args.q5_severity_source,
        "reviewer_note": row.get("note", ""),
        "source_split": row.get("source_split") or args.source_split,
    }


def command_prepare(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = load_jsonl(Path(args.input_jsonl))
    if args.limit is not None:
        rows = rows[: args.limit]
    candidates = [build_candidate(row, index, args) for index, row in enumerate(rows, start=1)]
    output_dir = Path(args.output_root) / "candidate_pool"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "candidates.jsonl", candidates)
    write_json(
        output_dir / "run_manifest.json",
        {
            "created_at": utc_now_iso(),
            "input_jsonl": args.input_jsonl,
            "output_jsonl": str(output_dir / "candidates.jsonl"),
            "candidate_rows": len(candidates),
            "source_policy": args.source_split,
            "q8_clean_counts": dict(Counter(str(row.get("q8_clean")) for row in candidates)),
            "selection_bucket_counts": dict(
                Counter(row.get("selection_bucket") for row in candidates)
            ),
            "q5_issue_tag_counts": dict(
                Counter(tag for row in candidates for tag in row.get("q5_issue_tags", []))
            ),
        },
    )
    print(f"[q5-corrections:prepare] wrote {len(candidates)} candidates")
    return candidates


def load_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidate_path = Path(args.output_root) / "candidate_pool" / "candidates.jsonl"
    if not candidate_path.exists():
        return command_prepare(args)
    candidates = load_jsonl(candidate_path)
    if args.limit is not None:
        candidates = candidates[: args.limit]
    return candidates


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
            "q5_severity_source": candidate.get("q5_severity_source", ""),
            "q5_issue_tags": ", ".join(candidate.get("q5_issue_tags", [])),
            "reviewer_note": candidate.get("reviewer_note", ""),
            "q5_summary": candidate.get("q5_summary", ""),
            "q8_summary": candidate.get("q8_summary", ""),
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
    if prompt_spec.get("plugins"):
        payload["plugins"] = prompt_spec["plugins"]
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
    return {
        **candidate,
        "correction_writer": "kimi",
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
    provider: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = build_writer_payload(
        candidate,
        prompt_spec=prompt_spec,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
    )
    parsed, raw_text, raw_response = request_json_with_retries(
        label=f"writer {candidate['candidate_id']}",
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
    output_dir = Path(args.output_root) / "kimi_writer_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_spec = load_prompt_spec(Path(args.writer_prompt))
    reasoning = reasoning_config(args.writer_reasoning_effort)
    request_payloads = [
        build_writer_payload(
            candidate,
            prompt_spec=prompt_spec,
            model=args.writer_model,
            temperature=args.temperature,
            max_output_tokens=args.writer_max_output_tokens,
            reasoning=reasoning,
        )
        for candidate in candidates
    ]
    write_jsonl(output_dir / "request_payloads.jsonl", request_payloads)
    manifest = {
        "created_at": utc_now_iso(),
        "mode": "prepare_only" if args.prepare_only else "live_run",
        "candidates_jsonl": str(Path(args.output_root) / "candidate_pool" / "candidates.jsonl"),
        "output_dir": str(output_dir),
        "prompt_path": args.writer_prompt,
        "prompt_version": prompt_spec["prompt_version"],
        "model": args.writer_model,
        "temperature": args.temperature,
        "max_output_tokens": args.writer_max_output_tokens,
        "workers": args.writer_workers,
        "reasoning": reasoning,
        "requested_candidates": len(candidates),
        "outputs": {
            "request_payloads": "request_payloads.jsonl",
            "corrections": "corrections.jsonl",
            "corrections_ordered": "corrections_ordered.jsonl",
            "failed_corrections": "failed_corrections.jsonl",
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    if args.prepare_only:
        print(f"[q5-corrections:writer] prepare-only payloads: {len(request_payloads)}")
        return []

    load_dotenv(Path(".env"))
    api_base = args.api_base or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for live writer run")
    output_path = output_dir / "corrections.jsonl"
    existing = load_jsonl(output_path) if output_path.exists() else []
    completed = {row["candidate_id"]: row for row in existing}
    pending = [row for row in candidates if row["candidate_id"] not in completed]
    print(
        f"[q5-corrections:writer] writing {len(pending)} pending "
        f"({len(existing)} existing) with {args.writer_workers} worker(s)...",
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
                f"[q5-corrections:writer] failed {candidate['candidate_id']}: {type(exc).__name__}",
                file=sys.stderr,
                flush=True,
            )
            continue
        completed[result["candidate_id"]] = result
        append_jsonl(output_path, result)
        print(
            f"[q5-corrections:writer] {len(completed)}/{len(candidates)} "
            f"{result['candidate_id']} {result['correction_decision']} "
            f"flags={','.join(result['correction_local_filter_flags']) or '-'}",
            file=sys.stderr,
            flush=True,
        )
    ordered = [completed[row["candidate_id"]] for row in candidates if row["candidate_id"] in completed]
    write_jsonl(output_dir / "corrections_ordered.jsonl", ordered)
    manifest.update(
        {
            "completed_corrections": len(ordered),
            "failed_corrections": len(failed_rows),
            "pending_corrections": len(candidates) - len(ordered) - len(failed_rows),
            "decision_counts": dict(Counter(row["correction_decision"] for row in ordered)),
            "local_filter_counts": dict(
                Counter(flag for row in ordered for flag in row["correction_local_filter_flags"])
            ),
        }
    )
    write_json(output_dir / "run_manifest.json", manifest)
    return ordered


def collect_local_eligible(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_dir = Path(args.output_root) / "local_eligible_for_qwen_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    corrections = load_jsonl(Path(args.output_root) / "kimi_writer_run" / "corrections_ordered.jsonl")
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
            "input_corrections": str(Path(args.output_root) / "kimi_writer_run" / "corrections_ordered.jsonl"),
            "eligible_rows": len(eligible),
            "excluded_rows": len(excluded),
            "excluded_decision_counts": dict(Counter(row.get("correction_decision") for row in excluded)),
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
            "q8_summary": correction.get("q8_summary", ""),
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
    if prompt_spec.get("plugins"):
        payload["plugins"] = prompt_spec["plugins"]
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
    provider: dict[str, Any] | None = None,
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
        label=f"verifier {correction['candidate_id']}",
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
    output_dir = Path(args.output_root) / "qwen_verification_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_spec = load_prompt_spec(Path(args.reviewer_prompt))
    if args.reviewer_disable_response_format:
        prompt_spec = dict(prompt_spec)
        prompt_spec.pop("response_format", None)
    reasoning = reasoning_config(args.reviewer_reasoning_effort)
    provider = provider_config(args.reviewer_provider)
    request_payloads = [
        build_reviewer_payload(
            correction,
            prompt_spec=prompt_spec,
            model=args.reviewer_model,
            temperature=args.temperature,
            max_output_tokens=args.reviewer_max_output_tokens,
            reasoning=reasoning,
        )
        for correction in corrections
    ]
    write_jsonl(output_dir / "request_payloads.jsonl", request_payloads)
    manifest = {
        "created_at": utc_now_iso(),
        "mode": "prepare_only" if args.prepare_only else "live_run",
        "corrections_jsonl": str(
            Path(args.output_root)
            / "local_eligible_for_qwen_verification"
            / "corrections_ordered.jsonl"
        ),
        "output_dir": str(output_dir),
        "prompt_path": args.reviewer_prompt,
        "prompt_version": prompt_spec["prompt_version"],
        "model": args.reviewer_model,
        "temperature": args.temperature,
        "max_output_tokens": args.reviewer_max_output_tokens,
        "workers": args.reviewer_workers,
        "reasoning": reasoning,
        "provider": provider,
        "requested_verifications": len(corrections),
        "outputs": {
            "request_payloads": "request_payloads.jsonl",
            "verifications": "verifications.jsonl",
            "verifications_ordered": "verifications_ordered.jsonl",
            "failed_verifications": "failed_verifications.jsonl",
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    if args.prepare_only:
        print(f"[q5-corrections:reviewer] prepare-only payloads: {len(request_payloads)}")
        return []

    load_dotenv(Path(".env"))
    api_base = args.api_base or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for live reviewer run")
    output_path = output_dir / "verifications.jsonl"
    existing = load_jsonl(output_path) if output_path.exists() else []
    completed = {row["candidate_id"]: row for row in existing}
    pending = [row for row in corrections if row["candidate_id"] not in completed]
    print(
        f"[q5-corrections:reviewer] verifying {len(pending)} pending "
        f"({len(existing)} existing) with {args.reviewer_workers} worker(s)...",
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
                f"[q5-corrections:reviewer] failed {correction['candidate_id']}: {type(exc).__name__}",
                file=sys.stderr,
                flush=True,
            )
            continue
        completed[result["candidate_id"]] = result
        append_jsonl(output_path, result)
        print(
            f"[q5-corrections:reviewer] {len(completed)}/{len(corrections)} "
            f"{result['candidate_id']} {result['verifier_decision']}",
            file=sys.stderr,
            flush=True,
        )
    ordered = [completed[row["candidate_id"]] for row in corrections if row["candidate_id"] in completed]
    write_jsonl(output_dir / "verifications_ordered.jsonl", ordered)
    manifest.update(
        {
            "completed_verifications": len(ordered),
            "failed_verifications": len(failed_rows),
            "pending_verifications": len(corrections) - len(ordered) - len(failed_rows),
            "decision_counts": dict(Counter(row["verifier_decision"] for row in ordered)),
            "issue_tag_counts": dict(
                Counter(tag for row in ordered for tag in row.get("verifier_issue_tags", []))
            ),
        }
    )
    write_json(output_dir / "run_manifest.json", manifest)
    return ordered


def training_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": row["paper_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "summary": row["correction_summary"],
        "primary_category": row.get("primary_category"),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date"),
        "teacher_model": row.get("correction_model"),
        "generator_prompt_version": row.get("correction_prompt_version"),
        "review_prompt_version": row.get("verifier_prompt_version"),
        "source_split": row.get("source_split") or DEFAULT_SOURCE_SPLIT,
        "q5_correction_candidate_id": row.get("candidate_id"),
        "q5_hardening_review_id": row.get("q5_hardening_review_id"),
        "q5_hardening_selection_bucket": row.get("selection_bucket"),
        "q5_hardening_axes": row.get("q5_hardening_axes", []),
        "q5_issue_tags": row.get("q5_issue_tags", []),
        "q5_severity_source": row.get("q5_severity_source"),
        "q5_summary": row.get("q5_summary"),
        "q8_summary": row.get("q8_summary"),
        "q8_clean": row.get("q8_clean"),
        "q8_severe": row.get("q8_severe"),
        "q8_mangled": row.get("q8_mangled"),
        "blind_review_note": row.get("reviewer_note"),
        "correction_writer": row.get("correction_writer"),
        "correction_model": row.get("correction_model"),
        "correction_prompt_version": row.get("correction_prompt_version"),
        "correction_decision": row.get("correction_decision"),
        "correction_issue_tags": row.get("correction_issue_tags", []),
        "correction_evidence_span": row.get("correction_evidence_span"),
        "correction_reason_span": row.get("correction_reason_span"),
        "verifier_model": row.get("verifier_model"),
        "verifier_decision": row.get("verifier_decision"),
        "verifier_issue_tags": row.get("verifier_issue_tags", []),
        "verifier_evidence_span": row.get("verifier_evidence_span"),
        "verifier_reason_span": row.get("verifier_reason_span"),
    }


def release_shape_flags(row: dict[str, Any]) -> list[str]:
    summary = normalize_text(str(row.get("correction_summary", "")))
    flags: list[str] = []
    if not summary.endswith("."):
        flags.append("missing_final_period")
    words = whitespace_word_count(summary)
    if words < 10 or words > 15:
        flags.append("outside_10_15_words")
    return flags


def command_materialize(args: argparse.Namespace) -> None:
    verification_path = Path(args.output_root) / "qwen_verification_run" / "verifications_ordered.jsonl"
    rows = load_jsonl(verification_path)
    accepted: list[dict[str, Any]] = []
    release_shape_excluded: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("verifier_decision") != "accept":
            rejected.append(row)
            continue
        shape_flags = release_shape_flags(row)
        if shape_flags:
            release_shape_excluded.append({**row, "release_shape_flags": shape_flags})
            continue
        accepted.append(row)
    release_dir = Path(args.release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)
    materialized_dir = Path(args.output_root) / "materialized_release"
    materialized_dir.mkdir(parents=True, exist_ok=True)
    training_rows = [training_row(row) for row in accepted]
    write_jsonl(release_dir / "training_rows.jsonl", training_rows)
    write_jsonl(materialized_dir / "selected_review_rows.jsonl", accepted)
    write_jsonl(materialized_dir / "excluded_review_rows.jsonl", rejected + release_shape_excluded)
    manifest = {
        "created_at": utc_now_iso(),
        "release_dir": str(release_dir),
        "verification_rows": len(rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "release_shape_excluded_rows": len(release_shape_excluded),
        "release_shape_excluded_counts": dict(
            Counter(flag for row in release_shape_excluded for flag in row.get("release_shape_flags", []))
        ),
        "selection_bucket_counts": dict(
            Counter(row.get("q5_hardening_selection_bucket") for row in training_rows)
        ),
        "q5_issue_tag_counts": dict(
            Counter(tag for row in training_rows for tag in row.get("q5_issue_tags", []))
        ),
        "outputs": {
            "training_rows": str(release_dir / "training_rows.jsonl"),
            "selected_review_rows": str(materialized_dir / "selected_review_rows.jsonl"),
            "excluded_review_rows": str(materialized_dir / "excluded_review_rows.jsonl"),
        },
    }
    write_json(release_dir / "run_manifest.json", manifest)
    write_json(materialized_dir / "run_manifest.json", manifest)
    print(
        f"[q5-corrections:materialize] accepted={len(accepted)} rejected={len(rejected)} "
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
