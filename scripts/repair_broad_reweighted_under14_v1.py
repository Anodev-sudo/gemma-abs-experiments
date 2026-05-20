#!/usr/bin/env python3
"""Repair broad reweighted SFT labels to an under-14-word target."""
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
    has_formula_shorthand,
    load_dotenv,
    load_jsonl,
    load_prompt_spec,
    normalize_text,
    request_json_with_retries,
    render_prompt,
    sentence_count,
    title_overlap_ratio,
)


DEFAULT_SOURCE_DATASET = Path(
    "hf_bucket/experiments/sft/q5_step2400_broad_reweighted_basin_v1/"
    "dataset_broad_base_low_repair_high_v1/train.jsonl"
)
DEFAULT_OUTPUT_ROOT = Path(
    "hf_bucket/experiments/sft/q8_promoted_broad_reweighted_under14_v1/"
    "kimi_qwen_under14_repair_v1"
)
DEFAULT_DATASET_DIR = Path(
    "hf_bucket/experiments/sft/q8_promoted_broad_reweighted_under14_v1/"
    "dataset_under14_kimi_qwen_v1"
)
DEFAULT_WRITER_PROMPT = Path(
    "prompts/synthetic_summary/kimi_k2_6_broad_reweighted_under14_repair_writer_v1.json"
)
DEFAULT_REVIEWER_PROMPT = Path(
    "prompts/synthetic_summary/qwen3_235b_a22b_2507_broad_reweighted_under14_repair_verifier_v1.json"
)
DEFAULT_WRITER_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_REVIEWER_MODEL = "qwen/qwen3-235b-a22b-2507"

VALID_WRITER_DECISIONS = {"rewrite", "drop"}
VALID_REVIEWER_DECISIONS = {"accept", "reject"}
APPEND_LOCK = threading.Lock()

BANNED_OPENING_RE = re.compile(
    r"^(?:this paper|the paper|the authors|the work|we propose|in this work|in this paper)\b",
    re.IGNORECASE,
)
ADJACENT_DUP_RE = re.compile(r"\b([A-Za-z][A-Za-z-]*)\s+\1\b", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You write one-sentence paper summaries for a mobile research feed. "
    "Keep them concrete, readable, and useful. Output only the summary sentence."
)
USER_TEMPLATE = (
    "The app already shows the title, author list, and date. "
    "Write one sentence that tells a reader why they might open this paper. "
    "Aim for 8 to 13 words. Avoid repeating the title. "
    'Avoid filler like "This paper" or "We propose". '
    "Do not mention authors or dates. "
    "Keep the wording natural and specific.\n\n"
    "Title:\n{title}\n\n"
    "Abstract:\n{abstract}"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with APPEND_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)


def done_ids(path: Path, key: str = "candidate_id") -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.open(encoding="utf-8"):
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


def surface_word_count(value: str) -> int:
    return len([part for part in normalize_text(value).split(" ") if part])


def current_summary(row: dict[str, Any]) -> str:
    if row.get("summary"):
        return normalize_text(str(row["summary"]))
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return normalize_text(str(messages[-1].get("content", "")))
    return ""


def hard_filter_flags(summary: str, title: str, title_overlap_threshold: float) -> list[str]:
    text = normalize_text(summary)
    lowered = text.lower()
    flags: list[str] = []
    wc = surface_word_count(text)
    if not text:
        flags.append("empty_summary")
    if wc < 8:
        flags.append("too_short")
    if wc > 13:
        flags.append("too_long")
    if sentence_count(text) != 1:
        flags.append("not_single_sentence")
    if text and not text.endswith("."):
        flags.append("not_period_ended")
    if text.endswith("..") or text.endswith("?.") or text.endswith("!."):
        flags.append("bad_terminal_punctuation")
    if ";" in text:
        flags.append("semicolon")
    if BANNED_OPENING_RE.search(lowered):
        flags.append("bad_opening")
    if has_formula_shorthand(text):
        flags.append("formula_or_symbol")
    if ADJACENT_DUP_RE.search(text):
        flags.append("adjacent_duplicate_word")
    if title_overlap_ratio(text, title) >= title_overlap_threshold:
        flags.append("title_echo")
    return sorted(set(flags))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair broad reweighted labels to an under-14-word target.")
    parser.add_argument("--stage", choices=["prepare", "write", "screen", "verify", "materialize", "audit", "all"], default="all")
    parser.add_argument("--source-dataset", type=Path, default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--writer-prompt", type=Path, default=DEFAULT_WRITER_PROMPT)
    parser.add_argument("--reviewer-prompt", type=Path, default=DEFAULT_REVIEWER_PROMPT)
    parser.add_argument("--writer-model", default=DEFAULT_WRITER_MODEL)
    parser.add_argument("--reviewer-model", default=DEFAULT_REVIEWER_MODEL)
    parser.add_argument("--writer-workers", type=int, default=4)
    parser.add_argument("--reviewer-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--writer-max-output-tokens", type=int, default=260)
    parser.add_argument("--reviewer-max-output-tokens", type=int, default=320)
    parser.add_argument("--writer-reasoning-effort", default="none")
    parser.add_argument("--reviewer-reasoning-effort", default="none")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--title-overlap-threshold", type=float, default=0.58)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    return parser.parse_args()


def command_prepare(args: argparse.Namespace) -> list[dict[str, Any]]:
    source_rows = load_jsonl(args.source_dataset)
    candidates: list[dict[str, Any]] = []
    for idx, row in enumerate(source_rows, start=1):
        summary = current_summary(row)
        flags = hard_filter_flags(summary, str(row.get("title", "")), args.title_overlap_threshold)
        if not flags:
            continue
        candidates.append(
            {
                "candidate_id": f"broad_under14_repair:{idx:05d}",
                "source_row_index": idx - 1,
                "paper_id": str(row.get("paper_id", "")),
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
                "primary_category": row.get("primary_category", ""),
                "categories": row.get("categories", []),
                "created_date": row.get("created_date", ""),
                "dataset_component": row.get("dataset_component", ""),
                "current_summary": summary,
                "current_word_count": surface_word_count(summary),
                "current_sentence_count": sentence_count(summary),
                "current_title_overlap": round(title_overlap_ratio(summary, str(row.get("title", ""))), 4),
                "issue_flags": flags,
                "loss_weight_hints": row.get("loss_weight_hints", {}),
            }
        )
    if args.limit is not None:
        candidates = candidates[: args.limit]
    out_dir = args.output_root / "candidate_pool"
    write_jsonl(out_dir / "candidates.jsonl", candidates)
    write_json(
        out_dir / "run_manifest.json",
        {
            "created_at": utc_now(),
            "source_dataset": str(args.source_dataset),
            "source_rows": len(source_rows),
            "candidate_rows": len(candidates),
            "limit": args.limit,
            "title_overlap_threshold": args.title_overlap_threshold,
            "issue_flag_counts": dict(collections.Counter(flag for row in candidates for flag in row["issue_flags"])),
            "component_counts": dict(collections.Counter(row.get("dataset_component", "") for row in candidates)),
        },
    )
    print(f"[prepare] source_rows={len(source_rows)} candidates={len(candidates)}")
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
            "current_summary": row.get("current_summary", ""),
            "issue_flags": ", ".join(row.get("issue_flags", [])),
            "dataset_component": row.get("dataset_component", ""),
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
        flags = hard_filter_flags(summary, str(candidate.get("title", "")), args.title_overlap_threshold) if summary else ["empty_summary"]
        return {
            **candidate,
            "writer_model": args.writer_model,
            "writer_prompt_version": spec["prompt_version"],
            "writer_decision": decision,
            "writer_issue_tags": parsed.get("issue_tags", []),
            "writer_evidence_span": normalize_text(parsed.get("evidence_span", "")),
            "repair_summary": summary,
            "writer_word_count": surface_word_count(summary),
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
    eligible_ids = {row["candidate_id"] for row in eligible}
    rejected = [row for row in writer_rows if row.get("candidate_id") not in eligible_ids]
    out_dir = args.output_root / "local_screen"
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
            "current_summary": row.get("current_summary", ""),
            "issue_flags": ", ".join(row.get("issue_flags", [])),
            "writer_decision": row.get("writer_decision", ""),
            "writer_issue_tags": ", ".join(row.get("writer_issue_tags", [])),
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
            reasoning=payload.get("reasoning"),
            attempt_log_dir=out_dir / "raw_json_attempts",
        )
        hard_flags = hard_filter_flags(row.get("repair_summary", ""), str(row.get("title", "")), args.title_overlap_threshold)
        issue_tags = parsed.get("issue_tags", [])
        normalized_issue_tags = {str(tag).strip().lower() for tag in issue_tags if str(tag).strip()}
        grounding_ok = bool(parsed.get("grounding_ok", False)) and not hard_flags
        style_ok = bool(parsed.get("style_ok", False)) and not hard_flags
        decision = str(parsed.get("decision", "")).strip().lower()
        if decision not in VALID_REVIEWER_DECISIONS:
            if grounding_ok and style_ok and normalized_issue_tags == {"none"}:
                decision = "accept"
            else:
                decision = "reject"
        if hard_flags:
            decision = "reject"
        return {
            **row,
            "reviewer_model": args.reviewer_model,
            "reviewer_prompt_version": spec["prompt_version"],
            "verifier_decision": decision,
            "verifier_issue_tags": issue_tags,
            "grounding_ok": grounding_ok,
            "style_ok": style_ok,
            "verifier_evidence_span": normalize_text(parsed.get("evidence_span", "") or parsed.get("quote", "")),
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
            "post_qwen_hard_filter_counts": dict(collections.Counter(flag for row in all_rows for flag in row.get("post_qwen_hard_filter_flags", []))),
        },
    )
    return all_rows


def rebuilt_messages(row: dict[str, Any], summary: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(title=row.get("title", ""), abstract=row.get("abstract", ""))},
        {"role": "assistant", "content": summary},
    ]


def with_repair(source_row: dict[str, Any], repair: dict[str, Any]) -> dict[str, Any]:
    out = dict(source_row)
    original_summary = current_summary(source_row)
    summary = normalize_text(repair["repair_summary"])
    out["summary"] = summary
    out["messages"] = rebuilt_messages(out, summary)
    out["teacher_model"] = repair.get("writer_model", DEFAULT_WRITER_MODEL)
    out["generator_prompt_version"] = repair.get("writer_prompt_version", "")
    out["review_prompt_version"] = repair.get("reviewer_prompt_version", "")
    out["under14_repair_v1"] = {
        "candidate_id": repair.get("candidate_id", ""),
        "original_summary": original_summary,
        "original_issue_flags": repair.get("issue_flags", []),
        "writer_evidence_span": repair.get("writer_evidence_span", ""),
        "writer_reason_span": repair.get("writer_reason_span", ""),
        "verifier_evidence_span": repair.get("verifier_evidence_span", ""),
        "verifier_reason_span": repair.get("verifier_reason_span", ""),
    }
    return out


def without_repair(source_row: dict[str, Any]) -> dict[str, Any]:
    out = dict(source_row)
    summary = current_summary(out)
    out["summary"] = summary
    out["messages"] = rebuilt_messages(out, summary)
    return out


def command_materialize(args: argparse.Namespace) -> list[dict[str, Any]]:
    source_rows = load_jsonl(args.source_dataset)
    candidates = load_candidates(args)
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
        and not hard_filter_flags(row.get("repair_summary", ""), str(row.get("title", "")), args.title_overlap_threshold)
    ]
    accepted_by_index = {int(row["source_row_index"]): row for row in accepted}
    candidate_indexes = {int(row["source_row_index"]) for row in candidates}
    unresolved = [
        row
        for row in candidates
        if int(row["source_row_index"]) in candidate_indexes
        and int(row["source_row_index"]) not in accepted_by_index
    ]
    unresolved_by_index = {int(row["source_row_index"]): row for row in unresolved}

    final_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []
    repaired_count = 0
    for idx, source_row in enumerate(source_rows):
        if idx in accepted_by_index:
            final_rows.append(with_repair(source_row, accepted_by_index[idx]))
            repaired_count += 1
        elif idx in unresolved_by_index:
            dropped_rows.append({**source_row, "under14_unresolved_candidate": unresolved_by_index[idx]})
        else:
            final_rows.append(without_repair(source_row))

    rejected_after_qwen = [row for row in verified if row.get("candidate_id") not in {item.get("candidate_id") for item in accepted}]
    out_dir = args.output_root / "materialized"
    write_jsonl(out_dir / "accepted_repairs.jsonl", accepted)
    write_jsonl(out_dir / "rejected_after_qwen.jsonl", rejected_after_qwen)
    write_jsonl(out_dir / "rejected_before_qwen.jsonl", local_rejected)
    write_jsonl(out_dir / "unresolved_dropped_source_rows.jsonl", dropped_rows)

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.dataset_dir / "train.jsonl", final_rows)
    audit = dataset_audit(final_rows, args.title_overlap_threshold)
    write_json(
        args.dataset_dir / "manifest.json",
        {
            "dataset_id": "q8_promoted_broad_reweighted_under14_kimi_qwen_v1",
            "created_at": utc_now(),
            "source_dataset": str(args.source_dataset),
            "repair_output_root": str(args.output_root),
            "rows": len(final_rows),
            "source_rows": len(source_rows),
            "candidate_rows": len(candidates),
            "repaired_rows": repaired_count,
            "dropped_unresolved_rows": len(dropped_rows),
            "accepted_repairs": len(accepted),
            "rejected_after_qwen": len(rejected_after_qwen),
            "rejected_before_qwen": len(local_rejected),
            "writer_model": args.writer_model,
            "reviewer_model": args.reviewer_model,
            "writer_prompt": str(args.writer_prompt),
            "reviewer_prompt": str(args.reviewer_prompt),
            "title_overlap_threshold": args.title_overlap_threshold,
            "chat_prompt_target": "8 to 13 words",
            "audit": audit,
        },
    )
    write_readme(args.dataset_dir, audit, len(source_rows), len(candidates), repaired_count, len(dropped_rows))
    print(
        f"[materialize] final_rows={len(final_rows)} repaired={repaired_count} "
        f"dropped_unresolved={len(dropped_rows)}"
    )
    return final_rows


def dataset_audit(rows: list[dict[str, Any]], title_overlap_threshold: float) -> dict[str, Any]:
    word_counts = collections.Counter()
    flag_counts = collections.Counter()
    component_counts = collections.Counter()
    repaired_by_component = collections.Counter()
    for row in rows:
        summary = current_summary(row)
        word_counts[surface_word_count(summary)] += 1
        component = str(row.get("dataset_component", ""))
        component_counts[component] += 1
        if row.get("under14_repair_v1"):
            repaired_by_component[component] += 1
        flag_counts.update(hard_filter_flags(summary, str(row.get("title", "")), title_overlap_threshold))
    return {
        "rows": len(rows),
        "word_count_distribution": dict(sorted(word_counts.items())),
        "hard_filter_flag_counts": dict(sorted(flag_counts.items())),
        "component_counts": dict(sorted(component_counts.items())),
        "repaired_by_component": dict(sorted(repaired_by_component.items())),
    }


def write_readme(
    dataset_dir: Path,
    audit: dict[str, Any],
    source_rows: int,
    candidate_rows: int,
    repaired_rows: int,
    dropped_rows: int,
) -> None:
    lines = [
        "# q8 Promoted Broad Reweighted Under-14 Dataset",
        "",
        "This dataset starts from the broad reweighted training mix and applies targeted Kimi/Qwen label repair.",
        "The source dataset is preserved; this directory is a new materialized training view.",
        "",
        "## Policy",
        "",
        "- One faithful, simple, readable sentence.",
        "- 8 to 13 words.",
        "- Avoid title rehash, boilerplate openings, formulas, and unsupported claims.",
        "- Preserve the original row weights where rows are kept.",
        "",
        "## Counts",
        "",
        f"- Source rows: {source_rows}",
        f"- Candidate rows sent to repair: {candidate_rows}",
        f"- Repaired rows accepted by Qwen: {repaired_rows}",
        f"- Unresolved candidate rows dropped from this materialized view: {dropped_rows}",
        f"- Final training rows: {audit['rows']}",
        "",
        "## Residual Hard Flags",
        "",
    ]
    if audit["hard_filter_flag_counts"]:
        for flag, count in audit["hard_filter_flag_counts"].items():
            lines.append(f"- {flag}: {count}")
    else:
        lines.append("- None")
    lines.extend(["", "## Word Counts", ""])
    for word_count, count in audit["word_count_distribution"].items():
        lines.append(f"- {word_count}: {count}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def command_audit(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = args.dataset_dir / "train.jsonl"
    if not dataset_path.exists():
        raise SystemExit(f"Missing materialized train.jsonl: {dataset_path}")
    rows = load_jsonl(dataset_path)
    audit = dataset_audit(rows, args.title_overlap_threshold)
    write_json(args.dataset_dir / "post_materialize_audit.json", audit)
    print(f"[audit] rows={audit['rows']} flags={audit['hard_filter_flag_counts']}")
    return audit


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
    if args.stage in {"audit", "all"}:
        command_audit(args)


if __name__ == "__main__":
    main()
