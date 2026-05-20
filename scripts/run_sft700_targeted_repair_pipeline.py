#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
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
    write_jsonl,
)

DEFAULT_ROOT = "hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/failure_mining/checkpoint700_10000_v1/repairs_targeted_v1"
DEFAULT_SCREENED = "hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/failure_mining/checkpoint700_10000_v1/heuristic_screen/screened_rows.jsonl"
DEFAULT_TRIAGE = "hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/failure_mining/checkpoint700_10000_v1/heuristic_screen/worker_triage_659.jsonl"
DEFAULT_WRITER_PROMPT = "prompts/synthetic_summary/kimi_k2_6_q5_sft700_targeted_repair_v1.json"
DEFAULT_REVIEWER_PROMPT = "prompts/synthetic_summary/qwen3_235b_a22b_2507_q5_sft700_targeted_repair_verifier_v1.json"
DEFAULT_WRITER_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_REVIEWER_MODEL = "qwen/qwen3-235b-a22b-2507"
VALID_WRITER_DECISIONS = {"rewrite", "drop"}
VALID_REVIEW_DECISIONS = {"accept", "reject"}
APPEND_JSONL_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with APPEND_JSONL_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def word_count(text: str) -> int:
    return len([p for p in normalize_text(text).split(" ") if p])


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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prepare", "write", "verify", "materialize", "all"], default="all")
    ap.add_argument("--screened-jsonl", default=DEFAULT_SCREENED)
    ap.add_argument("--triage-jsonl", default=DEFAULT_TRIAGE)
    ap.add_argument("--output-root", default=DEFAULT_ROOT)
    ap.add_argument("--writer-prompt", default=DEFAULT_WRITER_PROMPT)
    ap.add_argument("--reviewer-prompt", default=DEFAULT_REVIEWER_PROMPT)
    ap.add_argument("--writer-model", default=DEFAULT_WRITER_MODEL)
    ap.add_argument("--reviewer-model", default=DEFAULT_REVIEWER_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--writer-workers", type=int, default=4)
    ap.add_argument("--reviewer-workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--writer-max-output-tokens", type=int, default=220)
    ap.add_argument("--reviewer-max-output-tokens", type=int, default=256)
    ap.add_argument("--writer-reasoning-effort", default="none")
    ap.add_argument("--reviewer-reasoning-effort", default="none")
    ap.add_argument("--reviewer-provider", default=None)
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    ap.add_argument("--continue-on-error", action="store_true", default=True)
    return ap.parse_args()


def done_ids(path: Path, key: str = "candidate_id") -> set[str]:
    ids = set()
    if not path.exists():
        return ids
    for line in path.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get(key):
            ids.add(str(d[key]))
    return ids


def command_prepare(args: argparse.Namespace) -> list[dict[str, Any]]:
    screened = {r["paper_id"]: r for r in load_jsonl(Path(args.screened_jsonl))}
    triage_rows = load_jsonl(Path(args.triage_jsonl))
    candidates = []
    for idx, t in enumerate(triage_rows, start=1):
        if t.get("decision") in {"not_failure", "unclear"}:
            continue
        s = screened[t["paper_id"]]
        row = {
            "candidate_id": f"sft700_repair:{len(candidates)+1:04d}",
            "paper_id": t["paper_id"],
            "title": s.get("title", ""),
            "abstract": s.get("abstract", ""),
            "primary_category": s.get("primary_category", ""),
            "bad_summary": t.get("bad_summary") or s.get("summary", ""),
            "pre_dpo_summary": t.get("pre_dpo_summary") or s.get("pre_dpo_summary", ""),
            "triage_decision": t.get("decision", ""),
            "triage_severity": t.get("severity", ""),
            "failure_tags": t.get("failure_tags", []),
            "short_reason": t.get("short_reason", ""),
            "heuristic_numeric_tags": s.get("numeric_tags", []),
            "heuristic_semantic_tags": s.get("semantic_tags", []),
            "source_split": "q5_sft700_targeted_repair_v1",
        }
        candidates.append(row)
    if args.limit is not None:
        candidates = candidates[: args.limit]
    out_dir = Path(args.output_root) / "candidate_pool"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "candidates.jsonl", candidates)
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "created_at": utc_now(),
        "screened_jsonl": args.screened_jsonl,
        "triage_jsonl": args.triage_jsonl,
        "candidate_rows": len(candidates),
        "decision_counts": dict(Counter(c["triage_decision"] for c in candidates)),
        "failure_tag_counts": dict(Counter(tag for c in candidates for tag in c.get("failure_tags", []))),
    }, indent=2, sort_keys=True) + "\n")
    print(f"[prepare] wrote {len(candidates)} candidates")
    return candidates


def load_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = Path(args.output_root) / "candidate_pool" / "candidates.jsonl"
    if not path.exists():
        return command_prepare(args)
    rows = load_jsonl(path)
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def writer_payload(c: dict[str, Any], spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    prompt = render_prompt(spec, c, extra_variables={
        "bad_summary": c.get("bad_summary", ""),
        "pre_dpo_summary": c.get("pre_dpo_summary", ""),
        "triage_decision": c.get("triage_decision", ""),
        "triage_severity": c.get("triage_severity", ""),
        "failure_tags": ", ".join(c.get("failure_tags", [])),
        "short_reason": c.get("short_reason", ""),
    })
    payload = {
        "candidate_id": c["candidate_id"],
        "paper_id": c["paper_id"],
        "model": args.writer_model,
        "temperature": args.temperature,
        "max_output_tokens": args.writer_max_output_tokens,
        "prompt_version": spec["prompt_version"],
        "messages": build_messages(spec, prompt),
    }
    if spec.get("response_format"):
        payload["response_format"] = spec["response_format"]
    if (reasoning := reasoning_config(args.writer_reasoning_effort)):
        payload["reasoning"] = reasoning
    return payload


def command_write(args: argparse.Namespace) -> list[dict[str, Any]]:
    load_dotenv(Path(".env"))
    api_base = args.api_base or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    spec = load_prompt_spec(Path(args.writer_prompt))
    candidates = load_candidates(args)
    out_dir = Path(args.output_root) / "writer_run"
    req_path = out_dir / "requests.jsonl"
    out_path = out_dir / "outputs.jsonl"
    err_path = out_dir / "errors.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = done_ids(out_path)
    todo = [c for c in candidates if c["candidate_id"] not in done]
    print(f"[write] candidates={len(candidates)} done={len(done)} todo={len(todo)}")

    def run_one(c: dict[str, Any]) -> dict[str, Any]:
        payload = writer_payload(c, spec, args)
        append_jsonl(req_path, payload)
        parsed, raw_text, raw_response = request_json_with_retries(
            label=c["candidate_id"], api_base=api_base, api_key=api_key, model=args.writer_model,
            messages=payload["messages"], temperature=args.temperature, max_output_tokens=args.writer_max_output_tokens,
            timeout=args.timeout, max_retries=args.max_retries, retry_backoff_seconds=args.retry_backoff_seconds,
            response_format=payload.get("response_format"), reasoning=payload.get("reasoning"),
            attempt_log_dir=out_dir / "raw_json_attempts",
        )
        decision = str(parsed.get("decision", "")).strip().lower()
        if decision not in VALID_WRITER_DECISIONS:
            decision = "drop"
        raw_summary_value = ""
        if decision == "rewrite":
            raw_summary_value = str(parsed.get("summary") or parsed.get("label") or parsed.get("final_summary") or "")
        summary = normalize_text(raw_summary_value) if decision == "rewrite" else ""
        wc = word_count(summary)
        local_flags = local_filter_flags(summary, c.get("title", "")) if summary else []
        if decision == "rewrite" and (wc < 10 or wc > 15):
            local_flags.append("outside_10_15_words")
        row = {
            **c,
            "writer_model": args.writer_model,
            "writer_prompt_version": spec["prompt_version"],
            "writer_decision": decision,
            "writer_issue_tags": parsed.get("issue_tags", []),
            "number_policy": parsed.get("number_policy", ""),
            "repair_summary": summary,
            "writer_word_count": wc,
            "writer_reported_word_count": parsed.get("word_count", 0),
            "writer_evidence_span": str(parsed.get("evidence_span", "")),
            "writer_reason_span": str(parsed.get("reason_span", "")),
            "writer_local_filter_flags": local_flags,
            "raw_writer_text": raw_text,
        }
        return row

    results = []
    with ThreadPoolExecutor(max_workers=args.writer_workers) as ex:
        futs = {ex.submit(run_one, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), start=1):
            c = futs[fut]
            try:
                row = fut.result()
                append_jsonl(out_path, row)
                results.append(row)
                if i % 25 == 0:
                    print(f"[write] completed {i}/{len(todo)}")
            except Exception as exc:
                err = {"candidate_id": c["candidate_id"], "paper_id": c["paper_id"], "error": str(exc), "error_type": type(exc).__name__}
                append_jsonl(err_path, err)
                print(f"[write:error] {c['candidate_id']} {type(exc).__name__}: {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    raise
    all_rows = load_jsonl(out_path) if out_path.exists() else []
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "created_at": utc_now(), "rows": len(all_rows), "decision_counts": dict(Counter(r.get("writer_decision") for r in all_rows)),
        "local_flag_counts": dict(Counter(flag for r in all_rows for flag in r.get("writer_local_filter_flags", []))),
    }, indent=2, sort_keys=True) + "\n")
    return all_rows


def reviewer_payload(row: dict[str, Any], spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    prompt = render_prompt(spec, row, extra_variables={
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
    })
    payload = {
        "candidate_id": row["candidate_id"], "paper_id": row["paper_id"], "model": args.reviewer_model,
        "temperature": args.temperature, "max_output_tokens": args.reviewer_max_output_tokens,
        "prompt_version": spec["prompt_version"], "messages": build_messages(spec, prompt),
    }
    if spec.get("response_format"):
        payload["response_format"] = spec["response_format"]
    if (provider := provider_config(args.reviewer_provider)):
        payload["provider"] = provider
    if (reasoning := reasoning_config(args.reviewer_reasoning_effort)):
        payload["reasoning"] = reasoning
    return payload


def command_verify(args: argparse.Namespace) -> list[dict[str, Any]]:
    load_dotenv(Path(".env"))
    api_base = args.api_base or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    spec = load_prompt_spec(Path(args.reviewer_prompt))
    writer_rows = [r for r in load_jsonl(Path(args.output_root) / "writer_run" / "outputs.jsonl") if r.get("writer_decision") == "rewrite"]
    if args.limit is not None:
        writer_rows = writer_rows[:args.limit]
    out_dir = Path(args.output_root) / "reviewer_run"
    req_path = out_dir / "requests.jsonl"
    out_path = out_dir / "outputs.jsonl"
    err_path = out_dir / "errors.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = done_ids(out_path)
    todo = [r for r in writer_rows if r["candidate_id"] not in done]
    print(f"[verify] writer_rows={len(writer_rows)} done={len(done)} todo={len(todo)}")

    def run_one(r: dict[str, Any]) -> dict[str, Any]:
        payload = reviewer_payload(r, spec, args)
        append_jsonl(req_path, payload)
        parsed, raw_text, raw_response = request_json_with_retries(
            label=r["candidate_id"], api_base=api_base, api_key=api_key, model=args.reviewer_model,
            messages=payload["messages"], temperature=args.temperature, max_output_tokens=args.reviewer_max_output_tokens,
            timeout=args.timeout, max_retries=args.max_retries, retry_backoff_seconds=args.retry_backoff_seconds,
            response_format=payload.get("response_format"), provider=payload.get("provider"), reasoning=payload.get("reasoning"),
            attempt_log_dir=out_dir / "raw_json_attempts",
        )
        decision = str(parsed.get("decision", "")).strip().lower()
        if decision not in VALID_REVIEW_DECISIONS:
            decision = "reject"
        local_flags = r.get("writer_local_filter_flags", [])
        if local_flags:
            decision = "reject"
        return {
            **r,
            "reviewer_model": args.reviewer_model,
            "reviewer_prompt_version": spec["prompt_version"],
            "verifier_decision": decision,
            "verifier_issue_tags": parsed.get("issue_tags", []),
            "number_policy_ok": bool(parsed.get("number_policy_ok", False)),
            "verifier_evidence_span": str(parsed.get("evidence_span", "")),
            "verifier_reason_span": str(parsed.get("reason_span", "")),
            "raw_verifier_text": raw_text,
        }

    results = []
    with ThreadPoolExecutor(max_workers=args.reviewer_workers) as ex:
        futs = {ex.submit(run_one, r): r for r in todo}
        for i, fut in enumerate(as_completed(futs), start=1):
            r = futs[fut]
            try:
                row = fut.result()
                append_jsonl(out_path, row)
                results.append(row)
                if i % 25 == 0:
                    print(f"[verify] completed {i}/{len(todo)}")
            except Exception as exc:
                err = {"candidate_id": r["candidate_id"], "paper_id": r["paper_id"], "error": str(exc), "error_type": type(exc).__name__}
                append_jsonl(err_path, err)
                print(f"[verify:error] {r['candidate_id']} {type(exc).__name__}: {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    raise
    all_rows = load_jsonl(out_path) if out_path.exists() else []
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "created_at": utc_now(), "rows": len(all_rows), "decision_counts": dict(Counter(r.get("verifier_decision") for r in all_rows)),
        "tag_counts": dict(Counter(tag for r in all_rows for tag in r.get("verifier_issue_tags", []))),
    }, indent=2, sort_keys=True) + "\n")
    return all_rows


def training_row(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": r["paper_id"], "title": r["title"], "abstract": r["abstract"], "summary": r["repair_summary"],
        "primary_category": r.get("primary_category", ""), "categories": [], "created_date": "",
        "teacher_model": r.get("writer_model", ""),
        "generator_prompt_version": r.get("writer_prompt_version", ""),
        "review_prompt_version": r.get("reviewer_prompt_version", ""),
        "source_split": r.get("source_split", "q5_sft700_targeted_repair_v1"),
        "bad_summary_checkpoint700": r.get("bad_summary", ""),
        "pre_dpo_summary": r.get("pre_dpo_summary", ""),
        "failure_tags": r.get("failure_tags", []),
        "number_policy": r.get("number_policy", ""),
        "repair_candidate_id": r.get("candidate_id", ""),
    }


def command_materialize(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = load_jsonl(Path(args.output_root) / "reviewer_run" / "outputs.jsonl")
    accepted = [r for r in rows if r.get("verifier_decision") == "accept" and r.get("repair_summary")]
    out_dir = Path(args.output_root) / "materialized"
    out_dir.mkdir(parents=True, exist_ok=True)
    train = [training_row(r) for r in accepted]
    write_jsonl(out_dir / "training_rows.jsonl", train)
    write_jsonl(out_dir / "accepted_repairs.jsonl", accepted)
    rejected = [r for r in rows if r.get("verifier_decision") != "accept"]
    write_jsonl(out_dir / "rejected_repairs.jsonl", rejected)
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "created_at": utc_now(), "reviewed_rows": len(rows), "accepted_rows": len(train), "rejected_rows": len(rejected),
        "decision_counts": dict(Counter(r.get("verifier_decision") for r in rows)),
        "failure_tag_counts": dict(Counter(tag for r in accepted for tag in r.get("failure_tags", []))),
    }, indent=2, sort_keys=True) + "\n")
    print(f"[materialize] accepted={len(train)} rejected={len(rejected)}")
    return train


def main() -> None:
    args = parse_args()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    if args.stage in {"prepare", "all"}:
        command_prepare(args)
    if args.stage in {"write", "all"}:
        command_write(args)
    if args.stage in {"verify", "all"}:
        command_verify(args)
    if args.stage in {"materialize", "all"}:
        command_materialize(args)


if __name__ == "__main__":
    main()
