#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from faithfulness_cleanup_common import DEFAULT_SEED, load_jsonl, utc_now_iso, write_json, write_jsonl
from synthetic_summary_runtime import (
    build_messages,
    load_dotenv,
    load_prompt_spec,
    local_filter_flags,
    render_prompt,
    request_json_with_retries,
)

DEFAULT_ROOT = (
    "hf_bucket/experiments/faithfulness_cleanup/"
    "recent_3y_balanced_10746_faithfulness_overlay_1400_v2"
)
DEFAULT_PROMPT = (
    "prompts/synthetic_summary/"
    "qwen3_235b_a22b_2507_specificity_preserving_rewriter_v1.json"
)
DEFAULT_MODEL = "qwen/qwen3-235b-a22b-2507"
VALID_DECISIONS = {"keep", "rewrite", "drop"}
VALID_RISK_TAGS = {
    "supported_number",
    "supported_entity_label",
    "supported_comparator",
    "supported_mechanism",
    "unsupported_number",
    "unsupported_entity_label",
    "unsupported_mechanism",
    "unsupported_causal_claim",
    "overstrong_claim",
    "polarity_risk",
    "too_generic",
    "title_rehash",
    "style_only",
    "none",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draft specificity-preserving summary targets for manual review."
    )
    parser.add_argument(
        "--candidates-jsonl",
        default=str(Path(DEFAULT_ROOT) / "specificity_candidate_selection" / "selected_candidates.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(DEFAULT_ROOT) / "specificity_rewrite_run"),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=192)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--reasoning-exclude", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def normalize_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        tag = str(item).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def validate_rewrite(
    candidate: dict[str, Any],
    parsed: dict[str, Any],
    *,
    prompt_version: str,
    model: str,
    raw_text: str,
    raw_response: dict[str, Any],
) -> dict[str, Any]:
    decision = str(parsed.get("decision", "")).strip().lower()
    if decision not in VALID_DECISIONS:
        decision = "drop"
    risk_tags = [
        tag for tag in normalize_list(parsed.get("risk_tags", [])) if tag in VALID_RISK_TAGS
    ]
    if not risk_tags:
        risk_tags = ["none"] if decision == "keep" else []
    original_summary = str(candidate.get("original_summary", "")).strip()
    summary = " ".join(str(parsed.get("summary", "")).split())
    if decision == "keep":
        summary = original_summary
    if decision == "drop":
        summary = ""
    if decision == "rewrite" and not summary:
        decision = "drop"
    local_flags = local_filter_flags(summary, str(candidate.get("title", ""))) if summary else []

    return {
        "candidate_id": candidate["candidate_id"],
        "paper_id": candidate["paper_id"],
        "title": candidate["title"],
        "abstract": candidate["abstract"],
        "primary_category": candidate.get("primary_category"),
        "categories": candidate.get("categories", []),
        "created_date": candidate.get("created_date"),
        "source_split": candidate.get("source_split"),
        "teacher_model": candidate.get("teacher_model"),
        "generator_prompt_version": candidate.get("generator_prompt_version"),
        "review_prompt_version": candidate.get("review_prompt_version"),
        "original_summary": original_summary,
        "specificity_score": candidate.get("specificity_score"),
        "specificity_flags": candidate.get("specificity_flags", []),
        "supported_numbers": candidate.get("supported_numbers", []),
        "supported_labels": candidate.get("supported_labels", []),
        "unsupported_numbers": candidate.get("unsupported_numbers", []),
        "unsupported_labels": candidate.get("unsupported_labels", []),
        "specificity_prompt_version": prompt_version,
        "specificity_model": model,
        "specificity_decision": decision,
        "specificity_risk_tags": risk_tags,
        "specificity_evidence_span": " ".join(str(parsed.get("evidence_span", "")).split()),
        "specificity_summary": summary,
        "specificity_reason_span": " ".join(str(parsed.get("reason_span", "")).split()),
        "specificity_local_filter_flags": local_flags,
        "raw_specificity_response": parsed,
        "raw_specificity_text": raw_text,
        "raw_response_id": raw_response.get("id", ""),
    }


def build_request_payload(
    candidate: dict[str, Any],
    *,
    spec: dict[str, Any],
    model: str,
    temperature: float,
    max_output_tokens: int,
    reasoning: dict[str, Any] | None,
) -> dict[str, Any]:
    prompt = render_prompt(
        spec,
        candidate,
        extra_variables={
            "original_summary": candidate.get("original_summary", ""),
            "supported_numbers": ", ".join(candidate.get("supported_numbers", [])),
            "supported_labels": ", ".join(candidate.get("supported_labels", [])),
            "unsupported_numbers": ", ".join(candidate.get("unsupported_numbers", [])),
            "unsupported_labels": ", ".join(candidate.get("unsupported_labels", [])),
        },
    )
    payload: dict[str, Any] = {
        "candidate_id": candidate["candidate_id"],
        "paper_id": candidate["paper_id"],
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "prompt_version": spec["prompt_version"],
        "messages": build_messages(spec, prompt),
        "response_format": spec.get("response_format"),
        "plugins": spec.get("plugins"),
    }
    if reasoning:
        payload["reasoning"] = reasoning
    return payload


def run_one(
    candidate: dict[str, Any],
    *,
    spec: dict[str, Any],
    model: str,
    api_base: str,
    api_key: str,
    temperature: float,
    max_output_tokens: int,
    timeout: int,
    max_retries: int,
    retry_backoff_seconds: float,
    reasoning: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = build_request_payload(
        candidate,
        spec=spec,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        reasoning=reasoning,
    )
    parsed, raw_text, raw_response = request_json_with_retries(
        label=f"specificity {candidate['candidate_id']}",
        api_base=api_base,
        api_key=api_key,
        model=model,
        messages=payload["messages"],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        response_format=spec.get("response_format"),
        plugins=spec.get("plugins"),
        assistant_prefill=spec.get("assistant_prefill"),
        reasoning=reasoning,
    )
    return validate_rewrite(
        candidate,
        parsed,
        prompt_version=str(spec["prompt_version"]),
        model=model,
        raw_text=raw_text,
        raw_response=raw_response,
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv(Path(".env"))

    prompt_spec = load_prompt_spec(Path(args.prompt))
    candidates = load_jsonl(Path(args.candidates_jsonl))
    if args.limit is not None:
        candidates = candidates[: args.limit]

    reasoning = None
    if args.reasoning:
        reasoning = {"effort": args.reasoning_effort}
        if args.reasoning_exclude:
            reasoning["exclude"] = True

    request_payloads = [
        build_request_payload(
            candidate,
            spec=prompt_spec,
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            reasoning=reasoning,
        )
        for candidate in candidates
    ]
    write_jsonl(output_dir / "request_payloads.jsonl", request_payloads)

    output_path = output_dir / "specificity_rewrites.jsonl"
    existing = load_jsonl(output_path) if output_path.exists() else []
    existing_by_id = {row["candidate_id"]: row for row in existing}
    pending = [row for row in candidates if row["candidate_id"] not in existing_by_id]

    manifest: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "mode": "prepare_only" if args.prepare_only else "live_run",
        "candidates_jsonl": str(args.candidates_jsonl),
        "output_dir": str(output_dir),
        "prompt_path": args.prompt,
        "prompt_version": prompt_spec["prompt_version"],
        "model": args.model,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "workers": args.workers,
        "seed": args.seed,
        "requested_candidates": len(candidates),
        "existing_rewrites": len(existing),
        "pending_rewrites": len(pending),
        "reasoning_enabled": bool(reasoning),
        "reasoning_effort": reasoning.get("effort") if reasoning else "",
        "outputs": {
            "request_payloads": "request_payloads.jsonl",
            "specificity_rewrites": "specificity_rewrites.jsonl",
            "specificity_rewrites_ordered": "specificity_rewrites_ordered.jsonl",
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)

    if args.prepare_only:
        print(f"[specificity-rewriter] prepare-only request payloads: {len(request_payloads)}")
        print(f"[specificity-rewriter] wrote {output_dir / 'request_payloads.jsonl'}")
        return

    api_base = args.api_base or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for live specificity rewriting")

    print(
        f"[specificity-rewriter] rewriting {len(pending)} pending candidates "
        f"({len(existing)} already complete) with {args.workers} worker(s)...",
        file=sys.stderr,
        flush=True,
    )
    completed = dict(existing_by_id)
    futures: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending_iter = iter(pending)
        while len(futures) < args.workers:
            try:
                candidate = next(pending_iter)
            except StopIteration:
                break
            futures[
                executor.submit(
                    run_one,
                    candidate,
                    spec=prompt_spec,
                    model=args.model,
                    api_base=api_base,
                    api_key=api_key,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                    reasoning=reasoning,
                )
            ] = candidate

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                candidate = futures.pop(future)
                result = future.result()
                completed[result["candidate_id"]] = result
                append_jsonl(output_path, result)
                print(
                    f"[specificity-rewriter] {len(completed)}/{len(candidates)} "
                    f"{candidate['candidate_id']} {result['specificity_decision']}",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    next_candidate = next(pending_iter)
                except StopIteration:
                    continue
                futures[
                    executor.submit(
                        run_one,
                        next_candidate,
                        spec=prompt_spec,
                        model=args.model,
                        api_base=api_base,
                        api_key=api_key,
                        temperature=args.temperature,
                        max_output_tokens=args.max_output_tokens,
                        timeout=args.timeout,
                        max_retries=args.max_retries,
                        retry_backoff_seconds=args.retry_backoff_seconds,
                        reasoning=reasoning,
                    )
                ] = next_candidate

    ordered = [completed[row["candidate_id"]] for row in candidates if row["candidate_id"] in completed]
    write_jsonl(output_dir / "specificity_rewrites_ordered.jsonl", ordered)
    manifest["completed_rewrites"] = len(ordered)
    manifest["pending_rewrites"] = len(candidates) - len(ordered)
    write_json(output_dir / "run_manifest.json", manifest)
    print(f"[specificity-rewriter] completed {len(ordered)}/{len(candidates)}")


if __name__ == "__main__":
    main()
