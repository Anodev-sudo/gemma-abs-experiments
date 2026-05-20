#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
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

DEFAULT_PACKET = "hf_bucket/experiments/sft/q5_step2400_sft700_curated_v3/blind_review_ckpt700_vs_v3ckpt110_200_ignore_length/v3_wins_detail_packet.jsonl"
DEFAULT_PROMPT = "prompts/synthetic_summary/kimi_k2_6_q5_v3_win_concise_repair_v1.json"
DEFAULT_OUTPUT_ROOT = "hf_bucket/experiments/sft/q5_step2400_sft700_curated_v3/kimi_v3_win_concise_repairs_v1"
DEFAULT_MODEL = "moonshotai/kimi-k2.6"
VALID_DECISIONS = {"rewrite", "drop"}


def reasoning_config(effort: str | None) -> dict[str, Any] | None:
    effort = str(effort or "").strip()
    if not effort:
        return None
    return {"effort": effort, "exclude": True}

SYSTEM = (
    "You write one-sentence paper summaries for a mobile research feed. "
    "Keep them concrete, readable, and useful. Output only the summary sentence."
)
USER_TMPL = (
    "The app already shows the title, author list, and date. "
    "Write one sentence that tells a reader why they might open this paper. "
    "Aim for 10 to 15 words. Avoid repeating the title. "
    'Avoid filler like "This paper" or "We propose". '
    "Do not mention authors or dates. "
    "Keep the wording natural and specific.\n\n"
    "Title:\n{title}\n\n"
    "Abstract:\n{abstract}"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def word_count(text: str) -> int:
    return len([p for p in normalize_text(text).split(" ") if p])


def done_ids(path: Path) -> set[str]:
    ids = set()
    if not path.exists():
        return ids
    for line in path.open(encoding="utf-8"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("repair_id"):
            ids.add(str(d["repair_id"]))
    return ids


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", default=DEFAULT_PACKET)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=420)
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    ap.add_argument("--continue-on-error", action="store_true", default=True)
    return ap.parse_args()


def prepare_rows(packet_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(packet_rows, start=1):
        rows.append({
            **row,
            "repair_id": f"v3win_kimi_repair:{idx:04d}",
            "current_summary": row.get("current_checkpoint_700_summary", ""),
            "v3_summary": row.get("v3_checkpoint_110_summary", ""),
            "source_split": "q5_sft700_v3_win_kimi_concise_repair_v1",
        })
    return rows


def payload_for(row: dict[str, Any], spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    prompt = render_prompt(spec, row, extra_variables={
        "current_summary": row.get("current_summary", ""),
        "v3_summary": row.get("v3_summary", ""),
        "blind_short_reason": row.get("blind_short_reason", ""),
    })
    payload = {
        "repair_id": row["repair_id"],
        "paper_id": row["paper_id"],
        "model": args.model,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "prompt_version": spec["prompt_version"],
        "messages": build_messages(spec, prompt),
    }
    if spec.get("response_format"):
        payload["response_format"] = spec["response_format"]
    if (reasoning := reasoning_config(args.reasoning_effort)):
        payload["reasoning"] = reasoning
    return payload


def as_training_row(row: dict[str, Any]) -> dict[str, Any]:
    user = SYSTEM + "\n\n" + USER_TMPL.format(title=row["title"], abstract=row["abstract"])
    out = {
        "paper_id": row["paper_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "summary": row["repair_summary"],
        "primary_category": row.get("primary_category", ""),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date", ""),
        "teacher_model": row.get("writer_model", DEFAULT_MODEL),
        "generator_prompt_version": row.get("writer_prompt_version", ""),
        "review_prompt_version": "local_filters_only",
        "source_split": row.get("source_split", "q5_sft700_v3_win_kimi_concise_repair_v1"),
        "patch_component": "kimi_v3_win_concise_repair",
        "repair_id": row.get("repair_id", ""),
        "current_checkpoint_700_summary": row.get("current_summary", ""),
        "v3_checkpoint_110_summary": row.get("v3_summary", ""),
        "number_policy": row.get("number_policy", ""),
        "writer_evidence_span": row.get("writer_evidence_span", ""),
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["repair_summary"]},
        ],
    }
    return out


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    api_base = args.api_base or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    spec = load_prompt_spec(Path(args.prompt))
    packet = load_jsonl(Path(args.packet))
    if args.limit is not None:
        packet = packet[: args.limit]
    candidates = prepare_rows(packet)
    write_jsonl(output_root / "candidate_pool.jsonl", candidates)

    req_path = output_root / "requests.jsonl"
    out_path = output_root / "outputs.jsonl"
    err_path = output_root / "errors.jsonl"
    done = done_ids(out_path)
    todo = [r for r in candidates if r["repair_id"] not in done]
    print(f"[kimi-repair] candidates={len(candidates)} done={len(done)} todo={len(todo)}")

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        payload = payload_for(row, spec, args)
        append_jsonl(req_path, payload)
        parsed, raw_text, _ = request_json_with_retries(
            label=row["repair_id"],
            api_base=api_base,
            api_key=api_key,
            model=args.model,
            messages=payload["messages"],
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_backoff_seconds=args.retry_backoff_seconds,
            response_format=payload.get("response_format"),
            reasoning=payload.get("reasoning"),
            attempt_log_dir=output_root / "raw_json_attempts",
        )
        raw_summary = normalize_text(str(parsed.get("summary", "")))
        decision = str(parsed.get("decision", "")).strip().lower()
        if decision not in VALID_DECISIONS:
            decision = "rewrite" if raw_summary else "drop"
        if decision == "drop" and raw_summary:
            decision = "rewrite"
        summary = raw_summary if decision == "rewrite" else ""
        wc = word_count(summary)
        flags = local_filter_flags(summary, row.get("title", "")) if summary else ["empty_summary"]
        if decision == "rewrite" and not (10 <= wc <= 13):
            flags.append("outside_10_13_words")
        result = {
            **row,
            "writer_model": args.model,
            "writer_prompt_version": spec["prompt_version"],
            "writer_decision": decision,
            "writer_issue_tags": parsed.get("issue_tags", []),
            "number_policy": parsed.get("number_policy", ""),
            "repair_summary": summary,
            "writer_word_count": wc,
            "writer_reported_word_count": parsed.get("word_count", 0),
            "writer_evidence_span": str(parsed.get("evidence_span", "")),
            "writer_reason_span": str(parsed.get("reason_span", "")),
            "writer_local_filter_flags": flags,
            "raw_writer_text": raw_text,
        }
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, row): row for row in todo}
        for i, fut in enumerate(as_completed(futs), start=1):
            row = futs[fut]
            try:
                result = fut.result()
                append_jsonl(out_path, result)
                if i % 10 == 0:
                    print(f"[kimi-repair] completed {i}/{len(todo)}")
            except Exception as exc:
                err = {"repair_id": row["repair_id"], "paper_id": row["paper_id"], "error": str(exc), "error_type": type(exc).__name__}
                append_jsonl(err_path, err)
                print(f"[kimi-repair:error] {row['repair_id']} {type(exc).__name__}: {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    raise

    rows = load_jsonl(out_path) if out_path.exists() else []
    accepted = [r for r in rows if r.get("writer_decision") == "rewrite" and not r.get("writer_local_filter_flags")]
    rejected = [r for r in rows if r not in accepted]
    materialized_dir = output_root / "materialized"
    materialized_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(materialized_dir / "accepted_repairs.jsonl", accepted)
    write_jsonl(materialized_dir / "rejected_repairs.jsonl", rejected)
    write_jsonl(materialized_dir / "training_rows.jsonl", [as_training_row(r) for r in accepted])

    manifest = {
        "created_at": utc_now(),
        "prompt": args.prompt,
        "prompt_version": spec["prompt_version"],
        "model": args.model,
        "packet": args.packet,
        "candidate_rows": len(candidates),
        "output_rows": len(rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "decision_counts": dict(Counter(r.get("writer_decision") for r in rows)),
        "local_flag_counts": dict(Counter(flag for r in rows for flag in r.get("writer_local_filter_flags", []))),
        "word_count_counts": dict(Counter(r.get("writer_word_count") for r in accepted)),
        "number_policy_counts": dict(Counter(r.get("number_policy") for r in accepted)),
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
