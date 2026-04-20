#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import hashlib
import json
import os
import random
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import synthetic_summary_runtime as runtime

DEFAULT_INPUT_JSONL = "hf_dataset/corpora/recent_3y_balanced/arxiv_abstract_corpus_weighted_20000.jsonl"
DEFAULT_GENERATOR_PROMPT = "prompts/synthetic_summary/deepseek_v32_generator_v5_hybrid_sentence_case_minimal.json"
DEFAULT_REVIEWER_PROMPT = "prompts/synthetic_summary/qwen3_235b_a22b_2507_reviewer_classifier_v2_headline_baseline.json"
DEFAULT_SAMPLE_SOURCE_SPLIT = "smoke_test_100"
DEFAULT_FULL_SOURCE_SPLIT = "recent_3y_balanced_20000_full"
DEFAULT_GENERATOR_MAX_OUTPUT_TOKENS = 64
DEFAULT_REVIEWER_MAX_OUTPUT_TOKENS = 48

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
INVALID_JSON_ESCAPE_RE = re.compile(r'\\(?!["\\/bfnrtu])')
UGLY_FORMULA_RE = re.compile(
    r"""
    (?:
        \$|
        \\[A-Za-z]+|
        \^\(|\^\{|
        \bO\([^)]*\)|
        \bTheta\([^)]*\)|
        \bOmega\([^)]*\)|
        \b[a-zA-Z]+\^-[0-9]|
        \b[a-zA-Z]\^\([^)]+\)|
        𝔽[₀₁₂₃₄₅₆₇₈₉]+
    )
    """,
    re.VERBOSE,
)
BANNED_OPENINGS = (
    "this paper",
    "we ",
    "the authors",
    "in this work",
    "in this paper",
)
MIN_SUMMARY_WORDS = 8
MAX_SUMMARY_WORDS = 15
LOCAL_FILTER_DROP_REASONS = {
    "empty_summary": "empty summary",
    "too_short": "too short",
    "too_long": "over 15 words",
    "not_single_sentence": "not exactly one sentence",
    "banned_opening": "banned opening",
    "title_duplicate": "near-duplicate of title",
    "formula_shorthand": "formula-heavy notation",
}
LOCAL_FILTER_FLAGS = frozenset(LOCAL_FILTER_DROP_REASONS)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample arXiv abstracts, prepare DeepSeek prompt payloads, and optionally "
            "run a generator+reviewer synthetic summary pipeline."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        default=DEFAULT_INPUT_JSONL,
        help="Source abstract corpus in JSONL format.",
    )
    parser.add_argument(
        "--output-dir",
        default="hf_bucket/runs/deepseek_v32_smoke_100",
        help="Directory for run artifacts in bucket staging.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of source abstracts to sample.",
    )
    parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Select the full input corpus instead of drawing a random sample.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Number of deterministic shards for full-corpus runs.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index for full-corpus runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260407,
        help="Deterministic sampling seed.",
    )
    parser.add_argument(
        "--source-split",
        default=None,
        help="Label stored in final training rows.",
    )
    parser.add_argument(
        "--generator-prompt",
        default=DEFAULT_GENERATOR_PROMPT,
        help="Path to the generator prompt spec JSON.",
    )
    parser.add_argument(
        "--reviewer-prompt",
        default=DEFAULT_REVIEWER_PROMPT,
        help="Path to the reviewer prompt spec JSON.",
    )
    parser.add_argument(
        "--generator-model",
        default=os.getenv("OPENROUTER_MODEL", "deepseek-chat"),
        help="Model name for the generator call.",
    )
    parser.add_argument(
        "--reviewer-model",
        default=os.getenv("OPENROUTER_REVIEWER_MODEL") or "qwen/qwen3-235b-a22b-2507",
        help="Optional reviewer model override. Defaults to the generator model.",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv(
            "OPENROUTER_BASE_URL",
            os.getenv("DEEPSEEK_BASE_URL", "https://openrouter.ai/api/v1"),
        ),
        help="Base URL for an OpenAI-compatible chat completions API.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable that stores the API key.",
    )
    parser.add_argument(
        "--generator-temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for generator requests.",
    )
    parser.add_argument(
        "--reviewer-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for reviewer requests.",
    )
    parser.add_argument(
        "--reviewer-reasoning",
        action="store_true",
        help="Enable OpenRouter reasoning for reviewer requests.",
    )
    parser.add_argument(
        "--reviewer-reasoning-effort",
        default="medium",
        choices=("minimal", "low", "medium", "high", "xhigh"),
        help="Reasoning effort to request when reviewer reasoning is enabled.",
    )
    parser.add_argument(
        "--reviewer-reasoning-exclude",
        action="store_true",
        help="Use reviewer reasoning internally without returning reasoning text in the response.",
    )
    parser.add_argument(
        "--generator-max-output-tokens",
        type=int,
        default=None,
        help="Maximum completion tokens for generator requests.",
    )
    parser.add_argument(
        "--reviewer-max-output-tokens",
        type=int,
        default=None,
        help="Maximum completion tokens for reviewer requests.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Legacy fallback maximum completion tokens applied to both stages.",
    )
    parser.add_argument(
        "--generation-workers",
        type=int,
        default=1,
        help="Number of concurrent generator requests to run in-process.",
    )
    parser.add_argument(
        "--review-workers",
        type=int,
        default=1,
        help="Number of concurrent reviewer requests to run in-process.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds for each model call.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional delay between API calls.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum retries for transient request failures.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=5.0,
        help="Base backoff in seconds between request retries.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write sampled source rows and request payloads without calling the API.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_prompt_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = ("prompt_version", "role", "system_prompt", "user_prompt_template")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"Prompt spec {path} is missing keys: {', '.join(missing)}")
    return spec


def infer_reviewer_mode(spec: dict[str, Any]) -> str:
    role = str(spec.get("role", ""))
    if role == "reviewer_classifier":
        return "classifier"
    return "rewrite"


def sample_records(records: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > len(records):
        raise ValueError(
            f"sample_size {sample_size} exceeds source corpus size {len(records)}"
        )
    rng = random.Random(seed)
    chosen = rng.sample(records, sample_size)
    rng.shuffle(chosen)
    return chosen


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if args.shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    if args.generation_workers <= 0:
        raise ValueError("generation_workers must be positive")
    if args.review_workers <= 0:
        raise ValueError("review_workers must be positive")


def resolve_source_split(args: argparse.Namespace) -> str:
    if args.source_split:
        return args.source_split
    if args.full_corpus:
        return DEFAULT_FULL_SOURCE_SPLIT
    return DEFAULT_SAMPLE_SOURCE_SPLIT


def resolve_generator_max_output_tokens(args: argparse.Namespace) -> int:
    if args.generator_max_output_tokens is not None:
        return args.generator_max_output_tokens
    if args.max_output_tokens is not None:
        return args.max_output_tokens
    return DEFAULT_GENERATOR_MAX_OUTPUT_TOKENS


def resolve_reviewer_max_output_tokens(args: argparse.Namespace) -> int:
    if args.reviewer_max_output_tokens is not None:
        return args.reviewer_max_output_tokens
    if args.max_output_tokens is not None:
        return args.max_output_tokens
    return DEFAULT_REVIEWER_MAX_OUTPUT_TOKENS


def stable_shard_index(paper_id: str, shard_count: int) -> int:
    digest = hashlib.blake2b(paper_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % shard_count


def select_records(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], str]:
    if args.full_corpus:
        selected = [
            record
            for record in records
            if stable_shard_index(str(record.get("paper_id", "")), args.shard_count) == args.shard_index
        ]
        return selected, "full_corpus"
    return sample_records(records, args.sample_size, args.seed), "sample"


def render_prompt(
    spec: dict[str, Any],
    record: dict[str, Any],
    candidate_summary: str | None = None,
) -> dict[str, str]:
    variables = {
        "paper_id": record.get("paper_id", ""),
        "title": record.get("title", ""),
        "abstract": record.get("abstract", ""),
        "primary_category": record.get("primary_category", ""),
        "categories": ", ".join(record.get("categories", [])),
        "created_date": record.get("created_date", ""),
        "candidate_summary": candidate_summary or "",
    }
    return {
        "system": spec["system_prompt"],
        "user": spec["user_prompt_template"].format(**variables),
    }


def build_messages(spec: dict[str, Any], prompt: dict[str, str]) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]
    assistant_prefill = str(spec.get("assistant_prefill", ""))
    if assistant_prefill:
        messages.append({"role": "assistant", "content": assistant_prefill})
    return messages


def apply_assistant_prefill(raw_text: str, spec: dict[str, Any]) -> str:
    assistant_prefill = str(spec.get("assistant_prefill", ""))
    if not assistant_prefill:
        return raw_text
    stripped = raw_text.lstrip()
    if stripped.startswith(assistant_prefill):
        return stripped
    return assistant_prefill + stripped


def build_generation_request_rows(
    sample_rows: list[dict[str, Any]],
    generator_spec: dict[str, Any],
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in sample_rows:
        prompt = render_prompt(generator_spec, record)
        rows.append(
            {
                "paper_id": record["paper_id"],
                "prompt_version": generator_spec["prompt_version"],
                "model": model,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "response_format": generator_spec.get("response_format"),
                "plugins": generator_spec.get("plugins"),
                "messages": build_messages(generator_spec, prompt),
            }
        )
    return rows


def build_review_request_rows(
    reviewed_inputs: list[dict[str, Any]],
    reviewer_spec: dict[str, Any],
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in reviewed_inputs:
        if local_filter_flags(item["candidate_summary"], str(item["record"].get("title", ""))):
            continue
        prompt = render_prompt(
            reviewer_spec,
            item["record"],
            candidate_summary=item["candidate_summary"],
        )
        rows.append(
            {
                "paper_id": item["record"]["paper_id"],
                "prompt_version": reviewer_spec["prompt_version"],
                "model": model,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "response_format": reviewer_spec.get("response_format"),
                "plugins": reviewer_spec.get("plugins"),
                "candidate_summary": item["candidate_summary"],
                "messages": build_messages(reviewer_spec, prompt),
            }
        )
    return rows


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        healed = INVALID_JSON_ESCAPE_RE.sub(r"\\\\", cleaned)
        if healed != cleaned:
            try:
                return json.loads(healed)
            except json.JSONDecodeError:
                cleaned = healed
        match = JSON_BLOCK_RE.search(cleaned)
        if not match:
            raise
        block = match.group(0)
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            healed_block = INVALID_JSON_ESCAPE_RE.sub(r"\\\\", block)
            return json.loads(healed_block)


def extract_message_content(body: dict[str, Any]) -> str:
    message = body["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
                continue
            if isinstance(item.get("content"), str):
                parts.append(item["content"])
        return "".join(parts)
    if content is None:
        for key in ("output_text", "text"):
            value = message.get(key)
            if isinstance(value, str):
                return value
        raise ValueError("Model response did not include message content.")
    return str(content)


def post_chat_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_output_tokens: int,
    timeout: int,
    response_format: dict[str, Any] | None = None,
    plugins: list[dict[str, Any]] | None = None,
    reasoning: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }
    if response_format:
        payload["response_format"] = response_format
    if plugins:
        payload["plugins"] = plugins
    if reasoning:
        payload["reasoning"] = reasoning
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
    app_title = os.getenv("OPENROUTER_APP_TITLE", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if app_title:
        headers["X-Title"] = app_title
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = extract_message_content(body)
    return content, body


def is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, ValueError) and str(exc) == "Model response did not include message content.":
        return True
    if isinstance(exc, TimeoutError | socket.timeout):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        return isinstance(reason, TimeoutError | socket.timeout | OSError)
    return False


def request_with_retries(
    *,
    label: str,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_output_tokens: int,
    timeout: int,
    max_retries: int,
    retry_backoff_seconds: float,
    response_format: dict[str, Any] | None = None,
    plugins: list[dict[str, Any]] | None = None,
    reasoning: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    attempt = 0
    while True:
        attempt += 1
        try:
            return post_chat_completion(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
                response_format=response_format,
                plugins=plugins,
                reasoning=reasoning,
            )
        except Exception as exc:
            if attempt > max_retries or not is_retryable_error(exc):
                raise
            sleep_seconds = retry_backoff_seconds * attempt
            print(
                f"[retry] {label} failed on attempt {attempt}/{max_retries + 1}: "
                f"{type(exc).__name__}. sleeping {sleep_seconds:.1f}s before retry...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(sleep_seconds)


def request_json_with_retries(
    *,
    label: str,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_output_tokens: int,
    timeout: int,
    max_retries: int,
    retry_backoff_seconds: float,
    response_format: dict[str, Any] | None = None,
    plugins: list[dict[str, Any]] | None = None,
    assistant_prefill: str | None = None,
    reasoning: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    attempt = 0
    while True:
        attempt += 1
        raw_text, raw_response = request_with_retries(
            label=label,
            api_base=api_base,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            response_format=response_format,
            plugins=plugins,
            reasoning=reasoning,
        )
        try:
            parse_text = raw_text
            if assistant_prefill:
                parse_text = assistant_prefill + raw_text.lstrip()
            parsed = extract_json_object(parse_text)
            return parsed, parse_text, raw_response
        except json.JSONDecodeError:
            if attempt > max_retries:
                raise
            sleep_seconds = retry_backoff_seconds * attempt
            print(
                f"[retry] {label} returned malformed JSON on attempt {attempt}/{max_retries + 1}. "
                f"sleeping {sleep_seconds:.1f}s before retry...",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(sleep_seconds)


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def word_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    for raw in normalize_text(value).split():
        token = raw.strip(".,;:!?()[]{}<>\"'`“”‘’")
        if any(char.isalnum() for char in token):
            tokens.append(token)
    return tokens


def word_count(value: str) -> int:
    return len(word_tokens(value))


def normalized_token_signature(value: str) -> str:
    tokens = [token.casefold() for token in word_tokens(value)]
    return " ".join(tokens)


def sentence_count(value: str) -> int:
    normalized = normalize_text(value)
    if not normalized:
        return 0
    return len(re.findall(r"""[.!?]+(?:['")\]]+)?(?=\s|$)""", normalized))


def title_overlap_ratio(summary: str, title: str) -> float:
    summary_tokens = {token.casefold() for token in word_tokens(summary)}
    title_tokens = {token.casefold() for token in word_tokens(title)}
    if not summary_tokens:
        return 0.0
    if not title_tokens:
        return 0.0
    return len(summary_tokens & title_tokens) / len(summary_tokens)


def is_title_duplicate(summary: str, title: str) -> bool:
    summary_signature = normalized_token_signature(summary)
    title_signature = normalized_token_signature(title)
    if not summary_signature or not title_signature:
        return False
    if summary_signature == title_signature:
        return True
    return title_overlap_ratio(summary, title) >= 0.9


def has_formula_shorthand(summary: str) -> bool:
    return bool(UGLY_FORMULA_RE.search(summary))


def summary_flags(summary: str, title: str) -> list[str]:
    flags: list[str] = []
    normalized = normalize_text(summary).lower()
    if not normalized:
        return ["empty_summary"]
    words = word_count(summary)
    sentences = sentence_count(summary)
    overlap = title_overlap_ratio(summary, title)

    if words < MIN_SUMMARY_WORDS:
        flags.append("too_short")
    if words > MAX_SUMMARY_WORDS:
        flags.append("too_long")
    if sentences != 1:
        flags.append("not_single_sentence")
    if any(normalized.startswith(prefix) for prefix in BANNED_OPENINGS):
        flags.append("banned_opening")
    if has_formula_shorthand(summary):
        flags.append("formula_shorthand")
    if is_title_duplicate(summary, title):
        flags.append("title_duplicate")
    elif overlap >= 0.75:
        flags.append("high_title_overlap")
    return flags


def local_filter_flags(summary: str, title: str) -> list[str]:
    return [flag for flag in summary_flags(summary, title) if flag in LOCAL_FILTER_FLAGS]


def local_filter_reason(flags: list[str]) -> str:
    for flag in flags:
        reason = LOCAL_FILTER_DROP_REASONS.get(flag)
        if reason:
            return reason
    return "local_filter"


def iter_bounded_futures(
    items: list[Any],
    worker_count: int,
    submit_func: Any,
) -> tuple[int, Any, Future[Any]]:
    pending: dict[Future[Any], tuple[int, Any]] = {}
    next_index = 0

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        while next_index < len(items) and len(pending) < worker_count:
            item = items[next_index]
            pending[executor.submit(submit_func, item)] = (next_index, item)
            next_index += 1

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index, item = pending.pop(future)
                yield index, item, future
            while next_index < len(items) and len(pending) < worker_count:
                item = items[next_index]
                pending[executor.submit(submit_func, item)] = (next_index, item)
                next_index += 1


# Keep the runner behavior stable while using the shared runtime as the canonical
# implementation for builder and pipeline utility helpers.
load_dotenv = runtime.load_dotenv
load_jsonl = runtime.load_jsonl
write_jsonl = runtime.write_jsonl
append_jsonl_row = runtime.append_jsonl_row
load_prompt_spec = runtime.load_prompt_spec
render_prompt = runtime.render_prompt
build_messages = runtime.build_messages
extract_json_object = runtime.extract_json_object
request_json_with_retries = runtime.request_json_with_retries
normalize_text = runtime.normalize_text
word_tokens = runtime.word_tokens
word_count = runtime.word_count
sentence_count = runtime.sentence_count
title_overlap_ratio = runtime.title_overlap_ratio
summary_flags = runtime.summary_flags
local_filter_flags = runtime.local_filter_flags
local_filter_reason = runtime.local_filter_reason
iter_bounded_futures = runtime.iter_bounded_futures


def generate_candidate_request(
    record: dict[str, Any],
    generator_spec: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
) -> dict[str, Any]:
    prompt = render_prompt(generator_spec, record)
    parsed, raw_text, raw_response = request_json_with_retries(
        label=f"generation {record['paper_id']}",
        api_base=args.api_base,
        api_key=api_key,
        model=args.generator_model,
        messages=build_messages(generator_spec, prompt),
        temperature=args.generator_temperature,
        max_output_tokens=args.generator_max_output_tokens,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        response_format=generator_spec.get("response_format"),
        plugins=generator_spec.get("plugins"),
        assistant_prefill=generator_spec.get("assistant_prefill"),
    )
    candidate_summary = normalize_text(str(parsed.get("summary", "")))
    if args.sleep_seconds > 0:
        time.sleep(args.sleep_seconds)
    return {
        "paper_id": record["paper_id"],
        "prompt_version": generator_spec["prompt_version"],
        "model": args.generator_model,
        "raw_text": raw_text,
        "parsed": parsed,
        "candidate_summary": candidate_summary,
        "candidate_word_count": word_count(candidate_summary),
        "candidate_flags": summary_flags(candidate_summary, str(record["title"])),
        "api_response_id": raw_response.get("id"),
    }


def build_local_filter_artifacts(
    record: dict[str, Any],
    generation: dict[str, Any],
    reviewer_spec: dict[str, Any],
    args: argparse.Namespace,
    local_flags: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    review_row = {
        "paper_id": record["paper_id"],
        "prompt_version": reviewer_spec["prompt_version"],
        "model": "",
        "raw_text": "",
        "parsed": {},
        "decision": "drop",
        "summary": "",
        "summary_word_count": 0,
        "summary_flags": [],
        "drop_reason": local_filter_reason(local_flags),
        "candidate_summary": generation["candidate_summary"],
        "api_response_id": None,
        "reasoning_enabled": False,
        "reasoning_effort": "",
        "review_source": "local_filter",
        "local_filter_flags": local_flags,
    }
    dropped_row = {
        "paper_id": record["paper_id"],
        "title": record["title"],
        "abstract": record["abstract"],
        "summary": "",
        "primary_category": record["primary_category"],
        "categories": record.get("categories", []),
        "created_date": record.get("created_date"),
        "teacher_model": args.generator_model,
        "generator_prompt_version": generation["prompt_version"],
        "review_prompt_version": reviewer_spec["prompt_version"],
        "review_decision": "drop",
        "review_source": "local_filter",
        "source_split": args.source_split,
        "drop_reason": local_filter_reason(local_flags),
        "candidate_summary": generation["candidate_summary"],
        "local_filter_flags": local_flags,
    }
    return review_row, dropped_row


def review_candidate_request(
    item: tuple[dict[str, Any], dict[str, Any]],
    reviewer_spec: dict[str, Any],
    reviewer_model: str,
    reviewer_mode: str,
    reviewer_reasoning: dict[str, Any] | None,
    args: argparse.Namespace,
    api_key: str,
) -> dict[str, Any]:
    record, generation = item
    candidate_summary = generation["candidate_summary"]
    prompt = render_prompt(
        reviewer_spec,
        record,
        candidate_summary=candidate_summary,
    )
    parsed, raw_text, raw_response = request_json_with_retries(
        label=f"review {record['paper_id']}",
        api_base=args.api_base,
        api_key=api_key,
        model=reviewer_model,
        messages=build_messages(reviewer_spec, prompt),
        temperature=args.reviewer_temperature,
        max_output_tokens=args.reviewer_max_output_tokens,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        response_format=reviewer_spec.get("response_format"),
        plugins=reviewer_spec.get("plugins"),
        assistant_prefill=reviewer_spec.get("assistant_prefill"),
        reasoning=reviewer_reasoning,
    )
    decision = normalize_text(str(parsed.get("decision", "")).lower())
    final_summary = normalize_text(str(parsed.get("summary", "")))
    drop_reason = normalize_text(
        str(parsed.get("drop_reason", parsed.get("reason_span", "")))
    )
    if reviewer_mode == "classifier" and decision == "accept":
        final_summary = candidate_summary
    elif decision == "accept" and not final_summary:
        final_summary = candidate_summary
    flags = summary_flags(final_summary, str(record["title"])) if final_summary else []
    if args.sleep_seconds > 0:
        time.sleep(args.sleep_seconds)
    return {
        "paper_id": record["paper_id"],
        "parsed": parsed,
        "raw_text": raw_text,
        "api_response_id": raw_response.get("id"),
        "decision": decision,
        "final_summary": final_summary,
        "summary_flags": flags,
        "drop_reason": drop_reason,
        "candidate_summary": candidate_summary,
    }


def reviewer_expected_boolean_fields(spec: dict[str, Any]) -> list[str]:
    schema_fields = spec.get("response_schema", {})
    ordered_fields = []
    if isinstance(schema_fields, dict):
        ordered_fields.extend(schema_fields.keys())
    return [
        field
        for field in ordered_fields
        if field not in {"decision", "summary", "drop_reason", "reason_span"}
    ]


def generate_candidates(
    *,
    sample_rows: list[dict[str, Any]],
    generator_spec: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
    output_path: Path,
) -> list[dict[str, Any]]:
    existing_rows = load_jsonl(output_path) if output_path.exists() else []
    rows_by_id = {row["paper_id"]: row for row in existing_rows}
    if existing_rows:
        print(
            f"[generate] resuming with {len(existing_rows)}/{len(sample_rows)} candidates already saved.",
            file=sys.stderr,
            flush=True,
        )
    pending_records = [
        record for record in sample_rows if record["paper_id"] not in rows_by_id
    ]
    if pending_records:
        print(
            f"[generate] requesting {len(pending_records)} new candidates with {args.generation_workers} worker(s)...",
            file=sys.stderr,
            flush=True,
        )
        completed = len(rows_by_id)
        for _, record, future in iter_bounded_futures(
            pending_records,
            args.generation_workers,
            lambda item: generate_candidate_request(item, generator_spec, args, api_key),
        ):
            row = future.result()
            rows_by_id[record["paper_id"]] = row
            append_jsonl_row(output_path, row)
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(sample_rows):
                print(
                    f"[generate] saved candidate {completed}/{len(sample_rows)}.",
                    file=sys.stderr,
                    flush=True,
                )
    return [rows_by_id[record["paper_id"]] for record in sample_rows]


def review_candidates(
    *,
    sample_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    reviewer_spec: dict[str, Any],
    args: argparse.Namespace,
    api_key: str,
    review_output_path: Path,
    training_output_path: Path,
    dropped_output_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    generation_by_id = {row["paper_id"]: row for row in generation_rows}
    review_rows = load_jsonl(review_output_path) if review_output_path.exists() else []
    final_rows = load_jsonl(training_output_path) if training_output_path.exists() else []
    dropped_rows = load_jsonl(dropped_output_path) if dropped_output_path.exists() else []
    review_by_id = {row["paper_id"]: row for row in review_rows}
    final_by_id = {row["paper_id"]: row for row in final_rows}
    dropped_by_id = {row["paper_id"]: row for row in dropped_rows}

    reviewer_model = args.reviewer_model or args.generator_model
    reviewer_mode = infer_reviewer_mode(reviewer_spec)
    reviewer_reasoning: dict[str, Any] | None = None
    if args.reviewer_reasoning:
        reviewer_reasoning = {
            "enabled": True,
            "effort": args.reviewer_reasoning_effort,
        }
        if args.reviewer_reasoning_exclude:
            reviewer_reasoning["exclude"] = True
    if review_rows:
        print(
            f"[review] resuming with {len(review_rows)}/{len(sample_rows)} reviews already saved.",
            file=sys.stderr,
            flush=True,
        )
    for record in sample_rows:
        paper_id = record["paper_id"]
        if paper_id not in review_by_id:
            continue
        if paper_id in final_by_id or paper_id in dropped_by_id:
            continue
        generation = generation_by_id[paper_id]
        review_row = review_by_id[paper_id]
        decision = str(review_row.get("decision", "invalid")) or "invalid"
        final_summary = normalize_text(str(review_row.get("summary", "")))
        candidate_summary = str(review_row.get("candidate_summary", generation["candidate_summary"]))
        base_row = {
            "paper_id": record["paper_id"],
            "title": record["title"],
            "abstract": record["abstract"],
            "summary": final_summary,
            "primary_category": record["primary_category"],
            "categories": record.get("categories", []),
            "created_date": record.get("created_date"),
            "teacher_model": args.generator_model,
            "generator_prompt_version": generation["prompt_version"],
            "review_prompt_version": reviewer_spec["prompt_version"],
            "review_decision": decision,
            "review_source": review_row.get("review_source", "model"),
            "source_split": args.source_split,
        }
        if decision in {"rewrite", "accept"} and final_summary:
            final_by_id[paper_id] = base_row
            append_jsonl_row(training_output_path, base_row)
        else:
            dropped_row = {
                **base_row,
                "summary": "",
                "drop_reason": str(review_row.get("drop_reason", "")) or "reviewer_dropped",
                "candidate_summary": candidate_summary,
                "local_filter_flags": review_row.get("local_filter_flags", []),
            }
            dropped_by_id[paper_id] = dropped_row
            append_jsonl_row(dropped_output_path, dropped_row)

    pending_items: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in sample_rows:
        if record["paper_id"] in review_by_id:
            continue
        generation = generation_by_id[record["paper_id"]]
        candidate_summary = generation["candidate_summary"]
        local_flags = local_filter_flags(candidate_summary, str(record["title"]))
        if local_flags:
            review_row, dropped_row = build_local_filter_artifacts(
                record,
                generation,
                reviewer_spec,
                args,
                local_flags,
            )
            review_by_id[record["paper_id"]] = review_row
            dropped_by_id[record["paper_id"]] = dropped_row
            append_jsonl_row(review_output_path, review_row)
            append_jsonl_row(dropped_output_path, dropped_row)
            continue
        pending_items.append((record, generation))

    if pending_items:
        print(
            f"[review] sending {len(pending_items)} candidates to reviewer with {args.review_workers} worker(s)...",
            file=sys.stderr,
            flush=True,
        )
        completed = len(review_by_id)
        for _, item, future in iter_bounded_futures(
            pending_items,
            args.review_workers,
            lambda queued: review_candidate_request(
                queued,
                reviewer_spec,
                reviewer_model,
                reviewer_mode,
                reviewer_reasoning,
                args,
                api_key,
            ),
        ):
            record, generation = item
            result = future.result()
            decision = result["decision"]
            final_summary = result["final_summary"]
            drop_reason = result["drop_reason"]
            review_row = {
                "paper_id": record["paper_id"],
                "prompt_version": reviewer_spec["prompt_version"],
                "model": reviewer_model,
                "raw_text": result["raw_text"],
                "parsed": result["parsed"],
                "decision": decision,
                "summary": final_summary,
                "summary_word_count": word_count(final_summary),
                "summary_flags": result["summary_flags"],
                "drop_reason": drop_reason,
                "candidate_summary": result["candidate_summary"],
                "api_response_id": result["api_response_id"],
                "reasoning_enabled": bool(reviewer_reasoning),
                "reasoning_effort": reviewer_reasoning.get("effort") if reviewer_reasoning else "",
                "review_source": "model",
                "local_filter_flags": [],
            }
            if reviewer_mode == "classifier":
                for field in reviewer_expected_boolean_fields(reviewer_spec):
                    if field in result["parsed"]:
                        review_row[field] = result["parsed"][field]
                if "reason_span" in result["parsed"]:
                    review_row["reason_span"] = result["parsed"]["reason_span"]
            review_by_id[record["paper_id"]] = review_row
            append_jsonl_row(review_output_path, review_row)

            base_row = {
                "paper_id": record["paper_id"],
                "title": record["title"],
                "abstract": record["abstract"],
                "summary": final_summary,
                "primary_category": record["primary_category"],
                "categories": record.get("categories", []),
                "created_date": record.get("created_date"),
                "teacher_model": args.generator_model,
                "generator_prompt_version": generation["prompt_version"],
                "review_prompt_version": reviewer_spec["prompt_version"],
                "review_decision": decision or "invalid",
                "review_source": "model",
                "source_split": args.source_split,
            }

            if decision in {"rewrite", "accept"} and final_summary:
                final_by_id[record["paper_id"]] = base_row
                append_jsonl_row(training_output_path, base_row)
            else:
                dropped_row = {
                    **base_row,
                    "summary": "",
                    "drop_reason": drop_reason or "reviewer_dropped",
                    "candidate_summary": result["candidate_summary"],
                    "review_decision": decision or "invalid",
                }
                dropped_by_id[record["paper_id"]] = dropped_row
                append_jsonl_row(dropped_output_path, dropped_row)
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(sample_rows):
                print(
                    f"[review] saved review {completed}/{len(sample_rows)}.",
                    file=sys.stderr,
                    flush=True,
                )

    ordered_review_rows = [review_by_id[record["paper_id"]] for record in sample_rows]
    ordered_final_rows = [
        final_by_id[record["paper_id"]]
        for record in sample_rows
        if record["paper_id"] in final_by_id
    ]
    ordered_dropped_rows = [
        dropped_by_id[record["paper_id"]]
        for record in sample_rows
        if record["paper_id"] in dropped_by_id
    ]
    return ordered_review_rows, ordered_final_rows, ordered_dropped_rows


def write_manifest(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    generator_spec: dict[str, Any],
    reviewer_spec: dict[str, Any],
    sample_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]] | None,
    review_rows: list[dict[str, Any]] | None,
    final_rows: list[dict[str, Any]] | None,
    dropped_rows: list[dict[str, Any]] | None,
    mode: str,
) -> None:
    umbrella_counts = Counter(row.get("umbrella_category", "other") for row in sample_rows)
    category_counts = Counter(row.get("primary_category", "unknown") for row in sample_rows)

    manifest = {
        "mode": mode,
        "input_jsonl": args.input_jsonl,
        "output_dir": str(output_dir),
        "sample_size": len(sample_rows),
        "input_row_count": args.input_row_count,
        "selected_row_count": args.selected_row_count,
        "selection_mode": args.selection_mode,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "seed": args.seed,
        "source_split": args.source_split,
        "generator_prompt_version": generator_spec["prompt_version"],
        "review_prompt_version": reviewer_spec["prompt_version"],
        "reviewer_role": reviewer_spec.get("role"),
        "reviewer_mode": infer_reviewer_mode(reviewer_spec),
        "generator_model": args.generator_model,
        "reviewer_model": args.reviewer_model or args.generator_model,
        "generation_workers": args.generation_workers,
        "review_workers": args.review_workers,
        "generator_max_output_tokens": args.generator_max_output_tokens,
        "reviewer_max_output_tokens": args.reviewer_max_output_tokens,
        "reviewer_reasoning_enabled": bool(args.reviewer_reasoning),
        "reviewer_reasoning_effort": args.reviewer_reasoning_effort if args.reviewer_reasoning else "",
        "reviewer_reasoning_exclude": bool(
            args.reviewer_reasoning and args.reviewer_reasoning_exclude
        ),
        "sample_umbrella_counts": dict(sorted(umbrella_counts.items())),
        "sample_primary_category_counts": dict(sorted(category_counts.items())),
    }

    if generation_rows is not None:
        manifest["generated_rows"] = len(generation_rows)
        manifest["generator_flag_counts"] = dict(
            sorted(
                Counter(
                    flag
                    for row in generation_rows
                    for flag in row.get("candidate_flags", [])
                ).items()
            )
        )

    if review_rows is not None:
        manifest["reviewed_rows"] = len(review_rows)
        manifest["model_reviewed_rows"] = sum(
            row.get("review_source") == "model" for row in review_rows
        )
        manifest["locally_filtered_rows"] = sum(
            row.get("review_source") == "local_filter" for row in review_rows
        )
        manifest["review_decisions"] = dict(
            sorted(Counter(row.get("decision", "invalid") for row in review_rows).items())
        )
        manifest["local_filter_flag_counts"] = dict(
            sorted(
                Counter(
                    flag
                    for row in review_rows
                    for flag in row.get("local_filter_flags", [])
                ).items()
            )
        )
        manifest["review_flag_counts"] = dict(
            sorted(
                Counter(
                    flag
                    for row in review_rows
                    for flag in row.get("summary_flags", [])
                ).items()
            )
        )
        if infer_reviewer_mode(reviewer_spec) == "classifier":
            for field in reviewer_expected_boolean_fields(reviewer_spec):
                if not any(field in row for row in review_rows):
                    continue
                manifest[f"{field}_counts"] = dict(
                    sorted(
                        Counter(
                            str(row[field]).lower()
                            for row in review_rows
                            if field in row
                        ).items()
                    )
                )

    if final_rows is not None and dropped_rows is not None:
        manifest["final_training_rows"] = len(final_rows)
        manifest["dropped_rows"] = len(dropped_rows)

    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    load_dotenv(Path(".env"))
    args = parse_args()
    validate_args(args)
    args.source_split = resolve_source_split(args)
    args.generator_max_output_tokens = resolve_generator_max_output_tokens(args)
    args.reviewer_max_output_tokens = resolve_reviewer_max_output_tokens(args)
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator_spec = load_prompt_spec(Path(args.generator_prompt))
    reviewer_spec = load_prompt_spec(Path(args.reviewer_prompt))

    print("[run] loading source corpus...", file=sys.stderr, flush=True)
    records = load_jsonl(input_path)
    sample_rows, selection_mode = select_records(records, args)
    args.selection_mode = selection_mode
    args.input_row_count = len(records)
    args.selected_row_count = len(sample_rows)
    if selection_mode == "full_corpus":
        print(
            f"[run] loaded {len(records):,} source rows; selected shard {args.shard_index}/{args.shard_count} with {len(sample_rows):,} rows.",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"[run] loaded {len(records):,} source rows; sampled {len(sample_rows):,} rows.",
            file=sys.stderr,
            flush=True,
        )

    sample_path = output_dir / "sample_source_rows.jsonl"
    write_jsonl(sample_path, sample_rows)
    print(f"[run] wrote selected source rows to {sample_path}", file=sys.stderr, flush=True)

    generation_request_rows = build_generation_request_rows(
        sample_rows,
        generator_spec,
        args.generator_model,
        args.generator_temperature,
        args.generator_max_output_tokens,
    )
    write_jsonl(output_dir / "generation_requests.jsonl", generation_request_rows)
    generation_output_path = output_dir / "generation_outputs.jsonl"
    review_output_path = output_dir / "review_outputs.jsonl"
    training_output_path = output_dir / "training_rows.jsonl"
    dropped_output_path = output_dir / "dropped_rows.jsonl"

    api_key = os.getenv(args.api_key_env, "")
    if args.prepare_only or not api_key:
        if not args.prepare_only and not api_key:
            print(
                f"[run] {args.api_key_env} is not set; falling back to prepare-only mode.",
                file=sys.stderr,
                flush=True,
            )
        print("[run] wrote generator request payloads only.", file=sys.stderr, flush=True)
        write_manifest(
            output_dir=output_dir,
            args=args,
            generator_spec=generator_spec,
            reviewer_spec=reviewer_spec,
            sample_rows=sample_rows,
            generation_rows=None,
            review_rows=None,
            final_rows=None,
            dropped_rows=None,
            mode="prepare_only",
        )
        return

    generation_rows = generate_candidates(
        sample_rows=sample_rows,
        generator_spec=generator_spec,
        args=args,
        api_key=api_key,
        output_path=generation_output_path,
    )

    review_request_rows = build_review_request_rows(
        [
            {
                "record": row,
                "candidate_summary": generation_row["candidate_summary"],
            }
            for row, generation_row in zip(sample_rows, generation_rows, strict=True)
        ],
        reviewer_spec,
        args.reviewer_model or args.generator_model,
        args.reviewer_temperature,
        args.reviewer_max_output_tokens,
    )
    write_jsonl(output_dir / "review_requests.jsonl", review_request_rows)

    review_rows, final_rows, dropped_rows = review_candidates(
        sample_rows=sample_rows,
        generation_rows=generation_rows,
        reviewer_spec=reviewer_spec,
        args=args,
        api_key=api_key,
        review_output_path=review_output_path,
        training_output_path=training_output_path,
        dropped_output_path=dropped_output_path,
    )

    write_manifest(
        output_dir=output_dir,
        args=args,
        generator_spec=generator_spec,
        reviewer_spec=reviewer_spec,
        sample_rows=sample_rows,
        generation_rows=generation_rows,
        review_rows=review_rows,
        final_rows=final_rows,
        dropped_rows=dropped_rows,
        mode="full_run",
    )
    print(
        f"[run] finished full run with {len(final_rows)} kept and {len(dropped_rows)} dropped.",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
