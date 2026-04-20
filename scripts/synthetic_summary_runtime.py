from __future__ import annotations

import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

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


def render_prompt(
    spec: dict[str, Any],
    record: dict[str, Any],
    candidate_summary: str | None = None,
    extra_variables: dict[str, Any] | None = None,
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
    if extra_variables:
        variables.update(extra_variables)
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
