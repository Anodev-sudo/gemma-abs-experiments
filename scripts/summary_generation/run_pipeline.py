#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPTS_DIR.parent
for import_path in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

try:
    from synthetic_summary_runtime import (
        build_messages,
        exception_summary,
        has_formula_shorthand,
        iter_bounded_futures,
        load_dotenv,
        load_jsonl,
        load_prompt_spec,
        normalize_text,
        request_json_with_retries,
        sentence_count,
        title_overlap_ratio,
        word_count,
        write_jsonl,
    )
except ModuleNotFoundError:
    from scripts.synthetic_summary_runtime import (
        build_messages,
        exception_summary,
        has_formula_shorthand,
        iter_bounded_futures,
        load_dotenv,
        load_jsonl,
        load_prompt_spec,
        normalize_text,
        request_json_with_retries,
        sentence_count,
        title_overlap_ratio,
        word_count,
        write_jsonl,
    )


STAGES = (
    "prepare",
    "probe-routes",
    "write",
    "local-filter",
    "verify",
    "retry-failed",
    "materialize",
    "report",
    "all",
)
DECISION_ACCEPTED_WRITER = {"rewrite", "accept"}
DECISION_ACCEPTED_REVIEWER = {"accept"}
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_FORBIDDEN_PREFIXES = (
    "this paper",
    "we ",
    "the authors",
    "in this work",
    "in this paper",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in {".json"}:
        raise ValueError("Only JSON configs are supported; PyYAML is not a repo dependency.")
    config = read_json(path)
    config["_config_path"] = str(path)
    return config


def repo_path(value: str | Path) -> Path:
    return Path(value)


def get_path(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def list_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def string_list(value: Any) -> list[str]:
    return [str(item) for item in list_value(value) if str(item).strip()]


def normalize_for_prompt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def format_prompt(spec: dict[str, Any], record: dict[str, Any]) -> dict[str, str]:
    variables = {key: normalize_for_prompt(value) for key, value in record.items()}
    variables.setdefault("paper_id", "")
    variables.setdefault("title", "")
    variables.setdefault("abstract", "")
    variables.setdefault("primary_category", "")
    variables.setdefault("categories", "")
    variables.setdefault("created_date", "")
    variables.setdefault("candidate_summary", variables.get("generated_summary", ""))
    variables.setdefault("summary", variables.get("generated_summary", ""))
    return {
        "system": spec["system_prompt"],
        "user": spec["user_prompt_template"].format(**variables),
    }


def reasoning_config(route: dict[str, Any], role_config: dict[str, Any]) -> dict[str, Any] | None:
    effort = str(route.get("reasoning_effort", role_config.get("reasoning_effort", ""))).strip()
    if not effort:
        return None
    return {"effort": effort, "exclude": True}


def provider_config(route: dict[str, Any]) -> dict[str, Any] | None:
    provider = route.get("provider")
    if isinstance(provider, dict):
        return provider
    provider_name = str(provider or "").strip()
    if not provider_name:
        return None
    return {"order": [provider_name], "allow_fallbacks": False}


def route_id(route: dict[str, Any], index: int) -> str:
    return str(route.get("route_id") or route.get("id") or f"route_{index:02d}")


def route_model(route: dict[str, Any], role_config: dict[str, Any]) -> str:
    model = str(route.get("model") or role_config.get("model") or "").strip()
    if not model:
        raise ValueError("Each role route needs a model.")
    return model


def role_routes(role_config: dict[str, Any]) -> list[dict[str, Any]]:
    routes = list_value(role_config.get("routes"))
    if routes:
        return [dict(route) for route in routes]
    model = str(role_config.get("model") or "").strip()
    if not model:
        raise ValueError("Role config needs either routes or model.")
    return [{"route_id": "default", "model": model}]


def response_format_for_route(
    prompt_spec: dict[str, Any],
    route: dict[str, Any],
    role_config: dict[str, Any],
) -> dict[str, Any] | None:
    mode = str(route.get("response_format_mode", role_config.get("response_format_mode", "schema")))
    if mode in {"none", "plain_json", "disabled"}:
        return None
    return prompt_spec.get("response_format")


def selected_route_path(output_root: Path) -> Path:
    return output_root / "route_probe" / "route_report.json"


def stage_dir_name(role: str) -> str:
    return "writer_run" if role == "writer" else "reviewer_run"


def input_path_for_stage(config: dict[str, Any], role: str) -> Path:
    output_root = Path(config["output"]["run_root"])
    if role == "writer":
        return output_root / "candidate_pool" / "candidates.jsonl"
    return output_root / "local_eligible_for_review" / "writer_outputs_ordered.jsonl"


def output_path_for_stage(config: dict[str, Any], role: str) -> Path:
    output_root = Path(config["output"]["run_root"])
    if role == "writer":
        return output_root / "writer_run" / "writer_outputs.jsonl"
    return output_root / "reviewer_run" / "reviewer_outputs.jsonl"


def ordered_output_path_for_stage(config: dict[str, Any], role: str) -> Path:
    output_root = Path(config["output"]["run_root"])
    if role == "writer":
        return output_root / "writer_run" / "writer_outputs_ordered.jsonl"
    return output_root / "reviewer_run" / "reviewer_outputs_ordered.jsonl"


def failed_output_path_for_stage(config: dict[str, Any], role: str) -> Path:
    output_root = Path(config["output"]["run_root"])
    if role == "writer":
        return output_root / "writer_run" / "failed_writer_outputs.jsonl"
    return output_root / "reviewer_run" / "failed_reviewer_outputs.jsonl"


def normalize_candidate(
    row: dict[str, Any],
    *,
    index: int,
    config: dict[str, Any],
    cli_limit: int | None,
) -> dict[str, Any]:
    input_config = config.get("input", {})
    candidate_id_field = str(input_config.get("candidate_id_field") or "").strip()
    prefix = str(input_config.get("candidate_prefix") or "summary_generation").strip()
    if candidate_id_field and row.get(candidate_id_field):
        candidate_id = str(row[candidate_id_field])
    elif row.get("candidate_id"):
        candidate_id = str(row["candidate_id"])
    else:
        candidate_id = f"{prefix}:{index:04d}"
    candidate = dict(row)
    candidate["candidate_id"] = candidate_id
    candidate.setdefault("source_split", get_path(config, "materialize", "source_split", default=prefix))
    candidate.setdefault("paper_id", str(row.get("paper_id", "")))
    candidate.setdefault("title", str(row.get("title", "")))
    candidate.setdefault("abstract", str(row.get("abstract", "")))
    candidate.setdefault("categories", row.get("categories", []))
    candidate.setdefault("primary_category", row.get("primary_category"))
    candidate.setdefault("created_date", row.get("created_date"))
    candidate.setdefault("reviewer_note", row.get("note", row.get("blind_review_note", "")))
    candidate.setdefault("q5_hardening_review_id", row.get("review_id", row.get("q5_hardening_review_id")))
    if cli_limit is not None:
        candidate["_cli_limit"] = cli_limit
    return candidate


def command_prepare(config: dict[str, Any], *, cli_limit: int | None = None) -> list[dict[str, Any]]:
    input_config = config["input"]
    rows = load_jsonl(Path(input_config["path"]))
    configured_limit = input_config.get("limit")
    limit = cli_limit if cli_limit is not None else configured_limit
    if limit is not None:
        rows = rows[: int(limit)]
    candidates = [
        normalize_candidate(row, index=index, config=config, cli_limit=cli_limit)
        for index, row in enumerate(rows, start=1)
    ]
    output_root = Path(config["output"]["run_root"])
    output_dir = output_root / "candidate_pool"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "candidates.jsonl", candidates)
    count_fields = string_list(input_config.get("count_fields"))
    tag_fields = string_list(input_config.get("tag_fields"))
    manifest: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "config_path": config.get("_config_path", ""),
        "input_jsonl": input_config["path"],
        "output_jsonl": str(output_dir / "candidates.jsonl"),
        "candidate_rows": len(candidates),
        "limit": limit,
        "source_split": get_path(config, "materialize", "source_split", default=""),
    }
    for field in count_fields:
        manifest[f"{field}_counts"] = dict(Counter(str(row.get(field)) for row in candidates))
    for field in tag_fields:
        manifest[f"{field}_counts"] = dict(
            Counter(tag for row in candidates for tag in string_list(row.get(field)))
        )
    write_json(output_dir / "run_manifest.json", manifest)
    print(f"[summary-generation:prepare] wrote {len(candidates)} candidates")
    return candidates


def load_candidates(config: dict[str, Any], *, cli_limit: int | None = None) -> list[dict[str, Any]]:
    candidate_path = Path(config["output"]["run_root"]) / "candidate_pool" / "candidates.jsonl"
    if not candidate_path.exists():
        return command_prepare(config, cli_limit=cli_limit)
    candidates = load_jsonl(candidate_path)
    if cli_limit is not None:
        return candidates[:cli_limit]
    return candidates


def role_prompt_spec(role_config: dict[str, Any]) -> dict[str, Any]:
    prompt_spec = load_prompt_spec(Path(role_config["prompt_path"]))
    if bool_value(role_config.get("disable_response_format")):
        prompt_spec = dict(prompt_spec)
        prompt_spec.pop("response_format", None)
    return prompt_spec


def build_payload(
    row: dict[str, Any],
    *,
    role: str,
    role_config: dict[str, Any],
    prompt_spec: dict[str, Any],
    route: dict[str, Any],
    route_index: int,
) -> dict[str, Any]:
    prompt = format_prompt(prompt_spec, row)
    model = route_model(route, role_config)
    max_output_tokens = int(
        route.get("max_output_tokens", role_config.get("max_output_tokens", 192))
    )
    temperature = float(route.get("temperature", role_config.get("temperature", 0.0)))
    payload: dict[str, Any] = {
        "candidate_id": row["candidate_id"],
        "paper_id": row.get("paper_id", ""),
        "role": role,
        "route_id": route_id(route, route_index),
        "model": model,
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "prompt_version": prompt_spec["prompt_version"],
        "messages": build_messages(prompt_spec, prompt),
    }
    response_format = response_format_for_route(prompt_spec, route, role_config)
    if response_format:
        payload["response_format"] = response_format
    if prompt_spec.get("plugins"):
        payload["plugins"] = prompt_spec["plugins"]
    reasoning = reasoning_config(route, role_config)
    if reasoning:
        payload["reasoning"] = reasoning
    provider = provider_config(route)
    if provider:
        payload["provider"] = provider
    return payload


def route_ladder_from_report(
    config: dict[str, Any],
    *,
    role: str,
    no_route_probe: bool,
) -> list[tuple[int, dict[str, Any]]]:
    role_config = config[role]
    routes = role_routes(role_config)
    report_path = selected_route_path(Path(config["output"]["run_root"]))
    selected_id = ""
    if not no_route_probe and report_path.exists():
        report = read_json(report_path)
        selected_id = str(get_path(report, "selected_routes", role, "route_id", default=""))
    if selected_id:
        for index, route in enumerate(routes):
            if route_id(route, index) == selected_id:
                later = [
                    (other_index, other_route)
                    for other_index, other_route in enumerate(routes)
                    if other_index > index
                ]
                earlier = [
                    (other_index, other_route)
                    for other_index, other_route in enumerate(routes)
                    if other_index < index
                ]
                return [(index, route), *later, *earlier]
    return list(enumerate(routes))


def get_api_credentials(args: argparse.Namespace) -> tuple[str, str]:
    load_dotenv(Path(".env"))
    api_base = args.api_base or os.getenv("OPENROUTER_BASE_URL", DEFAULT_API_BASE)
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    return api_base, api_key


def probe_record(config: dict[str, Any]) -> dict[str, Any]:
    candidates = load_candidates(config)
    if candidates:
        record = dict(candidates[0])
    else:
        record = {"candidate_id": "probe:0001", "paper_id": "probe", "title": "", "abstract": ""}
    record.setdefault("generated_summary", record.get("q8_summary") or "A method is checked for faithful concise paper summaries.")
    record.setdefault("correction_summary", record["generated_summary"])
    record.setdefault("candidate_summary", record["generated_summary"])
    record.setdefault("correction_writer", "probe")
    record.setdefault("correction_issue_tags", "")
    record.setdefault("correction_evidence_span", "")
    record.setdefault("correction_reason_span", "")
    record.setdefault("q5_summary", record.get("q5_summary", ""))
    record.setdefault("q8_summary", record.get("q8_summary", ""))
    record.setdefault("q5_issue_tags", record.get("q5_issue_tags", []))
    record.setdefault("reviewer_note", record.get("reviewer_note", record.get("blind_review_note", "")))
    return record


def probe_one_route(
    *,
    role: str,
    role_config: dict[str, Any],
    prompt_spec: dict[str, Any],
    route: dict[str, Any],
    route_index: int,
    record: dict[str, Any],
    api_base: str,
    api_key: str,
    timeout: int,
) -> dict[str, Any]:
    started = time.monotonic()
    payload = build_payload(
        record,
        role=role,
        role_config=role_config,
        prompt_spec=prompt_spec,
        route=route,
        route_index=route_index,
    )
    route_name = route_id(route, route_index)
    try:
        parsed, raw_text, raw_response = request_json_with_retries(
            label=f"probe {role} {route_name}",
            api_base=api_base,
            api_key=api_key,
            model=payload["model"],
            messages=payload["messages"],
            temperature=float(payload["temperature"]),
            max_output_tokens=int(payload["max_output_tokens"]),
            timeout=timeout,
            max_retries=int(route.get("probe_retries", role_config.get("probe_retries", 0))),
            retry_backoff_seconds=float(
                route.get("retry_backoff_seconds", role_config.get("retry_backoff_seconds", 1.0))
            ),
            response_format=payload.get("response_format"),
            plugins=payload.get("plugins"),
            provider=payload.get("provider"),
            reasoning=payload.get("reasoning"),
            assistant_prefill=prompt_spec.get("assistant_prefill"),
            attempt_log_dir=Path(role_config.get("_probe_attempt_dir", "")) if role_config.get("_probe_attempt_dir") else None,
        )
        elapsed = time.monotonic() - started
        return {
            "role": role,
            "route_id": route_name,
            "model": payload["model"],
            "provider": payload.get("provider"),
            "response_format": bool(payload.get("response_format")),
            "ok": True,
            "latency_seconds": round(elapsed, 3),
            "parsed_keys": sorted(str(key) for key in parsed.keys()),
            "raw_response_id": raw_response.get("id", ""),
            "raw_text_preview": raw_text[:300],
        }
    except Exception as exc:
        elapsed = time.monotonic() - started
        return {
            "role": role,
            "route_id": route_name,
            "model": payload["model"],
            "provider": payload.get("provider"),
            "response_format": bool(payload.get("response_format")),
            "ok": False,
            "latency_seconds": round(elapsed, 3),
            "error_type": type(exc).__name__,
            "error": exception_summary(exc),
        }


def command_probe_routes(
    config: dict[str, Any],
    *,
    api_base: str,
    api_key: str,
    dry_run: bool,
    timeout: int,
) -> dict[str, Any]:
    output_dir = Path(config["output"]["run_root"]) / "route_probe"
    output_dir.mkdir(parents=True, exist_ok=True)
    if dry_run or not api_key:
        selected_routes = {}
        route_results = {}
        for role in ("writer", "reviewer"):
            routes = role_routes(config[role])
            first = routes[0]
            selected_routes[role] = {
                "route_id": route_id(first, 0),
                "model": route_model(first, config[role]),
                "dry_run_selected": True,
            }
            route_results[role] = []
        report = {
            "created_at": utc_now_iso(),
            "mode": "dry_run" if dry_run else "no_api_key",
            "selected_routes": selected_routes,
            "routes": route_results,
        }
        write_json(output_dir / "route_report.json", report)
        print("[summary-generation:probe-routes] selected first routes without live probe")
        return report

    record = probe_record(config)
    selected_routes: dict[str, Any] = {}
    route_results: dict[str, list[dict[str, Any]]] = {}
    for role in ("writer", "reviewer"):
        role_config = dict(config[role])
        role_config["_probe_attempt_dir"] = str(output_dir / f"{role}_raw_json_attempts")
        prompt_spec = role_prompt_spec(role_config)
        results: list[dict[str, Any]] = []
        for index, route in enumerate(role_routes(role_config)):
            result = probe_one_route(
                role=role,
                role_config=role_config,
                prompt_spec=prompt_spec,
                route=route,
                route_index=index,
                record=record,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
            )
            results.append(result)
            if result["ok"] and role not in selected_routes:
                selected_routes[role] = {
                    "route_id": result["route_id"],
                    "model": result["model"],
                    "provider": result.get("provider"),
                    "response_format": result.get("response_format"),
                    "latency_seconds": result.get("latency_seconds"),
                }
                break
        route_results[role] = results
        if role not in selected_routes:
            selected_routes[role] = {
                "route_id": route_id(role_routes(role_config)[0], 0),
                "model": route_model(role_routes(role_config)[0], role_config),
                "probe_failed_using_first_route": True,
            }
    report = {
        "created_at": utc_now_iso(),
        "mode": "live_probe",
        "selected_routes": selected_routes,
        "routes": route_results,
    }
    write_json(output_dir / "route_report.json", report)
    print(
        "[summary-generation:probe-routes] "
        f"writer={selected_routes['writer']['route_id']} "
        f"reviewer={selected_routes['reviewer']['route_id']}"
    )
    return report


def parse_tags(value: Any, allowed_tags: set[str] | None = None) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for item in string_list(value):
        tag = item.strip()
        if not tag or tag in seen:
            continue
        if allowed_tags is not None and tag not in allowed_tags:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def extract_numbers(value: str) -> set[str]:
    import re

    return set(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*(?:\.\d+)?%?", value))


def local_gate_flags(summary: str, row: dict[str, Any], config: dict[str, Any]) -> list[str]:
    gates = config.get("local_gates", {})
    title = str(row.get("title", ""))
    source_text = f"{title}\n{row.get('abstract', '')}"
    normalized = normalize_text(summary)
    lowered = normalized.casefold()
    flags: list[str] = []
    if not normalized:
        return ["empty_summary"]
    words = word_count(normalized)
    min_words = int(gates.get("min_words", 10))
    max_words = int(gates.get("max_words", 15))
    if words < min_words:
        flags.append("too_short")
    if words > max_words:
        flags.append("too_long")
    if bool_value(gates.get("require_single_sentence"), True) and sentence_count(normalized) != 1:
        flags.append("not_single_sentence")
    if bool_value(gates.get("require_final_period"), False) and not normalized.endswith("."):
        flags.append("missing_final_period")
    forbidden = tuple(str(prefix).casefold() for prefix in gates.get("forbidden_prefixes", DEFAULT_FORBIDDEN_PREFIXES))
    if any(lowered.startswith(prefix) for prefix in forbidden):
        flags.append("forbidden_prefix")
    max_title_overlap = gates.get("max_title_overlap")
    if max_title_overlap is not None and title_overlap_ratio(normalized, title) > float(max_title_overlap):
        flags.append("title_overlap")
    if bool_value(gates.get("reject_formula_shorthand"), True) and has_formula_shorthand(normalized):
        flags.append("formula_shorthand")
    if bool_value(gates.get("reject_unsupported_summary_numbers"), True):
        source_numbers = extract_numbers(source_text)
        summary_numbers = extract_numbers(normalized)
        if any(number not in source_numbers for number in summary_numbers):
            flags.append("unsupported_number_string")
    return flags


def model_call_for_routes(
    row: dict[str, Any],
    *,
    role: str,
    role_config: dict[str, Any],
    prompt_spec: dict[str, Any],
    route_ladder: list[tuple[int, dict[str, Any]]],
    api_base: str,
    api_key: str,
    timeout: int,
    attempt_log_dir: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any], int]:
    errors: list[dict[str, Any]] = []
    for route_index, route in route_ladder:
        payload = build_payload(
            row,
            role=role,
            role_config=role_config,
            prompt_spec=prompt_spec,
            route=route,
            route_index=route_index,
        )
        try:
            parsed, raw_text, raw_response = request_json_with_retries(
                label=f"{role} {row['candidate_id']} {payload['route_id']}",
                api_base=api_base,
                api_key=api_key,
                model=payload["model"],
                messages=payload["messages"],
                temperature=float(payload["temperature"]),
                max_output_tokens=int(payload["max_output_tokens"]),
                timeout=timeout,
                max_retries=int(route.get("max_retries", role_config.get("max_retries", 4))),
                retry_backoff_seconds=float(
                    route.get("retry_backoff_seconds", role_config.get("retry_backoff_seconds", 2.0))
                ),
                response_format=payload.get("response_format"),
                plugins=payload.get("plugins"),
                provider=payload.get("provider"),
                reasoning=payload.get("reasoning"),
                assistant_prefill=prompt_spec.get("assistant_prefill"),
                attempt_log_dir=attempt_log_dir,
            )
            return parsed, raw_text, raw_response, payload, route_index
        except Exception as exc:
            errors.append(
                {
                    "route_id": payload["route_id"],
                    "model": payload["model"],
                    "error_type": type(exc).__name__,
                    "error": exception_summary(exc),
                }
            )
            print(
                f"[summary-generation:{role}] {row['candidate_id']} failed "
                f"route {payload['route_id']}: {type(exc).__name__}: {exception_summary(exc)}",
                file=sys.stderr,
                flush=True,
            )
            continue
    raise RuntimeError(json.dumps({"route_failures": errors}, ensure_ascii=True))


def normalize_writer_result(
    row: dict[str, Any],
    parsed: dict[str, Any],
    *,
    prompt_spec: dict[str, Any],
    payload: dict[str, Any],
    raw_text: str,
    raw_response: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    decision = str(parsed.get("decision", "")).strip().lower()
    if decision not in {"rewrite", "drop", "accept", "reject"}:
        decision = "drop"
    summary = normalize_text(str(parsed.get("summary", "")))
    if decision in {"drop", "reject"}:
        summary = ""
    actual_words = len([part for part in summary.split(" ") if part])
    reported_words = int(parsed.get("word_count", actual_words if summary else 0) or 0)
    allowed_tags = set(string_list(get_path(config, "writer", "allowed_issue_tags", default=[]))) or None
    issue_tags = parse_tags(parsed.get("issue_tags", []), allowed_tags=allowed_tags)
    if not issue_tags and decision in DECISION_ACCEPTED_WRITER:
        issue_tags = ["none"]
    flags = local_gate_flags(summary, row, config) if summary else ["empty_summary"]
    format_flags: list[str] = []
    if reported_words != actual_words:
        format_flags.append("word_count_mismatch")
    return {
        **row,
        "generated_summary": summary,
        "correction_writer": get_path(config, "writer", "writer_label", default="writer"),
        "correction_summary": summary,
        "writer_model": payload["model"],
        "correction_model": payload["model"],
        "writer_route_id": payload["route_id"],
        "writer_prompt_version": prompt_spec["prompt_version"],
        "correction_prompt_version": prompt_spec["prompt_version"],
        "writer_decision": decision,
        "correction_decision": decision,
        "writer_issue_tags": issue_tags,
        "correction_issue_tags": issue_tags,
        "writer_evidence_span": normalize_text(str(parsed.get("evidence_span", ""))),
        "correction_evidence_span": normalize_text(str(parsed.get("evidence_span", ""))),
        "writer_reason_span": normalize_text(str(parsed.get("reason_span", ""))),
        "correction_reason_span": normalize_text(str(parsed.get("reason_span", ""))),
        "writer_reported_word_count": reported_words,
        "writer_actual_word_count": actual_words,
        "writer_format_flags": format_flags,
        "writer_local_filter_flags": flags,
        "raw_writer_response": parsed,
        "raw_writer_text": raw_text,
        "raw_writer_response_id": raw_response.get("id", ""),
    }


def normalize_reviewer_result(
    row: dict[str, Any],
    parsed: dict[str, Any],
    *,
    prompt_spec: dict[str, Any],
    payload: dict[str, Any],
    raw_text: str,
    raw_response: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    decision = str(parsed.get("decision", "")).strip().lower()
    if decision not in {"accept", "reject"}:
        decision = "reject"
    allowed_tags = set(string_list(get_path(config, "reviewer", "allowed_issue_tags", default=[]))) or None
    issue_tags = parse_tags(parsed.get("issue_tags", []), allowed_tags=allowed_tags)
    if not issue_tags and decision == "accept":
        issue_tags = ["none"]
    return {
        **row,
        "reviewer_model": payload["model"],
        "verifier_model": payload["model"],
        "reviewer_route_id": payload["route_id"],
        "reviewer_prompt_version": prompt_spec["prompt_version"],
        "verifier_prompt_version": prompt_spec["prompt_version"],
        "reviewer_decision": decision,
        "verifier_decision": decision,
        "reviewer_issue_tags": issue_tags,
        "verifier_issue_tags": issue_tags,
        "reviewer_evidence_span": normalize_text(str(parsed.get("evidence_span", ""))),
        "verifier_evidence_span": normalize_text(str(parsed.get("evidence_span", ""))),
        "reviewer_reason_span": normalize_text(str(parsed.get("reason_span", ""))),
        "verifier_reason_span": normalize_text(str(parsed.get("reason_span", ""))),
        "raw_reviewer_response": parsed,
        "raw_reviewer_text": raw_text,
        "raw_reviewer_response_id": raw_response.get("id", ""),
    }


def command_model_stage(
    config: dict[str, Any],
    *,
    role: str,
    args: argparse.Namespace,
    only_failed: bool = False,
) -> list[dict[str, Any]]:
    role_config = config[role]
    prompt_spec = role_prompt_spec(role_config)
    output_root = Path(config["output"]["run_root"])
    output_dir = output_root / stage_dir_name(role)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = load_jsonl(input_path_for_stage(config, role))
    if args.limit is not None:
        source_rows = source_rows[: args.limit]
    if only_failed:
        failed_path = failed_output_path_for_stage(config, role)
        failed_ids = {row["candidate_id"] for row in load_jsonl(failed_path)} if failed_path.exists() else set()
        source_rows = [row for row in source_rows if row["candidate_id"] in failed_ids]
    route_ladder = route_ladder_from_report(config, role=role, no_route_probe=args.no_route_probe)
    request_payloads = [
        build_payload(
            row,
            role=role,
            role_config=role_config,
            prompt_spec=prompt_spec,
            route=route_ladder[0][1],
            route_index=route_ladder[0][0],
        )
        for row in source_rows
    ]
    write_jsonl(output_dir / "request_payloads.jsonl", request_payloads)
    manifest = {
        "created_at": utc_now_iso(),
        "mode": "dry_run" if args.dry_run else "live_run",
        "role": role,
        "input_jsonl": str(input_path_for_stage(config, role)),
        "output_dir": str(output_dir),
        "prompt_path": role_config["prompt_path"],
        "prompt_version": prompt_spec["prompt_version"],
        "route_ladder": [
            {
                "route_id": route_id(route, index),
                "model": route_model(route, role_config),
                "provider": provider_config(route),
                "response_format_mode": route.get("response_format_mode", role_config.get("response_format_mode", "schema")),
            }
            for index, route in route_ladder
        ],
        "requested_rows": len(source_rows),
        "workers": int(role_config.get("workers", 4)),
        "only_failed": only_failed,
        "outputs": {
            "request_payloads": "request_payloads.jsonl",
            "outputs": output_path_for_stage(config, role).name,
            "outputs_ordered": ordered_output_path_for_stage(config, role).name,
            "failed_outputs": failed_output_path_for_stage(config, role).name,
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)
    if args.dry_run:
        print(f"[summary-generation:{role}] dry-run payloads: {len(request_payloads)}")
        return []
    api_base, api_key = get_api_credentials(args)
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for live generation stages")
    output_path = output_path_for_stage(config, role)
    existing = load_jsonl(output_path) if output_path.exists() else []
    completed = {row["candidate_id"]: row for row in existing}
    pending = [row for row in source_rows if row["candidate_id"] not in completed or only_failed]
    print(
        f"[summary-generation:{role}] running {len(pending)} pending "
        f"({len(existing)} existing) with {role_config.get('workers', 4)} worker(s)...",
        file=sys.stderr,
        flush=True,
    )

    def submit(row: dict[str, Any]) -> dict[str, Any]:
        parsed, raw_text, raw_response, payload, _ = model_call_for_routes(
            row,
            role=role,
            role_config=role_config,
            prompt_spec=prompt_spec,
            route_ladder=route_ladder,
            api_base=api_base,
            api_key=api_key,
            timeout=int(role_config.get("timeout", args.timeout)),
            attempt_log_dir=output_dir / "raw_json_attempts",
        )
        if role == "writer":
            return normalize_writer_result(
                row,
                parsed,
                prompt_spec=prompt_spec,
                payload=payload,
                raw_text=raw_text,
                raw_response=raw_response,
                config=config,
            )
        return normalize_reviewer_result(
            row,
            parsed,
            prompt_spec=prompt_spec,
            payload=payload,
            raw_text=raw_text,
            raw_response=raw_response,
            config=config,
        )

    failed_rows: list[dict[str, Any]] = []
    for _, row, future in iter_bounded_futures(
        pending,
        int(role_config.get("workers", 4)),
        submit,
    ):
        try:
            result = future.result()
        except Exception as exc:
            if not args.continue_on_error:
                raise
            failed = {
                "candidate_id": row["candidate_id"],
                "paper_id": row.get("paper_id", ""),
                "role": role,
                "error_type": type(exc).__name__,
                "error": exception_summary(exc),
            }
            failed_rows.append(failed)
            append_jsonl(failed_output_path_for_stage(config, role), failed)
            continue
        completed[result["candidate_id"]] = result
        append_jsonl(output_path, result)
        decision = result.get("writer_decision") if role == "writer" else result.get("reviewer_decision")
        print(
            f"[summary-generation:{role}] {len(completed)}/{len(source_rows)} "
            f"{result['candidate_id']} {decision}",
            file=sys.stderr,
            flush=True,
        )
    ordered = [completed[row["candidate_id"]] for row in source_rows if row["candidate_id"] in completed]
    write_jsonl(ordered_output_path_for_stage(config, role), ordered)
    manifest.update(
        {
            "completed_rows": len(ordered),
            "failed_rows": len(failed_rows),
            "pending_rows": len(source_rows) - len(ordered) - len(failed_rows),
            "decision_counts": dict(
                Counter(
                    row.get("writer_decision") if role == "writer" else row.get("reviewer_decision")
                    for row in ordered
                )
            ),
        }
    )
    write_json(output_dir / "run_manifest.json", manifest)
    return ordered


def command_local_filter(config: dict[str, Any]) -> list[dict[str, Any]]:
    output_root = Path(config["output"]["run_root"])
    output_dir = output_root / "local_eligible_for_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(output_root / "writer_run" / "writer_outputs_ordered.jsonl")
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        accepted_decisions = set(string_list(get_path(config, "writer", "accepted_decisions", default=[]))) or DECISION_ACCEPTED_WRITER
        if row.get("writer_decision") in accepted_decisions and not row.get("writer_local_filter_flags"):
            eligible.append(row)
        else:
            excluded.append(row)
    write_jsonl(output_dir / "writer_outputs_ordered.jsonl", eligible)
    write_jsonl(output_dir / "excluded.jsonl", excluded)
    manifest = {
        "created_at": utc_now_iso(),
        "input_writer_outputs": str(output_root / "writer_run" / "writer_outputs_ordered.jsonl"),
        "eligible_rows": len(eligible),
        "excluded_rows": len(excluded),
        "excluded_decision_counts": dict(Counter(row.get("writer_decision") for row in excluded)),
        "excluded_local_filter_counts": dict(
            Counter(flag for row in excluded for flag in row.get("writer_local_filter_flags", []))
        ),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(f"[summary-generation:local-filter] eligible={len(eligible)} excluded={len(excluded)}")
    return eligible


def materialized_training_row(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    materialize_config = config.get("materialize", {})
    output = {
        "paper_id": row.get("paper_id"),
        "title": row.get("title"),
        "abstract": row.get("abstract"),
        "summary": row.get("generated_summary"),
        "primary_category": row.get("primary_category"),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date"),
        "teacher_model": row.get("writer_model"),
        "reviewer_model": row.get("reviewer_model"),
        "generator_prompt_version": row.get("writer_prompt_version"),
        "review_prompt_version": row.get("reviewer_prompt_version"),
        "source_split": row.get("source_split") or materialize_config.get("source_split"),
        "summary_generation_candidate_id": row.get("candidate_id"),
        "writer_route_id": row.get("writer_route_id"),
        "reviewer_route_id": row.get("reviewer_route_id"),
    }
    for field in string_list(materialize_config.get("provenance_fields")):
        if field in row:
            output[field] = row[field]
    for field in string_list(materialize_config.get("review_fields")):
        if field in row:
            output[field] = row[field]
    return output


def release_shape_flags(row: dict[str, Any], config: dict[str, Any]) -> list[str]:
    summary = normalize_text(str(row.get("generated_summary", "")))
    return local_gate_flags(summary, row, config)


def command_materialize(config: dict[str, Any]) -> list[dict[str, Any]]:
    output_root = Path(config["output"]["run_root"])
    review_rows = load_jsonl(output_root / "reviewer_run" / "reviewer_outputs_ordered.jsonl")
    materialized_dir = output_root / "materialized_release"
    materialized_dir.mkdir(parents=True, exist_ok=True)
    release_dir_config = get_path(config, "output", "release_dir", default=None)
    release_dir = Path(release_dir_config) if release_dir_config else materialized_dir
    release_dir.mkdir(parents=True, exist_ok=True)
    accepted_decisions = set(string_list(get_path(config, "reviewer", "accepted_decisions", default=[]))) or DECISION_ACCEPTED_REVIEWER
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    shape_excluded: list[dict[str, Any]] = []
    for row in review_rows:
        if row.get("reviewer_decision") not in accepted_decisions:
            rejected.append(row)
            continue
        shape_flags = release_shape_flags(row, config)
        if shape_flags:
            shape_excluded.append({**row, "release_shape_flags": shape_flags})
            continue
        accepted.append(row)
    training_rows = [materialized_training_row(row, config) for row in accepted]
    write_jsonl(release_dir / "training_rows.jsonl", training_rows)
    write_jsonl(materialized_dir / "selected_review_rows.jsonl", accepted)
    write_jsonl(materialized_dir / "excluded_review_rows.jsonl", rejected + shape_excluded)
    manifest = {
        "created_at": utc_now_iso(),
        "release_dir": str(release_dir),
        "review_rows": len(review_rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "release_shape_excluded_rows": len(shape_excluded),
        "release_shape_excluded_counts": dict(
            Counter(flag for row in shape_excluded for flag in row.get("release_shape_flags", []))
        ),
        "outputs": {
            "training_rows": str(release_dir / "training_rows.jsonl"),
            "selected_review_rows": str(materialized_dir / "selected_review_rows.jsonl"),
            "excluded_review_rows": str(materialized_dir / "excluded_review_rows.jsonl"),
        },
    }
    write_json(release_dir / "run_manifest.json", manifest)
    write_json(materialized_dir / "run_manifest.json", manifest)
    print(f"[summary-generation:materialize] accepted={len(accepted)} release={release_dir}")
    return training_rows


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return len(load_jsonl(path))


def command_report(config: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(config["output"]["run_root"])
    report = {
        "created_at": utc_now_iso(),
        "run_root": str(output_root),
        "candidates": count_jsonl(output_root / "candidate_pool" / "candidates.jsonl"),
        "writer_completed": count_jsonl(output_root / "writer_run" / "writer_outputs_ordered.jsonl"),
        "writer_failed": count_jsonl(output_root / "writer_run" / "failed_writer_outputs.jsonl"),
        "local_eligible": count_jsonl(output_root / "local_eligible_for_review" / "writer_outputs_ordered.jsonl"),
        "local_excluded": count_jsonl(output_root / "local_eligible_for_review" / "excluded.jsonl"),
        "reviewer_completed": count_jsonl(output_root / "reviewer_run" / "reviewer_outputs_ordered.jsonl"),
        "reviewer_failed": count_jsonl(output_root / "reviewer_run" / "failed_reviewer_outputs.jsonl"),
        "released_training_rows": count_jsonl(Path(get_path(config, "output", "release_dir", default=output_root / "materialized_release")) / "training_rows.jsonl"),
    }
    candidates = max(int(report["candidates"]), 1)
    report["yield_rates"] = {
        "writer_completed": round(report["writer_completed"] / candidates, 4),
        "local_eligible": round(report["local_eligible"] / candidates, 4),
        "reviewer_completed": round(report["reviewer_completed"] / candidates, 4),
        "released_training_rows": round(report["released_training_rows"] / candidates, 4),
    }
    write_json(output_root / "yield_report.json", report)
    markdown = [
        "# Summary Generation Yield Report",
        "",
        f"- Run root: `{output_root}`",
        f"- Candidates: {report['candidates']}",
        f"- Writer completed: {report['writer_completed']}",
        f"- Writer failed: {report['writer_failed']}",
        f"- Local eligible: {report['local_eligible']}",
        f"- Local excluded: {report['local_excluded']}",
        f"- Reviewer completed: {report['reviewer_completed']}",
        f"- Reviewer failed: {report['reviewer_failed']}",
        f"- Released training rows: {report['released_training_rows']}",
        "",
        "## Yield Rates",
        "",
    ]
    markdown.extend(f"- {key}: {value:.1%}" for key, value in report["yield_rates"].items())
    (output_root / "yield_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"[summary-generation:report] released={report['released_training_rows']}")
    return report


def command_retry_failed(config: dict[str, Any], args: argparse.Namespace) -> None:
    if failed_output_path_for_stage(config, "writer").exists():
        command_model_stage(config, role="writer", args=args, only_failed=True)
        command_local_filter(config)
    if failed_output_path_for_stage(config, "reviewer").exists():
        command_model_stage(config, role="reviewer", args=args, only_failed=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reusable writer/reviewer summary data-generation pipeline."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-route-probe", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    api_base, api_key = get_api_credentials(args)
    if args.stage in {"prepare", "all"}:
        command_prepare(config, cli_limit=args.limit)
    if args.stage in {"probe-routes", "all"} and not args.no_route_probe:
        command_probe_routes(
            config,
            api_base=api_base,
            api_key=api_key,
            dry_run=args.dry_run,
            timeout=args.timeout,
        )
    if args.stage in {"write", "all"}:
        command_model_stage(config, role="writer", args=args)
    if args.dry_run and args.stage == "all":
        command_report(config)
        return
    if args.stage in {"local-filter", "all"}:
        command_local_filter(config)
    if args.stage in {"verify", "all"}:
        command_model_stage(config, role="reviewer", args=args)
    if args.stage == "retry-failed":
        command_retry_failed(config, args)
    if args.stage in {"materialize", "all"} and not args.dry_run:
        command_materialize(config)
    if args.stage in {"report", "all"}:
        command_report(config)


if __name__ == "__main__":
    main()
