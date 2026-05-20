#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import re
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "hf_bucket/experiments/mlx_lora/runtime_shift/mlx_swift_parity_root_cause_v1"
)
DEFAULT_PROMPT_SPEC = (
    REPO_ROOT / "prompts/mlx_lora/gemma_270m_it_abstract_to_glanceable_summary_v1.json"
)
DEFAULT_INPUT_MODE = "title_abstract"
DEFAULT_EVAL_INPUTS = REPO_ROOT / "hf_dataset/evals/q4_runtime_shift_200_v1/eval_inputs.jsonl"
DEFAULT_CANONICAL_OUTPUTS = (
    REPO_ROOT
    / "hf_bucket/experiments/mlx_lora/runtime_shift/q4_runtime_shift_200_v1/variant_outputs/canonical_mlx_q4_adapter.jsonl"
)
DEFAULT_SWIFT_OUTPUTS = (
    REPO_ROOT
    / "hf_bucket/experiments/mlx_lora/runtime_shift/q4_runtime_shift_200_v1/variant_outputs/mlx_swift_q4_fused.jsonl"
)
DEFAULT_REVIEWED_ROWS = (
    REPO_ROOT
    / "hf_bucket/experiments/mlx_lora/runtime_shift/q4_runtime_shift_200_v1/review_packets/"
    "canonical_mlx_q4_adapter_vs_mlx_swift_q4_fused/aggregation/reviewed_rows.jsonl"
)
DEFAULT_Q4_BASE = (
    REPO_ROOT / "hf_bucket/experiments/mlx_lora/quantized_models/google_gemma_3_270m_it/q4_g64"
)
DEFAULT_Q4_ADAPTER = (
    REPO_ROOT
    / "hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_10746_q4_hardening_combined_v1_lr_1e_6/adapters"
)
DEFAULT_Q4_CHECKPOINT = DEFAULT_Q4_ADAPTER / "0000300_adapters.safetensors"
DEFAULT_Q4_FUSED = (
    REPO_ROOT
    / "hf_bucket/experiments/litert_exports/q4_hardening_combined_v1_lr_1e_6_300/q4_fused_mlx"
)
MAX_TOKENS = 64
TEMP = 0.0
TOP_P = 1.0

CABLECAR_TITLE = "CAbLECAR: efficiently scheduling QLDPC codes on a tileable spin qubit chip with shuttling"
CABLECAR_ABSTRACT = """Semiconductor spin qubits are a promising platform for large-scale quantum computing, but have yet to take full advantage of the broad class of quantum low-density parity check (QLDPC) codes, which promise high encoding rates and efficient logic but require nonlocal connectivity between physical qubits. In this work, we investigate the implementation of QLDPC codes on a tileable, shuttling-based spin qubit architecture. By tailoring syndrome extraction circuits to the shuttling noise model, we significantly improve on previous surface code proposals and extend the feasible shuttling range of the architecture by 5-10x, enabling the implementation of more complex codes with long-range interactions. Taking inspiration from the field of robotics, we develop a coordinated shuttle scheduling algorithm that supports arbitrary codes and use it to benchmark the logical performance of a variety of promising code families. We find that the optimized schedules are up to 86% faster than hand-optimized schedules for certain code families.
Through detailed circuit-level simulations, we identify specific QLDPC codes that improve upon prior surface code implementations by orders of magnitude, increasing encoding efficiency and reducing logical error rates. This work demonstrates the potential of shuttling-based spin qubit hardware platforms for scalable and efficient fault-tolerant quantum computation."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any], *, overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path) if path.exists() else None,
    }


def word_count(value: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|\d+(?:[./:-]\d+)*%?", value or ""))


def normalize_whitespace(value: str) -> str:
    return " ".join((value or "").split())


def load_by_paper_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["paper_id"]): row for row in load_jsonl(path)}


def variant_summary(row: dict[str, Any]) -> str:
    return str(row.get("normalized_output") or row.get("normalized") or row.get("summary") or "")


def build_prompt(row: dict[str, Any], prompt_spec: dict[str, Any], input_mode: str) -> tuple[str, str]:
    system_prompt = str(prompt_spec["system_prompt"]).strip()
    user_template = str(prompt_spec["user_prompt_templates"][input_mode])
    user_prompt = user_template.format(
        title=str(row["title"]).strip(),
        abstract=str(row["abstract"]).strip(),
    )
    return system_prompt, user_prompt


def encode_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> list[int]:
    if getattr(tokenizer, "has_chat_template", False):
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return list(tokenizer.encode(rendered, add_special_tokens=False))
    return list(tokenizer.encode(f"{system_prompt}\n\n{user_prompt}"))


def build_artifact_manifest() -> dict[str, Any]:
    file_names = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "model.safetensors",
    ]
    return {
        "q4_base": {
            "path": str(DEFAULT_Q4_BASE.relative_to(REPO_ROOT)),
            "files": {name: file_record(DEFAULT_Q4_BASE / name) for name in file_names},
        },
        "q4_fused": {
            "path": str(DEFAULT_Q4_FUSED.relative_to(REPO_ROOT)),
            "files": {name: file_record(DEFAULT_Q4_FUSED / name) for name in file_names},
        },
        "q4_adapter": {
            "path": str(DEFAULT_Q4_ADAPTER.relative_to(REPO_ROOT)),
            "files": {
                "adapter_config.json": file_record(DEFAULT_Q4_ADAPTER / "adapter_config.json"),
                "adapters.safetensors": file_record(DEFAULT_Q4_ADAPTER / "adapters.safetensors"),
                "0000300_adapters.safetensors": file_record(DEFAULT_Q4_CHECKPOINT),
            },
        },
        "generation_settings": {
            "max_tokens": MAX_TOKENS,
            "temperature": TEMP,
            "top_p": TOP_P,
        },
    }


def fixture_row(row: dict[str, Any], reason: str) -> dict[str, Any]:
    copied = dict(row)
    copied["paper_id"] = str(copied["paper_id"])
    copied["eval_id"] = copied.get("eval_id") or copied["paper_id"]
    copied["parity_fixture_reason"] = reason
    return copied


def select_fixtures() -> list[dict[str, Any]]:
    source_rows = load_by_paper_id(DEFAULT_EVAL_INPUTS)
    canonical_outputs = load_by_paper_id(DEFAULT_CANONICAL_OUTPUTS)
    swift_outputs = load_by_paper_id(DEFAULT_SWIFT_OUTPUTS)
    reviewed_rows = load_by_paper_id(DEFAULT_REVIEWED_ROWS) if DEFAULT_REVIEWED_ROWS.exists() else {}

    selected: list[dict[str, Any]] = [
        {
            "paper_id": "fixture_cablecar",
            "eval_id": "fixture_cablecar",
            "title": CABLECAR_TITLE,
            "abstract": CABLECAR_ABSTRACT,
            "primary_category": "fixture",
            "categories": ["fixture"],
            "created_date": None,
            "selection_bucket": "fixture",
            "parity_fixture_reason": "cablecar_smoke",
        }
    ]
    selected_ids = {"fixture_cablecar"}

    length_candidates = []
    for paper_id, source_row in source_rows.items():
        canonical_words = word_count(variant_summary(canonical_outputs[paper_id]))
        swift_words = word_count(variant_summary(swift_outputs[paper_id]))
        length_candidates.append((swift_words - canonical_words, paper_id, source_row))
    for _, paper_id, row in sorted(length_candidates, reverse=True):
        if paper_id in selected_ids:
            continue
        selected.append(fixture_row(row, "swift_much_longer"))
        selected_ids.add(paper_id)
        if sum(1 for item in selected if item["parity_fixture_reason"] == "swift_much_longer") >= 3:
            break

    severe_candidates = [
        (paper_id, row)
        for paper_id, row in reviewed_rows.items()
        if row.get("mlx_swift_q4_fused_severe") or row.get("mlx_swift_q4_fused_mangled")
    ]
    for paper_id, _review in severe_candidates:
        if paper_id in selected_ids or paper_id not in source_rows:
            continue
        selected.append(fixture_row(source_rows[paper_id], "swift_review_severe_or_mangled"))
        selected_ids.add(paper_id)
        if sum(1 for item in selected if item["parity_fixture_reason"] == "swift_review_severe_or_mangled") >= 3:
            break

    bucket_counts = Counter(row.get("selection_bucket") for row in selected)
    for paper_id, row in source_rows.items():
        if paper_id in selected_ids:
            continue
        bucket = row.get("selection_bucket")
        if bucket in {"numeric", "entity"} and bucket_counts[bucket] < 1:
            selected.append(fixture_row(row, f"{bucket}_risk"))
            selected_ids.add(paper_id)
            bucket_counts[bucket] += 1
        if bucket_counts["numeric"] >= 1 and bucket_counts["entity"] >= 1:
            break

    for paper_id, row in source_rows.items():
        if len(selected) >= 11:
            break
        if paper_id in selected_ids:
            continue
        if row.get("selection_bucket") == "control":
            selected.append(fixture_row(row, "control"))
            selected_ids.add(paper_id)

    return selected[:11]


def command_prepare_fixtures(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    fixtures = select_fixtures()
    write_jsonl(output_root / "fixtures.jsonl", fixtures, overwrite=args.overwrite)
    write_json(output_root / "artifact_manifest.json", build_artifact_manifest(), overwrite=args.overwrite)
    print(f"[parity] wrote {len(fixtures)} fixtures to {output_root / 'fixtures.jsonl'}")


def trim_stop_token(token_ids: list[int], stop_ids: set[int]) -> list[int]:
    if token_ids and token_ids[-1] in stop_ids:
        return token_ids[:-1]
    return token_ids


def trace_stream_variant(
    *,
    label: str,
    model_path: Path,
    adapter_path: Path | None,
    fixtures: list[dict[str, Any]],
    prompt_spec: dict[str, Any],
    input_mode: str,
) -> list[dict[str, Any]]:
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    load_kwargs: dict[str, Any] = {"tokenizer_config": {"trust_remote_code": True}}
    if adapter_path is not None:
        load_kwargs["adapter_path"] = str(adapter_path)
    print(f"[parity:{label}] loading {model_path}")
    model, tokenizer = load(str(model_path), **load_kwargs)
    sampler = make_sampler(temp=TEMP, top_p=TOP_P)
    stop_ids = set(getattr(tokenizer, "eos_token_ids", []) or [])
    rows: list[dict[str, Any]] = []
    try:
        for index, fixture in enumerate(fixtures, start=1):
            system_prompt, user_prompt = build_prompt(fixture, prompt_spec, input_mode)
            prompt_token_ids = encode_prompt(tokenizer, system_prompt, user_prompt)
            generated_token_ids: list[int] = []
            chunks: list[str] = []
            finish_reason = None
            for response in stream_generate(
                model,
                tokenizer,
                prompt_token_ids,
                max_tokens=MAX_TOKENS,
                sampler=sampler,
            ):
                generated_token_ids.append(int(response.token))
                chunks.append(response.text)
                finish_reason = response.finish_reason or finish_reason
            content_token_ids = trim_stop_token(generated_token_ids, stop_ids)
            decoded_output = tokenizer.decode(content_token_ids)
            rows.append(
                {
                    "ok": bool(normalize_whitespace(decoded_output)),
                    "runtime": "mlx_python",
                    "generation_mode": "stream",
                    "variant": label,
                    "paper_id": str(fixture["paper_id"]),
                    "eval_id": fixture.get("eval_id"),
                    "title": fixture["title"],
                    "prompt_version": prompt_spec.get("prompt_version"),
                    "input_mode": input_mode,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "prompt_token_ids": prompt_token_ids,
                    "prompt_token_count": len(prompt_token_ids),
                    "generated_token_ids": generated_token_ids,
                    "content_token_ids": content_token_ids,
                    "decoded_output": decoded_output,
                    "stream_joined_output": "".join(chunks),
                    "normalized_output": normalize_whitespace(decoded_output),
                    "finish_reason": finish_reason,
                    "generation_settings": {
                        "max_tokens": MAX_TOKENS,
                        "temperature": TEMP,
                        "top_p": TOP_P,
                    },
                }
            )
            print(f"[parity:{label}] traced {index}/{len(fixtures)}")
    finally:
        del model
        del tokenizer
        del sampler
        gc.collect()
        mx.clear_cache()
    return rows


def trace_batch_variant(
    *,
    label: str,
    model_path: Path,
    adapter_path: Path,
    fixtures: list[dict[str, Any]],
    prompt_spec: dict[str, Any],
    input_mode: str,
) -> list[dict[str, Any]]:
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    print(f"[parity:{label}] loading {model_path}")
    model, tokenizer = load(
        str(model_path),
        adapter_path=str(adapter_path),
        tokenizer_config={"trust_remote_code": True},
    )
    sampler = make_sampler(temp=TEMP, top_p=TOP_P)
    stop_ids = set(getattr(tokenizer, "eos_token_ids", []) or [])
    prepared: list[tuple[dict[str, Any], str, str, list[int]]] = []
    for fixture in fixtures:
        system_prompt, user_prompt = build_prompt(fixture, prompt_spec, input_mode)
        prompt_token_ids = encode_prompt(tokenizer, system_prompt, user_prompt)
        prepared.append((fixture, system_prompt, user_prompt, prompt_token_ids))

    generator = BatchGenerator(
        model,
        stop_tokens=[[token] for token in stop_ids],
        sampler=sampler,
    )
    generated: dict[int, list[int]] = {}
    stop_tokens: dict[int, list[int]] = {}
    finish_reasons: dict[int, str] = {}
    try:
        uids = generator.insert([item[3] for item in prepared], [MAX_TOKENS] * len(prepared))
        generated = {uid: [] for uid in uids}
        stop_tokens = {uid: [] for uid in uids}
        while responses := generator.next_generated():
            for response in responses:
                token = int(response.token)
                if response.finish_reason == "stop":
                    stop_tokens[response.uid].append(token)
                    finish_reasons[response.uid] = "stop"
                else:
                    generated[response.uid].append(token)
                    if response.finish_reason:
                        finish_reasons[response.uid] = response.finish_reason
    finally:
        generator.close()

    rows: list[dict[str, Any]] = []
    try:
        for uid, (fixture, system_prompt, user_prompt, prompt_token_ids) in zip(uids, prepared, strict=True):
            content_token_ids = generated[uid]
            generated_token_ids = content_token_ids + stop_tokens.get(uid, [])
            decoded_output = tokenizer.decode(content_token_ids)
            rows.append(
                {
                    "ok": bool(normalize_whitespace(decoded_output)),
                    "runtime": "mlx_python",
                    "generation_mode": "batch",
                    "variant": label,
                    "paper_id": str(fixture["paper_id"]),
                    "eval_id": fixture.get("eval_id"),
                    "title": fixture["title"],
                    "prompt_version": prompt_spec.get("prompt_version"),
                    "input_mode": input_mode,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "prompt_token_ids": prompt_token_ids,
                    "prompt_token_count": len(prompt_token_ids),
                    "generated_token_ids": generated_token_ids,
                    "content_token_ids": content_token_ids,
                    "decoded_output": decoded_output,
                    "normalized_output": normalize_whitespace(decoded_output),
                    "finish_reason": finish_reasons.get(uid),
                    "generation_settings": {
                        "max_tokens": MAX_TOKENS,
                        "temperature": TEMP,
                        "top_p": TOP_P,
                    },
                }
            )
    finally:
        del model
        del tokenizer
        del sampler
        gc.collect()
        mx.clear_cache()
    return rows


def command_trace_python(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    fixtures = load_jsonl(output_root / "fixtures.jsonl")
    prompt_spec = json.loads(Path(args.prompt_spec).read_text(encoding="utf-8"))
    trace_root = output_root / "python_traces"

    canonical_stream = trace_stream_variant(
        label="canonical_mlx_q4_adapter_stream",
        model_path=DEFAULT_Q4_BASE,
        adapter_path=DEFAULT_Q4_ADAPTER,
        fixtures=fixtures,
        prompt_spec=prompt_spec,
        input_mode=args.input_mode,
    )
    write_jsonl(trace_root / "canonical_mlx_q4_adapter_stream.jsonl", canonical_stream, overwrite=args.overwrite)

    canonical_batch = trace_batch_variant(
        label="canonical_mlx_q4_adapter_batch",
        model_path=DEFAULT_Q4_BASE,
        adapter_path=DEFAULT_Q4_ADAPTER,
        fixtures=fixtures,
        prompt_spec=prompt_spec,
        input_mode=args.input_mode,
    )
    write_jsonl(trace_root / "canonical_mlx_q4_adapter_batch.jsonl", canonical_batch, overwrite=args.overwrite)

    fused_stream = trace_stream_variant(
        label="mlx_python_q4_fused_stream",
        model_path=DEFAULT_Q4_FUSED,
        adapter_path=None,
        fixtures=fixtures,
        prompt_spec=prompt_spec,
        input_mode=args.input_mode,
    )
    write_jsonl(trace_root / "mlx_python_q4_fused_stream.jsonl", fused_stream, overwrite=args.overwrite)


def traces_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["paper_id"]): row for row in load_jsonl(path)}


def decode_token(tokenizer: Any, token_id: int | None) -> str | None:
    if token_id is None:
        return None
    return tokenizer.decode([int(token_id)])


def decode_prefix(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return tokenizer.decode(token_ids)


def first_diff(left: list[int], right: list[int], tokenizer: Any) -> dict[str, Any]:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            prefix = left[:index]
            return {
                "index": index,
                "left_token": left_token,
                "right_token": right_token,
                "left_token_text": decode_token(tokenizer, left_token),
                "right_token_text": decode_token(tokenizer, right_token),
                "matching_prefix": prefix,
                "matching_prefix_text": decode_prefix(tokenizer, prefix),
            }
    if len(left) != len(right):
        index = min(len(left), len(right))
        left_token = left[index] if index < len(left) else None
        right_token = right[index] if index < len(right) else None
        prefix = left[:index]
        return {
            "index": index,
            "left_token": left_token,
            "right_token": right_token,
            "left_token_text": decode_token(tokenizer, left_token),
            "right_token_text": decode_token(tokenizer, right_token),
            "matching_prefix": prefix,
            "matching_prefix_text": decode_prefix(tokenizer, prefix),
        }
    return {
        "index": None,
        "left_token": None,
        "right_token": None,
        "left_token_text": None,
        "right_token_text": None,
        "matching_prefix": left,
        "matching_prefix_text": decode_prefix(tokenizer, left),
    }


def compare_pair(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    *,
    left_label: str,
    right_label: str,
    token_key: str,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for paper_id, left_row in left.items():
        right_row = right[paper_id]
        left_tokens = list(left_row.get(token_key, []))
        right_tokens = list(right_row.get(token_key, []))
        left_finish = left_row.get("finish_reason")
        right_finish = right_row.get("finish_reason")
        rows.append(
            {
                "paper_id": paper_id,
                "left_label": left_label,
                "right_label": right_label,
                "match": left_tokens == right_tokens,
                "first_diff": first_diff(left_tokens, right_tokens, tokenizer),
                "left_output": left_row.get("normalized_output"),
                "right_output": right_row.get("normalized_output"),
                "left_finish_reason": left_finish,
                "right_finish_reason": right_finish,
                "left_token_count": len(left_tokens),
                "right_token_count": len(right_tokens),
                "left_stopped_earlier_via_eos": bool(
                    left_finish == "stop" and len(left_tokens) < len(right_tokens)
                ),
                "right_stopped_earlier_via_eos": bool(
                    right_finish == "stop" and len(right_tokens) < len(left_tokens)
                ),
            }
        )
    return rows


def summarize_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches = [row for row in rows if not row["match"]]
    return {
        "rows": len(rows),
        "matches": len(rows) - len(mismatches),
        "mismatches": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
    }


def classify(summary: dict[str, Any]) -> str:
    if summary["prompt_tokens"]["mismatches"]:
        return "prompt/token mismatch"
    if summary["python_fuse"]["mismatches"]:
        return "Python fusion mismatch"
    if summary["python_batch_stream_text_mismatches"]:
        return "Python batch/stream mismatch"
    if summary["swift_runtime"]["mismatches"]:
        return "Swift runtime mismatch"
    return "no divergence on fixtures"


def command_report(args: argparse.Namespace) -> None:
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    output_root = Path(args.output_root)
    python_root = output_root / "python_traces"
    swift_trace_path = Path(args.swift_trace_jsonl)
    canonical_stream = traces_by_id(python_root / "canonical_mlx_q4_adapter_stream.jsonl")
    canonical_batch = traces_by_id(python_root / "canonical_mlx_q4_adapter_batch.jsonl")
    fused_stream = traces_by_id(python_root / "mlx_python_q4_fused_stream.jsonl")
    swift_fused = traces_by_id(swift_trace_path)
    tokenizer = load_tokenizer(
        DEFAULT_Q4_FUSED,
        tokenizer_config_extra={"trust_remote_code": True},
    )

    prompt_comparison = compare_pair(
        canonical_stream,
        swift_fused,
        left_label="python_canonical_stream",
        right_label="swift_fused",
        token_key="prompt_token_ids",
        tokenizer=tokenizer,
    )
    python_fuse_comparison = compare_pair(
        canonical_stream,
        fused_stream,
        left_label="python_canonical_stream",
        right_label="python_fused_stream",
        token_key="generated_token_ids",
        tokenizer=tokenizer,
    )
    swift_runtime_comparison = compare_pair(
        fused_stream,
        swift_fused,
        left_label="python_fused_stream",
        right_label="swift_fused",
        token_key="generated_token_ids",
        tokenizer=tokenizer,
    )
    batch_text_mismatches = [
        paper_id
        for paper_id, row in canonical_stream.items()
        if row.get("normalized_output") != canonical_batch[paper_id].get("normalized_output")
    ]
    summary = {
        "prompt_tokens": summarize_comparison(prompt_comparison),
        "python_fuse": summarize_comparison(python_fuse_comparison),
        "python_batch_stream_text_mismatches": len(batch_text_mismatches),
        "python_batch_stream_text_mismatch_paper_ids": batch_text_mismatches,
        "swift_runtime": summarize_comparison(swift_runtime_comparison),
    }
    classification = classify(summary)
    report = {
        "classification": classification,
        "summary": summary,
        "comparisons": {
            "prompt_tokens": prompt_comparison,
            "python_fuse": python_fuse_comparison,
            "swift_runtime": swift_runtime_comparison,
        },
        "swift_trace_jsonl": str(swift_trace_path),
    }
    write_json(output_root / "parity_report.json", report, overwrite=args.overwrite)
    write_markdown_report(output_root / "parity_report.md", report, overwrite=args.overwrite)
    print(f"[parity] classification: {classification}")
    print(f"[parity] wrote {output_root / 'parity_report.md'}")


def compact_mismatch(row: dict[str, Any] | None) -> str:
    if not row:
        return "none"
    diff = row["first_diff"]
    return (
        f"{row['paper_id']} at token {diff['index']} "
        f"({row['left_label']}={diff['left_token']} {diff['left_token_text']!r}, "
        f"{row['right_label']}={diff['right_token']} {diff['right_token_text']!r})"
    )


def write_markdown_report(path: Path, report: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite")
    summary = report["summary"]
    lines = [
        "# MLX-Swift Parity Root-Cause Report",
        "",
        f"Classification: `{report['classification']}`",
        "",
        "## Ladder Summary",
        "",
        "| Check | Matches | Mismatches | First mismatch |",
        "| --- | ---: | ---: | --- |",
    ]
    for key, label in [
        ("prompt_tokens", "Prompt tokens"),
        ("python_fuse", "Python canonical vs fused tokens"),
        ("swift_runtime", "Python fused vs Swift tokens"),
    ]:
        item = summary[key]
        lines.append(
            f"| {label} | {item['matches']} | {item['mismatches']} | {compact_mismatch(item['first_mismatch'])} |"
        )
    lines.extend(
        [
            f"| Python batch vs stream text | {len(report['comparisons']['python_fuse']) - summary['python_batch_stream_text_mismatches']} | {summary['python_batch_stream_text_mismatches']} | {', '.join(summary['python_batch_stream_text_mismatch_paper_ids'][:3]) or 'none'} |",
            "",
            "## Concrete Divergence Example",
            "",
        ]
    )
    if report["classification"] == "prompt/token mismatch":
        example = summary["prompt_tokens"]["first_mismatch"]
    elif report["classification"] == "Python fusion mismatch":
        example = summary["python_fuse"]["first_mismatch"]
    elif report["classification"] == "Swift runtime mismatch":
        example = summary["swift_runtime"]["first_mismatch"]
    else:
        example = None
    if example:
        lines.extend(
            [
                f"- Paper: `{example['paper_id']}`",
                f"- First differing token index: `{example['first_diff']['index']}`",
                f"- Previous matching token prefix IDs: `{example['first_diff']['matching_prefix']}`",
                f"- Previous matching token prefix text: {example['first_diff']['matching_prefix_text']!r}",
                f"- Left: `{example['left_label']}` token `{example['first_diff']['left_token']}` text {example['first_diff']['left_token_text']!r}",
                f"- Right: `{example['right_label']}` token `{example['first_diff']['right_token']}` text {example['first_diff']['right_token_text']!r}",
                f"- Stop status: left `{example.get('left_finish_reason')}` / right `{example.get('right_finish_reason')}`; left stopped earlier via EOS `{example.get('left_stopped_earlier_via_eos')}`, right stopped earlier via EOS `{example.get('right_stopped_earlier_via_eos')}`",
                f"- Left output: {example.get('left_output')}",
                f"- Right output: {example.get('right_output')}",
            ]
        )
    else:
        lines.append("No token divergence found on this fixture set.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace MLX Python vs MLX-Swift q4 parity.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
        subparser.add_argument("--overwrite", action="store_true")

    prepare = subparsers.add_parser("prepare-fixtures")
    add_common(prepare)
    prepare.set_defaults(func=command_prepare_fixtures)

    trace = subparsers.add_parser("trace-python")
    add_common(trace)
    trace.add_argument("--prompt-spec", default=str(DEFAULT_PROMPT_SPEC))
    trace.add_argument("--input-mode", default=DEFAULT_INPUT_MODE)
    trace.set_defaults(func=command_trace_python)

    report = subparsers.add_parser("report")
    add_common(report)
    report.add_argument("--swift-trace-jsonl", required=True)
    report.set_defaults(func=command_report)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
