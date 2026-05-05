#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import statistics
import sys
import time
from typing import Any
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sample_mlx_holdout_generations import (
    VariantSpec,
    build_prompt,
    generate_outputs,
    load_prompt_spec,
    materialize_checkpoint_view,
    normalize_whitespace,
    prepare_prompt_rows,
    slugify,
)

DEFAULT_PROMPT_SPEC = (
    "prompts/mlx_lora/gemma_270m_it_abstract_to_glanceable_summary_v1.json"
)
DEFAULT_INPUT_MODE = "title_abstract"
DEFAULT_MAX_TOKENS = 64
DEFAULT_TEMP = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 1
DEFAULT_PACKET_SIZE = 25
DEFAULT_SEED = 20260505

RUNTIME_ISSUE_TAGS = (
    "invented_number",
    "wrong_number_binding",
    "unsupported_entity",
    "unsupported_mechanism",
    "causal_drift",
    "polarity_flip",
    "overstrong_claim",
    "misses_point",
    "malformed_repetition",
    "prompt_echo",
    "empty_output",
    "truncation",
    "style_drift",
    "non_answer",
)

DECISION_HEADERS = (
    "review_id",
    "winner",
    "left_severe",
    "right_severe",
    "left_mangled",
    "right_mangled",
    "left_issue_tags",
    "right_issue_tags",
    "confidence",
    "note",
)

WINNERS = {"left", "right", "tie", "both_bad"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def open_output(path: Path, *, overwrite: bool) -> Any:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def coerce_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def parse_issue_tags(value: str) -> list[str]:
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def word_count(value: str) -> int:
    return len(
        re.findall(
            r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|\d+(?:[./:-]\d+)*%?",
            value or "",
        )
    )


def runtime_format_flags(summary: str) -> list[str]:
    text = (summary or "").strip()
    words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text.lower())
    flags: list[str] = []
    if not text:
        flags.append("empty_output")
    if re.search(r"write (a|one|the) (subtitle|summary)|abstract below|return only|title:", text, re.I):
        flags.append("prompt_echo")
    if len(words) < 5:
        flags.append("too_short")
    if len(words) > 28:
        flags.append("very_long")
    best_run = 0
    current_run = 0
    previous = None
    for word in words:
        current_run = current_run + 1 if word == previous else 1
        best_run = max(best_run, current_run)
        previous = word
    if best_run >= 3:
        flags.append("word_repeat3")
    bigrams = Counter(zip(words, words[1:]))
    if bigrams and max(bigrams.values()) >= 3:
        flags.append("bigram_repeat3")
    return flags


def summarize_lengths(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": min(values),
        "max": max(values),
    }


def command_generate_mlx(args: argparse.Namespace) -> None:
    rows = load_jsonl(Path(args.input_jsonl))
    prompt_spec = load_prompt_spec(Path(args.prompt_spec))
    prepared_rows = prepare_prompt_rows(rows, prompt_spec, args.input_mode)
    adapter_load_path = args.adapter_path
    checkpoint_path = args.checkpoint
    if args.adapter_path and checkpoint_path:
        adapter_load_path = str(
            materialize_checkpoint_view(
                label=args.label,
                adapter_dir=Path(args.adapter_path),
                checkpoint_path=Path(checkpoint_path),
                review_views_root=Path(args.review_views_root),
            )
        )

    variant = VariantSpec(
        label=args.label,
        kind="mlx_python",
        model_path=args.model_path,
        adapter_dir=args.adapter_path,
        checkpoint_path=checkpoint_path,
        adapter_load_path=adapter_load_path,
        model_repo_id=args.model_path,
    )
    outputs = generate_outputs(
        prepared_rows,
        model_path=args.model_path,
        variant=variant,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
    )

    output_rows: list[dict[str, Any]] = []
    for source_row, output in zip(rows, outputs, strict=True):
        output_rows.append(
            {
                "paper_id": str(source_row["paper_id"]),
                "eval_id": source_row.get("eval_id"),
                "label": args.label,
                "runtime": "mlx_python",
                "raw_output": output["raw"],
                "normalized_output": output["normalized"],
                "format_flags": runtime_format_flags(output["normalized"]),
            }
        )
    write_jsonl(Path(args.output_jsonl), output_rows)
    print(f"[runtime-shift:mlx] wrote {len(output_rows)} rows to {args.output_jsonl}")


def llama_request(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def command_generate_llama(args: argparse.Namespace) -> None:
    rows = load_jsonl(Path(args.input_jsonl))
    prompt_spec = load_prompt_spec(Path(args.prompt_spec))
    output_path = Path(args.output_jsonl)
    with open_output(output_path, overwrite=args.overwrite) as handle:
        for index, row in enumerate(rows, start=1):
            system_prompt, user_prompt = build_prompt(row, prompt_spec, args.input_mode)
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": args.max_tokens,
                "temperature": args.temp,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "stream": False,
            }
            start = time.perf_counter()
            try:
                response = llama_request(args.api_url, payload, args.timeout)
                raw_output = response["choices"][0]["message"]["content"]
                normalized_output = normalize_whitespace(raw_output)
                timing_ms = round((time.perf_counter() - start) * 1000, 3)
                output = {
                    "paper_id": str(row["paper_id"]),
                    "eval_id": row.get("eval_id"),
                    "label": args.label,
                    "runtime": "llama_cpp_server",
                    "ok": bool(normalized_output),
                    "raw_output": raw_output,
                    "normalized_output": normalized_output,
                    "wall_time_ms": timing_ms,
                    "timings": response.get("timings", {}),
                    "usage": response.get("usage", {}),
                    "format_flags": runtime_format_flags(normalized_output),
                }
            except Exception as exc:  # noqa: BLE001
                output = {
                    "paper_id": str(row["paper_id"]),
                    "eval_id": row.get("eval_id"),
                    "label": args.label,
                    "runtime": "llama_cpp_server",
                    "ok": False,
                    "raw_output": "",
                    "normalized_output": "",
                    "error": str(exc),
                    "format_flags": ["empty_output"],
                }
            handle.write(json.dumps(output, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            if index % 10 == 0:
                print(f"[runtime-shift:llama] generated {index}/{len(rows)} rows...")
    print(f"[runtime-shift:llama] wrote {len(rows)} rows to {output_path}")


def load_variant_outputs(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    outputs: dict[str, dict[str, Any]] = {}
    for row in rows:
        paper_id = str(row["paper_id"])
        if paper_id in outputs:
            raise ValueError(f"Duplicate paper_id {paper_id} in {path}")
        outputs[paper_id] = row
    return outputs


def variant_summary(row: dict[str, Any]) -> str:
    return str(
        row.get("normalized_output")
        or row.get("normalized")
        or row.get("summary")
        or ""
    )


def write_pair_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    blinded: bool,
) -> None:
    left_heading = "Left Summary" if blinded else manifest["left_label"]
    right_heading = "Right Summary" if blinded else manifest["right_label"]
    lines = [
        "# Runtime Pairwise Review",
        "",
        f"- Generated at: `{manifest['created_at']}`",
        f"- Sample size: `{manifest['sample_size']}`",
        f"- Prompt version: `{manifest['prompt_version']}`",
        f"- Input mode: `{manifest['input_mode']}`",
        f"- Matching outputs: `{manifest['matching_summary_count']}/{manifest['sample_size']}`",
    ]
    if not blinded:
        lines.extend(
            [
                f"- Left label: `{manifest['left_label']}`",
                f"- Right label: `{manifest['right_label']}`",
            ]
        )
    lines.append("")
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {index:03d}. {row['paper_id']}",
                "",
                f"**Title**: {row['title']}",
                "",
                f"**Category**: `{row.get('primary_category', '')}`",
                "",
                f"**{left_heading}**:",
                "",
                f"> {row['left_summary']}",
                "",
                f"**{right_heading}**:",
                "",
                f"> {row['right_summary']}",
                "",
                f"**Same Output**: `{'yes' if row['outputs_match'] else 'no'}`",
                "",
                "**Abstract**:",
                "",
                row["abstract"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def command_build_pairs(args: argparse.Namespace) -> None:
    source_rows = load_jsonl(Path(args.input_jsonl))
    prompt_spec = load_prompt_spec(Path(args.prompt_spec))
    variants = {
        "canonical_mlx_q4_adapter": load_variant_outputs(Path(args.canonical_outputs)),
        "gguf_q4_0_llama_cpp": load_variant_outputs(Path(args.gguf_outputs)),
        "mlx_swift_q4_fused": load_variant_outputs(Path(args.mlx_swift_outputs)),
    }
    variant_output_paths = {
        "canonical_mlx_q4_adapter": args.canonical_outputs,
        "gguf_q4_0_llama_cpp": args.gguf_outputs,
        "mlx_swift_q4_fused": args.mlx_swift_outputs,
    }
    pairs = [
        ("canonical_mlx_q4_adapter", "gguf_q4_0_llama_cpp"),
        ("canonical_mlx_q4_adapter", "mlx_swift_q4_fused"),
        ("gguf_q4_0_llama_cpp", "mlx_swift_q4_fused"),
    ]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    root_manifest = {
        "created_at": utc_now_iso(),
        "input_jsonl": args.input_jsonl,
        "prompt_spec": args.prompt_spec,
        "prompt_version": prompt_spec["prompt_version"],
        "input_mode": args.input_mode,
        "sample_size": len(source_rows),
        "variant_outputs": variant_output_paths,
        "pairs": [],
    }
    for left_label, right_label in pairs:
        pair_slug = f"{left_label}_vs_{right_label}"
        pair_dir = output_root / pair_slug
        pair_dir.mkdir(parents=True, exist_ok=True)
        generated_rows: list[dict[str, Any]] = []
        for row in source_rows:
            paper_id = str(row["paper_id"])
            left_output = variants[left_label][paper_id]
            right_output = variants[right_label][paper_id]
            left_summary = variant_summary(left_output)
            right_summary = variant_summary(right_output)
            generated_row = {
                "paper_id": paper_id,
                "eval_id": row.get("eval_id"),
                "title": str(row["title"]),
                "abstract": str(row["abstract"]),
                "primary_category": row.get("primary_category"),
                "categories": row.get("categories", []),
                "created_date": row.get("created_date"),
                "selection_bucket": row.get("selection_bucket") or row.get("split"),
                "q4_hardening_axes": row.get("q4_hardening_axes", []),
                "prompt_version": prompt_spec["prompt_version"],
                "input_mode": args.input_mode,
                "left_label": left_label,
                "right_label": right_label,
                "left_summary_raw": str(left_output.get("raw_output") or left_summary),
                "left_summary": left_summary,
                "right_summary_raw": str(right_output.get("raw_output") or right_summary),
                "right_summary": right_summary,
                "left_format_flags": left_output.get("format_flags", runtime_format_flags(left_summary)),
                "right_format_flags": right_output.get("format_flags", runtime_format_flags(right_summary)),
                "outputs_match": left_summary == right_summary,
            }
            generated_rows.append(generated_row)
        matching_count = sum(row["outputs_match"] for row in generated_rows)
        manifest = {
            "created_at": utc_now_iso(),
            "input_jsonl": args.input_jsonl,
            "prompt_spec": args.prompt_spec,
            "prompt_version": prompt_spec["prompt_version"],
            "input_mode": args.input_mode,
            "sample_size": len(generated_rows),
            "left_label": left_label,
            "right_label": right_label,
            "left_output_jsonl": variant_output_paths[left_label],
            "right_output_jsonl": variant_output_paths[right_label],
            "matching_summary_count": matching_count,
            "word_lengths": {
                left_label: summarize_lengths([word_count(row["left_summary"]) for row in generated_rows]),
                right_label: summarize_lengths([word_count(row["right_summary"]) for row in generated_rows]),
            },
            "format_flag_counts": {
                left_label: dict(Counter(flag for row in generated_rows for flag in row["left_format_flags"])),
                right_label: dict(Counter(flag for row in generated_rows for flag in row["right_format_flags"])),
            },
            "paper_ids": [row["paper_id"] for row in generated_rows],
        }
        write_jsonl(pair_dir / "generations.jsonl", generated_rows)
        write_json(pair_dir / "manifest.json", manifest)
        write_pair_markdown(pair_dir / "review.md", generated_rows, manifest, blinded=True)
        write_pair_markdown(pair_dir / "review_unblinded.md", generated_rows, manifest, blinded=False)
        root_manifest["pairs"].append({"pair": pair_slug, "manifest": str(pair_dir / "manifest.json")})
        print(f"[runtime-shift:pairs] wrote {len(generated_rows)} rows to {pair_dir}")
    write_json(output_root / "manifest.json", root_manifest)


def side_payload(row: dict[str, Any], source_side: str) -> dict[str, Any]:
    return {
        "label": row[f"{source_side}_label"],
        "summary": row[f"{source_side}_summary"],
        "summary_raw": row.get(f"{source_side}_summary_raw", ""),
        "format_flags": row.get(f"{source_side}_format_flags", []),
    }


def write_blank_decisions(path: Path, review_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECISION_HEADERS, delimiter="\t")
        writer.writeheader()
        for review_id in review_ids:
            writer.writerow({"review_id": review_id})


def command_prepare_packets(args: argparse.Namespace) -> None:
    generation_rows = load_jsonl(Path(args.generations_jsonl))
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists; pass --overwrite")
    rng = random.Random(args.seed)
    review_rows: list[dict[str, Any]] = []
    side_maps: list[dict[str, Any]] = []
    for index, row in enumerate(generation_rows, start=1):
        left_payload = side_payload(row, "left")
        right_payload = side_payload(row, "right")
        swap = rng.choice([True, False])
        blinded_left = right_payload if swap else left_payload
        blinded_right = left_payload if swap else right_payload
        review_id = f"{args.review_id_prefix}:{index:04d}"
        review_rows.append(
            {
                "review_id": review_id,
                "paper_id": row["paper_id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "primary_category": row.get("primary_category"),
                "categories": row.get("categories", []),
                "created_date": row.get("created_date"),
                "selection_bucket": row.get("selection_bucket"),
                "q4_hardening_axes": row.get("q4_hardening_axes", []),
                "prompt_version": row.get("prompt_version"),
                "input_mode": row.get("input_mode"),
                "left_summary": blinded_left["summary"],
                "right_summary": blinded_right["summary"],
                "review_instruction": (
                    "Blinded runtime pairwise review. Choose the better feed summary, "
                    "mark severe faithfulness issues, and separately mark mangled output "
                    "for repetition, prompt echo, empty output, truncation, or obvious formatting corruption."
                ),
                "issue_tag_options": list(RUNTIME_ISSUE_TAGS),
            }
        )
        side_maps.append(
            {
                "review_id": review_id,
                "paper_id": row["paper_id"],
                "left_source": blinded_left["label"],
                "right_source": blinded_right["label"],
                "left_original_side": "right" if swap else "left",
                "right_original_side": "left" if swap else "right",
                "source_left_label": row["left_label"],
                "source_right_label": row["right_label"],
                "source_left_summary": row["left_summary"],
                "source_right_summary": row["right_summary"],
            }
        )
    packets = [
        review_rows[start : start + args.packet_size]
        for start in range(0, len(review_rows), args.packet_size)
    ]
    write_jsonl(output_dir / "review_items.jsonl", review_rows)
    write_jsonl(output_dir / "side_map.jsonl", side_maps)
    for packet_index, packet_rows in enumerate(packets):
        packet_dir = output_dir / "primary" / f"packet_{packet_index:02d}"
        write_jsonl(packet_dir / "review_items.jsonl", packet_rows)
        write_blank_decisions(packet_dir / "decisions.tsv", [row["review_id"] for row in packet_rows])
    manifest = {
        "created_at": utc_now_iso(),
        "generations_jsonl": args.generations_jsonl,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "packet_size": args.packet_size,
        "review_rows": len(review_rows),
        "primary_packets": len(packets),
        "decision_headers": DECISION_HEADERS,
        "issue_tags": RUNTIME_ISSUE_TAGS,
        "side_counts": dict(Counter(row["left_source"] for row in side_maps)),
    }
    write_json(output_dir / "manifest.json", manifest)
    readme = (
        "# Runtime Pairwise Blinded Review Packets\n\n"
        "Fill each `decisions.tsv` row with:\n"
        "- `winner`: `left`, `right`, `tie`, or `both_bad`\n"
        "- `left_severe` / `right_severe`: `yes` or `no` for major faithfulness or usefulness failures\n"
        "- `left_mangled` / `right_mangled`: `yes` for prompt echo, repetition, truncation, empty output, or formatting corruption\n"
        "- `left_issue_tags` / `right_issue_tags`: comma-separated tags from `issue_tag_options`\n"
        "- `confidence`: `high`, `medium`, or `low`\n"
        "- `note`: short reason, especially for severe or mangled rows\n\n"
        "Do not infer which runtime produced which side; the side map is intentionally separate.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(f"[runtime-shift:packets] wrote {len(review_rows)} rows to {output_dir}")


def load_decisions(review_dir: Path) -> dict[str, dict[str, str]]:
    decisions: dict[str, dict[str, str]] = {}
    for decision_path in sorted((review_dir / "primary").glob("packet_*/decisions.tsv")):
        with decision_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                review_id = str(row.get("review_id", "")).strip()
                if not review_id:
                    continue
                winner = str(row.get("winner", "")).strip().lower()
                if not winner:
                    continue
                if winner not in WINNERS:
                    raise ValueError(f"{decision_path}: invalid winner {winner!r} for {review_id}")
                decisions[review_id] = {key: str(value or "").strip() for key, value in row.items()}
    return decisions


def command_aggregate(args: argparse.Namespace) -> None:
    review_dir = Path(args.review_dir)
    side_maps = {row["review_id"]: row for row in load_jsonl(review_dir / "side_map.jsonl")}
    review_rows = {row["review_id"]: row for row in load_jsonl(review_dir / "review_items.jsonl")}
    decisions = load_decisions(review_dir)
    output_dir = Path(args.output_dir or review_dir / "aggregation")
    output_dir.mkdir(parents=True, exist_ok=True)
    reviewed_rows: list[dict[str, Any]] = []
    counts: dict[str, Any] = {
        "reviewed_rows": len(decisions),
        "missing_decisions": len(review_rows) - len(decisions),
        "winner_counts": Counter(),
        "severe_counts": Counter(),
        "mangled_counts": Counter(),
        "issue_tag_counts": defaultdict(Counter),
    }
    for review_id, decision in sorted(decisions.items()):
        side_map = side_maps[review_id]
        review_row = review_rows[review_id]
        winner = decision["winner"]
        winner_label = winner
        if winner in {"left", "right"}:
            winner_label = side_map[f"{winner}_source"]
        counts["winner_counts"][winner_label] += 1
        output = {
            "review_id": review_id,
            "paper_id": side_map["paper_id"],
            "winner": winner_label,
            "raw_winner": winner,
            "left_label": side_map["left_source"],
            "right_label": side_map["right_source"],
            "left_summary": review_row["left_summary"],
            "right_summary": review_row["right_summary"],
            "confidence": decision.get("confidence", ""),
            "note": decision.get("note", ""),
        }
        for side in ("left", "right"):
            label = side_map[f"{side}_source"]
            severe = coerce_bool(decision.get(f"{side}_severe", ""))
            mangled = coerce_bool(decision.get(f"{side}_mangled", ""))
            tags = parse_issue_tags(decision.get(f"{side}_issue_tags", ""))
            output[f"{label}_severe"] = severe
            output[f"{label}_mangled"] = mangled
            output[f"{label}_issue_tags"] = tags
            if severe:
                counts["severe_counts"][label] += 1
            if mangled:
                counts["mangled_counts"][label] += 1
            counts["issue_tag_counts"][label].update(tags)
        reviewed_rows.append(output)
    summary = {
        "created_at": utc_now_iso(),
        "review_dir": str(review_dir),
        "summary": {
            "reviewed_rows": counts["reviewed_rows"],
            "missing_decisions": counts["missing_decisions"],
            "winner_counts": dict(counts["winner_counts"]),
            "severe_counts": dict(counts["severe_counts"]),
            "mangled_counts": dict(counts["mangled_counts"]),
            "issue_tag_counts": {
                label: dict(counter)
                for label, counter in counts["issue_tag_counts"].items()
            },
        },
    }
    write_jsonl(output_dir / "reviewed_rows.jsonl", reviewed_rows)
    write_json(output_dir / "review_summary.json", summary)
    print(json.dumps(summary["summary"], indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run q4 runtime-shift generation and review helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mlx = subparsers.add_parser("generate-mlx", help="Generate canonical MLX Python outputs.")
    mlx.add_argument("--input-jsonl", required=True)
    mlx.add_argument("--output-jsonl", required=True)
    mlx.add_argument("--label", default="canonical_mlx_q4_adapter")
    mlx.add_argument("--prompt-spec", default=DEFAULT_PROMPT_SPEC)
    mlx.add_argument("--input-mode", default=DEFAULT_INPUT_MODE)
    mlx.add_argument("--model-path", required=True)
    mlx.add_argument("--adapter-path", default=None)
    mlx.add_argument("--checkpoint", default=None)
    mlx.add_argument("--review-views-root", default="hf_bucket/experiments/mlx_lora/review_views")
    mlx.add_argument("--batch-size", type=int, default=8)
    mlx.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    mlx.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    mlx.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    mlx.set_defaults(func=command_generate_mlx)

    llama = subparsers.add_parser("generate-llama", help="Generate llama.cpp server outputs.")
    llama.add_argument("--input-jsonl", required=True)
    llama.add_argument("--output-jsonl", required=True)
    llama.add_argument("--label", default="gguf_q4_0_llama_cpp")
    llama.add_argument("--prompt-spec", default=DEFAULT_PROMPT_SPEC)
    llama.add_argument("--input-mode", default=DEFAULT_INPUT_MODE)
    llama.add_argument("--api-url", default="http://127.0.0.1:18083/v1/chat/completions")
    llama.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    llama.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    llama.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    llama.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    llama.add_argument("--timeout", type=float, default=60.0)
    llama.add_argument("--overwrite", action="store_true")
    llama.set_defaults(func=command_generate_llama)

    pairs = subparsers.add_parser("build-pairs", help="Build the three pairwise runtime bundles.")
    pairs.add_argument("--input-jsonl", required=True)
    pairs.add_argument("--output-root", required=True)
    pairs.add_argument("--canonical-outputs", required=True)
    pairs.add_argument("--gguf-outputs", required=True)
    pairs.add_argument("--mlx-swift-outputs", required=True)
    pairs.add_argument("--prompt-spec", default=DEFAULT_PROMPT_SPEC)
    pairs.add_argument("--input-mode", default=DEFAULT_INPUT_MODE)
    pairs.set_defaults(func=command_build_pairs)

    packets = subparsers.add_parser("prepare-packets", help="Prepare blinded review packets for one pair.")
    packets.add_argument("--generations-jsonl", required=True)
    packets.add_argument("--output-dir", required=True)
    packets.add_argument("--review-id-prefix", required=True)
    packets.add_argument("--seed", type=int, default=DEFAULT_SEED)
    packets.add_argument("--packet-size", type=int, default=DEFAULT_PACKET_SIZE)
    packets.add_argument("--overwrite", action="store_true")
    packets.set_defaults(func=command_prepare_packets)

    aggregate = subparsers.add_parser("aggregate", help="Aggregate filled runtime review packet decisions.")
    aggregate.add_argument("--review-dir", required=True)
    aggregate.add_argument("--output-dir", default=None)
    aggregate.set_defaults(func=command_aggregate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
