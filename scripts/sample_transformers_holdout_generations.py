#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from dpo_utils import (
    DEFAULT_INPUT_MODE,
    DEFAULT_MLX_HF_HOME,
    DEFAULT_MODEL_REPO_ID,
    DEFAULT_PROMPT_SPEC,
    DEFAULT_REUSE_SAMPLE_JSONL,
    build_prompt_messages,
    ensure_new_output_dir,
    load_jsonl,
    load_prompt_spec,
    normalize_summary,
    resolve_model_path,
    slugify,
    utc_timestamp_slug,
    write_jsonl,
)

try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - import guard for runtime setup
    raise SystemExit(
        "This script needs the DPO runtime packages. Run it with the DPO runtime "
        "Python after that environment is created."
    ) from exc


DEFAULT_REVIEW_ROOT = (
    "hf_bucket/experiments/dpo/review_comparisons/"
    "gemma_3_270m_it_title_abstract_dpo_hc100_v1"
)
DEFAULT_MAX_NEW_TOKENS = 40
DEFAULT_BATCH_SIZE = 4


def preferred_inference_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.backends.mps.is_available() else torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired holdout summaries with a Transformers/PEFT backend "
            "for DPO comparison work."
        )
    )
    parser.add_argument(
        "--left-label",
        default="baseline",
        help="Label for the left variant.",
    )
    parser.add_argument(
        "--right-label",
        default="candidate",
        help="Label for the right variant.",
    )
    parser.add_argument(
        "--left-adapter-dir",
        default=None,
        help="Optional PEFT adapter directory for the left variant.",
    )
    parser.add_argument(
        "--right-adapter-dir",
        default=None,
        help="Optional PEFT adapter directory for the right variant.",
    )
    parser.add_argument(
        "--reuse-sample-jsonl",
        default=DEFAULT_REUSE_SAMPLE_JSONL,
        help="Existing holdout generations.jsonl that defines the fixed sample set.",
    )
    parser.add_argument(
        "--prompt-spec",
        default=DEFAULT_PROMPT_SPEC,
        help="Prompt spec JSON used to build prompts.",
    )
    parser.add_argument(
        "--input-mode",
        default=DEFAULT_INPUT_MODE,
        choices=("abstract_only", "title_abstract"),
        help="Prompt mode to use.",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_MODEL_REPO_ID,
        help="Base model repo id.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional explicit local model snapshot path.",
    )
    parser.add_argument(
        "--hf-home",
        default=DEFAULT_MLX_HF_HOME,
        help="HF cache root used to resolve the local model snapshot.",
    )
    parser.add_argument(
        "--review-root",
        default=DEFAULT_REVIEW_ROOT,
        help="Default root for timestamped comparison directories.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of holdout rows to sample from the fixed set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum generated tokens per example.",
    )
    parser.add_argument(
        "--blinded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether review.md should use Left/Right headings.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in a non-empty output directory.",
    )
    return parser.parse_args()


def build_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    label_slug = f"{slugify(args.left_label)}_vs_{slugify(args.right_label)}"
    return Path(args.review_root) / f"{label_slug}_{utc_timestamp_slug()}"


def load_fixed_rows(path: Path, max_examples: int | None) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if max_examples is not None:
        rows = rows[:max_examples]
    prepared_rows: list[dict[str, Any]] = []
    for row in rows:
        prepared_rows.append(
            {
                "paper_id": str(row["paper_id"]),
                "title": str(row["title"]),
                "abstract": str(row["abstract"]),
                "primary_category": str(row.get("primary_category", "")),
                "categories": row.get("categories", []),
                "created_date": str(row.get("created_date", "")),
            }
        )
    return prepared_rows


def generate_variant_outputs(
    *,
    model_path: str,
    adapter_dir: Path | None,
    prompt_spec_path: Path,
    input_mode: str,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, str]]:
    prompt_spec = load_prompt_spec(prompt_spec_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=preferred_inference_dtype(),
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    model.eval()
    eos_token_id = model.generation_config.eos_token_id
    pad_token_id = (
        model.generation_config.pad_token_id
        if model.generation_config.pad_token_id is not None
        else tokenizer.pad_token_id
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    outputs: list[dict[str, str]] = []
    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        rendered_prompts: list[str] = []
        for row in batch_rows:
            messages = build_prompt_messages(row, prompt_spec, input_mode)
            if getattr(tokenizer, "chat_template", None):
                rendered_prompts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            else:
                rendered_prompts.append(
                    "\n\n".join(message["content"] for message in messages)
                )
        encoded = tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_width = int(encoded["input_ids"].shape[1])
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        for row_index in range(generated.shape[0]):
            continuation = generated[row_index, prompt_width:]
            raw = tokenizer.decode(continuation, skip_special_tokens=True)
            outputs.append(
                {
                    "raw": raw,
                    "normalized": normalize_summary(raw),
                }
            )

    del model
    del tokenizer
    gc.collect()
    return outputs


def build_rows(
    fixed_rows: list[dict[str, Any]],
    left_outputs: list[dict[str, str]],
    right_outputs: list[dict[str, str]],
    *,
    left_label: str,
    right_label: str,
    prompt_version: str,
    input_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_row, left_output, right_output in zip(fixed_rows, left_outputs, right_outputs):
        rows.append(
            {
                **base_row,
                "prompt_version": prompt_version,
                "input_mode": input_mode,
                "left_label": left_label,
                "right_label": right_label,
                "left_summary": left_output["normalized"],
                "left_summary_raw": left_output["raw"],
                "right_summary": right_output["normalized"],
                "right_summary_raw": right_output["raw"],
                "outputs_match": left_output["normalized"] == right_output["normalized"],
            }
        )
    return rows


def build_review_rows_blinded(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blinded_rows: list[dict[str, Any]] = []
    for row in rows:
        blinded_rows.append(
            {
                "paper_id": row["paper_id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "left_summary": row["left_summary"],
                "right_summary": row["right_summary"],
            }
        )
    return blinded_rows


def write_markdown_report(
    path: Path,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    left_heading: str,
    right_heading: str,
) -> None:
    lines = [
        f"# {manifest['comparison_name']}",
        "",
        f"- Left: `{left_heading}`",
        f"- Right: `{right_heading}`",
        f"- Rows: `{manifest['row_count']}`",
        f"- Prompt version: `{manifest['prompt_version']}`",
        f"- Input mode: `{manifest['input_mode']}`",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['paper_id']}",
                "",
                f"**Title**: {row['title']}",
                "",
                f"**Abstract**: {row['abstract']}",
                "",
                f"**{left_heading}**: {row['left_summary']}",
                "",
                f"**{right_heading}**: {row['right_summary']}",
                "",
            ]
        )
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    prompt_spec_path = Path(args.prompt_spec)
    fixed_rows = load_fixed_rows(Path(args.reuse_sample_jsonl), args.max_examples)
    prompt_spec = load_prompt_spec(prompt_spec_path)

    output_dir = build_output_dir(args)
    ensure_new_output_dir(output_dir, overwrite=args.overwrite)

    model_path = resolve_model_path(
        model_path=args.model_path,
        hf_home=Path(args.hf_home),
        repo_id=args.base_model,
    )

    print(f"[sample-transformers] model path: {model_path}")
    print(f"[sample-transformers] left adapter: {args.left_adapter_dir or '<base>'}")
    left_outputs = generate_variant_outputs(
        model_path=model_path,
        adapter_dir=Path(args.left_adapter_dir) if args.left_adapter_dir else None,
        prompt_spec_path=prompt_spec_path,
        input_mode=args.input_mode,
        rows=fixed_rows,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"[sample-transformers] right adapter: {args.right_adapter_dir or '<base>'}")
    right_outputs = generate_variant_outputs(
        model_path=model_path,
        adapter_dir=Path(args.right_adapter_dir) if args.right_adapter_dir else None,
        prompt_spec_path=prompt_spec_path,
        input_mode=args.input_mode,
        rows=fixed_rows,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    rows = build_rows(
        fixed_rows,
        left_outputs,
        right_outputs,
        left_label=args.left_label,
        right_label=args.right_label,
        prompt_version=str(prompt_spec["prompt_version"]),
        input_mode=args.input_mode,
    )
    blinded_rows = build_review_rows_blinded(rows)

    manifest = {
        "comparison_name": f"{args.left_label} vs {args.right_label}",
        "row_count": len(rows),
        "left_label": args.left_label,
        "right_label": args.right_label,
        "left_adapter_dir": args.left_adapter_dir,
        "right_adapter_dir": args.right_adapter_dir,
        "base_model_repo_id": args.base_model,
        "base_model_path_used": model_path,
        "sample_source_path": args.reuse_sample_jsonl,
        "prompt_spec_path": args.prompt_spec,
        "prompt_version": str(prompt_spec["prompt_version"]),
        "input_mode": args.input_mode,
        "matching_output_count": sum(1 for row in rows if row["outputs_match"]),
    }

    write_jsonl(output_dir / "generations.jsonl", rows)
    write_jsonl(output_dir / "review_rows_blinded.jsonl", blinded_rows)
    write_markdown_report(
        output_dir / "review.md",
        rows,
        manifest,
        left_heading="Left Summary" if args.blinded else args.left_label,
        right_heading="Right Summary" if args.blinded else args.right_label,
    )
    write_markdown_report(
        output_dir / "review_unblinded.md",
        rows,
        manifest,
        left_heading=args.left_label,
        right_heading=args.right_label,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"[sample-transformers] wrote comparison artifacts to {output_dir}")
    print(
        f"[sample-transformers] matching outputs: "
        f"{manifest['matching_output_count']}/{manifest['row_count']}"
    )


if __name__ == "__main__":
    main()
