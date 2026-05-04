#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

try:
    from scripts.sample_mlx_holdout_generations import (
        DEFAULT_HF_HOME,
        DEFAULT_INPUT_MODE,
        DEFAULT_MAX_TOKENS,
        DEFAULT_MODEL_REPO_ID,
        DEFAULT_PROMPT_SPEC,
        VariantSpec,
        generate_outputs,
        load_jsonl,
        load_prompt_spec,
        model_path_for_variant,
        prepare_prompt_rows,
        resolve_model_path,
        resolve_variant,
        slugify,
        write_jsonl,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from sample_mlx_holdout_generations import (
        DEFAULT_HF_HOME,
        DEFAULT_INPUT_MODE,
        DEFAULT_MAX_TOKENS,
        DEFAULT_MODEL_REPO_ID,
        DEFAULT_PROMPT_SPEC,
        VariantSpec,
        generate_outputs,
        load_jsonl,
        load_prompt_spec,
        model_path_for_variant,
        prepare_prompt_rows,
        resolve_model_path,
        resolve_variant,
        slugify,
        write_jsonl,
    )

DEFAULT_EVAL_JSONL = "hf_dataset/evals/safety_representative_2500_v1/eval_inputs.jsonl"
DEFAULT_PROMOTED_POINTER = (
    "hf_bucket/experiments/mlx_lora/promoted/"
    "gemma_3_270m_it_title_abstract_current.json"
)
DEFAULT_OUTPUT_ROOT = "hf_bucket/experiments/mlx_lora/safety_eval_generations"
DEFAULT_BATCH_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate summaries for a single MLX LoRA variant on a safety eval set."
    )
    parser.add_argument("--eval-jsonl", default=DEFAULT_EVAL_JSONL)
    parser.add_argument("--promoted-pointer", default=DEFAULT_PROMOTED_POINTER)
    parser.add_argument("--label", default=None)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt-spec", default=DEFAULT_PROMPT_SPEC)
    parser.add_argument(
        "--input-mode",
        choices=("abstract_only", "title_abstract"),
        default=DEFAULT_INPUT_MODE,
    )
    parser.add_argument("--hf-home", default=DEFAULT_HF_HOME)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def load_promoted_defaults(pointer_path: Path) -> dict[str, str]:
    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    if payload.get("status") != "promoted":
        raise ValueError(f"Expected promoted pointer at {pointer_path}")
    return {
        "label": str(payload["label"]),
        "adapter_dir": str(payload["adapter_dir"]),
        "checkpoint": str(payload["checkpoint_path"]),
        "base_model": str(payload.get("base_model", DEFAULT_MODEL_REPO_ID)),
    }


def build_output_dir(args: argparse.Namespace, label: str, sample_size: int) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    directory = f"{slugify(label)}_safety_eval_{sample_size}_{timestamp}"
    return Path(args.output_root) / directory


def build_manifest(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    prompt_spec: dict[str, Any],
    model_path: str,
    variant: VariantSpec,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_jsonl": args.eval_jsonl,
        "sample_size": len(rows),
        "by_split": {
            split: sum(1 for row in rows if row.get("split") == split)
            for split in sorted({str(row.get("split", "")) for row in rows})
        },
        "prompt_spec": args.prompt_spec,
        "prompt_version": prompt_spec["prompt_version"],
        "input_mode": args.input_mode,
        "base_model_path": model_path,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "temp": args.temp,
        "top_p": args.top_p,
        "variant": asdict(variant),
        "outputs": {
            "generations_jsonl": str(output_dir / "generations.jsonl"),
            "manifest_json": str(output_dir / "manifest.json"),
        },
    }


def main() -> None:
    args = parse_args()
    standalone_model = args.model_path and not args.adapter_dir and not args.checkpoint
    promoted: dict[str, str] | None = None
    if not standalone_model or args.label is None:
        promoted = load_promoted_defaults(Path(args.promoted_pointer))

    label = args.label or (promoted["label"] if promoted else "standalone_model")
    adapter_dir = args.adapter_dir or (promoted["adapter_dir"] if promoted else None)
    checkpoint = args.checkpoint or (promoted["checkpoint"] if promoted else None)

    rows = load_jsonl(Path(args.eval_jsonl))
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    prompt_spec = load_prompt_spec(Path(args.prompt_spec))
    prepared_rows = prepare_prompt_rows(rows, prompt_spec, args.input_mode)
    variant = resolve_variant(
        label=label,
        model_path_value=args.model_path,
        adapter_dir_value=adapter_dir,
        checkpoint_value=checkpoint,
        review_views_root=Path(DEFAULT_OUTPUT_ROOT) / "adapter_views",
    )
    default_model_path = None if variant.model_path else resolve_model_path(args, [variant])
    model_path = model_path_for_variant(default_model_path, variant)
    output_dir = build_output_dir(args, label, len(rows))
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = generate_outputs(
        prepared_rows,
        model_path=model_path,
        variant=variant,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
    )

    generation_rows: list[dict[str, Any]] = []
    for row, prepared_row, output in zip(rows, prepared_rows, outputs, strict=True):
        generation_rows.append(
            {
                "eval_id": row.get("eval_id"),
                "split": row.get("split"),
                "paper_id": row["paper_id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "primary_category": row.get("primary_category"),
                "umbrella_category": row.get("umbrella_category"),
                "categories": row.get("categories", []),
                "created_date": row.get("created_date"),
                "edge_topics": row.get("edge_topics", []),
                "prompt_version": prompt_spec["prompt_version"],
                "input_mode": args.input_mode,
                "user_prompt": prepared_row["user_prompt"],
                "model_label": label,
                "summary_raw": output["raw"],
                "summary": output["normalized"],
            }
        )
        for optional_key in (
            "source_paper_id",
            "adversarial_variant",
            "adversarial_source_paper_id",
        ):
            if optional_key in row:
                generation_rows[-1][optional_key] = row[optional_key]

    manifest = build_manifest(
        args=args,
        output_dir=output_dir,
        prompt_spec=prompt_spec,
        model_path=model_path,
        variant=variant,
        rows=rows,
    )
    write_jsonl(output_dir / "generations.jsonl", generation_rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"[safety-eval] wrote {len(generation_rows)} generations to {output_dir}")


if __name__ == "__main__":
    main()
