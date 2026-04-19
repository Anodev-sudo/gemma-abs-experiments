#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import random
from pathlib import Path
from typing import Any

from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

DEFAULT_RELEASE_DIR = "hf_dataset/releases/recent_3y_balanced_5000_pilot_v2"
DEFAULT_CORPUS_JSONL = (
    "hf_dataset/corpora/recent_3y_balanced/arxiv_abstract_corpus_weighted_20000.jsonl"
)
DEFAULT_PROMPT_SPEC = (
    "prompts/mlx_lora/gemma_270m_it_abstract_to_glanceable_summary_v1.json"
)
DEFAULT_ADAPTER_PATH = (
    "hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_v1/adapters"
)
DEFAULT_HF_HOME = "hf_bucket/experiments/mlx_lora/.hf-home"
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_BATCH_SIZE = 8
DEFAULT_SEED = 20260419
DEFAULT_MAX_TOKENS = 40
DEFAULT_INPUT_MODE = "title_abstract"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired base-model and adapter-model summary samples from the "
            "recent-balanced corpus after excluding the papers used in the current "
            "5k pilot training run."
        )
    )
    parser.add_argument(
        "--release-dir",
        default=DEFAULT_RELEASE_DIR,
        help="Promoted release directory with training_rows.jsonl.",
    )
    parser.add_argument(
        "--corpus-jsonl",
        default=DEFAULT_CORPUS_JSONL,
        help="Source corpus to sample held-out rows from.",
    )
    parser.add_argument(
        "--prompt-spec",
        default=DEFAULT_PROMPT_SPEC,
        help="Prompt spec JSON used to build inference prompts.",
    )
    parser.add_argument(
        "--adapter-path",
        default=DEFAULT_ADAPTER_PATH,
        help="Directory containing adapters.safetensors and adapter_config.json.",
    )
    parser.add_argument(
        "--hf-home",
        default=DEFAULT_HF_HOME,
        help="HF cache root used during training.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Optional explicit base model path or repo id. Defaults to the locally "
            "cached Gemma snapshot if present."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for review artifacts. Defaults to a timestamped directory "
            "under the current experiment."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of held-out papers to generate summaries for.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of prompts to generate per batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic sampling seed.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("abstract_only", "title_abstract"),
        default=DEFAULT_INPUT_MODE,
        help="Prompt mode to use for generation.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum generated tokens per paper.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature. Default 0 uses greedy decoding.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Optional nucleus sampling value when temp > 0.",
    )
    return parser.parse_args()


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
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def load_prompt_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = ("prompt_version", "system_prompt", "user_prompt_templates")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(
            f"Prompt spec {path} is missing required keys: {', '.join(missing)}"
        )
    return spec


def infer_raw_run_dir(release_dir: Path) -> Path:
    return Path("hf_bucket/runs") / release_dir.name


def collect_excluded_ids(release_dir: Path) -> tuple[set[str], str]:
    training_rows_path = release_dir / "training_rows.jsonl"
    if not training_rows_path.exists():
        raise FileNotFoundError(f"Missing training rows at {training_rows_path}")

    raw_run_dir = infer_raw_run_dir(release_dir)
    sample_source_path = raw_run_dir / "sample_source_rows.jsonl"
    if sample_source_path.exists():
        return (
            {str(row["paper_id"]) for row in load_jsonl(sample_source_path)},
            f"sample_source_rows from {sample_source_path}",
        )

    excluded_ids = {str(row["paper_id"]) for row in load_jsonl(training_rows_path)}
    exclusion_note = [f"kept training rows from {training_rows_path}"]

    dropped_rows_path = raw_run_dir / "dropped_rows.jsonl"
    if dropped_rows_path.exists():
        excluded_ids.update(str(row["paper_id"]) for row in load_jsonl(dropped_rows_path))
        exclusion_note.append(f"dropped rows from {dropped_rows_path}")

    return excluded_ids, " + ".join(exclusion_note)


def resolve_model_path(args: argparse.Namespace, adapter_config: dict[str, Any]) -> str:
    if args.model_path:
        return args.model_path

    hf_home = Path(args.hf_home)
    repo_id = str(adapter_config.get("model", "google/gemma-3-270m-it"))
    repo_cache_dir = hf_home / "hub" / "models--google--gemma-3-270m-it"
    ref_path = repo_cache_dir / "refs" / "main"
    if ref_path.exists():
        snapshot = ref_path.read_text(encoding="utf-8").strip()
        snapshot_path = repo_cache_dir / "snapshots" / snapshot
        if snapshot_path.exists():
            return str(snapshot_path)
    return repo_id


def build_prompt(
    row: dict[str, Any],
    prompt_spec: dict[str, Any],
    input_mode: str,
) -> tuple[str, str]:
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
        return tokenizer.encode(rendered, add_special_tokens=False)
    prompt = f"{system_prompt}\n\n{user_prompt}"
    return tokenizer.encode(prompt)


def build_markdown_report(
    output_path: Path,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# Holdout Generation Review",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Sample size: `{manifest['sample_size']}`",
        f"- Seed: `{manifest['seed']}`",
        f"- Input mode: `{manifest['input_mode']}`",
        f"- Prompt version: `{manifest['prompt_version']}`",
        f"- Exclusion source: `{manifest['exclusion_source']}`",
        f"- Base model path: `{manifest['base_model_path']}`",
        f"- Adapter path: `{manifest['adapter_path']}`",
        f"- Matching outputs: `{manifest['matching_summary_count']}/{manifest['sample_size']}`",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {index:03d}. {row['paper_id']}",
                "",
                f"**Title**: {row['title']}",
                "",
                f"**Category**: `{row['primary_category']}`",
                "",
                f"**Created**: `{row.get('created_date', '')}`",
                "",
                "**Base Summary**:",
                "",
                f"> {row['base_summary']}",
                "",
                "**Fine-Tuned Summary**:",
                "",
                f"> {row['tuned_summary']}",
                "",
                f"**Same Output**: `{'yes' if row['outputs_match'] else 'no'}`",
                "",
                "**Abstract**:",
                "",
                row["abstract"],
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def prepare_prompt_rows(
    rows: list[dict[str, Any]],
    prompt_spec: dict[str, Any],
    input_mode: str,
) -> list[dict[str, Any]]:
    prepared_rows: list[dict[str, Any]] = []
    for row in rows:
        system_prompt, user_prompt = build_prompt(row, prompt_spec, input_mode)
        prepared_rows.append(
            {
                "paper_id": row["paper_id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "primary_category": row["primary_category"],
                "categories": row.get("categories", []),
                "created_date": row.get("created_date"),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
    return prepared_rows


def generate_outputs(
    prepared_rows: list[dict[str, Any]],
    *,
    model_path: str,
    adapter_path: Path | None,
    batch_size: int,
    max_tokens: int,
    temp: float,
    top_p: float,
    run_label: str,
) -> list[dict[str, str]]:
    print(f"[sample:{run_label}] loading model from {model_path}")
    if adapter_path is None:
        print(f"[sample:{run_label}] running without adapters")
    else:
        print(f"[sample:{run_label}] loading adapters from {adapter_path}")

    load_kwargs: dict[str, Any] = {
        "tokenizer_config": {"trust_remote_code": True},
    }
    if adapter_path is not None:
        load_kwargs["adapter_path"] = str(adapter_path)
    model, tokenizer = load(model_path, **load_kwargs)

    sampler = make_sampler(temp=temp, top_p=top_p)
    generated_outputs: list[dict[str, str]] = []
    try:
        for batch_start in range(0, len(prepared_rows), batch_size):
            batch_rows = prepared_rows[batch_start : batch_start + batch_size]
            tokenized_prompts = [
                encode_prompt(tokenizer, row["system_prompt"], row["user_prompt"])
                for row in batch_rows
            ]
            response = batch_generate(
                model,
                tokenizer,
                tokenized_prompts,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )
            for generated_text in response.texts:
                generated_outputs.append(
                    {
                        "raw": generated_text,
                        "normalized": normalize_whitespace(generated_text),
                    }
                )
            print(
                f"[sample:{run_label}] generated {len(generated_outputs)}/"
                f"{len(prepared_rows)} summaries..."
            )
    finally:
        del model
        del tokenizer
        del sampler
        gc.collect()

    return generated_outputs


def main() -> None:
    args = parse_args()
    random_generator = random.Random(args.seed)

    release_dir = Path(args.release_dir)
    corpus_path = Path(args.corpus_jsonl)
    prompt_spec_path = Path(args.prompt_spec)
    adapter_path = Path(args.adapter_path)

    prompt_spec = load_prompt_spec(prompt_spec_path)
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing adapter config at {adapter_config_path}")
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))

    excluded_ids, exclusion_source = collect_excluded_ids(release_dir)
    corpus_rows = load_jsonl(corpus_path)
    eligible_rows = [
        row for row in corpus_rows if str(row["paper_id"]) not in excluded_ids
    ]
    if args.sample_size > len(eligible_rows):
        raise ValueError(
            f"Requested {args.sample_size} samples but only {len(eligible_rows)} "
            "rows remain after exclusions."
        )

    sampled_rows = random_generator.sample(eligible_rows, args.sample_size)
    prepared_rows = prepare_prompt_rows(sampled_rows, prompt_spec, args.input_mode)
    model_path = resolve_model_path(args, adapter_config)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            adapter_path.parent / "review_samples" / f"holdout_{args.sample_size}_{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_outputs = generate_outputs(
        prepared_rows,
        model_path=model_path,
        adapter_path=None,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
        run_label="base",
    )
    tuned_outputs = generate_outputs(
        prepared_rows,
        model_path=model_path,
        adapter_path=adapter_path,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
        run_label="tuned",
    )

    generated_rows: list[dict[str, Any]] = []
    for row, prepared_row, base_output, tuned_output in zip(
        sampled_rows,
        prepared_rows,
        base_outputs,
        tuned_outputs,
        strict=True,
    ):
        generated_rows.append(
            {
                "paper_id": row["paper_id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "primary_category": row["primary_category"],
                "categories": row.get("categories", []),
                "created_date": row.get("created_date"),
                "prompt_version": prompt_spec["prompt_version"],
                "input_mode": args.input_mode,
                "user_prompt": prepared_row["user_prompt"],
                "base_summary_raw": base_output["raw"],
                "base_summary": base_output["normalized"],
                "tuned_summary_raw": tuned_output["raw"],
                "tuned_summary": tuned_output["normalized"],
                "outputs_match": base_output["normalized"] == tuned_output["normalized"],
            }
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release_dir": str(release_dir),
        "corpus_jsonl": str(corpus_path),
        "prompt_spec": str(prompt_spec_path),
        "prompt_version": prompt_spec["prompt_version"],
        "adapter_path": str(adapter_path),
        "base_model_path": model_path,
        "sample_size": args.sample_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "input_mode": args.input_mode,
        "max_tokens": args.max_tokens,
        "temp": args.temp,
        "top_p": args.top_p,
        "excluded_paper_count": len(excluded_ids),
        "eligible_paper_count": len(eligible_rows),
        "exclusion_source": exclusion_source,
        "matching_summary_count": sum(
            1 for row in generated_rows if row["outputs_match"]
        ),
        "outputs": {
            "generations_jsonl": "generations.jsonl",
            "review_markdown": "review.md",
        },
    }

    write_jsonl(output_dir / "generations.jsonl", generated_rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    build_markdown_report(output_dir / "review.md", generated_rows, manifest)

    print(f"[sample] wrote review artifacts to {output_dir}")


if __name__ == "__main__":
    main()
