#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
import re
import shutil
from pathlib import Path
import random
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
DEFAULT_REUSE_SAMPLE_JSONL = (
    "hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_v1/"
    "review_samples/holdout_100_20260419-075505/generations.jsonl"
)
DEFAULT_HF_HOME = "hf_bucket/experiments/mlx_lora/.hf-home"
DEFAULT_COMPARISON_ROOT = "hf_bucket/experiments/mlx_lora/review_comparisons"
DEFAULT_REVIEW_VIEWS_ROOT = "hf_bucket/experiments/mlx_lora/review_views"
DEFAULT_MODEL_REPO_ID = "google/gemma-3-270m-it"
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_BATCH_SIZE = 8
DEFAULT_SEED = 20260419
DEFAULT_MAX_TOKENS = 40
DEFAULT_INPUT_MODE = "title_abstract"


@dataclass
class VariantSpec:
    label: str
    kind: str
    adapter_dir: str | None
    checkpoint_path: str | None
    adapter_load_path: str | None
    model_repo_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate blinded or unblinded paired summary comparison artifacts for "
            "two variants over a fixed or freshly sampled holdout set."
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
        help="Source corpus to sample held-out rows from when reuse-sample-jsonl is omitted.",
    )
    parser.add_argument(
        "--prompt-spec",
        default=DEFAULT_PROMPT_SPEC,
        help="Prompt spec JSON used to build inference prompts.",
    )
    parser.add_argument(
        "--reuse-sample-jsonl",
        default=None,
        help="Existing generations.jsonl to reuse as the fixed paper set and row order.",
    )
    parser.add_argument(
        "--paper-ids-json",
        default=None,
        help="Optional JSON file containing a paper_id list to filter the reused or sampled set.",
    )
    parser.add_argument(
        "--left-label",
        default="left",
        help="User-facing label for the left variant.",
    )
    parser.add_argument(
        "--right-label",
        default="right",
        help="User-facing label for the right variant.",
    )
    parser.add_argument(
        "--left-adapter-dir",
        default=None,
        help="Adapter directory for the left variant. Omit for the base model.",
    )
    parser.add_argument(
        "--right-adapter-dir",
        default=None,
        help="Adapter directory for the right variant. Omit for the base model.",
    )
    parser.add_argument(
        "--left-checkpoint",
        default=None,
        help="Optional checkpoint safetensors file for the left variant.",
    )
    parser.add_argument(
        "--right-checkpoint",
        default=None,
        help="Optional checkpoint safetensors file for the right variant.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help=argparse.SUPPRESS,
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
        "--comparison-root",
        default=DEFAULT_COMPARISON_ROOT,
        help="Default root for timestamped comparison artifact directories.",
    )
    parser.add_argument(
        "--review-views-root",
        default=DEFAULT_REVIEW_VIEWS_ROOT,
        help="Root directory for materialized review-only checkpoint views.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for comparison artifacts. Defaults under comparison-root.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of held-out papers to sample when reuse-sample-jsonl is omitted.",
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
        help="Deterministic sampling seed used when reuse-sample-jsonl is omitted.",
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
    parser.add_argument(
        "--blinded",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether review.md should be blinded as Left vs Right. Defaults to true.",
    )

    args = parser.parse_args()

    # Preserve legacy behavior: `--adapter-path` meant base vs tuned on the right.
    if args.adapter_path and not args.right_adapter_dir:
        args.right_adapter_dir = args.adapter_path
        if args.left_label == "left":
            args.left_label = "base"
        if args.right_label == "right":
            args.right_label = "tuned"
        if args.reuse_sample_jsonl is None:
            args.reuse_sample_jsonl = DEFAULT_REUSE_SAMPLE_JSONL

    return args


def slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    return normalized.strip("_") or "item"


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


def load_paper_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict) and isinstance(payload.get("paper_ids"), list):
        return [str(item) for item in payload["paper_ids"]]
    raise ValueError(f"Expected a JSON list or {{\"paper_ids\": [...]}} in {path}")


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
                "paper_id": str(row["paper_id"]),
                "title": str(row["title"]),
                "abstract": str(row["abstract"]),
                "primary_category": str(row["primary_category"]),
                "categories": row.get("categories", []),
                "created_date": row.get("created_date"),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
    return prepared_rows


def load_reused_rows(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    required = {"paper_id", "title", "abstract", "primary_category"}
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(
                f"Reused sample row for {row.get('paper_id', '<unknown>')} is missing "
                f"required keys: {', '.join(missing)}"
            )
    return rows


def select_rows_in_order(
    rows: list[dict[str, Any]],
    requested_paper_ids: list[str] | None,
) -> list[dict[str, Any]]:
    if requested_paper_ids is None:
        return rows

    requested = {str(paper_id) for paper_id in requested_paper_ids}
    selected = [row for row in rows if str(row["paper_id"]) in requested]
    found_ids = {str(row["paper_id"]) for row in selected}
    missing_ids = [paper_id for paper_id in requested_paper_ids if paper_id not in found_ids]
    if missing_ids:
        raise ValueError(
            "Some requested paper_ids were not found in the source rows: "
            + ", ".join(missing_ids)
        )
    return selected


def sample_rows_from_corpus(
    *,
    release_dir: Path,
    corpus_path: Path,
    sample_size: int,
    seed: int,
    requested_paper_ids: list[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded_ids, exclusion_source = collect_excluded_ids(release_dir)
    corpus_rows = load_jsonl(corpus_path)
    eligible_rows = [
        row for row in corpus_rows if str(row["paper_id"]) not in excluded_ids
    ]

    if requested_paper_ids is not None:
        selected_rows = select_rows_in_order(eligible_rows, requested_paper_ids)
    else:
        if sample_size > len(eligible_rows):
            raise ValueError(
                f"Requested {sample_size} samples but only {len(eligible_rows)} "
                "rows remain after exclusions."
            )
        random_generator = random.Random(seed)
        selected_rows = random_generator.sample(eligible_rows, sample_size)

    sample_meta = {
        "mode": "fresh_sample",
        "release_dir": str(release_dir),
        "corpus_jsonl": str(corpus_path),
        "excluded_paper_count": len(excluded_ids),
        "eligible_paper_count": len(eligible_rows),
        "exclusion_source": exclusion_source,
        "seed": seed,
    }
    return selected_rows, sample_meta


def load_source_rows(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested_paper_ids: list[str] | None = None
    if args.paper_ids_json:
        requested_paper_ids = load_paper_ids(Path(args.paper_ids_json))

    if args.reuse_sample_jsonl:
        reuse_path = Path(args.reuse_sample_jsonl)
        rows = load_reused_rows(reuse_path)
        selected_rows = select_rows_in_order(rows, requested_paper_ids)
        sample_meta = {
            "mode": "reused_sample",
            "reuse_sample_jsonl": str(reuse_path),
            "paper_ids_json": str(args.paper_ids_json) if args.paper_ids_json else None,
            "source_row_count": len(rows),
            "selected_row_count": len(selected_rows),
        }
        return selected_rows, sample_meta

    return sample_rows_from_corpus(
        release_dir=Path(args.release_dir),
        corpus_path=Path(args.corpus_jsonl),
        sample_size=args.sample_size,
        seed=args.seed,
        requested_paper_ids=requested_paper_ids,
    )


def load_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing adapter config at {adapter_config_path}")
    return json.loads(adapter_config_path.read_text(encoding="utf-8"))


def materialize_checkpoint_view(
    *,
    label: str,
    adapter_dir: Path,
    checkpoint_path: Path,
    review_views_root: Path,
) -> Path:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint weights at {checkpoint_path}")

    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing adapter config at {adapter_config_path}")

    try:
        adapter_slug = slugify(str(adapter_dir.resolve().relative_to(Path.cwd())))
    except ValueError:
        adapter_slug = slugify(str(adapter_dir.resolve()))
    checkpoint_slug = slugify(checkpoint_path.stem)
    label_slug = slugify(label)
    view_dir = review_views_root / adapter_slug / f"{checkpoint_slug}_{label_slug}"
    view_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(adapter_config_path, view_dir / "adapter_config.json")

    destination = view_dir / "adapters.safetensors"
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(checkpoint_path.resolve())
    except OSError:
        shutil.copy2(checkpoint_path, destination)

    return view_dir


def resolve_variant(
    *,
    label: str,
    adapter_dir_value: str | None,
    checkpoint_value: str | None,
    review_views_root: Path,
) -> VariantSpec:
    if adapter_dir_value is None:
        if checkpoint_value is not None:
            raise ValueError(
                f"{label} cannot use a checkpoint file without an adapter directory."
            )
        return VariantSpec(
            label=label,
            kind="base",
            adapter_dir=None,
            checkpoint_path=None,
            adapter_load_path=None,
            model_repo_id=DEFAULT_MODEL_REPO_ID,
        )

    adapter_dir = Path(adapter_dir_value)
    adapter_config = load_adapter_config(adapter_dir)
    model_repo_id = str(adapter_config.get("model", DEFAULT_MODEL_REPO_ID))

    if checkpoint_value is None:
        return VariantSpec(
            label=label,
            kind="final_adapter",
            adapter_dir=str(adapter_dir),
            checkpoint_path=None,
            adapter_load_path=str(adapter_dir),
            model_repo_id=model_repo_id,
        )

    checkpoint_path = Path(checkpoint_value)
    adapter_view_path = materialize_checkpoint_view(
        label=label,
        adapter_dir=adapter_dir,
        checkpoint_path=checkpoint_path,
        review_views_root=review_views_root,
    )
    return VariantSpec(
        label=label,
        kind="checkpoint_view",
        adapter_dir=str(adapter_dir),
        checkpoint_path=str(checkpoint_path),
        adapter_load_path=str(adapter_view_path),
        model_repo_id=model_repo_id,
    )


def resolve_model_path(args: argparse.Namespace, variants: list[VariantSpec]) -> str:
    if args.model_path:
        return args.model_path

    model_repo_ids = {variant.model_repo_id for variant in variants}
    if len(model_repo_ids) > 1:
        raise ValueError(
            "Variants point at different base model repo ids. Pass --model-path explicitly."
        )

    repo_id = next(iter(model_repo_ids), DEFAULT_MODEL_REPO_ID)
    hf_home = Path(args.hf_home)
    cache_dir = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}"
    ref_path = cache_dir / "refs" / "main"
    if ref_path.exists():
        snapshot = ref_path.read_text(encoding="utf-8").strip()
        snapshot_path = cache_dir / "snapshots" / snapshot
        if snapshot_path.exists():
            return str(snapshot_path)
    return repo_id


def generate_outputs(
    prepared_rows: list[dict[str, Any]],
    *,
    model_path: str,
    variant: VariantSpec,
    batch_size: int,
    max_tokens: int,
    temp: float,
    top_p: float,
) -> list[dict[str, str]]:
    print(f"[sample:{variant.label}] loading model from {model_path}")
    if variant.adapter_load_path is None:
        print(f"[sample:{variant.label}] running without adapters")
    else:
        print(
            f"[sample:{variant.label}] loading adapters from {variant.adapter_load_path}"
        )

    load_kwargs: dict[str, Any] = {
        "tokenizer_config": {"trust_remote_code": True},
    }
    if variant.adapter_load_path is not None:
        load_kwargs["adapter_path"] = variant.adapter_load_path
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
                f"[sample:{variant.label}] generated {len(generated_outputs)}/"
                f"{len(prepared_rows)} summaries..."
            )
    finally:
        del model
        del tokenizer
        del sampler
        gc.collect()

    return generated_outputs


def build_markdown_report(
    output_path: Path,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    left_heading: str,
    right_heading: str,
    blinded: bool,
) -> None:
    lines = [
        "# Holdout Generation Review",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Sample size: `{manifest['sample_size']}`",
        f"- Input mode: `{manifest['input_mode']}`",
        f"- Prompt version: `{manifest['prompt_version']}`",
        f"- Matching outputs: `{manifest['matching_summary_count']}/{manifest['sample_size']}`",
    ]

    sample_source = manifest["sample_source"]
    if sample_source["mode"] == "reused_sample":
        lines.append(f"- Sample source: reused `{sample_source['reuse_sample_jsonl']}`")
    else:
        lines.append(f"- Sample source: `{sample_source['corpus_jsonl']}`")
        lines.append(f"- Exclusion source: `{sample_source['exclusion_source']}`")
        lines.append(f"- Seed: `{sample_source['seed']}`")

    if not blinded:
        lines.extend(
            [
                f"- Base model path: `{manifest['base_model_path']}`",
                f"- Left label: `{manifest['left_variant']['label']}`",
                f"- Right label: `{manifest['right_variant']['label']}`",
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
                f"**Category**: `{row['primary_category']}`",
                "",
                f"**Created**: `{row.get('created_date', '')}`",
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
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_blinded_rows(generated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blinded_rows: list[dict[str, Any]] = []
    for row in generated_rows:
        blinded_rows.append(
            {
                "paper_id": row["paper_id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "primary_category": row["primary_category"],
                "categories": row.get("categories", []),
                "created_date": row.get("created_date"),
                "prompt_version": row["prompt_version"],
                "input_mode": row["input_mode"],
                "user_prompt": row["user_prompt"],
                "left_summary": row["left_summary"],
                "right_summary": row["right_summary"],
                "outputs_match": row["outputs_match"],
            }
        )
    return blinded_rows


def build_generated_rows(
    sampled_rows: list[dict[str, Any]],
    prepared_rows: list[dict[str, Any]],
    left_outputs: list[dict[str, str]],
    right_outputs: list[dict[str, str]],
    *,
    prompt_version: str,
    input_mode: str,
    left_label: str,
    right_label: str,
) -> list[dict[str, Any]]:
    generated_rows: list[dict[str, Any]] = []
    for row, prepared_row, left_output, right_output in zip(
        sampled_rows,
        prepared_rows,
        left_outputs,
        right_outputs,
        strict=True,
    ):
        generated_rows.append(
            {
                "paper_id": str(row["paper_id"]),
                "title": str(row["title"]),
                "abstract": str(row["abstract"]),
                "primary_category": str(row["primary_category"]),
                "categories": row.get("categories", []),
                "created_date": row.get("created_date"),
                "prompt_version": prompt_version,
                "input_mode": input_mode,
                "user_prompt": prepared_row["user_prompt"],
                "left_label": left_label,
                "right_label": right_label,
                "left_summary_raw": left_output["raw"],
                "left_summary": left_output["normalized"],
                "right_summary_raw": right_output["raw"],
                "right_summary": right_output["normalized"],
                "outputs_match": left_output["normalized"] == right_output["normalized"],
            }
        )
    return generated_rows


def default_output_dir(
    comparison_root: Path,
    *,
    left_label: str,
    right_label: str,
    sample_size: int,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    directory_name = (
        f"{slugify(left_label)}_vs_{slugify(right_label)}_holdout_{sample_size}_{timestamp}"
    )
    return comparison_root / directory_name


def main() -> None:
    args = parse_args()

    prompt_spec_path = Path(args.prompt_spec)
    prompt_spec = load_prompt_spec(prompt_spec_path)

    sampled_rows, sample_meta = load_source_rows(args)
    prepared_rows = prepare_prompt_rows(sampled_rows, prompt_spec, args.input_mode)

    review_views_root = Path(args.review_views_root)
    left_variant = resolve_variant(
        label=args.left_label,
        adapter_dir_value=args.left_adapter_dir,
        checkpoint_value=args.left_checkpoint,
        review_views_root=review_views_root,
    )
    right_variant = resolve_variant(
        label=args.right_label,
        adapter_dir_value=args.right_adapter_dir,
        checkpoint_value=args.right_checkpoint,
        review_views_root=review_views_root,
    )

    model_path = resolve_model_path(args, [left_variant, right_variant])

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = default_output_dir(
            Path(args.comparison_root),
            left_label=left_variant.label,
            right_label=right_variant.label,
            sample_size=len(sampled_rows),
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    left_outputs = generate_outputs(
        prepared_rows,
        model_path=model_path,
        variant=left_variant,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
    )
    right_outputs = generate_outputs(
        prepared_rows,
        model_path=model_path,
        variant=right_variant,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
    )

    generated_rows = build_generated_rows(
        sampled_rows,
        prepared_rows,
        left_outputs,
        right_outputs,
        prompt_version=prompt_spec["prompt_version"],
        input_mode=args.input_mode,
        left_label=left_variant.label,
        right_label=right_variant.label,
    )
    blinded_rows = build_blinded_rows(generated_rows)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_spec": str(prompt_spec_path),
        "prompt_version": prompt_spec["prompt_version"],
        "base_model_path": model_path,
        "sample_size": len(sampled_rows),
        "batch_size": args.batch_size,
        "input_mode": args.input_mode,
        "max_tokens": args.max_tokens,
        "temp": args.temp,
        "top_p": args.top_p,
        "blinded_review_default": args.blinded,
        "sample_source": sample_meta,
        "left_variant": asdict(left_variant),
        "right_variant": asdict(right_variant),
        "matching_summary_count": sum(
            1 for row in generated_rows if row["outputs_match"]
        ),
        "paper_ids": [row["paper_id"] for row in generated_rows],
        "outputs": {
            "generations_jsonl": "generations.jsonl",
            "review_rows_blinded_jsonl": "review_rows_blinded.jsonl",
            "review_markdown": "review.md",
            "review_unblinded_markdown": "review_unblinded.md",
        },
    }

    write_jsonl(output_dir / "generations.jsonl", generated_rows)
    write_jsonl(output_dir / "review_rows_blinded.jsonl", blinded_rows)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    build_markdown_report(
        output_dir / "review.md",
        generated_rows,
        manifest,
        left_heading="Left Summary" if args.blinded else left_variant.label,
        right_heading="Right Summary" if args.blinded else right_variant.label,
        blinded=args.blinded,
    )
    build_markdown_report(
        output_dir / "review_unblinded.md",
        generated_rows,
        manifest,
        left_heading=left_variant.label,
        right_heading=right_variant.label,
        blinded=False,
    )

    print(f"[sample] wrote comparison artifacts to {output_dir}")


if __name__ == "__main__":
    main()
