#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

DEFAULT_SOURCE = (
    "hf_dataset/releases/recent_3y_balanced_5000_pilot_v2"
)
DEFAULT_PROMPT_SPEC = (
    "prompts/mlx_lora/gemma_270m_it_abstract_to_glanceable_summary_v1.json"
)
DEFAULT_VALID_RATIO = 0.05
DEFAULT_TEST_RATIO = 0.05
DEFAULT_SEED = 20260417

REQUIRED_SOURCE_FIELDS = (
    "paper_id",
    "title",
    "abstract",
    "summary",
    "primary_category",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert reviewed summary rows into an MLX LoRA dataset directory with "
            "train.jsonl, valid.jsonl, and test.jsonl."
        )
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help=(
            "Path to a training_rows.jsonl file or a release directory that contains it."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to write MLX-ready JSONL files into. Defaults to a versioned "
            "subdirectory inside the source release."
        ),
    )
    parser.add_argument(
        "--prompt-spec",
        default=DEFAULT_PROMPT_SPEC,
        help="Path to the versioned prompt spec JSON.",
    )
    parser.add_argument(
        "--dataset-format",
        choices=("chat", "completions"),
        default="chat",
        help="MLX dataset format to emit. Default is chat for instruction-tuned Gemma.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("abstract_only", "title_abstract"),
        default="abstract_only",
        help="Whether the student sees only the abstract or both title and abstract.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=DEFAULT_VALID_RATIO,
        help="Fraction of rows to reserve for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Fraction of rows to reserve for test evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic shuffle seed.",
    )
    return parser.parse_args()


def resolve_source_paths(source_arg: str) -> tuple[Path, Path]:
    source_path = Path(source_arg)
    if source_path.is_dir():
        training_rows_path = source_path / "training_rows.jsonl"
        release_dir = source_path
    else:
        training_rows_path = source_path
        release_dir = source_path.parent
    if not training_rows_path.exists():
        raise FileNotFoundError(
            f"Could not find training_rows.jsonl at {training_rows_path}"
        )
    return training_rows_path, release_dir


def resolve_output_dir(
    output_dir_arg: str | None,
    release_dir: Path,
    dataset_format: str,
    input_mode: str,
    prompt_version: str,
) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg)
    return (
        release_dir
        / "mlx_lora"
        / f"{input_mode}_{dataset_format}_{prompt_version}"
    )


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


def load_prompt_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = ("prompt_version", "system_prompt", "user_prompt_templates")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(
            f"Prompt spec {path} is missing required keys: {', '.join(missing)}"
        )
    user_templates = spec["user_prompt_templates"]
    if not isinstance(user_templates, dict):
        raise ValueError(f"Prompt spec {path} has invalid user_prompt_templates")
    for mode in ("abstract_only", "title_abstract"):
        if mode not in user_templates:
            raise ValueError(f"Prompt spec {path} is missing template for {mode}")
    return spec


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def word_count(value: str) -> int:
    return len(normalize_whitespace(value).split())


def validate_rows(rows: list[dict[str, Any]]) -> None:
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        missing = [field for field in REQUIRED_SOURCE_FIELDS if not row.get(field)]
        if missing:
            raise ValueError(
                f"Row {index} is missing required fields: {', '.join(missing)}"
            )
        paper_id = str(row["paper_id"])
        if paper_id in seen_ids:
            raise ValueError(f"Duplicate paper_id found in source rows: {paper_id}")
        seen_ids.add(paper_id)


def build_user_prompt(
    row: dict[str, Any],
    prompt_spec: dict[str, Any],
    input_mode: str,
) -> str:
    template = prompt_spec["user_prompt_templates"][input_mode]
    return template.format(
        title=str(row["title"]).strip(),
        abstract=str(row["abstract"]).strip(),
    )


def build_chat_example(
    row: dict[str, Any],
    prompt_spec: dict[str, Any],
    input_mode: str,
) -> dict[str, Any]:
    user_prompt = build_user_prompt(row, prompt_spec, input_mode)
    return {
        "messages": [
            {
                "role": "system",
                "content": prompt_spec["system_prompt"],
            },
            {
                "role": "user",
                "content": user_prompt,
            },
            {
                "role": "assistant",
                "content": str(row["summary"]).strip(),
            },
        ],
        "paper_id": row["paper_id"],
        "primary_category": row["primary_category"],
        "created_date": row.get("created_date"),
        "source_split": row.get("source_split"),
        "teacher_model": row.get("teacher_model"),
        "generator_prompt_version": row.get("generator_prompt_version"),
        "review_prompt_version": row.get("review_prompt_version"),
        "prompt_version": prompt_spec["prompt_version"],
        "input_mode": input_mode,
    }


def build_completion_example(
    row: dict[str, Any],
    prompt_spec: dict[str, Any],
    input_mode: str,
) -> dict[str, Any]:
    user_prompt = build_user_prompt(row, prompt_spec, input_mode)
    prompt = f"{prompt_spec['system_prompt']}\n\n{user_prompt}"
    return {
        "prompt": prompt,
        "completion": str(row["summary"]).strip(),
        "paper_id": row["paper_id"],
        "primary_category": row["primary_category"],
        "created_date": row.get("created_date"),
        "source_split": row.get("source_split"),
        "teacher_model": row.get("teacher_model"),
        "generator_prompt_version": row.get("generator_prompt_version"),
        "review_prompt_version": row.get("review_prompt_version"),
        "prompt_version": prompt_spec["prompt_version"],
        "input_mode": input_mode,
    }


def build_example(
    row: dict[str, Any],
    prompt_spec: dict[str, Any],
    dataset_format: str,
    input_mode: str,
) -> dict[str, Any]:
    if dataset_format == "chat":
        return build_chat_example(row, prompt_spec, input_mode)
    return build_completion_example(row, prompt_spec, input_mode)


def compute_split_counts(
    total_rows: int,
    valid_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    if valid_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative.")
    if valid_ratio + test_ratio >= 1:
        raise ValueError("valid_ratio + test_ratio must be less than 1.")
    valid_count = round(total_rows * valid_ratio)
    test_count = round(total_rows * test_ratio)
    train_count = total_rows - valid_count - test_count
    if train_count <= 0:
        raise ValueError("Split ratios leave no rows for train.jsonl.")
    return train_count, valid_count, test_count


def split_rows(
    rows: list[dict[str, Any]],
    seed: int,
    valid_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    train_count, valid_count, test_count = compute_split_counts(
        total_rows=len(shuffled),
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )
    train_rows = shuffled[:train_count]
    valid_rows = shuffled[train_count : train_count + valid_count]
    test_rows = shuffled[train_count + valid_count : train_count + valid_count + test_count]
    assert len(train_rows) + len(valid_rows) + len(test_rows) == len(shuffled)
    return train_rows, valid_rows, test_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


def row_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary_word_counts = [word_count(str(row["summary"])) for row in rows]
    abstract_word_counts = [word_count(str(row["abstract"])) for row in rows]
    return {
        "row_count": len(rows),
        "summary_word_count": {
            "min": min(summary_word_counts),
            "max": max(summary_word_counts),
            "avg": round(mean(summary_word_counts), 2),
        },
        "abstract_word_count": {
            "min": min(abstract_word_counts),
            "max": max(abstract_word_counts),
            "avg": round(mean(abstract_word_counts), 2),
        },
    }


def build_manifest(
    *,
    source_path: Path,
    output_dir: Path,
    prompt_spec_path: Path,
    prompt_spec: dict[str, Any],
    dataset_format: str,
    input_mode: str,
    seed: int,
    valid_ratio: float,
    test_ratio: float,
    source_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_training_rows": str(source_path),
        "output_dir": str(output_dir),
        "prompt_spec_path": str(prompt_spec_path),
        "prompt_version": prompt_spec["prompt_version"],
        "dataset_format": dataset_format,
        "input_mode": input_mode,
        "seed": seed,
        "valid_ratio": valid_ratio,
        "test_ratio": test_ratio,
        "source_stats": row_stats(source_rows),
        "split_counts": {
            "train": len(train_rows),
            "valid": len(valid_rows),
            "test": len(test_rows),
        },
        "mlx_lm_notes": {
            "train_command_hint": "mlx_lm.lora --train --data <this directory> --mask-prompt",
            "eval_command_hint": "mlx_lm.lora --test --data <this directory>",
        },
    }


def main() -> None:
    args = parse_args()
    source_path, release_dir = resolve_source_paths(args.source)
    prompt_spec_path = Path(args.prompt_spec)
    prompt_spec = load_prompt_spec(prompt_spec_path)
    output_dir = resolve_output_dir(
        args.output_dir,
        release_dir,
        args.dataset_format,
        args.input_mode,
        str(prompt_spec["prompt_version"]),
    )

    source_rows = load_jsonl(source_path)
    validate_rows(source_rows)
    train_source_rows, valid_source_rows, test_source_rows = split_rows(
        source_rows,
        seed=args.seed,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples = [
        build_example(row, prompt_spec, args.dataset_format, args.input_mode)
        for row in train_source_rows
    ]
    valid_examples = [
        build_example(row, prompt_spec, args.dataset_format, args.input_mode)
        for row in valid_source_rows
    ]
    test_examples = [
        build_example(row, prompt_spec, args.dataset_format, args.input_mode)
        for row in test_source_rows
    ]

    write_jsonl(output_dir / "train.jsonl", train_examples)
    write_jsonl(output_dir / "valid.jsonl", valid_examples)
    write_jsonl(output_dir / "test.jsonl", test_examples)

    manifest = build_manifest(
        source_path=source_path,
        output_dir=output_dir,
        prompt_spec_path=prompt_spec_path,
        prompt_spec=prompt_spec,
        dataset_format=args.dataset_format,
        input_mode=args.input_mode,
        seed=args.seed,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        source_rows=source_rows,
        train_rows=train_source_rows,
        valid_rows=valid_source_rows,
        test_rows=test_source_rows,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(f"Prepared MLX dataset in {output_dir}")
    print(
        "Split counts: "
        f"train={len(train_examples)} "
        f"valid={len(valid_examples)} "
        f"test={len(test_examples)}"
    )
    print(f"Prompt version: {prompt_spec['prompt_version']}")
    print(f"Input mode: {args.input_mode}")
    print(f"Dataset format: {args.dataset_format}")


if __name__ == "__main__":
    main()
