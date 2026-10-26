#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sample_mlx_holdout_generations import (
    VariantSpec,
    generate_outputs,
    load_prompt_spec,
    load_reused_rows,
    materialize_checkpoint_view,
    prepare_prompt_rows,
)


DEFAULT_RUN_ROOT = Path(
    "hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_q8_fusion_aware_v1"
)
DEFAULT_RUNTIME_PYTHON = Path("hf_bucket/experiments/mlx_lora/runtime/py312/bin/python")
DEFAULT_SOURCE_MODEL = "ibm-granite/granite-4.0-350m"
DEFAULT_PROMPT_SPEC = Path("prompts/mlx_lora/gemma_270m_it_abstract_to_feed_blurb_v5.json")
DEFAULT_DATA_DIR = Path(
    "hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/mlx_lora/title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5"
)
DEFAULT_TRAINING_ROWS = Path(
    "hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_1000_milestone/training_rows.jsonl"
)
DEFAULT_BASELINE_GENERATIONS = Path(
    "hf_bucket/experiments/mlx_lora/granite_4_0_350m_natural_language_994_title_abstract_feed_blurb_v5_lora_lr5e6_v1/review_samples/blind100_gemma425_vs_granite425_title_abstract_feed_blurb_v5/generations.jsonl"
)
DEFAULT_PRIOR_Q8_SWIFT = Path(
    "hf_bucket/experiments/mlx_lora/granite_4_0_350m_v5_ckpt425_mlx_swift_q8_export_v1/generations/granite425_q8_fused_swift_blind100.jsonl"
)
DEFAULT_SWIFT_DERIVED_DATA = Path("/private/tmp/mlx-summary-probe-xcodebuild")
DEFAULT_SWIFT_BINARY = (
    DEFAULT_SWIFT_DERIVED_DATA / "Build/Products/Release/mlx-summary-probe"
)
STICKY_IDS = [
    "2405.03367",
    "2311.00575",
    "2308.04773",
    "2301.02200",
    "2007.03757",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("[granite-q8-ladder] " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def ensure_clean_q8_base(args: argparse.Namespace) -> None:
    base_dir = args.run_root / "base_q8_g64"
    if base_dir.exists() and any(base_dir.iterdir()) and not args.overwrite:
        print(f"[granite-q8-ladder] keeping existing clean q8 base: {base_dir}")
        return
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.runtime_python),
        "-m",
        "mlx_lm",
        "convert",
        "--hf-path",
        args.source_model,
        "--mlx-path",
        str(base_dir),
        "--quantize",
        "--q-bits",
        "8",
        "--q-group-size",
        "64",
    ]
    run(command)


def validate(args: argparse.Namespace) -> None:
    data_dir = args.data_dir
    counts = {
        "train": line_count(data_dir / "train.jsonl"),
        "valid": line_count(data_dir / "valid.jsonl"),
        "test": line_count(data_dir / "test.jsonl"),
        "training_rows": line_count(args.training_rows),
    }
    prompt_spec = load_prompt_spec(args.prompt_spec)
    problems: list[str] = []
    if counts["training_rows"] != 994:
        problems.append(f"training_rows.jsonl has {counts['training_rows']} rows, expected 994")
    if counts["train"] + counts["valid"] + counts["test"] != 994:
        problems.append(f"MLX split total is {counts['train'] + counts['valid'] + counts['test']}, expected 994")
    if prompt_spec.get("prompt_version") != "gemma_270m_it_abstract_to_feed_blurb_v5":
        problems.append(f"unexpected prompt version: {prompt_spec.get('prompt_version')}")
    if problems:
        raise SystemExit("\n".join(problems))
    args.run_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "counts": counts,
        "prompt_spec": str(args.prompt_spec),
        "prompt_version": prompt_spec["prompt_version"],
        "clean_q8_base": str(args.run_root / "base_q8_g64"),
        "source_model": args.source_model,
    }
    (args.run_root / "validation_manifest.json").write_text(
        json_dumps(manifest), encoding="utf-8"
    )
    print(f"[granite-q8-ladder] validation OK: {counts}")


def build_swift_probe(args: argparse.Namespace) -> None:
    if args.swift_binary.exists() and not args.rebuild_swift:
        return
    run(
        [
            "xcodebuild",
            "-quiet",
            "-scheme",
            "MLXSummaryProbe",
            "-configuration",
            "Release",
            "-destination",
            "platform=macOS,arch=arm64",
            "-derivedDataPath",
            str(args.swift_derived_data),
            "-skipMacroValidation",
            "build",
        ],
        cwd=Path("tools/mlx-summary-probe"),
    )


def make_variant(label: str, model_path: Path, adapter_path: Path | None = None) -> VariantSpec:
    return VariantSpec(
        label=label,
        kind="checkpoint_view" if adapter_path else "standalone_model",
        model_path=str(model_path),
        adapter_dir=None,
        checkpoint_path=None,
        adapter_load_path=str(adapter_path) if adapter_path else None,
        model_repo_id=DEFAULT_SOURCE_MODEL,
    )


def generate_variant(
    *,
    label: str,
    model_path: Path,
    adapter_path: Path | None,
    source_rows: list[dict[str, Any]],
    prompt_spec: dict[str, Any],
    output_path: Path,
    batch_size: int,
    max_tokens: int,
) -> None:
    prepared_rows = prepare_prompt_rows(source_rows, prompt_spec, "title_abstract")
    variant = make_variant(label, model_path, adapter_path)
    outputs = generate_outputs(
        prepared_rows,
        model_path=str(model_path),
        variant=variant,
        batch_size=batch_size,
        max_tokens=max_tokens,
        temp=0.0,
        top_p=0.0,
    )
    rows: list[dict[str, Any]] = []
    for source, output in zip(source_rows, outputs, strict=True):
        rows.append(
            {
                "ok": True,
                "paper_id": str(source["paper_id"]),
                "title": str(source["title"]),
                "primary_category": source.get("primary_category"),
                "normalized_output": output["normalized"],
                "raw_output": output["raw"],
            }
        )
    write_jsonl(output_path, rows)


def fuse_q8(
    *,
    runtime_python: Path,
    model_path: Path,
    adapter_path: Path,
    output_dir: Path,
    force: bool,
) -> None:
    if output_dir.exists():
        if not force:
            print(f"[granite-q8-ladder] keeping existing fused model: {output_dir}")
            return
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(runtime_python),
            "-m",
            "mlx_lm",
            "fuse",
            "--model",
            str(model_path),
            "--adapter-path",
            str(adapter_path),
            "--save-path",
            str(output_dir),
        ]
    )


def run_swift_eval(
    *,
    args: argparse.Namespace,
    model_dir: Path,
    output_path: Path,
) -> None:
    build_swift_probe(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(args.swift_binary),
            "--model-dir",
            str(model_dir),
            "--prompt-spec",
            str(args.prompt_spec),
            "--input-mode",
            "title_abstract",
            "--input-jsonl",
            str(args.baseline_generations),
            "--output-jsonl",
            str(output_path),
            "--json",
        ]
    )


def summarize(
    *,
    args: argparse.Namespace,
    label: str,
    variant_paths: OrderedDict[str, Path],
    output_dir: Path,
) -> None:
    command = [
        sys.executable,
        "scripts/summarize_runtime_generation_mechanics.py",
        "--source-generations",
        str(args.baseline_generations),
        "--baseline-side",
        args.baseline_side,
        "--baseline-label",
        args.baseline_label,
        "--output-dir",
        str(output_dir),
        "--examples-md",
        str(output_dir / "changed_examples.md"),
    ]
    for variant_label, path in variant_paths.items():
        command.extend(["--variant", f"{variant_label}={path}"])
    run(command)
    write_exact_match_report(
        baseline_generations=args.baseline_generations,
        baseline_side=args.baseline_side,
        variant_paths=variant_paths,
        output_path=output_dir / "exact_match_ladder.json",
        markdown_path=output_dir / "EXACT_MATCH_LADDER.md",
    )
    write_sticky_read(
        baseline_generations=args.baseline_generations,
        baseline_side=args.baseline_side,
        variant_paths=variant_paths,
        output_path=output_dir / "STICKY_READ.md",
    )
    print(f"[granite-q8-ladder] wrote mechanics for {label}: {output_dir}")


def output_summary(row: dict[str, Any]) -> str:
    return str(
        row.get("normalized_output")
        or row.get("normalizedOutput")
        or row.get("summary")
        or row.get("raw_output")
        or row.get("rawOutput")
        or ""
    )


def parse_label_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("Variant label cannot be empty")
    return label.strip(), Path(path)


def write_exact_match_report(
    *,
    baseline_generations: Path,
    baseline_side: str,
    variant_paths: OrderedDict[str, Path],
    output_path: Path,
    markdown_path: Path,
) -> None:
    baseline_rows = read_jsonl(baseline_generations)
    summaries: OrderedDict[str, dict[str, str]] = OrderedDict()
    baseline_label = str(baseline_rows[0][f"{baseline_side}_label"])
    summaries[baseline_label] = {
        str(row["paper_id"]): str(row[f"{baseline_side}_summary"]) for row in baseline_rows
    }
    for label, path in variant_paths.items():
        summaries[label] = {str(row["paper_id"]): output_summary(row) for row in read_jsonl(path)}

    labels = list(summaries)
    pairs: list[dict[str, Any]] = []
    ids = list(summaries[baseline_label])
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            exact = sum(1 for paper_id in ids if summaries[left].get(paper_id) == summaries[right].get(paper_id))
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "rows": len(ids),
                    "exact": exact,
                    "changed": len(ids) - exact,
                }
            )
    output_path.write_text(json_dumps({"pairs": pairs}), encoding="utf-8")
    lines = [
        "# Exact Match Ladder",
        "",
        "| left | right | rows | exact | changed |",
        "|---|---|---:|---:|---:|",
    ]
    for pair in pairs:
        lines.append(
            f"| {pair['left']} | {pair['right']} | {pair['rows']} | {pair['exact']} | {pair['changed']} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sticky_read(
    *,
    baseline_generations: Path,
    baseline_side: str,
    variant_paths: OrderedDict[str, Path],
    output_path: Path,
) -> None:
    baseline_rows = {str(row["paper_id"]): row for row in read_jsonl(baseline_generations)}
    variants = {
        label: {str(row["paper_id"]): output_summary(row) for row in read_jsonl(path)}
        for label, path in variant_paths.items()
    }
    baseline_label = str(next(iter(baseline_rows.values()))[f"{baseline_side}_label"])
    lines = ["# Sticky Row Read", ""]
    for paper_id in STICKY_IDS:
        row = baseline_rows.get(paper_id)
        if row is None:
            continue
        lines.extend(
            [
                f"## {paper_id}",
                "",
                f"Title: {row['title']}",
                "",
                f"- {baseline_label}: {row[f'{baseline_side}_summary']}",
            ]
        )
        for label, by_id in variants.items():
            lines.append(f"- {label}: {by_id.get(paper_id, '')}")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def command_eval_checkpoint(args: argparse.Namespace) -> None:
    prompt_spec = load_prompt_spec(args.prompt_spec)
    source_rows = load_reused_rows(args.baseline_generations)
    checkpoint_path = args.adapter_dir / f"{args.checkpoint_step:07d}_adapters.safetensors"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    eval_dir = args.run_root / "eval" / args.stage / f"checkpoint_{args.checkpoint_step:04d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    adapter_view = materialize_checkpoint_view(
        label=f"{args.stage}_ckpt{args.checkpoint_step}",
        adapter_dir=args.adapter_dir,
        checkpoint_path=checkpoint_path,
        review_views_root=eval_dir / "adapter_views",
    )
    quant_label = args.quant_label
    sidecar_out = eval_dir / "generations" / f"{args.stage}_ckpt{args.checkpoint_step:04d}_{quant_label}_sidecar_python.jsonl"
    fused_dir = eval_dir / f"fused_{quant_label}_g64"
    fused_python_out = eval_dir / "generations" / f"{args.stage}_ckpt{args.checkpoint_step:04d}_{quant_label}_fused_python.jsonl"
    swift_out = eval_dir / "generations" / f"{args.stage}_ckpt{args.checkpoint_step:04d}_{quant_label}_fused_swift.jsonl"

    if args.force or not sidecar_out.exists():
        generate_variant(
            label=f"{args.stage}_ckpt{args.checkpoint_step}_{quant_label}_sidecar_python",
            model_path=args.model,
            adapter_path=adapter_view,
            source_rows=source_rows,
            prompt_spec=prompt_spec,
            output_path=sidecar_out,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
        )
    fuse_q8(
        runtime_python=args.runtime_python,
        model_path=args.model,
        adapter_path=adapter_view,
        output_dir=fused_dir,
        force=args.force,
    )
    if args.force or not fused_python_out.exists():
        generate_variant(
            label=f"{args.stage}_ckpt{args.checkpoint_step}_{quant_label}_fused_python",
            model_path=fused_dir,
            adapter_path=None,
            source_rows=source_rows,
            prompt_spec=prompt_spec,
            output_path=fused_python_out,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
        )
    if args.force or not swift_out.exists():
        run_swift_eval(args=args, model_dir=fused_dir, output_path=swift_out)

    variant_paths: OrderedDict[str, Path] = OrderedDict()
    if args.prior_q8_swift.exists():
        variant_paths[args.prior_q8_swift_label] = args.prior_q8_swift
    for extra_label, extra_path in args.extra_variant:
        if extra_path.exists():
            variant_paths[extra_label] = extra_path
    variant_paths[f"{args.stage}_ckpt{args.checkpoint_step}_{quant_label}_sidecar_python"] = sidecar_out
    variant_paths[f"{args.stage}_ckpt{args.checkpoint_step}_{quant_label}_fused_python"] = fused_python_out
    variant_paths[f"{args.stage}_ckpt{args.checkpoint_step}_{quant_label}_fused_swift"] = swift_out
    summarize(args=args, label=f"{args.stage}_ckpt{args.checkpoint_step}", variant_paths=variant_paths, output_dir=eval_dir / "reports")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "checkpoint_step": args.checkpoint_step,
        "model": str(args.model),
        "adapter_dir": str(args.adapter_dir),
        "checkpoint_path": str(checkpoint_path),
        "adapter_view": str(adapter_view),
        "fused_dir": str(fused_dir),
        "quant_label": quant_label,
        "outputs": {label: str(path) for label, path in variant_paths.items()},
        "reports": str(eval_dir / "reports"),
    }
    (eval_dir / "manifest.json").write_text(json_dumps(manifest), encoding="utf-8")
    if args.delete_fused_after_eval and fused_dir.exists():
        shutil.rmtree(fused_dir)
        print(f"[granite-q8-ladder] deleted reproducible fused model after eval: {fused_dir}")


def command_promote_fused(args: argparse.Namespace) -> None:
    source = args.source_fused
    destination = args.destination
    if not source.exists():
        raise FileNotFoundError(f"Missing source fused model: {source}")
    if destination.exists():
        if not args.force:
            raise FileExistsError(f"Destination exists: {destination}")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, symlinks=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_fused": str(source),
        "destination": str(destination),
    }
    (destination.parent / f"{destination.name}_manifest.json").write_text(
        json_dumps(manifest), encoding="utf-8"
    )
    print(f"[granite-q8-ladder] copied {source} -> {destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Granite q8 fusion-aware eval helpers.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--runtime-python", type=Path, default=DEFAULT_RUNTIME_PYTHON)
    sub = parser.add_subparsers(dest="command", required=True)

    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    validate_parser.add_argument("--training-rows", type=Path, default=DEFAULT_TRAINING_ROWS)
    validate_parser.add_argument("--prompt-spec", type=Path, default=DEFAULT_PROMPT_SPEC)
    validate_parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    validate_parser.add_argument("--ensure-clean-q8-base", action="store_true")
    validate_parser.add_argument("--overwrite", action="store_true")

    eval_parser = sub.add_parser("eval-checkpoint")
    eval_parser.add_argument("--stage", required=True)
    eval_parser.add_argument("--model", type=Path, required=True)
    eval_parser.add_argument("--adapter-dir", type=Path, required=True)
    eval_parser.add_argument("--checkpoint-step", type=int, required=True)
    eval_parser.add_argument("--baseline-generations", type=Path, default=DEFAULT_BASELINE_GENERATIONS)
    eval_parser.add_argument("--baseline-side", choices=("left", "right"), default="right")
    eval_parser.add_argument("--baseline-label", default="granite350m-v5-ckpt425-original-mlx")
    eval_parser.add_argument("--prior-q8-swift", type=Path, default=DEFAULT_PRIOR_Q8_SWIFT)
    eval_parser.add_argument("--prior-q8-swift-label", default="prior_granite425_q8_fused_swift")
    eval_parser.add_argument(
        "--extra-variant",
        type=parse_label_path,
        action="append",
        default=[],
        help="Additional compact comparison variant as LABEL=PATH.",
    )
    eval_parser.add_argument("--prompt-spec", type=Path, default=DEFAULT_PROMPT_SPEC)
    eval_parser.add_argument("--quant-label", default="q8")
    eval_parser.add_argument("--batch-size", type=int, default=8)
    eval_parser.add_argument("--max-tokens", type=int, default=36)
    eval_parser.add_argument("--swift-derived-data", type=Path, default=DEFAULT_SWIFT_DERIVED_DATA)
    eval_parser.add_argument("--swift-binary", type=Path, default=DEFAULT_SWIFT_BINARY)
    eval_parser.add_argument("--rebuild-swift", action="store_true")
    eval_parser.add_argument("--force", action="store_true")
    eval_parser.add_argument(
        "--delete-fused-after-eval",
        action="store_true",
        help="Remove the large fused model directory after JSONL outputs and reports are written.",
    )

    promote_parser = sub.add_parser("copy-fused")
    promote_parser.add_argument("--source-fused", type=Path, required=True)
    promote_parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_RUN_ROOT / "fused_round1_q8_g64",
    )
    promote_parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "validate":
        validate(args)
        if args.ensure_clean_q8_base:
            ensure_clean_q8_base(args)
    elif args.command == "eval-checkpoint":
        command_eval_checkpoint(args)
    elif args.command == "copy-fused":
        command_promote_fused(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
