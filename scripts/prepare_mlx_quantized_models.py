#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_HF_MODEL = (
    "hf_bucket/experiments/litert_exports/"
    "faith_overlay_1400_v2_final_600/merged_hf_model"
)
DEFAULT_OUTPUT_ROOT = "hf_bucket/experiments/mlx_lora/quantized_models"
DEFAULT_LABEL = "faith_overlay_1400_v2_final_600"
DEFAULT_RUNTIME_PYTHON = "hf_bucket/experiments/mlx_lora/runtime/py312/bin/python"
DEFAULT_Q_BITS = (8, 4)
DEFAULT_Q_GROUP_SIZE = 64


@dataclass(frozen=True)
class QuantizedModelSpec:
    bits: int
    group_size: int
    source_hf_model: str
    output_dir: str
    command: list[str]


def json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def build_output_dir(
    *,
    output_root: Path,
    label: str,
    bits: int,
    group_size: int,
) -> Path:
    return output_root / label / f"q{bits}_g{group_size}"


def build_convert_command(
    *,
    runtime_python: Path,
    source_hf_model: Path,
    output_dir: Path,
    bits: int,
    group_size: int,
    trust_remote_code: bool,
) -> list[str]:
    command = [
        str(runtime_python),
        "-m",
        "mlx_lm",
        "convert",
        "--hf-path",
        str(source_hf_model),
        "--mlx-path",
        str(output_dir),
        "--quantize",
        "--q-bits",
        str(bits),
        "--q-group-size",
        str(group_size),
    ]
    if trust_remote_code:
        command.append("--trust-remote-code")
    return command


def collect_file_sizes(path: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for item in sorted(path.rglob("*")):
        if item.is_file():
            sizes[str(item.relative_to(path))] = item.stat().st_size
    return sizes


def prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        return
    if any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite non-empty MLX model directory: {path}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(path)
    else:
        path.rmdir()
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_bits(values: list[int] | None) -> list[int]:
    bits = list(values or DEFAULT_Q_BITS)
    if not bits:
        raise ValueError("At least one --q-bits value is required.")
    invalid = [value for value in bits if value <= 0]
    if invalid:
        raise ValueError(f"Quantization bits must be positive: {invalid}")
    return bits


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare standalone MLX quantized exports of the promoted model."
    )
    parser.add_argument("--source-hf-model", default=DEFAULT_SOURCE_HF_MODEL)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--runtime-python", default=DEFAULT_RUNTIME_PYTHON)
    parser.add_argument(
        "--q-bits",
        type=int,
        action="append",
        default=None,
        help="Quantized bit widths to generate. Defaults to 8 and 4.",
    )
    parser.add_argument("--q-group-size", type=int, default=DEFAULT_Q_GROUP_SIZE)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the conversion commands and write no model files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source_hf_model = Path(args.source_hf_model)
    output_root = Path(args.output_root)
    runtime_python = Path(args.runtime_python)
    bits_to_generate = parse_bits(args.q_bits)

    if not source_hf_model.exists():
        raise FileNotFoundError(f"Missing source HF model: {source_hf_model}")
    if not runtime_python.exists():
        raise FileNotFoundError(f"Missing runtime Python: {runtime_python}")

    specs: list[QuantizedModelSpec] = []
    for bits in bits_to_generate:
        output_dir = build_output_dir(
            output_root=output_root,
            label=str(args.label),
            bits=bits,
            group_size=int(args.q_group_size),
        )
        command = build_convert_command(
            runtime_python=runtime_python,
            source_hf_model=source_hf_model,
            output_dir=output_dir,
            bits=bits,
            group_size=int(args.q_group_size),
            trust_remote_code=bool(args.trust_remote_code),
        )
        specs.append(
            QuantizedModelSpec(
                bits=bits,
                group_size=int(args.q_group_size),
                source_hf_model=str(source_hf_model),
                output_dir=str(output_dir),
                command=command,
            )
        )

    if args.dry_run:
        for spec in specs:
            print(" ".join(spec.command))
        return

    generated: list[dict[str, Any]] = []
    for spec in specs:
        output_dir = Path(spec.output_dir)
        prepare_output_dir(output_dir, overwrite=bool(args.overwrite))
        print(f"[prepare-mlx-quant] q{spec.bits}: {' '.join(spec.command)}")
        subprocess.run(spec.command, check=True)
        files = collect_file_sizes(output_dir)
        generated.append(
            {
                **asdict(spec),
                "files": files,
                "total_bytes": sum(files.values()),
            }
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": str(args.label),
        "source_hf_model": str(source_hf_model),
        "output_root": str(output_root),
        "runtime_python": str(runtime_python),
        "quantized_models": generated,
    }
    manifest_path = output_root / str(args.label) / "mlx_quantization_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json_dumps(manifest), encoding="utf-8")
    print(f"[prepare-mlx-quant] wrote {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[prepare-mlx-quant] error: {exc}", file=sys.stderr)
        raise
