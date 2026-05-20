#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import mlx.core as mx


COPY_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "README.md",
]


def touched_modules(adapter_path: Path) -> set[str]:
    weights = mx.load(str(adapter_path / "adapters.safetensors"))
    modules: set[str] = set()
    for key in weights:
        if key.endswith(".lora_a"):
            modules.add(key.removesuffix(".lora_a"))
        elif key.endswith(".lora_b"):
            modules.add(key.removesuffix(".lora_b"))
    if not modules:
        raise ValueError(f"No LoRA tensors found in {adapter_path}")
    return modules


def belongs_to_module(weight_key: str, modules: set[str]) -> str | None:
    for module in modules:
        if weight_key == f"{module}.weight":
            return module
        if weight_key == f"{module}.scales":
            return module
        if weight_key == f"{module}.biases":
            return module
    return None


def copy_model_sidecars(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in COPY_FILES:
        source_path = source / name
        if source_path.exists():
            shutil.copy2(source_path, destination / name)


def write_index(path: Path, weights: dict[str, mx.array]) -> None:
    total_size = sum(value.nbytes for value in weights.values())
    total_parameters = sum(value.size for value in weights.values())
    payload = {
        "metadata": {
            "total_size": total_size,
            "total_parameters": total_parameters,
        },
        "weight_map": {key: "model.safetensors" for key in sorted(weights)},
    }
    (path / "model.safetensors.index.json").write_text(
        json.dumps(payload, indent=4, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_mixed_model(base_model: Path, dequantized_fused_model: Path, adapter_path: Path, output: Path) -> None:
    modules = touched_modules(adapter_path)
    base_weights = mx.load(str(base_model / "model.safetensors"))
    fused_weights = mx.load(str(dequantized_fused_model / "model.safetensors"))

    mixed: dict[str, mx.array] = {}
    replaced = 0
    skipped_quant_tensors = 0
    for key, value in base_weights.items():
        module = belongs_to_module(key, modules)
        if module is None:
            mixed[key] = value
            continue

        if key.endswith(".weight"):
            mixed[key] = fused_weights[key]
            replaced += 1
        else:
            skipped_quant_tensors += 1

    output.mkdir(parents=True, exist_ok=True)
    copy_model_sidecars(base_model, output)
    mx.save_safetensors(str(output / "model.safetensors"), mixed, metadata={"format": "mlx"})
    write_index(output, mixed)

    manifest = {
        "base_model": str(base_model),
        "dequantized_fused_model": str(dequantized_fused_model),
        "adapter_path": str(adapter_path),
        "output": str(output),
        "touched_modules": sorted(modules),
        "dense_fused_modules": replaced,
        "removed_quantization_tensors": skipped_quant_tensors,
        "weight_count": len(mixed),
    }
    (output / "mixed_fusion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an MLX mixed q4/dense fused model where LoRA-touched modules "
            "come from a dequantized fused model and untouched modules remain q4."
        )
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dequantized-fused-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} already exists; pass --overwrite")
        shutil.rmtree(output)

    build_mixed_model(
        base_model=Path(args.base_model),
        dequantized_fused_model=Path(args.dequantized_fused_model),
        adapter_path=Path(args.adapter_path),
        output=output,
    )


if __name__ == "__main__":
    main()
