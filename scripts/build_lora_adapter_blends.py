#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


def parse_blend(value: str) -> tuple[str, Path, float]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Use label=path=weight for each blend input.")
    label, path, weight = parts
    try:
        parsed_weight = float(weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid weight {weight!r}.") from exc
    if parsed_weight <= 0:
        raise argparse.ArgumentTypeError("Weights must be positive.")
    return label, Path(path), parsed_weight


def load_config(path: Path) -> dict[str, Any]:
    return json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))


def lora_pair_key(key: str, suffix: str) -> str:
    if suffix not in key:
        raise ValueError(f"Expected {suffix!r} in {key!r}.")
    return key.replace(suffix, ".lora_B.weight" if suffix.endswith("lora_A.weight") else ".lora_A.weight")


def write_blend(output_dir: Path, blend: list[tuple[str, Path, float]]) -> None:
    configs = [load_config(path) for _, path, _ in blend]
    ranks = {int(config["r"]) for config in configs}
    alphas = {int(config["lora_alpha"]) for config in configs}
    scalings = {int(config["lora_alpha"]) / int(config["r"]) for config in configs}
    if len(ranks) != 1 or len(alphas) != 1 or len(scalings) != 1:
        raise ValueError(f"Incompatible blend configs: ranks={ranks}, alphas={alphas}, scalings={scalings}")

    tensors_by_label = [(label, load_file(str(path / "adapter_model.safetensors")), weight) for label, path, weight in blend]
    keys = [set(tensors.keys()) for _, tensors, _ in tensors_by_label]
    if any(keys[0] != key_set for key_set in keys[1:]):
        raise ValueError("Blend inputs do not have matching tensor keys.")

    output_tensors: dict[str, torch.Tensor] = {}
    for key in sorted(keys[0]):
        if key.endswith(".lora_A.weight"):
            parts = []
            for _, tensors, weight in tensors_by_label:
                parts.append(tensors[key] * math.sqrt(weight))
            output_tensors[key] = torch.cat(parts, dim=0)
        elif key.endswith(".lora_B.weight"):
            parts = []
            for _, tensors, weight in tensors_by_label:
                parts.append(tensors[key] * math.sqrt(weight))
            output_tensors[key] = torch.cat(parts, dim=1)
        else:
            raise ValueError(f"Unexpected tensor key: {key}")

    base_config = dict(configs[0])
    base_rank = int(base_config["r"])
    base_alpha = int(base_config["lora_alpha"])
    output_rank = base_rank * len(blend)
    output_alpha = base_alpha * len(blend)
    base_config["r"] = output_rank
    base_config["lora_alpha"] = output_alpha
    base_config["inference_mode"] = True

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(output_tensors, str(output_dir / "adapter_model.safetensors"))
    (output_dir / "adapter_config.json").write_text(
        json.dumps(base_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "blend": [
                    {"label": label, "path": str(path), "weight": weight}
                    for label, path, weight in blend
                ],
                "rank": output_rank,
                "lora_alpha": output_alpha,
                "method": "positive weighted exact LoRA delta concatenation",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build positive weighted PEFT LoRA blends.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input", action="append", type=parse_blend, required=True)
    parser.add_argument("--also-copy-to", type=Path, default=None)
    args = parser.parse_args()

    total = sum(weight for _, _, weight in args.input)
    blend = [(label, path, weight / total) for label, path, weight in args.input]
    write_blend(args.output_dir, blend)
    if args.also_copy_to is not None:
        if args.also_copy_to.exists():
            shutil.rmtree(args.also_copy_to)
        shutil.copytree(args.output_dir, args.also_copy_to)


if __name__ == "__main__":
    main()
