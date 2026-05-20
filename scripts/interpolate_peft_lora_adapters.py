#!/usr/bin/env python3
"""Interpolate two PEFT LoRA adapters with matching tensor keys."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import load_file, save_file


COPY_FILES = [
    "adapter_config.json",
    "README.md",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpolate two matching PEFT LoRA adapters.")
    parser.add_argument("--base-adapter", type=Path, required=True, help="Adapter at alpha=0.")
    parser.add_argument("--other-adapter", type=Path, required=True, help="Adapter at alpha=1.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def adapter_weights(path: Path) -> dict[str, torch.Tensor]:
    weights_path = path / "adapter_model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {weights_path}")
    return load_file(str(weights_path))


def copy_sidecars(source: Path, output: Path) -> None:
    for name in COPY_FILES:
        src = source / name
        if src.exists():
            shutil.copy2(src, output / name)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} already exists; pass --overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    base = adapter_weights(args.base_adapter)
    other = adapter_weights(args.other_adapter)
    if set(base) != set(other):
        missing_from_other = sorted(set(base) - set(other))
        missing_from_base = sorted(set(other) - set(base))
        raise ValueError(
            "Adapter tensor keys do not match: "
            f"missing_from_other={missing_from_other[:5]} "
            f"missing_from_base={missing_from_base[:5]}"
        )

    mixed: dict[str, torch.Tensor] = {}
    for key in sorted(base):
        left = base[key]
        right = other[key]
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch for {key}: {tuple(left.shape)} vs {tuple(right.shape)}")
        if not left.is_floating_point():
            if not torch.equal(left, right):
                raise ValueError(f"Non-floating tensor differs for {key}")
            mixed[key] = left
            continue
        mixed[key] = left.lerp(right.to(dtype=left.dtype), args.alpha)

    copy_sidecars(args.base_adapter, args.output)
    save_file(mixed, str(args.output / "adapter_model.safetensors"))
    manifest = {
        "created_by": "scripts/interpolate_peft_lora_adapters.py",
        "base_adapter": str(args.base_adapter),
        "other_adapter": str(args.other_adapter),
        "output": str(args.output),
        "alpha": args.alpha,
        "tensor_count": len(mixed),
    }
    (args.output / "interpolation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[interpolate-peft] wrote {args.output} alpha={args.alpha:g} tensors={len(mixed)}")


if __name__ == "__main__":
    main()
