#!/usr/bin/env python3
"""Convert a PEFT LoRA adapter checkpoint into MLX LoRA adapter format."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from safetensors.numpy import load_file


PEFT_KEY_RE = re.compile(
    r"^base_model\.model\.model\.layers\.(?P<layer>\d+)\.self_attn\."
    r"(?P<proj>[qkvo]_proj)\.lora_(?P<side>[AB])\.weight$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PEFT LoRA weights to MLX format.")
    parser.add_argument("--peft-adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="google/gemma-3-270m-it")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--scale", type=float, default=10.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--source-config", default=None)
    return parser.parse_args()


def convert_key(key: str) -> str:
    match = PEFT_KEY_RE.match(key)
    if not match:
        raise ValueError(f"Unsupported PEFT key: {key}")
    side = "lora_a" if match.group("side") == "A" else "lora_b"
    return (
        f"model.layers.{match.group('layer')}.self_attn."
        f"{match.group('proj')}.{side}"
    )


def convert_weights(peft_weights: dict[str, np.ndarray]) -> dict[str, mx.array]:
    converted: dict[str, mx.array] = {}
    for key, value in peft_weights.items():
        converted_key = convert_key(key)
        converted[converted_key] = mx.array(np.ascontiguousarray(value.T))
    return converted


def write_adapter_config(args: argparse.Namespace, output_dir: Path, source_count: int) -> None:
    payload: dict[str, Any] = {
        "adapter_path": str(output_dir),
        "converted_at": datetime.now(timezone.utc).isoformat(),
        "converted_by": "scripts/convert_peft_lora_to_mlx.py",
        "source_peft_adapter_dir": str(args.peft_adapter_dir),
        "source_config": args.source_config,
        "model": args.model,
        "fine_tune_type": "lora",
        "lora_parameters": {
            "keys": [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
            ],
            "rank": args.rank,
            "scale": args.scale,
            "dropout": args.dropout,
        },
        "source_weight_count": source_count,
    }
    (output_dir / "adapter_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source_path = args.peft_adapter_dir / "adapter_model.safetensors"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing PEFT weights: {source_path}")

    peft_weights = load_file(str(source_path))
    converted = convert_weights(peft_weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.output_dir / "adapters.safetensors"),
        converted,
        metadata={"format": "mlx"},
    )
    write_adapter_config(args, args.output_dir, len(peft_weights))
    manifest = {
        "converted_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source_path),
        "output_weights": str(args.output_dir / "adapters.safetensors"),
        "output_config": str(args.output_dir / "adapter_config.json"),
        "source_weight_count": len(peft_weights),
        "converted_weight_count": len(converted),
        "first_keys": sorted(converted)[:8],
    }
    (args.output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[peft-to-mlx] converted {len(converted)} weights to "
        f"{args.output_dir / 'adapters.safetensors'}"
    )


if __name__ == "__main__":
    main()
