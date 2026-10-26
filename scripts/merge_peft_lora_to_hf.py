#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT LoRA adapter into a local Hugging Face CausalLM checkpoint."
    )
    parser.add_argument("--base-model", required=True, help="Base model path or repo id.")
    parser.add_argument("--adapter-dir", required=True, help="PEFT adapter directory.")
    parser.add_argument("--output-dir", required=True, help="Merged model output directory.")
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16", "auto"),
        default="float32",
        help="Load dtype for the merge.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing output files.")
    return parser.parse_args()


def resolve_dtype(value: str):
    if value == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[value]


def ensure_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, overwrite=args.overwrite)

    dtype = resolve_dtype(args.dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        dtype=dtype,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir, is_trainable=False)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    manifest = {
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "output_dir": str(output_dir),
        "dtype": args.dtype,
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[merge-peft] wrote merged checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
