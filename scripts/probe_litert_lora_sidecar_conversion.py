#!/usr/bin/env python3
"""Probe MediaPipe/LiteRT LoRA sidecar conversion for the current Gemma adapter."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

from mediapipe.tasks.python.genai import converter


DEFAULT_BASE = (
    "hf_bucket/experiments/mlx_lora/.hf-home/hub/"
    "models--google--gemma-3-270m-it/snapshots/"
    "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"
)
DEFAULT_ADAPTER = (
    "hf_bucket/experiments/litert_exports/"
    "faith_overlay_1400_v2_final_600/peft_adapter"
)
DEFAULT_OUTPUT_ROOT = (
    "hf_bucket/experiments/litert_lora_sidecar_probe_20260507"
)


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_full_conversion(base: Path, adapter: Path, output_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "step": "full_conversion",
        "base": str(base),
        "adapter": str(adapter),
        "model_type": "GEMMA3_300M",
        "backend": "gpu",
        "rank": 16,
        "alpha": 160,
    }
    start = time.time()
    base_tflite = output_root / "base_gpu_q8.tflite"
    lora_tflite = output_root / "lora_rank16_alpha160.tflite"
    try:
        config = converter.ConversionConfig(
            input_ckpt=str(base),
            ckpt_format="safetensors",
            model_type="GEMMA3_300M",
            backend="gpu",
            output_dir=str(output_root / "converter_work"),
            vocab_model_file=str(base / "tokenizer.model"),
            output_tflite_file=str(base_tflite),
            lora_ckpt=str(adapter),
            lora_rank=16,
            lora_alpha=160,
            lora_output_tflite_file=str(lora_tflite),
        )
        converter.convert_checkpoint(config)
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001 - diagnostic script records failures.
        result["ok"] = False
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        result["elapsed_seconds"] = round(time.time() - start, 3)
        result["base_tflite"] = file_record(base_tflite)
        result["lora_tflite"] = file_record(lora_tflite)
        write_json(output_root / "conversion_result.json", result)
    return result


def run_packaging_variants(base: Path, output_root: Path) -> list[dict[str, Any]]:
    work = output_root / "converter_work"
    variants = [
        {
            "name": "gemma3_300m_plain",
            "init_model_type": "GEMMA3_300M",
        },
        {
            "name": "gemma3_300m_with_submodel",
            "init_model_type": "GEMMA3_300M",
            "submodel_type": "GEMMA3_300M",
            "lora_main_model_type": "GEMMA3_300M",
        },
        {
            "name": "functiongemma3_300m_mutated",
            "init_model_type": "GEMMA3_300M",
            "mutate_model_type": "FUNCTIONGEMMA3_300M",
        },
    ]
    attempts: list[dict[str, Any]] = []
    for variant in variants:
        base_tflite = output_root / f"{variant['name']}_base.tflite"
        lora_tflite = output_root / f"{variant['name']}_lora.tflite"
        record = {"step": "packaging_variant", **variant}
        start = time.time()
        try:
            config = converter.ConversionConfig(
                input_ckpt=str(base),
                ckpt_format="safetensors",
                model_type=str(variant["init_model_type"]),
                backend="gpu",
                output_dir=str(work),
                combine_file_only=True,
                vocab_model_file=str(base / "tokenizer.model"),
                output_tflite_file=str(base_tflite),
                lora_ckpt=str(work),
                lora_rank=16,
                lora_alpha=160,
                lora_output_tflite_file=str(lora_tflite),
                lora_main_model_type=variant.get("lora_main_model_type"),
                submodel_type=variant.get("submodel_type"),
            )
            if "mutate_model_type" in variant:
                config.model_type = str(variant["mutate_model_type"])
            converter.convert_checkpoint(config)
            record["ok"] = True
        except Exception as exc:  # noqa: BLE001 - diagnostic script records failures.
            record["ok"] = False
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
            record["traceback"] = traceback.format_exc()
        finally:
            record["elapsed_seconds"] = round(time.time() - start, 3)
            record["base_tflite"] = file_record(base_tflite)
            record["lora_tflite"] = file_record(lora_tflite)
            attempts.append(record)
    write_json(output_root / "packaging_attempts.json", attempts)
    return attempts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path(DEFAULT_BASE))
    parser.add_argument("--adapter", type=Path, default=Path(DEFAULT_ADAPTER))
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.exists() and not args.keep_existing:
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    full = run_full_conversion(args.base, args.adapter, args.output_root)
    attempts = []
    if (args.output_root / "converter_work").exists():
        attempts = run_packaging_variants(args.base, args.output_root)

    summary = {
        "full_conversion_ok": full.get("ok", False),
        "packaging_variant_ok": any(row.get("ok", False) for row in attempts),
        "full_conversion": full,
        "packaging_attempts": attempts,
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["full_conversion_ok"] or summary["packaging_variant_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
