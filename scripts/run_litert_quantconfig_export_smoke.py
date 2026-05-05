#!/usr/bin/env python3
"""Smoke-test LiteRT Torch-side generative quantization through export_hf.

This keeps the Hugging Face export pipeline but injects Converter.convert's
`quant_config` argument before LiteRT lowering. It intentionally does not pass
export_hf's `quantization_recipe`, because that path quantizes the exported
TFLite flatbuffer after conversion.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any

from litert_torch._convert import interface as converter_interface
from litert_torch.generative.export_hf import export as export_hf
from litert_torch.generative.export_hf.core import export_lib
from litert_torch.generative.quantize import quant_recipes
from litert_torch.generative.quantize.quant_attrs import Dtype, Granularity
from litert_torch import model as litert_model


def build_quant_config(recipe: str):
    if recipe == "dynamic_int4_blockwise32":
        return quant_recipes.full_dynamic_recipe(
            weight_dtype=Dtype.INT4,
            granularity=Granularity.BLOCKWISE_32,
        )
    if recipe == "weight_only_int4_blockwise32":
        return quant_recipes.full_weight_only_recipe(
            weight_dtype=Dtype.INT4,
            granularity=Granularity.BLOCKWISE_32,
        )
    if recipe == "dynamic_int8_channelwise":
        return quant_recipes.full_dynamic_recipe(
            weight_dtype=Dtype.INT8,
            granularity=Granularity.CHANNELWISE,
        )
    raise ValueError(f"Unsupported quant config recipe: {recipe}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--recipe",
        choices=[
            "dynamic_int4_blockwise32",
            "weight_only_int4_blockwise32",
            "dynamic_int8_channelwise",
        ],
        default="dynamic_int4_blockwise32",
    )
    parser.add_argument("--prefill-length", type=int, default=1024)
    parser.add_argument("--cache-length", type=int, default=1024)
    parser.add_argument("--externalize-embedder", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    quant_config = build_quant_config(args.recipe)
    original_convert = converter_interface.Converter.convert
    original_model_content = litert_model.LiteRTModel.model_content
    original_update_export_config = export_lib.update_export_config

    def convert_with_quant_config(self, *convert_args: Any, **convert_kwargs: Any):
        if convert_kwargs.get("quant_config") is None:
            convert_kwargs["quant_config"] = quant_config
        return original_convert(self, *convert_args, **convert_kwargs)

    def model_content_as_bytes(self):
        content = original_model_content(self)
        return bytes(content) if isinstance(content, bytearray) else content

    def update_export_config_without_post_quant(*update_args: Any, **update_kwargs: Any):
        config = original_update_export_config(*update_args, **update_kwargs)
        return dataclasses.replace(config, quantization_recipe=None)

    converter_interface.Converter.convert = convert_with_quant_config
    litert_model.LiteRTModel.model_content = model_content_as_bytes
    export_lib.update_export_config = update_export_config_without_post_quant

    export_hf.export(
        model=str(args.model_dir),
        output_dir=str(args.output_dir),
        prefill_lengths=[args.prefill_length],
        cache_length=args.cache_length,
        externalize_embedder=args.externalize_embedder,
        use_jinja_template=True,
        bundle_litert_lm=True,
        trust_remote_code=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
