#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from dpo_utils import (
    DEFAULT_INPUT_MODE,
    DEFAULT_MLX_HF_HOME,
    DEFAULT_MODEL_REPO_ID,
    DEFAULT_PROMPT_SPEC,
    DEFAULT_REUSE_SAMPLE_JSONL,
    build_prompt_messages,
    ensure_new_output_dir,
    load_jsonl,
    load_prompt_spec,
    normalize_summary,
    resolve_model_path,
)

try:
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - import guard for runtime setup
    raise SystemExit(
        "This script needs the DPO runtime packages. Run it with the DPO runtime "
        "Python after that environment is created."
    ) from exc


DEFAULT_MLX_ADAPTER_FILE = (
    "hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_v1/"
    "adapters/0001400_adapters.safetensors"
)
DEFAULT_MLX_ADAPTER_CONFIG = (
    "hf_bucket/experiments/mlx_lora/gemma_3_270m_it_title_abstract_v1/"
    "adapters/adapter_config.json"
)
DEFAULT_OUTPUT_DIR = (
    "hf_bucket/experiments/dpo/"
    "gemma_3_270m_it_title_abstract_attn_1400_peft_v1"
)
DEFAULT_MLX_PYTHON = "hf_bucket/experiments/mlx_lora/runtime/py312/bin/python"

EXPECTED_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
EXPECTED_RANK = 16
EXPECTED_LORA_SCALE = 10.0
EXPECTED_LORA_ALPHA = int(EXPECTED_LORA_SCALE * EXPECTED_RANK)
DEFAULT_MAX_NEW_TOKENS = 40
DEFAULT_PARITY_SAMPLE_COUNT = 3


def preferred_inference_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.backends.mps.is_available() else torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an MLX LoRA checkpoint into a PEFT LoRA adapter for the "
            "local DPO pilot."
        )
    )
    parser.add_argument(
        "--mlx-adapter-file",
        default=DEFAULT_MLX_ADAPTER_FILE,
        help="MLX checkpoint safetensors file to convert.",
    )
    parser.add_argument(
        "--mlx-adapter-config",
        default=DEFAULT_MLX_ADAPTER_CONFIG,
        help="MLX adapter_config.json that matches the checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the converted PEFT adapter.",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_MODEL_REPO_ID,
        help="Base model repo id for the converted adapter config.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional explicit model path for Transformers loading.",
    )
    parser.add_argument(
        "--hf-home",
        default=DEFAULT_MLX_HF_HOME,
        help="HF cache root used to resolve a local model snapshot.",
    )
    parser.add_argument(
        "--prompt-spec",
        default=DEFAULT_PROMPT_SPEC,
        help="Prompt spec used during parity checks.",
    )
    parser.add_argument(
        "--parity-source-jsonl",
        default=DEFAULT_REUSE_SAMPLE_JSONL,
        help="JSONL source for the fixed parity prompts.",
    )
    parser.add_argument(
        "--parity-sample-count",
        type=int,
        default=DEFAULT_PARITY_SAMPLE_COUNT,
        help="Number of prompts to compare during parity checks.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum generated tokens during parity checks.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("abstract_only", "title_abstract"),
        default=DEFAULT_INPUT_MODE,
        help="Prompt mode used during parity checks.",
    )
    parser.add_argument(
        "--mlx-python",
        default=DEFAULT_MLX_PYTHON,
        help="Python interpreter inside the existing MLX runtime.",
    )
    parser.add_argument(
        "--run-parity-check",
        action="store_true",
        help="Run a 3-prompt greedy parity check after conversion.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in a non-empty output directory.",
    )
    return parser.parse_args()


def load_mlx_adapter_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    lora_parameters = metadata.get("lora_parameters", {})
    keys = tuple(
        item.split(".")[-1] for item in lora_parameters.get("keys", []) if item.startswith("self_attn.")
    )
    if keys != EXPECTED_TARGET_MODULES:
        raise ValueError(
            f"Expected target modules {EXPECTED_TARGET_MODULES}, found {keys}."
        )
    if int(lora_parameters.get("rank", -1)) != EXPECTED_RANK:
        raise ValueError(
            f"Expected rank {EXPECTED_RANK}, found {lora_parameters.get('rank')}."
        )
    if float(lora_parameters.get("scale", -1.0)) != EXPECTED_LORA_SCALE:
        raise ValueError(
            f"Expected scale {EXPECTED_LORA_SCALE}, found {lora_parameters.get('scale')}."
        )
    if float(lora_parameters.get("dropout", -1.0)) != 0.0:
        raise ValueError(
            f"Expected dropout 0.0, found {lora_parameters.get('dropout')}."
        )
    model_name = str(metadata.get("model", ""))
    if model_name != DEFAULT_MODEL_REPO_ID:
        raise ValueError(f"Expected base model {DEFAULT_MODEL_REPO_ID}, found {model_name}.")
    return metadata


def prepare_peft_skeleton(
    *,
    model_path: str,
    base_model_id: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.float32,
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(EXPECTED_TARGET_MODULES),
        r=EXPECTED_RANK,
        lora_alpha=EXPECTED_LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        inference_mode=False,
    )
    peft_model = get_peft_model(model, peft_config)
    with tempfile.TemporaryDirectory(prefix="peft-skeleton-") as temp_dir:
        temp_path = Path(temp_dir)
        peft_model.save_pretrained(temp_path, safe_serialization=True)
        skeleton_state = load_file(str(temp_path / "adapter_model.safetensors"), device="cpu")
        adapter_config = json.loads((temp_path / "adapter_config.json").read_text(encoding="utf-8"))
    adapter_config["base_model_name_or_path"] = base_model_id
    del peft_model
    del model
    gc.collect()
    return skeleton_state, adapter_config


def find_matching_peft_key(
    expected_keys: set[str],
    *,
    layer_index: str,
    projection_name: str,
    suffix: str,
) -> str:
    needle = f".layers.{layer_index}.self_attn.{projection_name}.lora_{suffix}"
    matches = [
        key
        for key in expected_keys
        if needle in key and (key.endswith(".weight") or ".weight" in key.split(needle, 1)[1])
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one PEFT key for layer={layer_index} proj={projection_name} "
            f"suffix={suffix}, found {matches}."
        )
    return matches[0]


def convert_state_dict(
    *,
    mlx_state_path: Path,
    skeleton_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    mlx_state = load_file(str(mlx_state_path), device="cpu")
    expected_keys = set(skeleton_state)
    converted: dict[str, torch.Tensor] = {}
    for key, tensor in mlx_state.items():
        parts = key.split(".")
        if len(parts) != 6 or parts[:2] != ["model", "layers"] or parts[3] != "self_attn":
            raise ValueError(f"Unexpected MLX adapter tensor key: {key}")
        layer_index = parts[2]
        projection_name = parts[4]
        raw_suffix = parts[5]
        if projection_name not in EXPECTED_TARGET_MODULES:
            raise ValueError(f"Unexpected projection in MLX adapter key: {key}")
        if raw_suffix == "lora_a":
            peft_suffix = "A"
        elif raw_suffix == "lora_b":
            peft_suffix = "B"
        else:
            raise ValueError(f"Unexpected LoRA suffix in MLX adapter key: {key}")
        target_key = find_matching_peft_key(
            expected_keys,
            layer_index=layer_index,
            projection_name=projection_name,
            suffix=peft_suffix,
        )
        mapped = tensor.transpose(0, 1).contiguous()
        if mapped.shape != skeleton_state[target_key].shape:
            raise ValueError(
                f"Shape mismatch for {key} -> {target_key}: "
                f"{tuple(mapped.shape)} != {tuple(skeleton_state[target_key].shape)}"
            )
        converted[target_key] = mapped

    if set(converted) != expected_keys:
        missing = sorted(expected_keys - set(converted))
        extra = sorted(set(converted) - expected_keys)
        raise ValueError(
            "Converted adapter keys do not match PEFT skeleton. "
            f"Missing={missing} Extra={extra}"
        )
    return converted


def build_parity_rows(path: Path, count: int) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if len(rows) < count:
        raise ValueError(f"Need at least {count} parity rows in {path}, found {len(rows)}.")
    return [
        {
            "paper_id": str(row["paper_id"]),
            "title": str(row["title"]),
            "abstract": str(row["abstract"]),
        }
        for row in rows[:count]
    ]


def generate_transformers_summaries(
    *,
    model_path: str,
    adapter_dir: Path,
    prompt_spec_path: Path,
    rows: list[dict[str, Any]],
    input_mode: str,
    max_new_tokens: int,
) -> list[str]:
    prompt_spec = load_prompt_spec(prompt_spec_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=preferred_inference_dtype(),
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    model.eval()
    eos_token_id = model.generation_config.eos_token_id
    pad_token_id = (
        model.generation_config.pad_token_id
        if model.generation_config.pad_token_id is not None
        else tokenizer.pad_token_id
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    rendered_prompts: list[str] = []
    for row in rows:
        messages = build_prompt_messages(row, prompt_spec, input_mode)
        if getattr(tokenizer, "chat_template", None):
            rendered_prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        else:
            rendered_prompts.append(
                "\n\n".join(message["content"] for message in messages)
            )

    outputs: list[str] = []
    for rendered_prompt in rendered_prompts:
        encoded = tokenizer(
            [rendered_prompt],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_width = int(encoded["input_ids"].shape[1])
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        continuation = generated[0, prompt_width:]
        text = tokenizer.decode(continuation, skip_special_tokens=True)
        outputs.append(normalize_summary(text))

    del model
    del base_model
    del tokenizer
    gc.collect()
    return outputs


def generate_mlx_summaries(
    *,
    mlx_python: Path,
    model_path: str,
    adapter_file: Path,
    adapter_config: Path,
    prompt_spec_path: Path,
    rows: list[dict[str, Any]],
    input_mode: str,
    max_new_tokens: int,
) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="mlx-parity-") as temp_dir:
        temp_path = Path(temp_dir)
        adapter_view = temp_path / "adapter_view"
        adapter_view.mkdir(parents=True)
        shutil.copy2(adapter_config, adapter_view / "adapter_config.json")
        shutil.copy2(adapter_file, adapter_view / "adapters.safetensors")

        payload_path = temp_path / "payload.json"
        payload_path.write_text(
            json.dumps(
                {
                    "model_path": model_path,
                    "adapter_path": str(adapter_view),
                    "prompt_spec_path": str(prompt_spec_path),
                    "rows": rows,
                    "input_mode": input_mode,
                    "max_new_tokens": max_new_tokens,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        code = r"""
import json
import sys
from pathlib import Path

from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
prompt_spec = json.loads(Path(payload["prompt_spec_path"]).read_text(encoding="utf-8"))
model, tokenizer = load(
    payload["model_path"],
    adapter_path=payload["adapter_path"],
    tokenizer_config={"trust_remote_code": True},
)

def encode_prompt(messages):
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(rendered, add_special_tokens=False)

prompts = []
for row in payload["rows"]:
    user_prompt = prompt_spec["user_prompt_templates"][payload["input_mode"]].format(
        title=row["title"],
        abstract=row["abstract"],
    )
    prompts.append(
        [
            {"role": "system", "content": prompt_spec["system_prompt"]},
            {"role": "user", "content": user_prompt},
        ]
    )

tokenized = [encode_prompt(messages) for messages in prompts]
response = batch_generate(
    model,
    tokenizer,
    tokenized,
    max_tokens=payload["max_new_tokens"],
    sampler=make_sampler(temp=0.0, top_p=0.0),
    verbose=False,
)
print(json.dumps([" ".join(text.strip().split()) for text in response.texts]))
"""
        result = subprocess.run(
            [str(mlx_python), "-c", code, str(payload_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout.strip())


def run_parity_check(
    *,
    mlx_python: Path,
    model_path: str,
    mlx_adapter_file: Path,
    mlx_adapter_config: Path,
    converted_adapter_dir: Path,
    prompt_spec_path: Path,
    parity_rows: list[dict[str, Any]],
    input_mode: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    mlx_outputs = generate_mlx_summaries(
        mlx_python=mlx_python,
        model_path=model_path,
        adapter_file=mlx_adapter_file,
        adapter_config=mlx_adapter_config,
        prompt_spec_path=prompt_spec_path,
        rows=parity_rows,
        input_mode=input_mode,
        max_new_tokens=max_new_tokens,
    )
    transformers_outputs = generate_transformers_summaries(
        model_path=model_path,
        adapter_dir=converted_adapter_dir,
        prompt_spec_path=prompt_spec_path,
        rows=parity_rows,
        input_mode=input_mode,
        max_new_tokens=max_new_tokens,
    )
    mismatches = []
    for row, mlx_output, transformers_output in zip(
        parity_rows, mlx_outputs, transformers_outputs
    ):
        if mlx_output != transformers_output:
            mismatches.append(
                {
                    "paper_id": row["paper_id"],
                    "mlx_output": mlx_output,
                    "transformers_output": transformers_output,
                }
            )
    return {
        "rows_checked": len(parity_rows),
        "matches_token_for_token": not mismatches,
        "mismatches": mismatches,
    }


def main() -> None:
    args = parse_args()

    mlx_adapter_file = Path(args.mlx_adapter_file)
    mlx_adapter_config = Path(args.mlx_adapter_config)
    output_dir = Path(args.output_dir)
    prompt_spec_path = Path(args.prompt_spec)
    parity_source_path = Path(args.parity_source_jsonl)
    mlx_python = Path(args.mlx_python)
    hf_home = Path(args.hf_home)

    ensure_new_output_dir(output_dir, overwrite=args.overwrite)
    metadata = load_mlx_adapter_metadata(mlx_adapter_config)
    model_path = resolve_model_path(
        model_path=args.model_path,
        hf_home=hf_home,
        repo_id=args.base_model,
    )

    print(f"[convert-peft] using base model path: {model_path}")
    print(f"[convert-peft] building PEFT skeleton for {args.base_model}")
    skeleton_state, adapter_config = prepare_peft_skeleton(
        model_path=model_path,
        base_model_id=args.base_model,
    )

    print(f"[convert-peft] converting MLX checkpoint: {mlx_adapter_file}")
    converted_state = convert_state_dict(
        mlx_state_path=mlx_adapter_file,
        skeleton_state=skeleton_state,
    )
    save_file(converted_state, str(output_dir / "adapter_model.safetensors"))
    (output_dir / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "base_model_repo_id": args.base_model,
        "base_model_path_used": model_path,
        "mlx_adapter_file": str(mlx_adapter_file),
        "mlx_adapter_config": str(mlx_adapter_config),
        "output_dir": str(output_dir),
        "rank": EXPECTED_RANK,
        "lora_alpha": EXPECTED_LORA_ALPHA,
        "target_modules": list(EXPECTED_TARGET_MODULES),
        "source_scale": EXPECTED_LORA_SCALE,
        "mlx_training_config": str(metadata.get("config", "")),
        "parity_check": None,
    }

    if args.run_parity_check:
        print("[convert-peft] running MLX vs Transformers parity check")
        parity_rows = build_parity_rows(parity_source_path, args.parity_sample_count)
        parity_result = run_parity_check(
            mlx_python=mlx_python,
            model_path=model_path,
            mlx_adapter_file=mlx_adapter_file,
            mlx_adapter_config=mlx_adapter_config,
            converted_adapter_dir=output_dir,
            prompt_spec_path=prompt_spec_path,
            parity_rows=parity_rows,
            input_mode=args.input_mode,
            max_new_tokens=args.max_new_tokens,
        )
        manifest["parity_check"] = parity_result
        if not parity_result["matches_token_for_token"]:
            (output_dir / "conversion_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            mismatch_count = len(parity_result["mismatches"])
            raise SystemExit(
                f"Parity check failed with {mismatch_count} mismatches. "
                f"See {output_dir / 'conversion_manifest.json'} for details."
            )

    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[convert-peft] wrote PEFT adapter to {output_dir}")
    if manifest["parity_check"]:
        print(
            "[convert-peft] parity passed on "
            f"{manifest['parity_check']['rows_checked']} prompts"
        )


if __name__ == "__main__":
    main()
