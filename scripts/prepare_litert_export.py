#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PROMOTED_POINTER = (
    "hf_bucket/experiments/mlx_lora/promoted/"
    "gemma_3_270m_it_title_abstract_current.json"
)
DEFAULT_OUTPUT_ROOT = "hf_bucket/experiments/litert_exports"
DEFAULT_DPO_PYTHON = "hf_bucket/experiments/dpo/runtime/py312/bin/python"
DEFAULT_MLX_PYTHON = "hf_bucket/experiments/mlx_lora/runtime/py312/bin/python"
DEFAULT_HF_HOME = "hf_bucket/experiments/mlx_lora/.hf-home"
DEFAULT_PREFILL_LENGTH = 1024
DEFAULT_CACHE_LENGTH = 1024
DEFAULT_QUANTIZATION = "dynamic_wi8_afp32"
DEFAULT_MAX_SHARD_SIZE = "2GB"
DEFAULT_PROMPT_PREFIX = "<start_of_turn>user\n"
DEFAULT_PROMPT_SUFFIX = "<end_of_turn>\n<start_of_turn>model\n"
DEFAULT_STOP_TOKENS = ("<end_of_turn>", "<eos>")


@dataclass(frozen=True)
class PromotedAdapter:
    pointer_path: Path
    raw: dict[str, Any]
    label: str
    base_model: str
    checkpoint_path: Path
    adapter_config_path: Path


@dataclass(frozen=True)
class ReleasePaths:
    release_dir: Path
    peft_adapter_dir: Path
    merged_model_dir: Path
    litert_output_dir: Path
    command_doc_path: Path
    manifest_path: Path


def slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    return normalized.strip("_") or "litert_export"


def json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def resolve_pointer_path(pointer_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    pointer_relative = pointer_path.parent / candidate
    if pointer_relative.exists():
        return pointer_relative
    return candidate


def require_string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Promoted pointer is missing non-empty string key: {key}")
    return value


def load_promoted_pointer(pointer_path: Path) -> PromotedAdapter:
    raw = json.loads(pointer_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected promoted pointer object in {pointer_path}.")
    if raw.get("status") != "promoted":
        raise ValueError(
            f"Expected {pointer_path} to have status='promoted', found {raw.get('status')!r}."
        )

    label = require_string(raw, "label")
    base_model = require_string(raw, "base_model")
    checkpoint_path = resolve_pointer_path(
        pointer_path, require_string(raw, "checkpoint_path")
    )
    adapter_config_path = resolve_pointer_path(
        pointer_path, require_string(raw, "adapter_config_path")
    )
    for path, label_for_error in (
        (checkpoint_path, "checkpoint_path"),
        (adapter_config_path, "adapter_config_path"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Promoted pointer {label_for_error} does not exist: {path}"
            )

    return PromotedAdapter(
        pointer_path=pointer_path,
        raw=raw,
        label=label,
        base_model=base_model,
        checkpoint_path=checkpoint_path,
        adapter_config_path=adapter_config_path,
    )


def build_release_paths(output_root: Path, label: str) -> ReleasePaths:
    release_dir = output_root / slugify(label)
    return ReleasePaths(
        release_dir=release_dir,
        peft_adapter_dir=release_dir / "peft_adapter",
        merged_model_dir=release_dir / "merged_hf_model",
        litert_output_dir=release_dir / "litert",
        command_doc_path=release_dir / "litert_export_commands.md",
        manifest_path=release_dir / "litert_readiness_manifest.json",
    )


def ensure_release_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite non-empty export directory: {path}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def resolve_model_path(*, model_path: str | None, hf_home: Path, repo_id: str) -> str:
    if model_path:
        return model_path
    cache_dir = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}"
    ref_path = cache_dir / "refs" / "main"
    if ref_path.exists():
        snapshot = ref_path.read_text(encoding="utf-8").strip()
        snapshot_path = cache_dir / "snapshots" / snapshot
        if snapshot_path.exists():
            return str(snapshot_path)
    return repo_id


def run_command(command: list[str], *, label: str) -> None:
    print(f"[prepare-litert] {label}")
    print("[prepare-litert] " + " ".join(command))
    subprocess.run(command, check=True)


def module_available(python: Path, module_name: str) -> bool:
    result = subprocess.run(
        [
            str(python),
            "-c",
            (
                "import importlib.util, sys; "
                f"sys.exit(0 if importlib.util.find_spec({module_name!r}) else 1)"
            ),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def convert_promoted_adapter(
    *,
    runtime_python: Path,
    mlx_python: Path,
    promoted: PromotedAdapter,
    paths: ReleasePaths,
    hf_home: Path,
    model_path: str | None,
    run_parity_check: bool,
    overwrite: bool,
) -> None:
    command = [
        str(runtime_python),
        "scripts/convert_mlx_lora_to_peft.py",
        "--mlx-adapter-file",
        str(promoted.checkpoint_path),
        "--mlx-adapter-config",
        str(promoted.adapter_config_path),
        "--output-dir",
        str(paths.peft_adapter_dir),
        "--base-model",
        promoted.base_model,
        "--hf-home",
        str(hf_home),
        "--mlx-python",
        str(mlx_python),
    ]
    if model_path:
        command.extend(["--model-path", model_path])
    if run_parity_check:
        command.append("--run-parity-check")
    if overwrite:
        command.append("--overwrite")
    run_command(command, label="converting promoted MLX adapter to PEFT")


def merge_peft_adapter(
    *,
    runtime_python: Path,
    model_path_used: str,
    peft_adapter_dir: Path,
    output_dir: Path,
    max_shard_size: str,
) -> None:
    payload = {
        "model_path": model_path_used,
        "peft_adapter_dir": str(peft_adapter_dir),
        "output_dir": str(output_dir),
        "max_shard_size": max_shard_size,
    }
    code = r"""
import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

payload = json.loads(sys.argv[1])
model_path = payload["model_path"]
adapter_dir = Path(payload["peft_adapter_dir"])
output_dir = Path(payload["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

print(f"[prepare-litert:merge] loading base model from {model_path}", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.float32,
)
print(f"[prepare-litert:merge] loading PEFT adapter from {adapter_dir}", flush=True)
model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
print("[prepare-litert:merge] merging adapter into base weights", flush=True)
merged = model.merge_and_unload()
merged.eval()
print(f"[prepare-litert:merge] writing merged model to {output_dir}", flush=True)
merged.save_pretrained(
    output_dir,
    safe_serialization=True,
    max_shard_size=payload["max_shard_size"],
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.save_pretrained(output_dir)

source_path = Path(model_path)
for file_name in ("chat_template.jinja", "generation_config.json"):
    source_file = source_path / file_name
    target_file = output_dir / file_name
    if source_file.exists() and not target_file.exists():
        shutil.copy2(source_file, target_file)
"""
    run_command(
        [str(runtime_python), "-c", code, json.dumps(payload)],
        label="merging PEFT adapter into a full Hugging Face model",
    )


def render_litert_command_doc(
    *,
    paths: ReleasePaths,
    base_model: str,
    label: str,
    prompt_prefix: str,
    prompt_suffix: str,
    stop_tokens: list[str] | tuple[str, ...],
    prefill_length: int = DEFAULT_PREFILL_LENGTH,
    cache_length: int = DEFAULT_CACHE_LENGTH,
    quantization: str = DEFAULT_QUANTIZATION,
) -> str:
    stop_tokens_json = json.dumps(list(stop_tokens))
    prompt_prefix_json = json.dumps(prompt_prefix)
    prompt_suffix_json = json.dumps(prompt_suffix)
    recommended_litert_output_dir = paths.release_dir / f"litert_prefill{prefill_length}"
    return f"""# LiteRT Export Commands

Prepared adapter: `{label}`

Base model: `{base_model}`

Merged Hugging Face model:

```text
{paths.merged_model_dir}
```

The local preparation step writes a full merged Transformers model. Run the
actual LiteRT Torch export in a Linux Python 3.12 environment with
`litert-torch` installed.

## Docker/Colima Export

Build the reusable Linux export image:

```bash
docker build --platform linux/amd64 \\
  -t gemma-litert-export:0.9.0 \\
  -f docker/litert-export/Dockerfile \\
  docker/litert-export
```

Then run the export from the repo root:

```bash
docker run --rm --platform linux/amd64 \\
  -v "$PWD":/workspace \\
  -w /workspace \\
  gemma-litert-export:0.9.0 \\
  python -m litert_torch.generative.export_hf \\
    {paths.merged_model_dir} \\
    {recommended_litert_output_dir} \\
    --prefill_lengths={prefill_length}, \\
    --cache_length={cache_length} \\
    --quantization_recipe={quantization} \\
    --externalize_embedder=True \\
    --use_jinja_template=True \\
    --bundle_litert_lm=True \\
    --trust_remote_code=True
```

## Export `.litertlm` Without Docker

```bash
python -m litert_torch.generative.export_hf \\
  {paths.merged_model_dir} \\
  {recommended_litert_output_dir} \\
  --prefill_lengths={prefill_length}, \\
  --cache_length={cache_length} \\
  --quantization_recipe={quantization} \\
  --externalize_embedder=True \\
  --use_jinja_template=True \\
  --bundle_litert_lm=True \\
  --trust_remote_code=True
```

## Prompt Formatting Defaults

Use the Gemma chat template when possible. If a downstream bundler requires
explicit strings, start with:

```json
{{
  "prompt_prefix": {prompt_prefix_json},
  "prompt_suffix": {prompt_suffix_json},
  "stop_tokens": {stop_tokens_json}
}}
```

## Expected Outputs

The export directory should contain the LiteRT-LM model payload, typically
including a `model.litertlm` file. Validate output quality against the fixed
holdout before treating the artifact as app-ready.
"""


def build_readiness_manifest(
    *,
    promoted: PromotedAdapter,
    paths: ReleasePaths,
    base_model_path_used: str,
    litert_torch_available: bool,
    parity_check_requested: bool,
    merged_model_written: bool,
    prefill_length: int = DEFAULT_PREFILL_LENGTH,
) -> dict[str, Any]:
    recommended_litert_output_dir = paths.release_dir / f"litert_prefill{prefill_length}"
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "litert_ready_inputs_prepared",
        "label": promoted.label,
        "base_model": promoted.base_model,
        "base_model_path_used": base_model_path_used,
        "promoted_pointer": str(promoted.pointer_path),
        "source_checkpoint": str(promoted.checkpoint_path),
        "source_adapter_config": str(promoted.adapter_config_path),
        "artifacts": {
            "release_dir": str(paths.release_dir),
            "peft_adapter_dir": str(paths.peft_adapter_dir),
            "merged_hf_model_dir": str(paths.merged_model_dir),
            "litert_output_dir": str(paths.litert_output_dir),
            "recommended_litert_output_dir": str(recommended_litert_output_dir),
            "litert_command_doc": str(paths.command_doc_path),
        },
        "parity_check_requested": parity_check_requested,
        "merged_model_written": merged_model_written,
        "litert_torch_available": litert_torch_available,
        "notes": [
            "LiteRT export is a separate Linux-runtime step.",
            "Validate generated summaries on the fixed holdout before app promotion.",
        ],
    }


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the currently promoted MLX LoRA adapter as a merged "
            "Transformers model ready for LiteRT Torch export."
        )
    )
    parser.add_argument("--promoted-pointer", default=DEFAULT_PROMOTED_POINTER)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--runtime-python", default=DEFAULT_DPO_PYTHON)
    parser.add_argument("--mlx-python", default=DEFAULT_MLX_PYTHON)
    parser.add_argument("--hf-home", default=DEFAULT_HF_HOME)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-shard-size", default=DEFAULT_MAX_SHARD_SIZE)
    parser.add_argument("--prefill-length", type=int, default=DEFAULT_PREFILL_LENGTH)
    parser.add_argument("--cache-length", type=int, default=DEFAULT_CACHE_LENGTH)
    parser.add_argument("--quantization", default=DEFAULT_QUANTIZATION)
    parser.add_argument(
        "--run-parity-check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the existing MLX-vs-Transformers parity check during PEFT "
            "conversion. This requires a working MLX/Metal runtime."
        ),
    )
    parser.add_argument(
        "--merge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge the PEFT adapter into full base weights for LiteRT export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty release directory.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    promoted = load_promoted_pointer(Path(args.promoted_pointer))
    paths = build_release_paths(Path(args.output_root), promoted.label)
    model_path_used = resolve_model_path(
        model_path=args.model_path,
        hf_home=Path(args.hf_home),
        repo_id=promoted.base_model,
    )
    litert_torch_available = module_available(Path(args.runtime_python), "litert_torch")

    print(f"[prepare-litert] promoted label: {promoted.label}")
    print(f"[prepare-litert] checkpoint: {promoted.checkpoint_path}")
    print(f"[prepare-litert] base model path: {model_path_used}")
    print(f"[prepare-litert] output: {paths.release_dir}")

    ensure_release_dir(paths.release_dir, overwrite=args.overwrite)
    convert_promoted_adapter(
        runtime_python=Path(args.runtime_python),
        mlx_python=Path(args.mlx_python),
        promoted=promoted,
        paths=paths,
        hf_home=Path(args.hf_home),
        model_path=args.model_path,
        run_parity_check=bool(args.run_parity_check),
        overwrite=bool(args.overwrite),
    )

    merged_model_written = False
    if args.merge:
        merge_peft_adapter(
            runtime_python=Path(args.runtime_python),
            model_path_used=model_path_used,
            peft_adapter_dir=paths.peft_adapter_dir,
            output_dir=paths.merged_model_dir,
            max_shard_size=str(args.max_shard_size),
        )
        merged_model_written = True

    command_doc = render_litert_command_doc(
        paths=paths,
        base_model=promoted.base_model,
        label=promoted.label,
        prompt_prefix=DEFAULT_PROMPT_PREFIX,
        prompt_suffix=DEFAULT_PROMPT_SUFFIX,
        stop_tokens=DEFAULT_STOP_TOKENS,
        prefill_length=int(args.prefill_length),
        cache_length=int(args.cache_length),
        quantization=str(args.quantization),
    )
    write_text(paths.command_doc_path, command_doc)

    manifest = build_readiness_manifest(
        promoted=promoted,
        paths=paths,
        base_model_path_used=model_path_used,
        litert_torch_available=litert_torch_available,
        parity_check_requested=bool(args.run_parity_check),
        merged_model_written=merged_model_written,
        prefill_length=int(args.prefill_length),
    )
    write_text(paths.manifest_path, json_dumps(manifest))

    print(f"[prepare-litert] wrote {paths.command_doc_path}")
    print(f"[prepare-litert] wrote {paths.manifest_path}")
    if not litert_torch_available:
        print(
            "[prepare-litert] litert_torch is not installed in the runtime; "
            "use the generated command doc in a Linux LiteRT export environment."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[prepare-litert] error: {exc}", file=sys.stderr)
        raise
