#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Any

from dpo_utils import DEFAULT_MODEL_REPO_ID, resolve_model_path

try:
    import torch
    import yaml
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer
except ImportError as exc:  # pragma: no cover - import guard for runtime setup
    raise SystemExit(
        "This script needs the DPO runtime packages. Run it with the DPO runtime "
        "Python after that environment is created."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local Gemma DPO pilot from a YAML config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config for the DPO pilot.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the training output directory.",
    )
    parser.add_argument(
        "--max-steps-override",
        type=int,
        default=None,
        help="Optional max_steps override for smoke tests.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    required = (
        "model",
        "hf_home",
        "converted_adapter_dir",
        "dataset_dir",
        "output_dir",
        "seed",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "num_train_epochs",
        "learning_rate",
        "beta",
        "loss_type",
        "warmup_ratio",
        "max_length",
        "eval_steps",
        "save_steps",
        "save_total_limit",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")
    return config


def load_preference_split(path: Path) -> Any:
    return load_dataset("json", data_files=str(path), split="train")


def resolve_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model_dtype {name!r}. Expected one of: {', '.join(sorted(mapping))}."
        ) from exc


def render_prompt_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n\n".join(message["content"] for message in messages)


def validate_prompt_lengths(dataset: Any, tokenizer: Any, max_prompt_length: int | None) -> dict[str, Any]:
    if max_prompt_length is None:
        return {
            "checked_rows": len(dataset),
            "max_prompt_tokens": None,
            "over_limit_count": 0,
        }
    max_seen = 0
    over_limit = 0
    for row in dataset:
        rendered = render_prompt_text(tokenizer, row["prompt"])
        prompt_tokens = tokenizer(
            rendered,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        prompt_length = len(prompt_tokens)
        max_seen = max(max_seen, prompt_length)
        if prompt_length > max_prompt_length:
            over_limit += 1
    if over_limit:
        raise ValueError(
            f"Found {over_limit} prompts over max_prompt_length={max_prompt_length}. "
            f"Largest prompt was {max_seen} tokens."
        )
    return {
        "checked_rows": len(dataset),
        "max_prompt_tokens": max_seen,
        "over_limit_count": 0,
    }


def build_training_args(config: dict[str, Any], output_dir: str) -> DPOConfig:
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "seed": int(config["seed"]),
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(config.get("per_device_eval_batch_size", config["per_device_train_batch_size"])),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "num_train_epochs": float(config["num_train_epochs"]),
        "learning_rate": float(config["learning_rate"]),
        "beta": float(config["beta"]),
        "loss_type": str(config["loss_type"]),
        "warmup_ratio": float(config["warmup_ratio"]),
        "max_length": int(config["max_length"]),
        "logging_steps": int(config.get("logging_steps", 1)),
        "eval_strategy": str(config.get("eval_strategy", "steps")),
        "eval_steps": int(config["eval_steps"]),
        "save_strategy": str(config.get("save_strategy", "steps")),
        "save_steps": int(config["save_steps"]),
        "save_total_limit": int(config["save_total_limit"]),
        "remove_unused_columns": False,
        "report_to": list(config.get("report_to", [])),
        "max_grad_norm": float(config.get("max_grad_norm", 1.0)),
        "dataloader_num_workers": int(config.get("dataloader_num_workers", 0)),
        "disable_dropout": bool(config.get("disable_dropout", True)),
    }
    if "max_steps" in config:
        kwargs["max_steps"] = int(config["max_steps"])
    if "torch_empty_cache_steps" in config:
        kwargs["torch_empty_cache_steps"] = int(config["torch_empty_cache_steps"])
    return DPOConfig(**kwargs)


def write_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_eval_dataset(trainer: Any, dataset: Any, dataset_name: str) -> Any:
    first_row = next(iter(dataset))
    if {"prompt_ids", "chosen_ids", "rejected_ids"}.issubset(first_row.keys()):
        return dataset
    return trainer._prepare_dataset(
        dataset,
        trainer.processing_class,
        trainer.args,
        dataset_name,
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    output_dir = Path(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_steps_override is not None:
        config["max_steps"] = args.max_steps_override
        output_dir = output_dir.parent / f"{output_dir.name}_smoke_{args.max_steps_override}_steps"
        output_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(
        model_path=config.get("model_path"),
        hf_home=Path(str(config["hf_home"])),
        repo_id=str(config.get("model", DEFAULT_MODEL_REPO_ID)),
    )
    dataset_dir = Path(str(config["dataset_dir"]))
    converted_adapter_dir = Path(str(config["converted_adapter_dir"]))

    train_dataset = load_preference_split(dataset_dir / "train.jsonl")
    valid_dataset = load_preference_split(dataset_dir / "valid.jsonl")
    test_dataset = load_preference_split(dataset_dir / "test.jsonl")

    print(f"[run-dpo] loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompt_length_summary = {
        "train": validate_prompt_lengths(
            train_dataset, tokenizer, config.get("max_prompt_length")
        ),
        "valid": validate_prompt_lengths(
            valid_dataset, tokenizer, config.get("max_prompt_length")
        ),
        "test": validate_prompt_lengths(
            test_dataset, tokenizer, config.get("max_prompt_length")
        ),
    }

    print(f"[run-dpo] loading base model from {model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=resolve_torch_dtype(str(config.get("model_dtype", "float32"))),
    )

    print(f"[run-dpo] loading train adapter from {converted_adapter_dir}")
    model = PeftModel.from_pretrained(
        base_model,
        converted_adapter_dir,
        is_trainable=True,
    )

    trainer_args = build_training_args(config, str(output_dir))
    manifest: dict[str, Any] = {
        "config_path": str(config_path),
        "base_model_repo_id": str(config.get("model", DEFAULT_MODEL_REPO_ID)),
        "base_model_path_used": model_path,
        "dataset_dir": str(dataset_dir),
        "converted_adapter_dir": str(converted_adapter_dir),
        "output_dir": str(output_dir),
        "train_rows": len(train_dataset),
        "valid_rows": len(valid_dataset),
        "test_rows": len(test_dataset),
        "hyperparameters": {
            "per_device_train_batch_size": trainer_args.per_device_train_batch_size,
            "gradient_accumulation_steps": trainer_args.gradient_accumulation_steps,
            "num_train_epochs": trainer_args.num_train_epochs,
            "learning_rate": trainer_args.learning_rate,
            "beta": trainer_args.beta,
            "loss_type": trainer_args.loss_type,
            "warmup_ratio": trainer_args.warmup_ratio,
            "max_prompt_length": config.get("max_prompt_length"),
            "max_length": trainer_args.max_length,
            "eval_steps": trainer_args.eval_steps,
            "save_steps": trainer_args.save_steps,
            "save_total_limit": trainer_args.save_total_limit,
            "max_steps": trainer_args.max_steps,
        },
        "device": (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
        "reference_mode": "trl_auto_ref_copy_of_default_adapter",
        "prompt_length_summary": prompt_length_summary,
        "train_metrics": None,
        "valid_metrics": None,
        "test_metrics": None,
    }
    write_run_manifest(output_dir / "run_manifest.json", manifest)

    trainer = DPOTrainer(
        model=model,
        args=trainer_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    print(
        f"[run-dpo] training on {len(train_dataset)} train rows "
        f"with {len(valid_dataset)} valid rows"
    )
    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    manifest["train_metrics"] = train_result.metrics
    manifest["checkpoint_dirs"] = sorted(
        str(path.name)
        for path in output_dir.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-")
    )
    write_run_manifest(output_dir / "run_manifest.json", manifest)

    if config.get("copy_config_into_output", True):
        shutil.copy2(config_path, output_dir / config_path.name)

    if bool(config.get("skip_final_eval", False)):
        print("[run-dpo] skipping final valid/test evaluation")
        valid_metrics = None
        test_metrics = None
    else:
        print("[run-dpo] running final valid/test evaluation")
        valid_metrics = trainer.evaluate(metric_key_prefix="valid")
        prepared_test_dataset = prepare_eval_dataset(trainer, test_dataset, "test")
        test_metrics = trainer.evaluate(eval_dataset=prepared_test_dataset, metric_key_prefix="test")

    manifest["valid_metrics"] = valid_metrics
    manifest["test_metrics"] = test_metrics
    write_run_manifest(output_dir / "run_manifest.json", manifest)

    del trainer
    del model
    del base_model
    del tokenizer
    gc.collect()

    print("[run-dpo] done")
    print(f"[run-dpo] output dir: {output_dir}")
    print(f"[run-dpo] valid metrics: {valid_metrics}")
    print(f"[run-dpo] test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
