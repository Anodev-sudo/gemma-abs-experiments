#!/usr/bin/env python3
"""Run SFT fine-tuning on Gemma 3 270M from a PEFT adapter checkpoint."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path
from typing import Any

try:
    import torch
    import yaml
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTConfig, SFTTrainer
except ImportError as exc:
    raise SystemExit(
        "This script needs the DPO runtime packages."
    ) from exc


DEFAULT_MODEL_REPO_ID = "google/gemma-3-270m-it"

GEMMA_TRAINING_CHAT_TEMPLATE = """{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if message['role'] == 'assistant' -%}
        {{ '<start_of_turn>model\n' }}
        {% generation %}{{ message['content'] | trim }}{% endgeneration %}
        {{ '<end_of_turn>\n' }}
    {%- else -%}
        {{ '<start_of_turn>' + message['role'] + '\n' + (first_user_prefix if loop.first else "") }}
        {%- if message['content'] is string -%}
            {{ message['content'] | trim }}
        {%- elif message['content'] is iterable -%}
            {%- for item in message['content'] -%}
                {%- if item['type'] == 'image' -%}
                    {{ '<start_of_image>' }}
                {%- elif item['type'] == 'text' -%}
                    {{ item['text'] | trim }}
                {%- endif -%}
            {%- endfor -%}
        {%- else -%}
            {{ raise_exception("Invalid content type") }}
        {%- endif -%}
        {{ '<end_of_turn>\n' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model\n'}}
{%- endif -%}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT fine-tuning from a YAML config.")
    parser.add_argument("--config", required=True, help="YAML config.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--max-steps-override", type=int, default=None, help="Max steps override for smoke.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    required = (
        "model", "hf_home", "converted_adapter_dir", "dataset_path",
        "output_dir", "seed", "per_device_train_batch_size",
        "gradient_accumulation_steps", "num_train_epochs", "learning_rate",
        "warmup_ratio", "max_seq_length", "save_steps", "save_total_limit",
    )
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    return config


def resolve_model_path(model_path: str | None, hf_home: Path, repo_id: str) -> str:
    if model_path:
        return model_path
    candidate = sorted(
        (hf_home / "hub" / "models--google--gemma-3-270m-it" / "snapshots").iterdir(),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidate:
        return str(candidate[0])
    return repo_id


def resolve_torch_dtype(name: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    return mapping[name]


def build_training_args(config: dict[str, Any], output_dir: str) -> TrainingArguments | SFTConfig:
    common = dict(
        output_dir=output_dir,
        seed=int(config["seed"]),
        do_train=True,
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        num_train_epochs=float(config["num_train_epochs"]),
        learning_rate=float(config["learning_rate"]),
        warmup_ratio=float(config["warmup_ratio"]),
        logging_steps=int(config.get("logging_steps", 1)),
        save_strategy=str(config.get("save_strategy", "steps")),
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config["save_total_limit"]),
        remove_unused_columns=False,
        report_to=list(config.get("report_to", [])),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        dataloader_num_workers=int(config.get("dataloader_num_workers", 0)),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
        bf16=False,
        fp16=False,
    )
    if "max_steps" in config:
        common["max_steps"] = int(config["max_steps"])
    if config.get("assistant_only_loss", False):
        return SFTConfig(
            **common,
            max_length=int(config["max_seq_length"]),
            assistant_only_loss=True,
            completion_only_loss=False,
            packing=False,
        )
    return TrainingArguments(**common)


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
        config.get("model_path"),
        Path(str(config["hf_home"])),
        str(config.get("model", DEFAULT_MODEL_REPO_ID)),
    )
    dataset_path = Path(str(config["dataset_path"]))
    converted_adapter_dir = Path(str(config["converted_adapter_dir"]))

    # Load dataset
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    print(f"[run-sft] loaded {len(dataset)} training rows from {dataset_path}")

    # Load tokenizer
    print(f"[run-sft] loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if bool(config.get("assistant_only_loss", False)) and "{% generation %}" not in str(tokenizer.chat_template):
        tokenizer.chat_template = GEMMA_TRAINING_CHAT_TEMPLATE
        print("[run-sft] installed Gemma training chat template with assistant generation masks")

    # Load model
    print(f"[run-sft] loading base model from {model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=resolve_torch_dtype(str(config.get("model_dtype", "float32"))),
    )

    print(f"[run-sft] loading adapter from {converted_adapter_dir}")
    model = PeftModel.from_pretrained(base_model, converted_adapter_dir, is_trainable=True)

    assistant_only_loss = bool(config.get("assistant_only_loss", False))
    if assistant_only_loss:
        if "messages" not in dataset.column_names:
            raise ValueError("assistant_only_loss requires a conversational dataset with a messages column")
        print(f"[run-sft] using assistant-only loss over conversational messages")
    else:
        # Format dataset using chat template
        def format_chat(example):
            msgs = example["messages"]
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            return {"text": text}

        dataset = dataset.map(format_chat, remove_columns=dataset.column_names)
        print(f"[run-sft] formatted {len(dataset)} rows with chat template")

    trainer_args = build_training_args(config, str(output_dir))

    trainer = SFTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    n = len(dataset)
    bs = trainer_args.per_device_train_batch_size * trainer_args.gradient_accumulation_steps
    steps_per_epoch = n // bs
    total_steps = steps_per_epoch * int(config["num_train_epochs"])
    print(f"[run-sft] training on {n} rows, batch_size={bs}, {steps_per_epoch} steps/epoch, ~{total_steps} total steps")

    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Save manifest
    manifest = {
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "converted_adapter_dir": str(converted_adapter_dir),
        "output_dir": str(output_dir),
        "train_rows": len(dataset),
        "hyperparameters": {
            "per_device_train_batch_size": trainer_args.per_device_train_batch_size,
            "gradient_accumulation_steps": trainer_args.gradient_accumulation_steps,
            "num_train_epochs": trainer_args.num_train_epochs,
            "learning_rate": trainer_args.learning_rate,
            "warmup_ratio": trainer_args.warmup_ratio,
            "max_seq_length": int(config["max_seq_length"]),
            "assistant_only_loss": bool(config.get("assistant_only_loss", False)),
        },
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "train_metrics": train_result.metrics,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if config.get("copy_config_into_output", True):
        shutil.copy2(config_path, output_dir / config_path.name)

    del trainer, model, base_model, tokenizer
    gc.collect()

    print(f"[run-sft] done. output: {output_dir}")
    print(f"[run-sft] metrics: {train_result.metrics}")


if __name__ == "__main__":
    main()