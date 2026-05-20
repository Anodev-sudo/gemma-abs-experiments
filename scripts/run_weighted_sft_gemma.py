#!/usr/bin/env python3
"""Run assistant-only weighted SFT on Gemma 3 270M with PEFT LoRA."""
from __future__ import annotations

import argparse
import gc
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    import yaml
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
except ImportError as exc:  # pragma: no cover
    raise SystemExit("This script needs the DPO runtime packages.") from exc

from run_sft_gemma import DEFAULT_MODEL_REPO_ID, GEMMA_TRAINING_CHAT_TEMPLATE, resolve_model_path, resolve_torch_dtype


class WeightedChatDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features = [self._encode(row) for row in rows]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.features[idx]

    @staticmethod
    def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
        return a0 < b1 and b0 < a1

    def _encode(self, row: dict[str, Any]) -> dict[str, Any]:
        messages = row.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Row {row.get('paper_id')} missing messages")
        summary = str(row.get("summary") or messages[-1].get("content", "")).strip()
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        start = text.rfind(summary)
        if start < 0:
            raise ValueError(f"Could not locate summary in rendered chat for {row.get('paper_id')}: {summary!r}")
        end = start + len(summary)

        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        input_ids = list(enc["input_ids"])
        attention_mask = list(enc["attention_mask"])
        offsets = list(enc["offset_mapping"])
        labels = [-100] * len(input_ids)
        weights = [0.0] * len(input_ids)

        hints = row.get("loss_weight_hints") or {}
        span_hints = []
        if bool(hints.get("enabled", False)):
            for sp in hints.get("spans", []):
                try:
                    span_hints.append((start + int(sp["start"]), start + int(sp["end"]), float(sp["weight"])))
                except Exception:
                    continue

        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end:
                continue
            if not self._overlaps(int(tok_start), int(tok_end), start, end):
                continue
            labels[i] = input_ids[i]
            weight = float(hints.get("base_weight", 1.0)) if hints else 1.0
            for hs, he, hw in span_hints:
                if self._overlaps(int(tok_start), int(tok_end), hs, he):
                    weight = max(weight, hw)
            weights[i] = weight

        if all(x == -100 for x in labels):
            raise ValueError(f"No assistant labels after tokenization for {row.get('paper_id')}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_weights": weights,
            "paper_id": row.get("paper_id", ""),
            "row_type": row.get("weighted_numeric_patch_row_type", ""),
        }


@dataclass
class WeightedCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch: dict[str, list[list[int] | list[float]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "loss_weights": [],
        }
        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
            batch["loss_weights"].append(f["loss_weights"] + [0.0] * pad)
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "loss_weights": torch.tensor(batch["loss_weights"], dtype=torch.float32),
        }


class WeightedLossTrainer(Trainer):
    def __init__(self, *args: Any, weighted_loss_normalization: str = "label_count", **kwargs: Any):
        super().__init__(*args, **kwargs)
        if weighted_loss_normalization not in {"label_count", "weight_sum"}:
            raise ValueError("weighted_loss_normalization must be label_count or weight_sum")
        self.weighted_loss_normalization = weighted_loss_normalization

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        loss_weights = inputs.pop("loss_weights")
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = loss_weights[..., 1:].to(shift_logits.device).contiguous()

        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        valid = shift_labels.ne(-100)
        weighted = token_losses * shift_weights
        if self.weighted_loss_normalization == "weight_sum":
            denom = shift_weights.masked_select(valid).sum().clamp_min(1.0)
        else:
            denom = valid.sum().clamp_min(1).to(weighted.dtype)
        loss = weighted.sum() / denom
        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weighted assistant-only SFT from YAML config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps-override", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected mapping in {path}")
    required = (
        "model", "hf_home", "converted_adapter_dir", "dataset_path", "output_dir", "seed",
        "per_device_train_batch_size", "gradient_accumulation_steps", "num_train_epochs", "learning_rate",
        "warmup_ratio", "max_seq_length", "save_steps", "save_total_limit",
    )
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    return config


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_training_args(config: dict[str, Any], output_dir: str) -> TrainingArguments:
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
    if args.resume_from_checkpoint is not None:
        config["resume_from_checkpoint"] = args.resume_from_checkpoint

    model_path = resolve_model_path(
        config.get("model_path"),
        Path(str(config["hf_home"])),
        str(config.get("model", DEFAULT_MODEL_REPO_ID)),
    )
    dataset_path = Path(str(config["dataset_path"]))
    adapter_config_value = config.get("converted_adapter_dir")
    adapter_dir = Path(str(adapter_config_value)) if adapter_config_value else None
    rows = load_rows(dataset_path)
    print(f"[weighted-sft] loaded {len(rows)} training rows from {dataset_path}")

    print(f"[weighted-sft] loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.chat_template = GEMMA_TRAINING_CHAT_TEMPLATE
    print("[weighted-sft] installed Gemma training chat template")

    dataset = WeightedChatDataset(rows, tokenizer, int(config["max_seq_length"]))
    labeled_tokens = sum(sum(1 for x in f["labels"] if x != -100) for f in dataset.features)
    weighted_tokens = sum(sum(1 for x in f["loss_weights"] if x > 1.0) for f in dataset.features)
    weight_sum = sum(sum(float(x) for x in f["loss_weights"] if x > 0) for f in dataset.features)
    print(
        f"[weighted-sft] encoded rows={len(dataset)} labeled_tokens={labeled_tokens} "
        f"weighted_tokens={weighted_tokens} weight_sum={weight_sum:.1f}"
    )

    print(f"[weighted-sft] loading base model from {model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=resolve_torch_dtype(str(config.get("model_dtype", "float32"))),
    )
    if adapter_dir is not None:
        print(f"[weighted-sft] loading adapter from {adapter_dir}")
        model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=True)
        adapter_initialization: dict[str, Any] = {
            "mode": "from_adapter",
            "converted_adapter_dir": str(adapter_dir),
        }
    else:
        lora_config = config.get("lora") or {}
        target_modules = list(lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]))
        rank = int(lora_config.get("r", lora_config.get("rank", 16)))
        alpha = int(lora_config.get("lora_alpha", lora_config.get("alpha", 160)))
        dropout = float(lora_config.get("lora_dropout", lora_config.get("dropout", 0.0)))
        print(
            "[weighted-sft] initializing fresh LoRA "
            f"target_modules={target_modules} r={rank} alpha={alpha} dropout={dropout}"
        )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias=str(lora_config.get("bias", "none")),
            inference_mode=False,
        )
        model = get_peft_model(base_model, peft_config)
        adapter_initialization = {
            "mode": "fresh_lora",
            "target_modules": target_modules,
            "r": rank,
            "lora_alpha": alpha,
            "lora_dropout": dropout,
            "bias": str(lora_config.get("bias", "none")),
        }
    model.print_trainable_parameters()

    trainer_args = build_training_args(config, str(output_dir))
    trainer = WeightedLossTrainer(
        model=model,
        args=trainer_args,
        train_dataset=dataset,
        data_collator=WeightedCollator(int(tokenizer.pad_token_id)),
        processing_class=tokenizer,
        weighted_loss_normalization=str(config.get("weighted_loss_normalization", "label_count")),
    )

    bs = trainer_args.per_device_train_batch_size * trainer_args.gradient_accumulation_steps
    steps_per_epoch = len(dataset) // bs
    total_steps = steps_per_epoch * int(float(config["num_train_epochs"]))
    print(
        f"[weighted-sft] training rows={len(dataset)}, batch_size={bs}, "
        f"{steps_per_epoch} steps/epoch, ~{total_steps} total steps, "
        f"normalization={config.get('weighted_loss_normalization', 'label_count')}"
    )

    resume_from_checkpoint = config.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        print(f"[weighted-sft] resuming from checkpoint: {resume_from_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None)
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    manifest = {
        "config_path": str(config_path),
        "model": str(config.get("model", DEFAULT_MODEL_REPO_ID)),
        "model_path": str(model_path),
        "model_dtype": str(config.get("model_dtype", "float32")),
        "dataset_path": str(dataset_path),
        "converted_adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
        "adapter_initialization": adapter_initialization,
        "output_dir": str(output_dir),
        "train_rows": len(dataset),
        "weighted_sft": True,
        "weight_stats": {
            "labeled_tokens": labeled_tokens,
            "weighted_tokens_gt_1": weighted_tokens,
            "weight_sum": weight_sum,
            "weighted_loss_normalization": str(config.get("weighted_loss_normalization", "label_count")),
        },
        "hyperparameters": {
            "per_device_train_batch_size": trainer_args.per_device_train_batch_size,
            "gradient_accumulation_steps": trainer_args.gradient_accumulation_steps,
            "num_train_epochs": trainer_args.num_train_epochs,
            "learning_rate": trainer_args.learning_rate,
            "warmup_ratio": trainer_args.warmup_ratio,
            "max_seq_length": int(config["max_seq_length"]),
            "assistant_only_loss": True,
        },
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "train_metrics": train_result.metrics,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if config.get("copy_config_into_output", True):
        shutil.copy2(config_path, output_dir / config_path.name)

    del trainer, model, base_model, tokenizer
    gc.collect()
    print(f"[weighted-sft] done. output: {output_dir}")
    print(f"[weighted-sft] metrics: {train_result.metrics}")


if __name__ == "__main__":
    main()
