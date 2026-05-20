#!/usr/bin/env python3
"""Generate eval summaries with a Transformers base model plus optional PEFT adapter."""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dpo_utils import (
    DEFAULT_INPUT_MODE,
    DEFAULT_MLX_HF_HOME,
    DEFAULT_MODEL_REPO_ID,
    DEFAULT_PROMPT_SPEC,
    build_prompt_messages,
    load_jsonl,
    load_prompt_spec,
    normalize_summary,
    resolve_model_path,
    write_jsonl,
)

try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - import guard for runtime setup.
    raise SystemExit(
        "This script needs the PEFT/Transformers runtime packages."
    ) from exc


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate single-variant eval outputs with Transformers/PEFT."
    )
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--base-model", default=DEFAULT_MODEL_REPO_ID)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--hf-home", default=DEFAULT_MLX_HF_HOME)
    parser.add_argument("--prompt-spec", default=DEFAULT_PROMPT_SPEC)
    parser.add_argument(
        "--input-mode",
        choices=("abstract_only", "title_abstract"),
        default=DEFAULT_INPUT_MODE,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--model-dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


def render_prompts(
    *,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    prompt_spec: dict[str, Any],
    input_mode: str,
) -> tuple[list[str], list[str]]:
    rendered: list[str] = []
    user_prompts: list[str] = []
    for row in rows:
        messages = build_prompt_messages(row, prompt_spec, input_mode)
        user_prompts.append(messages[-1]["content"])
        if getattr(tokenizer, "chat_template", None):
            rendered.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        else:
            rendered.append("\n\n".join(message["content"] for message in messages))
    return rendered, user_prompts


def generate_outputs(
    *,
    rows: list[dict[str, Any]],
    model_path: str,
    adapter_dir: Path | None,
    prompt_spec: dict[str, Any],
    input_mode: str,
    batch_size: int,
    max_new_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[list[dict[str, str]], list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[transformers-generate] loading base model from {model_path}")
    print(f"[transformers-generate] dtype={dtype} device={device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
    )
    if adapter_dir is not None:
        print(f"[transformers-generate] loading adapter from {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    else:
        print("[transformers-generate] running without adapter")

    model.eval()
    model.to(device)

    eos_token_id = model.generation_config.eos_token_id
    pad_token_id = (
        model.generation_config.pad_token_id
        if model.generation_config.pad_token_id is not None
        else tokenizer.pad_token_id
    )

    outputs: list[dict[str, str]] = []
    all_user_prompts: list[str] = []
    try:
        for batch_start in range(0, len(rows), batch_size):
            batch_rows = rows[batch_start : batch_start + batch_size]
            rendered_prompts, user_prompts = render_prompts(
                tokenizer=tokenizer,
                rows=batch_rows,
                prompt_spec=prompt_spec,
                input_mode=input_mode,
            )
            all_user_prompts.extend(user_prompts)
            encoded = tokenizer(
                rendered_prompts,
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
            for row_index in range(generated.shape[0]):
                continuation = generated[row_index, prompt_width:]
                raw = tokenizer.decode(continuation, skip_special_tokens=True)
                outputs.append({"raw": raw, "normalized": normalize_summary(raw)})

            print(
                f"[transformers-generate] generated {len(outputs)}/{len(rows)} summaries...",
                flush=True,
            )
    finally:
        del model
        del tokenizer
        gc.collect()

    return outputs, all_user_prompts


def build_generation_rows(
    *,
    rows: list[dict[str, Any]],
    outputs: list[dict[str, str]],
    user_prompts: list[str],
    label: str,
    prompt_version: str,
    input_mode: str,
) -> list[dict[str, Any]]:
    generation_rows: list[dict[str, Any]] = []
    for row, output, user_prompt in zip(rows, outputs, user_prompts, strict=True):
        generation_rows.append(
            {
                "eval_id": row.get("eval_id"),
                "split": row.get("split"),
                "paper_id": str(row["paper_id"]),
                "title": str(row["title"]),
                "abstract": str(row["abstract"]),
                "primary_category": row.get("primary_category"),
                "umbrella_category": row.get("umbrella_category"),
                "categories": row.get("categories", []),
                "created_date": row.get("created_date"),
                "prompt_version": prompt_version,
                "input_mode": input_mode,
                "user_prompt": user_prompt,
                "model_label": label,
                "summary_raw": output["raw"],
                "summary": output["normalized"],
            }
        )
    return generation_rows


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    rows = load_jsonl(args.eval_jsonl)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    prompt_spec = load_prompt_spec(Path(args.prompt_spec))
    model_path = resolve_model_path(
        model_path=args.model_path,
        hf_home=Path(args.hf_home),
        repo_id=args.base_model,
    )
    device = select_device(args.device)
    dtype = DTYPES[args.model_dtype]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs, user_prompts = generate_outputs(
        rows=rows,
        model_path=model_path,
        adapter_dir=args.adapter_dir,
        prompt_spec=prompt_spec,
        input_mode=args.input_mode,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        dtype=dtype,
        device=device,
    )
    generation_rows = build_generation_rows(
        rows=rows,
        outputs=outputs,
        user_prompts=user_prompts,
        label=args.label,
        prompt_version=str(prompt_spec["prompt_version"]),
        input_mode=args.input_mode,
    )
    write_jsonl(args.output_dir / "generations.jsonl", generation_rows)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "scripts/run_transformers_peft_eval_generations.py",
        "eval_jsonl": str(args.eval_jsonl),
        "sample_size": len(rows),
        "label": args.label,
        "base_model": args.base_model,
        "base_model_path": model_path,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "prompt_spec": args.prompt_spec,
        "prompt_version": prompt_spec["prompt_version"],
        "input_mode": args.input_mode,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "model_dtype": args.model_dtype,
        "device": str(device),
        "outputs": {
            "generations_jsonl": str(args.output_dir / "generations.jsonl"),
            "manifest_json": str(args.output_dir / "manifest.json"),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[transformers-generate] wrote {len(generation_rows)} generations to {args.output_dir}")


if __name__ == "__main__":
    main()
