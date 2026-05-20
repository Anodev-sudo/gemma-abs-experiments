#!/usr/bin/env python3
"""Generate summaries from a Transformers/PEFT checkpoint for eval JSONL rows."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dpo_utils import resolve_model_path
from generate_llama_stream import build_prompt, load_prompt_spec


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("ok") and row.get("eval_id"):
                done.add(str(row["eval_id"]))
    return done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--base-model", default="google/gemma-3-270m-it")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--hf-home", type=Path, default=Path("hf_bucket/experiments/mlx_lora/.hf-home"))
    parser.add_argument("--prompt-spec", type=Path, default=Path("prompts/mlx_lora/gemma_270m_it_abstract_to_glanceable_summary_v1.json"))
    parser.add_argument("--input-mode", default="title_abstract")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def main() -> None:
    args = parse_args()
    if args.force and args.output.exists():
        args.output.unlink()
    rows = load_jsonl(args.input)
    completed = done_ids(args.output)
    todo = [row for row in rows if str(row.get("eval_id", "")) not in completed]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(
        model_path=args.model_path,
        hf_home=args.hf_home,
        repo_id=args.base_model,
    )
    print(f"[peft-generate] model_path={model_path}")
    print(f"[peft-generate] adapter_dir={args.adapter_dir}")
    print(f"[peft-generate] rows={len(rows)} done={len(completed)} todo={len(todo)}")
    if not todo:
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch_dtype(args.dtype),
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    prompt_spec = load_prompt_spec(str(args.prompt_spec))
    end_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = [tokenizer.eos_token_id]
    if isinstance(end_turn_id, int) and end_turn_id >= 0:
        eos_ids.append(end_turn_id)

    start_time = time.time()
    written = 0
    with args.output.open("a", encoding="utf-8") as out:
        for batch_start in range(0, len(todo), args.batch_size):
            batch = todo[batch_start : batch_start + args.batch_size]
            prompts = [
                build_prompt(
                    str(row.get("title", "")),
                    str(row.get("abstract", "")),
                    prompt_spec=prompt_spec,
                    input_mode=args.input_mode,
                )
                for row in batch
            ]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            prompt_width = int(encoded["input_ids"].shape[1])
            batch_t0 = time.time()
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_ids,
                )
            for index, row in enumerate(batch):
                raw = tokenizer.decode(generated[index, prompt_width:], skip_special_tokens=True).strip()
                normalized = " ".join(raw.split())
                out.write(
                    json.dumps(
                        {
                            "eval_id": row.get("eval_id", ""),
                            "paper_id": row.get("paper_id", ""),
                            "label": args.label,
                            "ok": True,
                            "raw_output": raw,
                            "normalized_output": normalized,
                            "runtime": f"transformers_peft_{args.dtype}",
                            "wall_time_ms_batch": round((time.time() - batch_t0) * 1000, 3),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                written += 1
            out.flush()
            total_done = len(completed) + written
            if total_done % args.progress_every == 0 or total_done == len(rows):
                elapsed = max(time.time() - start_time, 1e-6)
                print(f"[peft-generate] completed={total_done}/{len(rows)} rate={written / elapsed:.2f}/s")

    del model, base, tokenizer
    gc.collect()


if __name__ == "__main__":
    main()
