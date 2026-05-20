#!/usr/bin/env python3
"""Convert and evaluate q5 four-stress overfit diagnostic checkpoints."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_four_stress_overfit_diag_v1_lr2e6")
DEFAULT_EVAL_INPUT = DEFAULT_ROOT / "dataset_repeats50_fulltarget_weighted_v1/eval_input_4.jsonl"

CONVERTER = Path(
    "hf_bucket/experiments/mlx_lora/runtime_shift/"
    "llama_cpp_q4_lora_sidecar_benchmark_20260507/"
    "llama_cpp_b9020_src/convert_lora_to_gguf.py"
)
BASE_CONFIG = Path(
    "hf_bucket/experiments/mlx_lora/.hf-home/hub/models--google--gemma-3-270m-it/"
    "snapshots/ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"
)
RUNTIME_BASE = Path(
    "hf_bucket/experiments/mlx_lora/runtime_shift/"
    "llama_cpp_q4_lora_sidecar_benchmark_20260507/"
    "gguf_base/gemma_3_270m_it.Q5_K_M.gguf"
)

DIGIT_RE = re.compile(r"\d")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def checkpoint_dir(root: Path, step: int) -> Path:
    return root / f"checkpoint-{step}"


def lora_path(root: Path, step: int) -> Path:
    return root / "gguf_adapter/checkpoints" / f"checkpoint_{step}_lora.f16.gguf"


def output_path(root: Path, step: int) -> Path:
    return root / "eval/four_stress_outputs" / f"checkpoint_{step}_outputs_4.jsonl"


def convert_checkpoint(root: Path, step: int, python: str, force: bool) -> None:
    out = lora_path(root, step)
    if out.exists() and not force:
        print(f"[four-stress-eval] reusing {out}")
        return
    ckpt = checkpoint_dir(root, step)
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        str(CONVERTER),
        "--base",
        str(BASE_CONFIG),
        "--outtype",
        "f16",
        "--outfile",
        str(out),
        str(ckpt),
    ]
    log_out = root / "eval/logs"
    log_out.mkdir(parents=True, exist_ok=True)
    log_path = log_out / f"convert_four_stress_{step}.log"
    print(f"[four-stress-eval] converting checkpoint-{step}; log={log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)


def run_runtime_eval(root: Path, eval_input: Path, step: int, python: str, port: int, force: bool) -> None:
    out = output_path(root, step)
    if out.exists() and force:
        out.unlink()
    if out.exists():
        done = {row.get("paper_id") for row in load_jsonl(out) if row.get("ok")}
        if len(done) >= 4:
            print(f"[four-stress-eval] reusing {out}")
            return
    cmd = [
        python,
        "scripts/generate_llama_stream.py",
        "--input",
        str(eval_input),
        "--output",
        str(out),
        "--model",
        str(RUNTIME_BASE),
        "--lora",
        str(lora_path(root, step)),
        "--label",
        f"four_stress_checkpoint_{step}",
        "--port",
        str(port),
        "--progress-every",
        "1",
    ]
    print(f"[four-stress-eval] running llama.cpp eval for checkpoint-{step}")
    subprocess.run(cmd, check=True)


def summarize_step(root: Path, step: int, targets: dict[str, str]) -> dict[str, Any]:
    rows = load_jsonl(output_path(root, step))
    result_rows = []
    for row in rows:
        pid = row.get("paper_id", "")
        output = str(row.get("normalized_output", "")).strip()
        target = targets.get(pid, "")
        result_rows.append(
            {
                "paper_id": pid,
                "ok": bool(row.get("ok")),
                "output": output,
                "target": target,
                "has_digit": bool(DIGIT_RE.search(output)),
                "word_count": len(WORD_RE.findall(output)),
                "exact_target": output.casefold() == target.casefold(),
            }
        )
    return {
        "step": step,
        "rows": result_rows,
        "ok_rows": sum(1 for row in result_rows if row["ok"]),
        "digit_rows": sum(1 for row in result_rows if row["has_digit"]),
        "exact_target_rows": sum(1 for row in result_rows if row["exact_target"]),
    }


def write_summary(root: Path, summaries: list[dict[str, Any]]) -> None:
    summary_json = root / "eval/four_stress_summary.json"
    summary_md = root / "eval/FOUR_STRESS_TAKEAWAY.md"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps({"checkpoints": summaries}, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Four-Stress Overfit Diagnostic",
        "",
        "Diagnostic only; not a promotion candidate.",
        "",
        "| checkpoint | digit rows | exact target rows | outputs |",
        "|---:|---:|---:|---|",
    ]
    for summary in summaries:
        outputs = "<br>".join(
            f"`{row['paper_id']}`: {row['output']}" for row in summary["rows"]
        )
        lines.append(
            f"| {summary['step']} | {summary['digit_rows']} | "
            f"{summary['exact_target_rows']} | {outputs} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[four-stress-eval] wrote {summary_json}")
    print(f"[four-stress-eval] wrote {summary_md}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate four-stress overfit checkpoints.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--eval-input", type=Path, default=DEFAULT_EVAL_INPUT)
    parser.add_argument("--steps", nargs="+", type=int, default=[10, 20, 50, 100])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = {row["paper_id"]: row["target_summary"] for row in load_jsonl(args.eval_input)}
    summaries = []
    for index, step in enumerate(args.steps):
        convert_checkpoint(args.root, step, args.python, args.force)
        run_runtime_eval(args.root, args.eval_input, step, args.python, args.port + index, args.force)
        summaries.append(summarize_step(args.root, step, targets))
    write_summary(args.root, summaries)


if __name__ == "__main__":
    main()
