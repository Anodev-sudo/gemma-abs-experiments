#!/usr/bin/env python3
"""Convert and evaluate q5 LoRA checkpoints through llama.cpp runtime."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

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

STRESS_IDS = {"1901.06548", "2111.10635", "2303.09722", "2406.04373"}
DIGIT_RE = re.compile(r"\d")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9@+'-]*")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def output_dir(root: Path, eval_name: str) -> Path:
    return root / "eval" / f"{eval_name}_outputs"


def output_path(root: Path, eval_name: str, label: str, row_count: int) -> Path:
    return output_dir(root, eval_name) / f"{label}_outputs_{row_count}.jsonl"


def checkpoint_dir(root: Path, step: int) -> Path:
    return root / f"checkpoint-{step}"


def lora_path(root: Path, step: int) -> Path:
    return root / "gguf_adapter/checkpoints" / f"checkpoint_{step}_lora.f16.gguf"


def convert_peft_dir(root: Path, peft_dir: Path, label: str, python: str, force: bool) -> Path:
    out = root / "gguf_adapter/checkpoints" / f"{label}_lora.f16.gguf"
    if out.exists() and not force:
        print(f"[runtime-sweep] reusing {out}")
        return out
    if not peft_dir.exists():
        raise FileNotFoundError(f"Missing PEFT adapter dir: {peft_dir}")
    out.parent.mkdir(parents=True, exist_ok=True)
    log_dir = root / "eval/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"convert_{label}.log"
    cmd = [
        python,
        str(CONVERTER),
        "--base",
        str(BASE_CONFIG),
        "--outtype",
        "f16",
        "--outfile",
        str(out),
        str(peft_dir),
    ]
    print(f"[runtime-sweep] converting {label}; log={log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    return out


def convert_checkpoint(root: Path, step: int, python: str, force: bool) -> Path:
    return convert_peft_dir(root, checkpoint_dir(root, step), f"checkpoint_{step}", python, force)


def run_generation(
    *,
    python: str,
    eval_input: Path,
    output: Path,
    lora: Path,
    label: str,
    port: int,
    force: bool,
    prompt_spec: Path | None,
    input_mode: str,
    max_tokens: int,
) -> None:
    if output.exists() and force:
        output.unlink()
    if output.exists():
        ok_rows = [row for row in load_jsonl(output) if row.get("ok")]
        expected = len(load_jsonl(eval_input))
        if len(ok_rows) >= expected:
            print(f"[runtime-sweep] reusing {output}")
            return
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        "scripts/generate_llama_stream.py",
        "--input",
        str(eval_input),
        "--output",
        str(output),
        "--model",
        str(RUNTIME_BASE),
        "--lora",
        str(lora),
        "--label",
        label,
        "--port",
        str(port),
        "--input-mode",
        input_mode,
        "--max-tokens",
        str(max_tokens),
        "--progress-every",
        "25",
    ]
    if prompt_spec is not None:
        cmd.extend(["--prompt-spec", str(prompt_spec)])
    print(f"[runtime-sweep] generating {label} on {eval_input}")
    subprocess.run(cmd, check=True)


def has_repeated_ngram(text: str, n: int = 3, max_count: int = 3) -> bool:
    words = [w.lower() for w in WORD_RE.findall(text)]
    if len(words) < n * max_count:
        return False
    counts: dict[tuple[str, ...], int] = {}
    for idx in range(len(words) - n + 1):
        gram = tuple(words[idx : idx + n])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= max_count:
            return True
    return False


def summarize_outputs(label: str, output: Path, inputs: list[dict[str, Any]]) -> dict[str, Any]:
    input_by_pid = {str(row["paper_id"]): row for row in inputs}
    result_rows = []
    for row in load_jsonl(output):
        pid = str(row.get("paper_id", ""))
        source = input_by_pid.get(pid, {})
        text = norm(row.get("normalized_output") or row.get("raw_output"))
        target = norm(source.get("target_summary", ""))
        word_count = len(WORD_RE.findall(text))
        flags: list[str] = []
        if not row.get("ok"):
            flags.append("not_ok")
        if not text:
            flags.append("empty")
        if DIGIT_RE.search(text):
            flags.append("has_digit")
        if word_count < 10:
            flags.append("under10")
        if word_count > 15:
            flags.append("over15")
        if word_count > 35:
            flags.append("too_long_malformed")
        if has_repeated_ngram(text):
            flags.append("repeated_ngram")
        exact_target = bool(target) and text.casefold() == target.casefold()
        result_rows.append(
            {
                "paper_id": pid,
                "eval_id": source.get("eval_id", ""),
                "split": source.get("split", ""),
                "ok": bool(row.get("ok")),
                "output": text,
                "target": target,
                "word_count": word_count,
                "has_digit": "has_digit" in flags,
                "exact_target": exact_target,
                "flags": flags,
            }
        )
    return {
        "label": label,
        "output_path": str(output),
        "rows": result_rows,
        "ok_rows": sum(1 for row in result_rows if row["ok"]),
        "digit_rows": sum(1 for row in result_rows if row["has_digit"]),
        "exact_target_rows": sum(1 for row in result_rows if row["exact_target"]),
        "malformed_rows": sum(1 for row in result_rows if any(flag in row["flags"] for flag in ("not_ok", "empty", "too_long_malformed", "repeated_ngram"))),
    }


def hard77_gate(summary: dict[str, Any]) -> dict[str, Any]:
    rows = {row["paper_id"]: row for row in summary["rows"]}
    stress_digit_free = all(pid in rows and not rows[pid]["has_digit"] for pid in STRESS_IDS)
    p2303 = rows.get("2303.09722")
    p2303_ok = bool(p2303 and not p2303["has_digit"] and p2303["exact_target"])
    pass_gate = (
        p2303_ok
        and stress_digit_free
        and int(summary["digit_rows"]) <= 5
        and int(summary["malformed_rows"]) == 0
    )
    return {
        "label": summary["label"],
        "pass": pass_gate,
        "p2303_digit_free_exact": p2303_ok,
        "stress_digit_free": stress_digit_free,
        "digit_rows": summary["digit_rows"],
        "malformed_rows": summary["malformed_rows"],
    }


def write_summary(root: Path, eval_name: str, summaries: list[dict[str, Any]], gates: list[dict[str, Any]]) -> None:
    eval_root = root / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)
    summary_json = eval_root / f"{eval_name}_summary.json"
    summary_md = eval_root / f"{eval_name.upper()}_SUMMARY.md"
    selected = next((gate for gate in gates if gate["pass"]), None)
    summary_json.write_text(
        json.dumps({"selected_gate": selected, "gates": gates, "checkpoints": summaries}, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [f"# {eval_name} runtime summary", "", "| label | digit rows | exact target rows | malformed rows | gate |", "|---|---:|---:|---:|---|"]
    gate_by_label = {gate["label"]: gate for gate in gates}
    for summary in summaries:
        gate = gate_by_label.get(summary["label"])
        gate_text = "pass" if gate and gate["pass"] else "fail" if gate else "-"
        lines.append(
            f"| {summary['label']} | {summary['digit_rows']} | {summary['exact_target_rows']} | {summary['malformed_rows']} | {gate_text} |"
        )
    if selected:
        lines.extend(["", f"Selected earliest passing checkpoint: `{selected['label']}`"])
    lines.append("")
    for summary in summaries:
        lines.append(f"## {summary['label']}")
        for row in summary["rows"]:
            if row["flags"] or row["paper_id"] in STRESS_IDS:
                lines.append(f"- `{row['paper_id']}` flags={row['flags']} exact={row['exact_target']}: {row['output']}")
        lines.append("")
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[runtime-sweep] wrote {summary_json}")
    print(f"[runtime-sweep] wrote {summary_md}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate q5 checkpoints through llama.cpp.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--eval-input", type=Path, required=True)
    parser.add_argument("--eval-name", required=True)
    parser.add_argument("--steps", nargs="*", type=int, default=[])
    parser.add_argument("--external-lora", type=Path, default=None)
    parser.add_argument("--external-label", default="external")
    parser.add_argument("--include-final", action="store_true", help="Also convert and evaluate the final adapter saved at --root.")
    parser.add_argument("--final-label", default="final")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--port", type=int, default=8120)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--hard77-gate", action="store_true")
    parser.add_argument("--prompt-spec", type=Path, default=None)
    parser.add_argument("--input-mode", default="title_abstract")
    parser.add_argument("--max-tokens", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = load_jsonl(args.eval_input)
    row_count = len(inputs)
    summaries: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []

    if args.external_lora:
        out = output_path(args.root, args.eval_name, args.external_label, row_count)
        run_generation(
            python=args.python,
            eval_input=args.eval_input,
            output=out,
            lora=args.external_lora,
            label=args.external_label,
            port=args.port,
            force=args.force,
            prompt_spec=args.prompt_spec,
            input_mode=args.input_mode,
            max_tokens=args.max_tokens,
        )
        summary = summarize_outputs(args.external_label, out, inputs)
        summaries.append(summary)
        if args.hard77_gate:
            gates.append(hard77_gate(summary))

    if args.include_final:
        label = str(args.final_label)
        lora = convert_peft_dir(args.root, args.root, label, args.python, args.force)
        out = output_path(args.root, args.eval_name, label, row_count)
        run_generation(
            python=args.python,
            eval_input=args.eval_input,
            output=out,
            lora=lora,
            label=label,
            port=args.port + (1 if args.external_lora else 0),
            force=args.force,
            prompt_spec=args.prompt_spec,
            input_mode=args.input_mode,
            max_tokens=args.max_tokens,
        )
        summary = summarize_outputs(label, out, inputs)
        summaries.append(summary)
        if args.hard77_gate:
            gates.append(hard77_gate(summary))

    for index, step in enumerate(args.steps):
        label = f"checkpoint_{step}"
        lora = convert_checkpoint(args.root, step, args.python, args.force)
        out = output_path(args.root, args.eval_name, label, row_count)
        run_generation(
            python=args.python,
            eval_input=args.eval_input,
            output=out,
            lora=lora,
            label=label,
            port=args.port + index + (1 if args.external_lora else 0) + (1 if args.include_final else 0),
            force=args.force,
            prompt_spec=args.prompt_spec,
            input_mode=args.input_mode,
            max_tokens=args.max_tokens,
        )
        summary = summarize_outputs(label, out, inputs)
        summaries.append(summary)
        if args.hard77_gate:
            gates.append(hard77_gate(summary))

    write_summary(args.root, args.eval_name, summaries, gates)


if __name__ == "__main__":
    main()
