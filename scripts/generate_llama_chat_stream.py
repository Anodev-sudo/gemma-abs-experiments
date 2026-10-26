#!/usr/bin/env python3
"""Stream llama.cpp chat-completion generations for JSONL paper rows."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


def load_prompt_spec(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_messages(row: dict[str, Any], prompt_spec: dict[str, Any], input_mode: str) -> list[dict[str, str]]:
    templates = prompt_spec.get("user_prompt_templates", {})
    if input_mode not in templates:
        raise ValueError(f"Prompt spec does not define input mode {input_mode!r}")
    user = str(templates[input_mode]).format(
        title=str(row.get("title", "")),
        abstract=str(row.get("abstract", "")),
    )
    return [
        {"role": "system", "content": str(prompt_spec["system_prompt"]).strip()},
        {"role": "user", "content": user},
    ]


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("ok") and row.get("paper_id"):
            done.add(str(row["paper_id"]))
    return done


def start_server(args: argparse.Namespace) -> subprocess.Popen:
    cmd = [
        "llama-server",
        "-m",
        args.model,
        "--port",
        str(args.port),
        "--ctx-size",
        str(args.ctx_size),
        "--n-gpu-layers",
        str(args.gpu_layers),
        "--host",
        "127.0.0.1",
        "--no-webui",
        "--jinja",
    ]
    if args.chat_template:
        cmd.extend(["--chat-template", args.chat_template])
    cmd.extend(args.server_arg)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_server(port: int, timeout: float = 90.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("llama-server did not become healthy")


def generate(port: int, messages: list[dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": "local",
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=args.request_timeout) as resp:
        return json.loads(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prompt-spec", required=True)
    parser.add_argument("--input-mode", default="title_abstract")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--ctx-size", type=int, default=2048)
    parser.add_argument("--gpu-layers", default="99")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--chat-template", default=None)
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Additional llama-server argument; repeat once per argument.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_spec = load_prompt_spec(Path(args.prompt_spec))
    done = load_done(output_path)
    print(f"[llama-chat] resume: {len(done)} completed rows in {output_path}")

    proc = start_server(args)
    start = time.time()
    try:
        wait_for_server(args.port)
        print(f"[llama-chat] server ready on port {args.port}")
        seen = 0
        written = 0
        skipped = 0
        with input_path.open(encoding="utf-8") as source, output_path.open("a", encoding="utf-8") as out:
            for line in source:
                if args.limit is not None and seen >= args.limit:
                    break
                if not line.strip():
                    continue
                seen += 1
                paper = json.loads(line)
                paper_id = str(paper["paper_id"])
                if paper_id in done:
                    skipped += 1
                    continue
                messages = build_messages(paper, prompt_spec, args.input_mode)
                t0 = time.time()
                try:
                    response = generate(args.port, messages, args)
                    raw = response["choices"][0]["message"]["content"].strip()
                    row = {
                        "eval_id": paper.get("eval_id", ""),
                        "paper_id": paper_id,
                        "title": paper.get("title", ""),
                        "label": args.label,
                        "ok": True,
                        "raw_output": raw,
                        "normalized_output": " ".join(raw.split()),
                        "runtime": "llama_cpp_chat_server",
                        "usage": response.get("usage", {}),
                        "wall_time_ms": round((time.time() - t0) * 1000, 3),
                    }
                except Exception as exc:
                    row = {
                        "eval_id": paper.get("eval_id", ""),
                        "paper_id": paper_id,
                        "title": paper.get("title", ""),
                        "label": args.label,
                        "ok": False,
                        "raw_output": "",
                        "normalized_output": "",
                        "runtime": "llama_cpp_chat_server",
                        "error": str(exc),
                        "wall_time_ms": round((time.time() - t0) * 1000, 3),
                    }
                out.write(json.dumps(row, sort_keys=True) + "\n")
                out.flush()
                done.add(paper_id)
                written += 1
                if (written + skipped) % args.progress_every == 0:
                    rate = (written + skipped) / max(time.time() - start, 1e-6)
                    print(
                        f"[llama-chat] processed={written + skipped} written={written} "
                        f"skipped={skipped} rate={rate:.2f}/s last={paper_id}"
                    )
        print(f"[llama-chat] done seen={seen} written={written} skipped={skipped} output={output_path}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
