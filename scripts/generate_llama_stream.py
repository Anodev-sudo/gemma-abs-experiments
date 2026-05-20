#!/usr/bin/env python3
"""Stream llama.cpp generations for JSONL paper rows with resume support."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SYSTEM_PROMPT = (
    "You write one-sentence paper summaries for a mobile research feed. "
    "Keep them concrete, readable, and useful. Output only the summary sentence."
)
USER_TEMPLATE = (
    "The app already shows the title, author list, and date. "
    "Write one sentence that tells a reader why they might open this paper. "
    "Aim for 10 to 15 words. Avoid repeating the title. "
    'Avoid filler like "This paper" or "We propose". '
    "Do not mention authors or dates. "
    "Keep the wording natural and specific.\n\n"
    "Title:\n{title}\n\n"
    "Abstract:\n{abstract}"
)


def load_prompt_spec(path: str | None) -> dict | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_prompt(title: str, abstract: str, prompt_spec: dict | None = None, input_mode: str = "title_abstract") -> str:
    if prompt_spec:
        templates = prompt_spec.get("user_prompt_templates", {})
        if input_mode not in templates:
            raise ValueError(f"Prompt spec does not define input mode {input_mode!r}")
        system = str(prompt_spec["system_prompt"]).strip()
        user = str(templates[input_mode]).format(title=title, abstract=abstract)
    else:
        system = SYSTEM_PROMPT
        user = USER_TEMPLATE.format(title=title, abstract=abstract)
    return f"<bos><start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n"


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("ok") and d.get("paper_id"):
                done.add(d["paper_id"])
    return done


def start_server(model: str, lora: str, port: int, n_ctx: int) -> subprocess.Popen:
    cmd = [
        "llama-server",
        "-m", model,
        "--lora", lora,
        "--port", str(port),
        "--ctx-size", str(n_ctx),
        "--n-gpu-layers", "99",
        "--host", "127.0.0.1",
        "--no-webui",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_server(port: int, timeout: float = 60.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("llama-server did not become healthy")


def generate(port: int, prompt: str, max_tokens: int, temperature: float) -> dict:
    payload = json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["<end_of_turn>", "<eos>"],
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--ctx-size", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--prompt-spec", default=None)
    ap.add_argument("--input-mode", default="title_abstract")
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    print(f"[generate] resume: {len(done)} completed rows in {out_path}")
    prompt_spec = load_prompt_spec(args.prompt_spec)

    proc = start_server(args.model, args.lora, args.port, args.ctx_size)
    start = time.time()
    try:
        wait_for_server(args.port)
        print(f"[generate] server ready on port {args.port}")
        total = 0
        written = 0
        skipped = 0
        with in_path.open() as inp, out_path.open("a") as out:
            for line in inp:
                if args.limit is not None and total >= args.limit:
                    break
                total += 1
                paper = json.loads(line)
                pid = paper["paper_id"]
                if pid in done:
                    skipped += 1
                    continue
                prompt = build_prompt(
                    paper["title"],
                    paper["abstract"],
                    prompt_spec=prompt_spec,
                    input_mode=args.input_mode,
                )
                t0 = time.time()
                try:
                    resp = generate(args.port, prompt, args.max_tokens, args.temperature)
                    raw = resp["choices"][0]["text"].strip()
                    norm = " ".join(raw.split())
                    row = {
                        "eval_id": paper.get("eval_id", ""),
                        "paper_id": pid,
                        "label": args.label,
                        "ok": True,
                        "raw_output": raw,
                        "normalized_output": norm,
                        "runtime": "llama_cpp_server",
                        "usage": resp.get("usage", {}),
                        "wall_time_ms": round((time.time() - t0) * 1000, 3),
                    }
                except Exception as exc:
                    row = {
                        "eval_id": paper.get("eval_id", ""),
                        "paper_id": pid,
                        "label": args.label,
                        "ok": False,
                        "raw_output": "",
                        "normalized_output": "",
                        "runtime": "llama_cpp_server",
                        "error": str(exc),
                        "wall_time_ms": round((time.time() - t0) * 1000, 3),
                    }
                out.write(json.dumps(row) + "\n")
                out.flush()
                written += 1
                done.add(pid)
                if (written + skipped) % args.progress_every == 0:
                    rate = (written + skipped) / max(time.time() - start, 1e-6)
                    print(f"[generate] processed={written+skipped} written={written} skipped={skipped} rate={rate:.2f}/s last={pid}")
        print(f"[generate] done total_seen={total} written={written} skipped={skipped} output={out_path}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
