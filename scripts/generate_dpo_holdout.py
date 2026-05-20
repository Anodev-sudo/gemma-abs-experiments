#!/usr/bin/env python3
"""Generate summaries for holdout papers using llama.cpp with a LoRA sidecar."""
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


def build_chat_prompt(title: str, abstract: str) -> str:
    """Build a Gemma-3 chat-formatted prompt from title and abstract."""
    user_text = USER_TEMPLATE.format(title=title, abstract=abstract)
    # Gemma-3 applies system prompt as part of the first user turn
    return (
        "<bos><start_of_turn>user\n"
        f"{SYSTEM_PROMPT}\n\n{user_text}"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def load_papers(path: Path, limit: int) -> list[dict]:
    papers = []
    with open(path) as f:
        for line in f:
            if len(papers) >= limit:
                break
            d = json.loads(line)
            papers.append(d)
    return papers


def start_server(model: str, lora: str, port: int, n_ctx: int = 2048) -> subprocess.Popen:
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
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc


def wait_for_server(port: int, timeout: float = 30.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
            with urllib.request.urlopen(req) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def generate_one(port: int, prompt: str, max_tokens: int = 50, temperature: float = 0.0) -> dict:
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


def main():
    parser = argparse.ArgumentParser(description="Generate DPO holdout summaries via llama.cpp")
    parser.add_argument("--papers", required=True, help="JSONL file with holdout papers")
    parser.add_argument("--limit", type=int, default=100, help="Number of papers to generate")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--model", required=True, help="Path to base GGUF")
    parser.add_argument("--lora", required=True, help="Path to LoRA GGUF sidecar")
    parser.add_argument("--port", type=int, default=8089, help="Port for llama-server")
    parser.add_argument("--label", default="dpo_checkpoint", help="Label for output rows")
    args = parser.parse_args()

    papers = load_papers(Path(args.papers), args.limit)
    print(f"Loaded {len(papers)} papers from {args.papers}")

    print(f"Starting llama-server on port {args.port}...")
    proc = start_server(args.model, args.lora, args.port)
    try:
        if not wait_for_server(args.port, timeout=60):
            print("ERROR: server did not start in time", file=sys.stderr)
            sys.exit(1)
        print("Server is ready.")

        results = []
        for i, paper in enumerate(papers):
            pid = paper["paper_id"]
            title = paper["title"]
            abstract = paper["abstract"]
            prompt = build_chat_prompt(title, abstract)

            try:
                resp = generate_one(args.port, prompt)
                choice = resp["choices"][0]
                raw = choice["text"].strip()
                # Normalize: strip leading/trailing whitespace, collapse multi-line
                normalized = " ".join(raw.split())
                usage = resp.get("usage", {})
                result = {
                    "paper_id": pid,
                    "eval_id": paper.get("eval_id", ""),
                    "label": args.label,
                    "ok": True,
                    "raw_output": raw,
                    "normalized_output": normalized,
                    "runtime": "llama_cpp_server",
                    "usage": usage,
                    "wall_time_ms": 0,
                }
                results.append(result)
                print(f"  [{i+1}/{len(papers)}] {pid}: {normalized[:80]}...")
            except Exception as e:
                print(f"  [{i+1}/{len(papers)}] {pid}: ERROR - {e}", file=sys.stderr)
                results.append({
                    "paper_id": pid,
                    "eval_id": paper.get("eval_id", ""),
                    "label": args.label,
                    "ok": False,
                    "raw_output": "",
                    "normalized_output": "",
                    "runtime": "llama_cpp_server",
                    "error": str(e),
                })

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(results)} results to {out_path}")
    finally:
        print("Stopping server...")
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()