#!/usr/bin/env python3
"""Smoke-test a LiteRT-LM text generation bundle.

The default input is the CAbLECAR paper that exposed the q4 deployment
repetition failure. The script prints the raw summary and exits non-zero when
the output is empty or obviously degenerate if --fail-on-degenerate is set.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import litert_lm
from litert_lm import Backend, SamplerConfig


SYSTEM_PROMPT = (
    "You write one-sentence paper summaries for a mobile research feed. "
    "Keep them concrete, readable, and useful. Output only the summary sentence."
)

USER_TEMPLATE = """The app already shows the title, author list, and date. Write one sentence that tells a reader why they might open this paper. Aim for 10 to 15 words. Avoid repeating the title. Avoid filler like "This paper" or "We propose". Do not mention authors or dates. Keep the wording natural and specific.

Title:
{title}

Abstract:
{abstract}"""

DEFAULT_TITLE = (
    "CAbLECAR: efficiently scheduling QLDPC codes on a tileable spin qubit "
    "chip with shuttling"
)

DEFAULT_ABSTRACT = """Semiconductor spin qubits are a promising platform for large-scale quantum computing, but have yet to take full advantage of the broad class of quantum low-density parity check (QLDPC) codes, which promise high encoding rates and efficient logic but require nonlocal connectivity between physical qubits. In this work, we investigate the implementation of QLDPC codes on a tileable, shuttling-based spin qubit architecture. By tailoring syndrome extraction circuits to the shuttling noise model, we significantly improve on previous surface code proposals and extend the feasible shuttling range of the architecture by 5-10x, enabling the implementation of more complex codes with long-range interactions. Taking inspiration from the field of robotics, we develop a coordinated shuttle scheduling algorithm that supports arbitrary codes and use it to benchmark the logical performance of a variety of promising code families. We find that the optimized schedules are up to 86% faster than hand-optimized schedules for certain code families.
Through detailed circuit-level simulations, we identify specific QLDPC codes that improve upon prior surface code implementations by orders of magnitude, increasing encoding efficiency and reducing logical error rates. This work demonstrates the potential of shuttling-based spin qubit hardware platforms for scalable and efficient fault-tolerant quantum computation."""


def looks_degenerate(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if "write a subtitle" in lowered or "write one sentence" in lowered:
        return True
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]*", lowered)
    if len(words) >= 8:
        counts: dict[str, int] = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        if max(counts.values(), default=0) >= 5:
            return True
    if re.search(r"\b([a-zA-Z][a-zA-Z'-]{1,})\b(?:\W+\1\b){3,}", lowered):
        return True
    if re.search(r"([a-zA-Z]-){5,}", stripped):
        return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--abstract", default=DEFAULT_ABSTRACT)
    parser.add_argument("--cache-dir", default="/tmp/litert-cache")
    parser.add_argument("--max-num-tokens", type=int, default=768)
    parser.add_argument("--fail-on-degenerate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = litert_lm.Engine(
        str(args.model),
        backend=Backend.CPU,
        max_num_tokens=args.max_num_tokens,
        cache_dir=args.cache_dir,
        enable_speculative_decoding=False,
    )
    conversation = engine.create_conversation(
        system_message=SYSTEM_PROMPT,
        sampler_config=SamplerConfig(
            top_k=1,
            top_p=1.0,
            temperature=0.0,
            seed=0,
        ),
    )
    user_prompt = USER_TEMPLATE.format(title=args.title, abstract=args.abstract)
    response = conversation.send_message(
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
    )
    if isinstance(response, str):
        text = response
    else:
        content = response.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                part.get("text", str(part)) if isinstance(part, dict) else str(part)
                for part in content
            )
        else:
            text = str(content)
    degenerate = looks_degenerate(text)
    print("SUMMARY:", text.strip())
    print("DEGENERATE:", "yes" if degenerate else "no")
    return 2 if degenerate and args.fail_on_degenerate else 0


if __name__ == "__main__":
    raise SystemExit(main())
