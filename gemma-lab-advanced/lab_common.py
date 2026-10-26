"""Shared helpers for generating + scoring feed briefs across runs.

Kept tiny and dependency-light so every experiment uses the SAME prompt
construction, output cleaning, and word counting — comparability across runs."""
from __future__ import annotations
import re
from pathlib import Path

LAB = Path(__file__).resolve().parent
TEST_SPLIT = (
    LAB / "gold_pipeline/05_milestone_assembly/gold_set/mlx_lora"
    / "title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5/test.jsonl"
)
STOPS = ("<end_of_turn>", "<eos>", "<|end_of_text|>", "</s>")


def build_prompt(tokenizer, messages):
    """Apply the model's chat template to the system+user turns (drop assistant).
    Falls back to folding the system turn into the user turn if the template
    rejects a standalone system role."""
    convo = [m for m in messages if m["role"] != "assistant"]
    try:
        return tokenizer.apply_chat_template(
            convo, add_generation_prompt=True, tokenize=True
        )
    except Exception:
        sys_txt = "\n\n".join(m["content"] for m in convo if m["role"] == "system")
        folded = []
        for m in convo:
            if m["role"] == "system":
                continue
            if m["role"] == "user" and sys_txt:
                folded.append({"role": "user", "content": f"{sys_txt}\n\n{m['content']}"})
                sys_txt = ""
            else:
                folded.append(m)
        return tokenizer.apply_chat_template(
            folded, add_generation_prompt=True, tokenize=True
        )


def clean(text: str) -> str:
    text = text.strip()
    for stop in STOPS:
        text = text.split(stop)[0]
    text = text.strip().strip('"').strip()
    return text.splitlines()[0].strip() if text else text


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))
