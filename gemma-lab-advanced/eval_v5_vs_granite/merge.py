#!/usr/bin/env python3
"""Merge the per-variant generations with the source abstract/title/gold into a
single combined.jsonl + a readable combined.md for judging."""
from __future__ import annotations
import json
import re
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
TEST = (
    LAB / "gold_pipeline/05_milestone_assembly/gold_set/mlx_lora"
    / "title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5/test.jsonl"
)
OUT = Path(__file__).resolve().parent / "out"


def parse_title_abstract(user_content: str):
    title = abstract = ""
    m = re.search(r"Title:\n(.*?)\n\nAbstract:\n(.*)\Z", user_content, re.S)
    if m:
        title, abstract = m.group(1).strip(), m.group(2).strip()
    return title, abstract


def main():
    rows = [json.loads(l) for l in TEST.read_text().splitlines() if l.strip()]
    variants = {v: {} for v in ("gemma425", "gemma500", "granite")}
    for v in variants:
        for l in (OUT / f"{v}.jsonl").read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                variants[v][r["idx"]] = r

    combined = []
    for i, row in enumerate(rows):
        user = next(m["content"] for m in row["messages"] if m["role"] == "user")
        title, abstract = parse_title_abstract(user)
        gold = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
        combined.append({
            "idx": i,
            "paper_id": row.get("paper_id"),
            "primary_category": row.get("primary_category"),
            "title": title,
            "abstract": abstract,
            "gold": gold,
            "gemma425": variants["gemma425"][i]["brief"],
            "gemma500": variants["gemma500"][i]["brief"],
            "granite": variants["granite"][i]["brief"],
        })

    (OUT / "combined.jsonl").write_text(
        "\n".join(json.dumps(c) for c in combined) + "\n"
    )

    # readable markdown
    lines = ["# Test-split briefs: gemma vs shipped Granite\n"]
    for c in combined:
        lines.append(f"## [{c['idx']:02d}] {c['paper_id']} ({c['primary_category']})")
        lines.append(f"**Title:** {c['title']}\n")
        lines.append(f"**Abstract:** {c['abstract']}\n")
        lines.append(f"- **gold (teacher):** {c['gold']}")
        lines.append(f"- **gemma425:** {c['gemma425']}")
        lines.append(f"- **gemma500:** {c['gemma500']}")
        lines.append(f"- **granite(shipped):** {c['granite']}")
        lines.append("")
    (OUT / "combined.md").write_text("\n".join(lines))
    print(f"wrote {len(combined)} rows -> {OUT/'combined.jsonl'} and combined.md")


if __name__ == "__main__":
    main()
