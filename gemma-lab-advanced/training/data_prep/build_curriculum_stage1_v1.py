#!/usr/bin/env python3
"""Stage-1 grounding data for the curriculum run.

Summaries from the wide ~10.7k DB (deduped, leakage-excluded) rendered UNDER THE
BRIEF system+user prompt, with the summary as the target. Builds grounding in the
exact mode we infer in (unlike the failed multitask run, which quarantined the
summary register behind its own prompt). valid/test = brief 50/50 (track brief perf).

Stage 2 uses the existing v5 brief split directly (no new data needed)."""
from __future__ import annotations
import json
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent.parent
SIB = Path(__file__).resolve().parents[3]  # parent repo root
BRIEF_SPLIT = (LAB / "gold_pipeline/05_milestone_assembly/gold_set/mlx_lora"
               / "title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5")
WIDE = SIB / "hf_dataset/releases/recent_3y_balanced_10746_v1/training_rows.jsonl"
OUT = LAB / "training/data/curriculum_stage1_summaries_briefprompt_v1"

# the v5 brief prompt (identical to stage-2 briefs, so stage 2 is pure continuation)
BRIEF_SYSTEM = "You write short, faithful paper-feed blurbs. Output only one sentence."
BRIEF_USER = ('The title is already visible in the app. Write one readable feed blurb, '
              '14 words or less, that gives a broad reader a reason to open this paper. '
              'Never exceed 14 words. Prefer the main takeaway over implementation detail. '
              'Use exact numbers only when they are central and easy to bind. Avoid '
              'repeating the title. Avoid jargon unless necessary. Avoid dense numbers, '
              'equations, shorthand, author/date references, and filler like "This paper" '
              'or "We propose".\n\nTitle:\n{title}\n\nAbstract:\n{abstract}')


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main():
    exclude = set()
    for sp in ("train", "valid", "test"):
        for r in load(BRIEF_SPLIT / f"{sp}.jsonl"):
            exclude.add(r.get("paper_id"))

    seen, rows = set(), []
    for r in load(WIDE):
        pid = r.get("paper_id")
        s = (r.get("summary") or "").strip()
        ab = (r.get("abstract") or "").strip()
        if not s or not ab or pid in exclude or pid in seen:
            continue
        seen.add(pid)
        rows.append({"messages": [
            {"role": "system", "content": BRIEF_SYSTEM},
            {"role": "user", "content": BRIEF_USER.format(title=(r.get("title") or "").strip(), abstract=ab)},
            {"role": "assistant", "content": s},
        ], "paper_id": pid})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    for sp in ("valid", "test"):
        (OUT / f"{sp}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in load(BRIEF_SPLIT / f"{sp}.jsonl")) + "\n")
    (OUT / "manifest.json").write_text(json.dumps({
        "stage": 1, "purpose": "grounding pretrain (summaries under brief prompt)",
        "train_rows": len(rows), "excluded_brief_paper_ids": len(exclude),
        "valid_test": "brief 50/50",
    }, indent=2))
    print(f"stage-1 train rows: {len(rows)} -> {OUT}")


if __name__ == "__main__":
    main()
