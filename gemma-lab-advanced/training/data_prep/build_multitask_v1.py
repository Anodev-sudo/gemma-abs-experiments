#!/usr/bin/env python3
"""Build a dual-prompt multi-task chat dataset:

  - SUMMARY mode: the wide ~10.7k summary DB (deduped, leakage-excluded) behind a
    distinct 'summary' system prompt + summary-register user instruction.
  - BRIEF mode:   the 894 brief rows as-is (their original brief system prompt),
    upsampled to keep brief voice present in training.

Grounding is taught broadly via the summaries; brief *voice* stays bound to the
brief system prompt (used at inference). Valid/test are brief-mode only.

Knobs: --summary-cap (0=all), --brief-upsample N. All recorded in manifest.json.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent.parent
SIB = Path.home() / "myApps/model-experiments/gemma-abstract-finetuning"
BRIEF_SPLIT = (LAB / "gold_pipeline/05_milestone_assembly/gold_set/mlx_lora"
               / "title_abstract_chat_gemma_270m_it_abstract_to_feed_blurb_v5")
WIDE = SIB / "hf_dataset/releases/recent_3y_balanced_10746_v1/training_rows.jsonl"

SUMMARY_SYSTEM = ("You write short, faithful one-sentence summaries of research "
                  "papers. Output only one sentence.")
SUMMARY_USER = ("Write one faithful one-sentence summary of this paper, 14 words "
                "or less, stating its main result.\n\nTitle:\n{title}\n\n"
                "Abstract:\n{abstract}")


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def brief_paper_ids():
    ids = set()
    for sp in ("train", "valid", "test"):
        for r in load_jsonl(BRIEF_SPLIT / f"{sp}.jsonl"):
            ids.add(r.get("paper_id"))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-cap", type=int, default=0, help="0 = use all eligible")
    ap.add_argument("--brief-upsample", type=int, default=3)
    ap.add_argument("--out", default="training/data/multitask_summary10k_brief994_v1")
    ap.add_argument("--seed", type=int, default=20260613)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    exclude = brief_paper_ids()

    # --- summary-mode rows (deduped + leakage-excluded) ---
    seen, summary_rows = set(), []
    for r in load_jsonl(WIDE):
        pid = r.get("paper_id")
        s = (r.get("summary") or "").strip()
        if not s or pid in exclude or pid in seen:
            continue
        seen.add(pid)
        title = (r.get("title") or "").strip()
        abstract = (r.get("abstract") or "").strip()
        if not abstract:
            continue
        summary_rows.append({
            "messages": [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": SUMMARY_USER.format(title=title, abstract=abstract)},
                {"role": "assistant", "content": s},
            ],
            "paper_id": pid, "mode": "summary",
        })
    rng.shuffle(summary_rows)
    if args.summary_cap and len(summary_rows) > args.summary_cap:
        summary_rows = summary_rows[: args.summary_cap]

    # --- brief-mode rows (as-is, upsampled) ---
    brief_train = load_jsonl(BRIEF_SPLIT / "train.jsonl")
    for r in brief_train:
        r["mode"] = "brief"
    brief_up = []
    for _ in range(args.brief_upsample):
        brief_up.extend(brief_train)

    train = summary_rows + brief_up
    rng.shuffle(train)

    out = LAB / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n")
    # valid/test = brief-mode only (track brief performance)
    for sp in ("valid", "test"):
        rows = load_jsonl(BRIEF_SPLIT / f"{sp}.jsonl")
        (out / f"{sp}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    manifest = {
        "summary_eligible": len(seen),
        "summary_used": len(summary_rows),
        "summary_cap": args.summary_cap,
        "brief_unique": len(brief_train),
        "brief_upsample": args.brief_upsample,
        "brief_rows": len(brief_up),
        "train_total": len(train),
        "brief_fraction": round(len(brief_up) / len(train), 3),
        "excluded_brief_paper_ids": len(exclude),
        "summary_system": SUMMARY_SYSTEM,
        "summary_user_template": SUMMARY_USER,
        "seed": args.seed,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
