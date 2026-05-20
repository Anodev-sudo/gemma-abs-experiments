#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from faithfulness_cleanup_common import unsupported_numbers
except Exception:
    unsupported_numbers = None

NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?(?:\s?[%x×]|e[-+]?\d+)?|\b\d+[A-Za-z]+|\b[A-Za-z]+\d+[A-Za-z0-9-]*")
DIGIT_RE = re.compile(r"\d")
FORMULA_RE = re.compile(r"<[^>]+>|\$|\\[A-Za-z]+|\bO\([^)]*\)|\^|[≤≥≈]")
BANNED_OPENINGS = ("this paper", "the paper", "the authors", "we propose", "in this work", "in this paper")
STRESS_IDS = {"1901.06548", "2111.10635", "2303.09722", "2406.04373"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def wc(text: str) -> int:
    return len(str(text).split())


def norm(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def title_overlap(summary: str, title: str) -> int:
    words = lambda s: {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z-]{2,}", s)}
    return len(words(summary) & words(title))


def heur(summary: str, row: dict[str, Any]) -> dict[str, Any]:
    summary = norm(summary)
    nums = NUM_RE.findall(summary)
    unsup = []
    if unsupported_numbers is not None:
        try:
            unsup = unsupported_numbers(summary, row.get("title", "") + "\n" + row.get("abstract", ""))
        except Exception:
            unsup = []
    low = summary.lower()
    flags = []
    words = wc(summary)
    if words < 10:
        flags.append("under10")
    if words > 15:
        flags.append("over15")
    if DIGIT_RE.search(summary):
        flags.append("has_digit")
    if len(nums) >= 2:
        flags.append("number_dense")
    if FORMULA_RE.search(summary):
        flags.append("formula_or_html")
    if any(low.startswith(prefix) for prefix in BANNED_OPENINGS):
        flags.append("banned_opening")
    if unsup:
        flags.append("unsupported_number")
    return {
        "summary": summary,
        "word_count": words,
        "numbers": nums,
        "number_count": len(nums),
        "unsupported_numbers": unsup,
        "title_overlap": title_overlap(summary, row.get("title", "")),
        "flags": flags,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    counts = Counter(flag for r in rows for flag in r["heuristics"]["flags"])
    wc_list = [r["heuristics"]["word_count"] for r in rows]
    return {
        "rows": len(rows),
        "avg_wc": round(mean(wc_list), 2),
        "under10": counts.get("under10", 0),
        "over15": counts.get("over15", 0),
        "digit_rows": counts.get("has_digit", 0),
        "number_dense_rows": counts.get("number_dense", 0),
        "unsupported_number_rows": counts.get("unsupported_number", 0),
        "formula_or_html_rows": counts.get("formula_or_html", 0),
        "banned_opening_rows": counts.get("banned_opening", 0),
        "avg_title_overlap": round(mean([r["heuristics"]["title_overlap"] for r in rows]), 2),
    }


def load_baseline_summaries(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = load_jsonl(path)
    return {str(r.get("paper_id")): norm(r.get("summary", "")) for r in rows if r.get("paper_id") and r.get("summary")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--inputs", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--ckpts", nargs="+", type=int, default=[25, 50, 75, 100, 125, 150, 175, 200])
    ap.add_argument("--baseline-candidates", default="hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/failure_mining/checkpoint700_10000_v1/heuristic_screen/numeric_candidates.jsonl")
    ap.add_argument("--worst-ckpt", type=int, default=150)
    args = ap.parse_args()

    run = Path(args.run)
    inputs = Path(args.inputs) if args.inputs else run / "eval/heuristic_inputs/combined_454.jsonl"
    outdir = Path(args.outdir) if args.outdir else run / "eval/heuristic_outputs"
    report = run / "eval/HEURISTIC_SWEEP_REPORT.md"
    summary_json = run / "eval/heuristic_sweep_summary.json"
    rows_jsonl = run / "eval/heuristic_sweep_rows.jsonl"

    input_rows = {r["paper_id"]: r for r in load_jsonl(inputs)}
    baseline_by_pid = load_baseline_summaries(Path(args.baseline_candidates))
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    baseline_numeric_heur: dict[str, dict[str, Any]] = {}

    for pid, base_summary in baseline_by_pid.items():
        row = input_rows.get(pid)
        if row and row.get("split") == "numeric_ckpt700_risk_pool350":
            baseline_numeric_heur[pid] = heur(base_summary, row)

    for ck in args.ckpts:
        out_path = outdir / f"checkpoint_{ck}_outputs_454.jsonl"
        outputs = load_jsonl(out_path)
        ck_rows = []
        for out in outputs:
            row = input_rows[out["paper_id"]]
            h = heur(out.get("normalized_output") or out.get("raw_output") or "", row)
            record = {
                "checkpoint": ck,
                "paper_id": out["paper_id"],
                "eval_id": row.get("eval_id", ""),
                "split": row.get("split", ""),
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
                "summary": h["summary"],
                "baseline_summary": baseline_by_pid.get(out["paper_id"], ""),
                "heuristics": h,
            }
            ck_rows.append(record)
            all_rows.append(record)
        by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in ck_rows:
            by_split[r["split"]].append(r)
        summary[str(ck)] = {
            "all": summarize(ck_rows),
            **{split: summarize(rows) for split, rows in sorted(by_split.items())},
        }
        numeric = by_split.get("numeric_ckpt700_risk_pool350", [])
        if numeric:
            summary[str(ck)]["baseline_comparison"] = {
                "exact_unchanged_vs_ckpt700": sum(1 for r in numeric if r["summary"] == baseline_by_pid.get(r["paper_id"], "")),
                "rows_with_fewer_numbers": sum(1 for r in numeric if r["heuristics"]["number_count"] < baseline_numeric_heur.get(r["paper_id"], {}).get("number_count", 999)),
                "dense_fixed": sum(1 for r in numeric if "number_dense" in baseline_numeric_heur.get(r["paper_id"], {}).get("flags", []) and "number_dense" not in r["heuristics"]["flags"]),
            }

    write_jsonl(rows_jsonl, all_rows)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    lines = ["# Heuristic sweep report", "", "Eval set: 350 checkpoint-700 numeric-risk rows, 100 fresh normal rows, 4 remembered stress rows.", ""]
    lines.append("## Summary by checkpoint")
    lines.append("")
    lines.append("| ckpt | split | rows | avg wc | <10 | >15 | digit | dense nums | unsupported nums | formula/html | title ovlp |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ck in args.ckpts:
        for split in ["all", "numeric_ckpt700_risk_pool350", "fresh_recent_normal_100", "remembered_stress"]:
            s = summary[str(ck)].get(split, {})
            if not s:
                continue
            lines.append(
                f"| {ck} | {split} | {s['rows']} | {s['avg_wc']} | {s['under10']} | {s['over15']} | {s['digit_rows']} | {s['number_dense_rows']} | {s['unsupported_number_rows']} | {s['formula_or_html_rows']} | {s['avg_title_overlap']} |"
            )
    if baseline_numeric_heur:
        base_rows = [{"heuristics": h} for h in baseline_numeric_heur.values()]
        bs = summarize(base_rows)
        lines.extend(["", "## Checkpoint-700 numeric baseline comparison", "", "Baseline is the checkpoint-700 mined bad summary stored in the 350-row numeric candidate pool.", ""])
        lines.append("| model | digit rows | dense-number rows | unsupported-number rows | over15 | exact unchanged vs ckpt700 | rows with fewer numbers | dense fixed |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        lines.append(f"| ckpt700 baseline | {bs['digit_rows']} | {bs['number_dense_rows']} | {bs['unsupported_number_rows']} | {bs['over15']} | {len(baseline_numeric_heur)} | 0 | 0 |")
        for ck in args.ckpts:
            ns = summary[str(ck)].get("numeric_ckpt700_risk_pool350", {})
            bc = summary[str(ck)].get("baseline_comparison", {})
            lines.append(f"| checkpoint-{ck} | {ns.get('digit_rows', 0)} | {ns.get('number_dense_rows', 0)} | {ns.get('unsupported_number_rows', 0)} | {ns.get('over15', 0)} | {bc.get('exact_unchanged_vs_ckpt700', 0)} | {bc.get('rows_with_fewer_numbers', 0)} | {bc.get('dense_fixed', 0)} |")

    lines.extend(["", "## Remembered stress outputs"])
    for ck in args.ckpts:
        lines.append(f"\n### checkpoint-{ck}")
        for r in [x for x in all_rows if x["checkpoint"] == ck and x["paper_id"] in STRESS_IDS]:
            h = r["heuristics"]
            lines.append(f"- `{r['paper_id']}` ({h['word_count']}w; flags={h['flags']}): {r['summary']}")

    lines.extend(["", f"## Worst flagged examples at checkpoint-{args.worst_ckpt}"])
    ck_rows = [r for r in all_rows if r["checkpoint"] == args.worst_ckpt]
    flagged = [r for r in ck_rows if r["heuristics"]["flags"]]
    priority = sorted(flagged, key=lambda r: ("unsupported_number" not in r["heuristics"]["flags"], "number_dense" not in r["heuristics"]["flags"], -r["heuristics"]["word_count"]))[:30]
    for r in priority:
        h = r["heuristics"]
        lines.append(f"- `{r['paper_id']}` `{r['split']}` {h['word_count']}w flags={h['flags']} nums={h['numbers']} unsup={h['unsupported_numbers']}: {r['summary']}")

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
