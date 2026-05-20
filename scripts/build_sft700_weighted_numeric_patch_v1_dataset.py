#!/usr/bin/env python3
"""Build a style-anchored numeric weighted-loss patch dataset for SFT700.

The dataset is intentionally 75% normal concise style anchors and 25% numeric
challenge rows. Numeric rows carry char-span weight hints for a later custom
weighted-loss trainer; normal rows are plain assistant-only style anchors.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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

FILLER_RE = re.compile(
    r"^(this paper|we propose|we present|we introduce|we study|we show|we demonstrate|"
    r"we develop|we investigate|in this paper|this work|the paper|this article)\b",
    re.I,
)
DIGIT_RE = re.compile(r"\d")
NUMBER_RE = re.compile(
    r"(?<![\w.])[+\-−]?((?:\d+(?:\.\d+)?)|(?:\.\d+))"
    r"(?:\s?(?:%|x|×|nm|μm|um|mm|cm|m|km|ms|s|sec|seconds|mV|V|kV|Hz|kHz|MHz|GHz|"
    r"KB|MB|GB|TB|hours?|days?|tokens?|samples?|parameters?|params?|datasets?))?",
    re.I,
)
SYMBOL_RE = re.compile(r"(?<=\d)[+\-−](?=\d)|(?<!\w)[+\-−](?!\w)|[≤≥±%]")

DIRECTION_TERMS = {
    "improve", "improves", "improved", "improving", "improvement",
    "reduce", "reduces", "reduced", "reducing", "reduction",
    "outperform", "outperforms", "outperformed", "outperforming",
    "increase", "increases", "increased", "increasing",
    "decrease", "decreases", "decreased", "decreasing",
    "higher", "lower", "lowering", "raises", "raising", "better", "worse", "faster", "slower",
    "fewer", "smaller", "larger", "cheaper", "costlier",
    "cuts", "cutting", "cut", "boosts", "boost", "boosted",
    "speeds", "speeding", "accelerates", "accelerate", "accelerated", "saves", "saving",
    "negative", "positive", "no", "not", "without", "under", "over",
    "beats", "beat", "beating", "exceeds", "exceed", "lags", "lagging",
}

CONCEPT_TERMS = {
    "accuracy", "f1", "precision", "recall", "auc", "bleu", "score", "scores",
    "throughput", "runtime", "latency", "speed", "speedup", "slowdown",
    "cost", "costs", "energy", "memory", "parameters", "parameter", "params",
    "datasets", "dataset", "error", "errors", "loss", "efficiency", "rate", "rates",
    "risk", "risks", "performance", "quality", "training", "inference", "time",
    "tokens", "samples", "workloads", "devices", "gpu", "gpus", "dnn", "llm",
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "is", "of", "on",
    "or", "than", "the", "to", "under", "while", "with", "without", "over", "across",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def word_count(text: str) -> int:
    return len(str(text).split())


def umbrella(primary: str) -> str:
    prefix = (primary or "").split(".")[0]
    if prefix == "cs":
        return "cs"
    if prefix == "math":
        return "math"
    if prefix in {
        "astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th",
        "nucl-ex", "nucl-th", "physics", "quant-ph",
    }:
        return "physics"
    return "other"


def exact_messages(title: str, abstract: str, summary: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n" + USER_TEMPLATE.format(title=title, abstract=abstract)},
        {"role": "assistant", "content": summary.strip()},
    ]


def span(start: int, end: int, text: str, weight: float, kind: str) -> dict[str, Any]:
    return {"start": start, "end": end, "text": text[start:end], "weight": weight, "kind": kind}


def word_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"[A-Za-z][A-Za-z0-9+'-]*", text)]


def overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return int(a["start"]) < int(b["end"]) and int(b["start"]) < int(a["end"])


def add_span(spans: list[dict[str, Any]], new: dict[str, Any]) -> None:
    for old in spans:
        if overlaps(old, new):
            if float(new["weight"]) > float(old["weight"]):
                old.update(new)
            elif new["kind"] not in str(old.get("kind", "")).split("+"):
                old["kind"] = f"{old['kind']}+{new['kind']}"
            return
    spans.append(new)


def build_weight_hints(summary: str, row_type: str) -> dict[str, Any]:
    if row_type == "normal_style_anchor":
        return {
            "enabled": False,
            "base_weight": 1.0,
            "spans": [],
            "policy": "normal rows are unweighted style anchors",
        }

    spans: list[dict[str, Any]] = []
    number_matches = list(NUMBER_RE.finditer(summary))
    for m in number_matches:
        add_span(spans, span(m.start(), m.end(), summary, 6.0, "number_or_unit"))

    for m in SYMBOL_RE.finditer(summary):
        add_span(spans, span(m.start(), m.end(), summary, 6.0, "numeric_symbol"))

    for start, end, token in word_spans(summary):
        low = token.lower()
        if low in DIRECTION_TERMS:
            add_span(spans, span(start, end, summary, 5.0, "direction_or_polarity"))
        if low in CONCEPT_TERMS:
            add_span(spans, span(start, end, summary, 3.0, "metric_concept"))

    words = word_spans(summary)
    for m in number_matches:
        nearby = [i for i, (s, e, _tok) in enumerate(words) if abs(s - m.start()) < 80 or abs(e - m.end()) < 80]
        if nearby:
            lo = max(0, min(nearby) - 4)
            hi = min(len(words), max(nearby) + 5)
            for s, e, token in words[lo:hi]:
                low = token.lower().strip(".,;:()[]{}")
                if len(low) < 3 or low in STOPWORDS or DIGIT_RE.search(low):
                    continue
                add_span(spans, span(s, e, summary, 3.0, "near_number_content_word"))

    spans.sort(key=lambda x: (int(x["start"]), int(x["end"])))
    return {
        "enabled": True,
        "base_weight": 1.0,
        "default_numeric_weight": 6.0,
        "default_direction_polarity_weight": 5.0,
        "default_concept_neighborhood_weight": 3.0,
        "spans": spans,
        "policy": "apply only to assistant target tokens; prompt/source tokens stay masked out",
    }


def normalize_row(row: dict[str, Any], component: str, source_path: str, row_type: str) -> dict[str, Any]:
    title = str(row["title"]).strip()
    abstract = str(row["abstract"]).strip()
    summary = str(row["summary"]).strip()
    out = dict(row)
    out["title"] = title
    out["abstract"] = abstract
    out["summary"] = summary
    out["messages"] = exact_messages(title, abstract, summary)
    out["patch_component"] = component
    out["weighted_numeric_patch_row_type"] = row_type
    out["weighted_numeric_patch_source_path"] = source_path
    out["summary_word_count"] = word_count(summary)
    out["summary_has_digit"] = bool(DIGIT_RE.search(summary))
    out["loss_weight_hints"] = build_weight_hints(summary, row_type)
    return out


def good_summary(row: dict[str, Any], allow_digits: bool) -> bool:
    summary = str(row.get("summary", "")).strip()
    if not 10 <= word_count(summary) <= 15:
        return False
    if FILLER_RE.search(summary):
        return False
    if not allow_digits and DIGIT_RE.search(summary):
        return False
    if not row.get("title") or not row.get("abstract"):
        return False
    return True


def select_by_umbrella(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    targets = {"cs": 222, "physics": 168, "math": 96, "other": 114}
    assert sum(targets.values()) == n
    rnd = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[umbrella(str(row.get("primary_category", "")))].append(row)
    selected: list[dict[str, Any]] = []
    for key, target in targets.items():
        bucket = buckets[key]
        rnd.shuffle(bucket)
        if len(bucket) < target:
            raise ValueError(f"Not enough {key} rows: need {target}, have {len(bucket)}")
        selected.extend(bucket[:target])
    rnd.shuffle(selected)
    return selected


def round_robin_by_key(rows: list[dict[str, Any]], key: str, n: int, seed: int) -> list[dict[str, Any]]:
    rnd = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key) or "unknown")].append(row)
    for bucket in buckets.values():
        rnd.shuffle(bucket)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets, key=lambda k: (-len(buckets[k]), k))
    while len(selected) < n and keys:
        progressed = False
        for k in list(keys):
            if buckets[k]:
                selected.append(buckets[k].pop())
                progressed = True
                if len(selected) == n:
                    break
            else:
                keys.remove(k)
        if not progressed:
            break
    if len(selected) < n:
        raise ValueError(f"Could only select {len(selected)} rows for key={key}, need {n}")
    rnd.shuffle(selected)
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal-source", default="hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/train.jsonl")
    ap.add_argument("--numeric-source", default="hf_bucket/experiments/sft/q5_step2400_sft700_numeric_role_curriculum_v1/dataset_majority_synthetic_v1/train.jsonl")
    ap.add_argument("--stress-source", default="hf_bucket/experiments/sft/q5_step2400_sft700_numeric_binding_poc_exact_prompt_v1/dataset/train.jsonl")
    ap.add_argument("--output-dir", default="hf_bucket/experiments/sft/q5_step2400_sft700_weighted_numeric_patch_v1/dataset_75normal_25numeric_v1")
    ap.add_argument("--seed", type=int, default=20260515)
    args = ap.parse_args()

    normal_source = Path(args.normal_source)
    numeric_source = Path(args.numeric_source)
    stress_source = Path(args.stress_source)
    out_dir = Path(args.output_dir)

    normal_raw = load_jsonl(normal_source)
    numeric_raw = load_jsonl(numeric_source)
    stress_raw = load_jsonl(stress_source)

    # Keep normal anchors visually style-stable: 10-15 words, no target digits, no filler.
    normal_candidates = []
    seen_pids: set[str] = set()
    for row in normal_raw:
        pid = str(row.get("paper_id", ""))
        if not pid or pid in seen_pids:
            continue
        if good_summary(row, allow_digits=False):
            normal_candidates.append(row)
            seen_pids.add(pid)

    normal_selected = select_by_umbrella(normal_candidates, 600, args.seed)
    normal_rows = [
        normalize_row(row, "normal_style_anchor", str(normal_source), "normal_style_anchor")
        for row in normal_selected
    ]

    real_numeric = [
        row for row in numeric_raw
        if row.get("patch_component") == "real_mined_numeric_repair_subagent_accepted"
        and good_summary(row, allow_digits=True)
        and DIGIT_RE.search(str(row.get("summary", "")))
    ]
    real_numeric.sort(key=lambda r: (str(r.get("number_policy", "")), str(r.get("paper_id", ""))))

    synthetic_candidates = [
        row for row in numeric_raw
        if row.get("patch_component") == "synthetic_numeric_role_current_failed_v5"
        and row.get("current_sft700_failed_numeric_policy") is True
        and good_summary(row, allow_digits=True)
        and DIGIT_RE.search(str(row.get("summary", "")))
    ]

    # One unique row for each severe known stress paper, not the repeated POC replicas.
    stress_by_source: dict[str, dict[str, Any]] = {}
    for row in stress_raw:
        sid = str(row.get("source_paper_id") or row.get("paper_id", "")).split("#")[0]
        if sid in {"1901.06548", "2111.10635"} and sid not in stress_by_source and good_summary(row, allow_digits=True):
            stress_by_source[sid] = row
    stress_selected = [stress_by_source["1901.06548"], stress_by_source["2111.10635"]]

    need_synth = 200 - len(real_numeric) - len(stress_selected)
    synthetic_selected = round_robin_by_key(synthetic_candidates, "numeric_pattern", need_synth, args.seed + 1)

    numeric_rows = []
    for row in real_numeric:
        numeric_rows.append(normalize_row(row, "numeric_challenge_real_mined_current_failed", str(numeric_source), "numeric_challenge"))
    for row in synthetic_selected:
        numeric_rows.append(normalize_row(row, "numeric_challenge_synthetic_current_failed", str(numeric_source), "numeric_challenge"))
    for row in stress_selected:
        numeric_rows.append(normalize_row(row, "numeric_challenge_known_stress_current_failed", str(stress_source), "numeric_challenge"))

    if len(normal_rows) != 600 or len(numeric_rows) != 200:
        raise AssertionError((len(normal_rows), len(numeric_rows)))

    all_rows = normal_rows + numeric_rows
    rnd = random.Random(args.seed + 2)
    rnd.shuffle(all_rows)
    for i, row in enumerate(all_rows):
        row["weighted_numeric_patch_dataset_id"] = "q5_step2400_sft700_weighted_numeric_patch_75normal_25numeric_v1"
        row["weighted_numeric_patch_mix_index"] = i

    write_jsonl(out_dir / "train.jsonl", all_rows)
    write_jsonl(out_dir / "normal_style_anchor_rows.jsonl", normal_rows)
    write_jsonl(out_dir / "numeric_challenge_rows.jsonl", numeric_rows)

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    manifest = {
        "dataset_id": "q5_step2400_sft700_weighted_numeric_patch_75normal_25numeric_v1",
        "purpose": "Style-stable checkpoint-700 patch dataset: 75% normal concise anchors, 25% current-failed numeric challenge rows with assistant-target loss weight hints.",
        "base_checkpoint_intent": "hf_bucket/experiments/sft/q5_step2400_sft_correction_v1/checkpoint-700",
        "output_train_jsonl": str(out_dir / "train.jsonl"),
        "seed": args.seed,
        "row_count": len(all_rows),
        "normal_rows": len(normal_rows),
        "numeric_rows": len(numeric_rows),
        "mix": {"normal_style_anchor": 0.75, "numeric_challenge": 0.25},
        "sources": {
            "normal_source": str(normal_source),
            "numeric_source": str(numeric_source),
            "stress_source": str(stress_source),
        },
        "component_counts": dict(Counter(row["patch_component"] for row in all_rows)),
        "row_type_counts": dict(Counter(row["weighted_numeric_patch_row_type"] for row in all_rows)),
        "normal_umbrella_counts": dict(Counter(umbrella(str(row.get("primary_category", ""))) for row in normal_rows)),
        "numeric_pattern_counts": dict(Counter(str(row.get("numeric_pattern") or "real_or_stress") for row in numeric_rows)),
        "numeric_number_policy_counts": dict(Counter(str(row.get("number_policy") or "none") for row in numeric_rows)),
        "summary_word_counts": dict(sorted(Counter(row["summary_word_count"] for row in all_rows).items())),
        "avg_summary_words": avg([row["summary_word_count"] for row in all_rows]),
        "avg_normal_summary_words": avg([row["summary_word_count"] for row in normal_rows]),
        "avg_numeric_summary_words": avg([row["summary_word_count"] for row in numeric_rows]),
        "summary_digit_rows": sum(bool(row["summary_has_digit"]) for row in all_rows),
        "loss_weighting": {
            "normal_rows": "base assistant-token weight 1.0 only",
            "numeric_rows": "base assistant-token weight 1.0 plus char-span hints: numbers/symbols 6.0, direction/polarity 5.0, metric concepts and near-number content words 3.0",
            "important": "weights should be applied only to assistant target tokens; source/prompt tokens remain masked",
        },
        "current_failure_gate": {
            "synthetic_numeric": "current_sft700_failed_numeric_policy == true from llama.cpp Q5 screen",
            "real_numeric": "accepted rows from checkpoint-700 failure mining/subagent review",
            "stress_numeric": "known checkpoint-700 severe failures from exact-prompt numeric POC",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    examples = []
    for component in [
        "normal_style_anchor",
        "numeric_challenge_real_mined_current_failed",
        "numeric_challenge_synthetic_current_failed",
        "numeric_challenge_known_stress_current_failed",
    ]:
        sample = [r for r in all_rows if r["patch_component"] == component][:5]
        examples.append(f"## {component}\n")
        for row in sample:
            examples.append(f"- `{row['paper_id']}` ({row['summary_word_count']}w): {row['summary']}\n")
            if row["loss_weight_hints"].get("enabled"):
                hints = ", ".join(f"{s['text']}:{s['weight']}" for s in row["loss_weight_hints"]["spans"][:12])
                examples.append(f"  - weight hints: {hints}\n")
        examples.append("\n")

    readme = (
        "# Weighted numeric patch v1 dataset\n\n"
        "800 rows for a checkpoint-700 patch: 600 normal style anchors and 200 numeric challenge rows. "
        "Numeric rows include assistant-target char-span weight hints for a custom weighted-loss trainer.\n\n"
        "See `manifest.json` for counts and policy details.\n\n"
        + "".join(examples)
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
