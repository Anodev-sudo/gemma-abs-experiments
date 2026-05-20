#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Callable

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

TRAINING_DOMAINS = [
    ("cs.LG", "adaptive training scheduler", "DNN training", "training jobs"),
]
SERVING_DOMAINS = [
    ("cs.DC", "heterogeneous serving planner", "LLM serving", "inference requests"),
    ("cs.NI", "traffic routing controller", "datacenter traffic", "network flows"),
    ("eess.SP", "wireless signal reconstructor", "channel recovery", "wireless signals"),
]
QUALITY_DOMAINS = [
    ("cs.CL", "retrieval reranker", "question answering", "retrieved passages"),
    ("cs.CV", "video recognition pipeline", "activity detection", "video streams"),
    ("eess.SP", "wireless signal reconstructor", "channel recovery", "wireless signals"),
]
OPTICS_DOMAINS = [
    ("physics.optics", "3D microfiber coupler", "fiber-to-chip coupling", "silicon nitride waveguides"),
    ("physics.optics", "photonic taper design", "waveguide coupling", "integrated photonic chips"),
]
DECIMAL_DOMAINS = [
    ("cs.DM", "golden-ratio tangle algorithm", "height-optimal tangles", "swap lists", "Golden-ratio analysis speeds structured tangle search without copying decimal constants."),
    ("math.OC", "packing approximation method", "geometric packing", "structured instances", "The packing method gives bounded approximations without copying decimal constants."),
    ("cs.NA", "damping-stability analysis", "numerical solvers", "structured systems", "The damping analysis improves stability without turning decimals into counts."),
    ("eess.SP", "spectral calibration rule", "spectral calibration", "noisy spectra", "The calibration rule stabilizes noisy spectra without copying decimal constants."),
]
OMIT_SCENARIOS = [
    {
        "cat": "cs.RO",
        "method": "self-supervised calibration",
        "task": "robot sensing",
        "distractors": ["12 cameras", "48 sensors", "7 collection days"],
        "summary": "Self-supervised calibration improves cross-device transfer without relying on setup counts.",
        "contribution": "a calibration strategy that transfers across sensor suites without hand-tuned feature rules",
    },
    {
        "cat": "cs.LG",
        "method": "benchmark audit",
        "task": "small-model evaluation",
        "distractors": ["31 tasks", "6 baselines", "4 prompt formats"],
        "summary": "The benchmark audit reveals small-model failures without emphasizing setup counts.",
        "contribution": "an analysis protocol that separates model weaknesses from benchmark bookkeeping",
    },
    {
        "cat": "cs.DB",
        "method": "table-cleaning pipeline",
        "task": "scientific table extraction",
        "distractors": ["9 filters", "24 thresholds", "5 validation folds"],
        "summary": "The table-cleaning pipeline standardizes noisy records without copying setup counts.",
        "contribution": "a cleaning strategy that standardizes noisy records without manual schema rules",
    },
    {
        "cat": "cs.DM",
        "method": "graph proof technique",
        "task": "hardness reductions",
        "distractors": ["3 lemmas", "17 cases", "2 auxiliary constants"],
        "summary": "The graph proof technique establishes hardness without foregrounding incidental constants.",
        "contribution": "a reduction strategy that makes the hardness argument easier to verify",
    },
]


def fmt_num(x: float, places: int = 1) -> str:
    return f"{x:.{places}f}"


def wc(text: str) -> int:
    return len(text.split())


def article(phrase: str) -> str:
    first = phrase.strip().lower()
    if first.startswith(("adaptive", "accuracy", "f1", "inference", "energy")):
        return "an"
    return "a"


def clean_summary(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    if not text.endswith("."):
        text += "."
    return text[0].upper() + text[1:]


def assert_label(summary: str) -> None:
    n = wc(summary)
    if not (10 <= n <= 13):
        raise ValueError(f"bad label word count {n}: {summary}")


def build_messages(title: str, abstract: str, summary: str) -> list[dict[str, str]]:
    user = SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE.format(title=title, abstract=abstract)
    return [{"role": "user", "content": user}, {"role": "assistant", "content": summary}]


def make_base_row(
    idx: int,
    title: str,
    abstract: str,
    summary: str,
    cat: str,
    number_policy: str,
    pattern: str,
    required_numbers: list[str],
    forbidden_numbers: list[str],
    central_roles: list[dict[str, str]] | None = None,
    distractor_numbers: list[str] | None = None,
) -> dict[str, Any]:
    summary = clean_summary(summary)
    assert_label(summary)
    pid = f"synthetic.numrole.v5.{idx:04d}"
    return {
        "paper_id": pid,
        "title": title,
        "abstract": abstract,
        "summary": summary,
        "primary_category": cat,
        "categories": [cat],
        "created_date": "2026-05-13",
        "source_split": "synthetic_numeric_role_candidate_v5",
        "numeric_pattern": pattern,
        "number_policy": number_policy,
        "required_numbers": required_numbers,
        "forbidden_numbers": forbidden_numbers,
        "central_roles": central_roles or [],
        "distractor_numbers": distractor_numbers or [],
        "teacher_model": "deterministic_synthetic_numeric_role_v5",
        "generator_prompt_version": "deterministic_numeric_role_candidates_v5_exact_runtime_prompt",
        "review_prompt_version": "unverified_candidate",
        "messages": build_messages(title, abstract, summary),
    }


def make_two_metric(idx: int, rng: random.Random) -> dict[str, Any]:
    pattern = rng.choice(["throughput_cost", "speed_memory", "accuracy_latency", "recall_runtime", "throughput_energy"])
    if pattern in {"accuracy_latency", "recall_runtime"}:
        cat, method, task, object_noun = rng.choice(QUALITY_DOMAINS)
    elif pattern == "throughput_cost":
        cat, method, task, object_noun = rng.choice(TRAINING_DOMAINS + SERVING_DOMAINS)
    else:
        cat, method, task, object_noun = rng.choice(SERVING_DOMAINS)

    a = fmt_num(rng.uniform(2.3, 18.9), 1)
    b = fmt_num(rng.uniform(12.0, 78.0), 1)
    if pattern == "throughput_cost":
        n1, n2 = f"{a}x", f"{b}%"
        phrase1, phrase2 = "throughput improvement", "cost reduction"
        claim = f"{n1} higher throughput and {n2} lower deployment cost"
        summary = f"Role-aware scheduling improves {task} throughput {a}x while cutting deployment costs {b}%."
    elif pattern == "speed_memory":
        n1, n2 = f"{a}x", f"{b}%"
        phrase1, phrase2 = "inference speedup", "memory-traffic reduction"
        claim = f"{n1} faster inference and {n2} lower memory traffic"
        summary = f"Role-aware execution speeds {task} {a}x while reducing memory traffic {b}%."
    elif pattern == "accuracy_latency":
        a = fmt_num(rng.uniform(2.0, 9.8), 1)
        n1, n2 = f"{a} points", f"{b}%"
        phrase1, phrase2 = "accuracy gain", "latency reduction"
        claim = f"{n1} higher accuracy and {n2} lower latency"
        summary = f"Role-aware processing improves {task} accuracy {a} points while reducing latency {b}%."
    elif pattern == "recall_runtime":
        a = fmt_num(rng.uniform(2.0, 9.8), 1)
        n1, n2 = f"{a} points", f"{b}%"
        phrase1, phrase2 = "recall gain", "runtime reduction"
        claim = f"{n1} higher recall and {n2} lower runtime"
        summary = f"Role-aware processing raises {task} recall {a} points while reducing runtime {b}%."
    else:
        n1, n2 = f"{a}x", f"{b}%"
        phrase1, phrase2 = "throughput improvement", "energy reduction"
        claim = f"{n1} higher throughput and {n2} lower energy per query"
        summary = f"Role-aware batching raises {task} throughput {a}x while lowering energy {b}%."

    title = f"Role-aware scheduling for {task}"
    abstract = (
        f"We study {article(method)} {method} for {task} under realistic deployment constraints. "
        f"The method separates resource-heavy stages from lightweight control decisions and assigns each stage to a suitable device. "
        f"Across controlled experiments, the system achieves {claim} compared with a tuned production baseline. "
        f"Additional ablations with 8 devices and 3 workloads show that the gains come from role-aware scheduling rather than larger models. "
        f"The first reported number is the {phrase1}, while the second is the {phrase2}."
    )
    return make_base_row(
        idx, title, abstract, summary, cat, "keep_both", f"two_metric_role_binding:{pattern}",
        [n1, n2], [], [{"value": n1, "role": phrase1}, {"value": n2, "role": phrase2}],
    )


def make_keep_one(idx: int, rng: random.Random) -> dict[str, Any]:
    kind = rng.choice(["accuracy", "f1", "runtime", "efficiency"])
    if kind == "efficiency":
        cat, method, task, object_noun = rng.choice(OPTICS_DOMAINS)
        b = fmt_num(rng.uniform(88.0, 98.7), 1)
        keep = f"{b}%"
        distractors = rng.sample(["1542 nm", "768 nm bandwidth", "1-dB cutoff", "three taper sections"], 3)
        title = f"Robust {method} for {task}"
        abstract = (
            f"This work introduces a robust {method} for {task}. "
            f"The evaluation mentions {distractors[0]}, {distractors[1]}, and {distractors[2]} to describe the optical setup. "
            f"The central result is coupling efficiency of {keep} over a carefully matched baseline. "
            f"Qualitative analysis suggests that the design improves coupling rather than exploiting a narrow wavelength setting."
        )
        summary = f"The {method} achieves {keep} coupling efficiency across practical chip setups."
        central_role = "coupling efficiency"
    else:
        if kind in {"accuracy", "f1"}:
            cat, method, task, object_noun = rng.choice(QUALITY_DOMAINS)
        else:
            cat, method, task, object_noun = rng.choice(SERVING_DOMAINS)
        distractors = rng.sample(["8 GPUs", "3 datasets", "48 hours", "12 sensors", "7 days", "2024 benchmark"], 3)
        title = f"Robust {method} for {task}"
        if kind == "accuracy":
            a = fmt_num(rng.uniform(2.0, 9.8), 1)
            keep = f"{a} points"
            metric_phrase = "accuracy gain"
            summary = f"The {method} improves {task} accuracy {a} points across noisy deployments."
        elif kind == "f1":
            a = fmt_num(rng.uniform(2.0, 9.8), 1)
            keep = f"{a} points"
            metric_phrase = "F1 gain"
            summary = f"The {method} raises {task} F1 {a} points across varied domains."
        else:
            b = fmt_num(rng.uniform(28.0, 88.0), 1)
            keep = f"{b}%"
            metric_phrase = "runtime reduction"
            summary = f"The {method} reduces {task} runtime {b}% under deployment noise."
        abstract = (
            f"This work introduces a robust {method} for {task}. "
            f"The evaluation uses {distractors[0]}, {distractors[1]}, and {distractors[2]} to stress deployment variability. "
            f"Despite these setup details, the central result is {article(metric_phrase)} {metric_phrase} of {keep} over a carefully matched baseline. "
            f"Qualitative analysis suggests that the method improves the core decision process rather than exploiting benchmark artifacts."
        )
        central_role = metric_phrase
    return make_base_row(
        idx, title, abstract, summary, cat, "keep_one", f"keep_metric_omit_distractors:{kind}",
        [keep], distractors, [{"value": keep, "role": central_role}], distractors,
    )


def make_omit(idx: int, rng: random.Random) -> dict[str, Any]:
    scenario = rng.choice(OMIT_SCENARIOS)
    cat = scenario["cat"]
    method = scenario["method"]
    task = scenario["task"]
    distractors = list(scenario["distractors"])
    rng.shuffle(distractors)
    title = f"{method.title()} for {task}"
    abstract = (
        f"We present a {method} for {task}. "
        f"The experimental setup includes {distractors[0]}, {distractors[1]}, and {distractors[2]}, but these counts describe only the study protocol. "
        f"The main contribution is {scenario['contribution']}. "
        f"Results show more reliable behavior under domain shift and easier deployment for small research teams."
    )
    return make_base_row(
        idx, title, abstract, scenario["summary"], cat, "omit_all", "omit_incidental_numbers",
        [], distractors, [], distractors,
    )


def make_decimal(idx: int, rng: random.Random) -> dict[str, Any]:
    cat, method, task, object_noun, summary = rng.choice(DECIMAL_DOMAINS)
    if "golden" in method:
        dec, role = "1.618", "phi-based search bound"
    elif "packing" in method:
        dec, role = "1.732", "approximation ratio"
    elif "damping" in method:
        dec, role = "0.625", "stability threshold"
    else:
        dec, role = "0.041", "regularization coefficient"
    title = f"{method.title()} for {task}"
    abstract = (
        f"We analyze a {method} for {task}. "
        f"The derivation contains a parameter approximately equal to {dec}, which acts as a {role} rather than a dataset size or bit length. "
        f"The main result is a simpler algorithmic argument with improved practical behavior on {object_noun}. "
        f"Experiments confirm that the method is useful without requiring the decimal constant to appear in a feed summary."
    )
    return make_base_row(
        idx, title, abstract, summary, cat, "omit_decimal", "decimal_corruption_prevention",
        [], [dec, dec.replace(".", "")], [], [dec],
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="hf_bucket/experiments/sft/q5_step2400_sft700_numeric_role_curriculum_v1/candidates_v5")
    ap.add_argument("--count", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260513)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    makers: list[Callable[[int, random.Random], dict[str, Any]]] = [
        make_two_metric,
        make_two_metric,
        make_keep_one,
        make_keep_one,
        make_omit,
        make_decimal,
    ]
    rows = [makers[i % len(makers)](i + 1, rng) for i in range(args.count)]
    out = Path(args.output_dir)
    write_jsonl(out / "synthetic_numeric_role_candidates.jsonl", rows)
    wc_counts: dict[str, int] = {}
    for r in rows:
        wc_counts[str(wc(r["summary"]))] = wc_counts.get(str(wc(r["summary"])), 0) + 1
    manifest = {
        "dataset_id": "synthetic_numeric_role_candidates_v5",
        "purpose": "Candidate pool only. Rows must be run through current checkpoint-700 and fail numeric-policy checks before training eligibility.",
        "row_count": len(rows),
        "seed": args.seed,
        "exact_runtime_prompt": True,
        "quality_changes_vs_v4": [
            "fixed article grammar in abstracts",
            "excluded DNN training from inference-speed patterns",
            "updated row IDs and source metadata to v5",
            "kept exact runtime prompt and 10-13 word labels",
        ],
        "word_count_counts": wc_counts,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
