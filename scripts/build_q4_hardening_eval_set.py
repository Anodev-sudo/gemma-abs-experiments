#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
from typing import Any

DEFAULT_CORPUS_JSONL = (
    "hf_dataset/corpora/recent_3y_balanced/arxiv_abstract_corpus_weighted_20000.jsonl"
)
DEFAULT_OUTPUT_DIR = "hf_dataset/evals/q4_hardening_mine_1000_v1"
DEFAULT_SEED = 20260501

DEFAULT_EXCLUDE_RELEASES = (
    "hf_dataset/releases/recent_3y_balanced_10746_v1",
    "hf_dataset/releases/recent_3y_balanced_10746_faithfulness_overlay_1400_v2",
)
DEFAULT_PRIOR_REVIEW_DIRS = (
    "hf_bucket/experiments/mlx_lora/review_comparisons/model_e_vs_model_f_holdout_100_q4lineage",
    "hf_bucket/experiments/mlx_lora/review_comparisons/model_g_vs_model_h_fresh_100_q4lineage_seed_20260501",
    "hf_bucket/experiments/mlx_lora/review_comparisons/model_m_vs_model_n_fresh_100_q4lineage_seed_2026050102_no_prior",
    "hf_bucket/experiments/mlx_lora/review_comparisons/model_i_vs_model_j_holdout_100_q4lineage_posthoc_check",
    "hf_bucket/experiments/mlx_lora/review_comparisons/model_k_vs_model_l_fresh_100_q4lineage_posthoc_check_seed_20260501",
)

DEFAULT_BUCKET_TARGETS = {
    "numeric": 350,
    "polarity": 250,
    "entity": 200,
    "long_notation": 100,
    "control": 100,
}

NUMERIC_RE = re.compile(
    r"\b\d+(?:,\d{3})*(?:\.\d+)?(?:%|x|X|×|k|K|m|M|b|B|ms|s|nm|kg|GB|MB)?\b|"
    r"\$[^$]{1,80}\$|\\(?:frac|sum|int|alpha|beta|gamma|lambda|nu|sigma|theta)\b|"
    r"\b(?:theorem|lemma|proposition|corollary|equation|percent|percentage|fold|factor)\b",
    re.IGNORECASE,
)
POLARITY_RE = re.compile(
    r"\b(?:not|no|without|never|neither|nor|fail(?:s|ed)?|cannot|can't|"
    r"does not|do not|did not|rather than|instead of|although|however|"
    r"but|whereas|despite|contrast|contradict|overestimat\w*|underestimat\w*|"
    r"increase\w*|decrease\w*|reduce\w*|improve\w*|worse|better|less|more)\b",
    re.IGNORECASE,
)
ENTITY_ACRONYM_RE = re.compile(
    r"\b(?:[A-Z]{2,}[A-Za-z0-9-]*|"
    r"APT|NASA|ESA|TESS|GAIA|LAMOST|CNN|XGBoost|LLM|FPGA|GPU|CPU|"
    r"SARS-CoV-2|COVID-19)\b"
)
ENTITY_KEYWORD_RE = re.compile(
    r"\b(?:malware|ransomware|cyber|trojan|russia|ukraine|china|india|pakistan|"
    r"clinical|patient|cancer|tumou?r|disease|protein|virus)\b",
    re.IGNORECASE,
)
LONG_NOTATION_RE = re.compile(
    r"\$[^$]{20,}\$|\\[A-Za-z]+|[{}_^]|(?:\b\w+\b[,\s]+){180,}|"
    r"\b(?:appendix|proof|notation|operator|Hamiltonian|Lagrangian|manifold|"
    r"non-Hermitian|quasiperiodic|hypergeometric)\b",
    re.IGNORECASE,
)

BUCKET_REGEXES = {
    "numeric": NUMERIC_RE,
    "polarity": POLARITY_RE,
    "long_notation": LONG_NOTATION_RE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic held-out q4 severe-case mining set."
    )
    parser.add_argument("--corpus-jsonl", default=DEFAULT_CORPUS_JSONL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--numeric-count", type=int, default=DEFAULT_BUCKET_TARGETS["numeric"])
    parser.add_argument("--polarity-count", type=int, default=DEFAULT_BUCKET_TARGETS["polarity"])
    parser.add_argument("--entity-count", type=int, default=DEFAULT_BUCKET_TARGETS["entity"])
    parser.add_argument(
        "--long-notation-count",
        type=int,
        default=DEFAULT_BUCKET_TARGETS["long_notation"],
    )
    parser.add_argument("--control-count", type=int, default=DEFAULT_BUCKET_TARGETS["control"])
    parser.add_argument(
        "--exclude-release",
        action="append",
        default=list(DEFAULT_EXCLUDE_RELEASES),
        help="Release directory or training_rows.jsonl whose paper_ids must be excluded.",
    )
    parser.add_argument(
        "--exclude-review-dir",
        action="append",
        default=list(DEFAULT_PRIOR_REVIEW_DIRS),
        help="Prior comparison directory whose manifest/generations paper_ids must be excluded.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def paper_text(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(row.get("title", "")),
            str(row.get("abstract", "")),
        ]
    )


def row_axes(row: dict[str, Any]) -> list[str]:
    text = paper_text(row)
    axes = [axis for axis, pattern in BUCKET_REGEXES.items() if pattern.search(text)]
    if ENTITY_ACRONYM_RE.search(text) or ENTITY_KEYWORD_RE.search(text):
        axes.append("entity")
    return axes


def release_training_path(path: Path) -> Path:
    return path / "training_rows.jsonl" if path.is_dir() else path


def collect_training_ids(paths: list[str]) -> dict[str, set[str]]:
    ids_by_source: dict[str, set[str]] = {}
    for raw_path in paths:
        path = release_training_path(Path(raw_path))
        if not path.exists():
            ids_by_source[str(path)] = set()
            continue
        ids_by_source[str(path)] = {str(row["paper_id"]) for row in load_jsonl(path)}
    return ids_by_source


def collect_prior_review_ids(paths: list[str]) -> dict[str, set[str]]:
    ids_by_source: dict[str, set[str]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        ids: set[str] = set()
        manifest_path = path / "manifest.json"
        generations_path = path / "generations.jsonl"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            ids.update(str(item) for item in manifest.get("paper_ids", []))
        if generations_path.exists():
            ids.update(str(row["paper_id"]) for row in load_jsonl(generations_path))
        ids_by_source[str(path)] = ids
    return ids_by_source


def sample_axis(
    rows: list[dict[str, Any]],
    *,
    axis: str,
    count: int,
    selected_ids: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows if row["paper_id"] not in selected_ids and axis in row["q4_hardening_axes"]
    ]
    if count > len(candidates):
        raise ValueError(f"Need {count} rows for {axis}, found {len(candidates)}")
    rng.shuffle(candidates)
    chosen = candidates[:count]
    selected_ids.update(str(row["paper_id"]) for row in chosen)
    return chosen


def sample_controls(
    rows: list[dict[str, Any]],
    *,
    count: int,
    selected_ids: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row["paper_id"] not in selected_ids]
    if count > len(candidates):
        raise ValueError(f"Need {count} control rows, found {len(candidates)}")

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        buckets[str(row.get("umbrella_category") or "unknown")].append(row)

    targets: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    total = len(candidates)
    for bucket, bucket_rows in buckets.items():
        exact = count * len(bucket_rows) / total
        base = int(exact)
        targets[bucket] = base
        remainders.append((exact - base, bucket))

    for _, bucket in sorted(remainders, reverse=True)[: count - sum(targets.values())]:
        targets[bucket] += 1

    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for bucket in sorted(buckets):
        bucket_rows = buckets[bucket][:]
        rng.shuffle(bucket_rows)
        take = min(targets[bucket], len(bucket_rows))
        selected.extend(bucket_rows[:take])
        leftovers.extend(bucket_rows[take:])

    if len(selected) < count:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: count - len(selected)])

    rng.shuffle(selected)
    chosen = selected[:count]
    selected_ids.update(str(row["paper_id"]) for row in chosen)
    return chosen


def build_eval_row(row: dict[str, Any], bucket: str, index: int) -> dict[str, Any]:
    return {
        "eval_id": f"{bucket}_{index:04d}",
        "split": bucket,
        "selection_bucket": bucket,
        "q4_hardening_axes": row.get("q4_hardening_axes", []),
        "paper_id": row["paper_id"],
        "title": row["title"],
        "abstract": row["abstract"],
        "primary_category": row.get("primary_category"),
        "umbrella_category": row.get("umbrella_category"),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date"),
    }


def build_eval_rows(
    corpus_rows: list[dict[str, Any]],
    *,
    excluded_ids: set[str],
    bucket_targets: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    eligible_rows: list[dict[str, Any]] = []
    for row in corpus_rows:
        paper_id = str(row["paper_id"])
        if paper_id in excluded_ids:
            continue
        copied = dict(row)
        copied["paper_id"] = paper_id
        copied["q4_hardening_axes"] = row_axes(copied)
        eligible_rows.append(copied)

    selected_ids: set[str] = set()
    selected_by_bucket: dict[str, list[dict[str, Any]]] = {}
    for axis in ("numeric", "polarity", "entity", "long_notation"):
        selected_by_bucket[axis] = sample_axis(
            eligible_rows,
            axis=axis,
            count=bucket_targets[axis],
            selected_ids=selected_ids,
            rng=rng,
        )
    selected_by_bucket["control"] = sample_controls(
        eligible_rows,
        count=bucket_targets["control"],
        selected_ids=selected_ids,
        rng=rng,
    )

    eval_rows: list[dict[str, Any]] = []
    for bucket in ("numeric", "polarity", "entity", "long_notation", "control"):
        bucket_rows = selected_by_bucket[bucket][:]
        rng.shuffle(bucket_rows)
        eval_rows.extend(
            build_eval_row(row, bucket, index)
            for index, row in enumerate(bucket_rows, start=1)
        )
    return eval_rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "by_selection_bucket": dict(Counter(row["selection_bucket"] for row in rows)),
        "by_umbrella_category": dict(Counter(row.get("umbrella_category") for row in rows)),
        "by_primary_category_top_25": dict(
            Counter(row.get("primary_category") for row in rows).most_common(25)
        ),
        "by_axis": dict(
            Counter(axis for row in rows for axis in row.get("q4_hardening_axes", []))
        ),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{output_dir} already exists and is not empty; pass --overwrite")

    training_ids_by_source = collect_training_ids(args.exclude_release)
    prior_ids_by_source = collect_prior_review_ids(args.exclude_review_dir)
    excluded_ids = set().union(*training_ids_by_source.values(), *prior_ids_by_source.values())

    bucket_targets = {
        "numeric": args.numeric_count,
        "polarity": args.polarity_count,
        "entity": args.entity_count,
        "long_notation": args.long_notation_count,
        "control": args.control_count,
    }
    corpus_rows = load_jsonl(Path(args.corpus_jsonl))
    eval_rows = build_eval_rows(
        corpus_rows,
        excluded_ids=excluded_ids,
        bucket_targets=bucket_targets,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "eval_inputs.jsonl", eval_rows)
    write_json(output_dir / "paper_ids.json", {"paper_ids": [row["paper_id"] for row in eval_rows]})
    manifest = {
        "created_at": utc_now_iso(),
        "corpus_jsonl": args.corpus_jsonl,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "bucket_targets": bucket_targets,
        "exclude_releases": args.exclude_release,
        "exclude_review_dirs": args.exclude_review_dir,
        "excluded_training_ids_by_source": {
            source: len(ids) for source, ids in training_ids_by_source.items()
        },
        "excluded_prior_review_ids_by_source": {
            source: len(ids) for source, ids in prior_ids_by_source.items()
        },
        "excluded_unique_paper_ids": len(excluded_ids),
        "summary": summarize(eval_rows),
    }
    write_json(output_dir / "manifest.json", manifest)
    print(f"[q4-hardening-eval] wrote {len(eval_rows)} rows to {output_dir}")
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
