#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CORPUS = Path("hf_dataset/corpora/recent_3y_balanced/arxiv_abstract_corpus_weighted_20000.jsonl")
DEFAULT_BASE_GOLD = Path(
    "hf_dataset/releases/"
    "recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_604_expansion_pool/"
    "training_rows.jsonl"
)
DEFAULT_RUN_NAME = "deepseek_v4_pro_v17_ratio_repair_800"
DEFAULT_NONMATH_TEMPLATE = Path(
    "configs/synthetic_summary_generation/"
    "natural_language_teacher_deepseek_v4_pro_v10_ratio_fill_500_nonmath.json"
)
DEFAULT_NONMATH_AUDIT_TEMPLATE = Path(
    "configs/synthetic_summary_generation/"
    "natural_language_teacher_deepseek_v4_pro_v10_ratio_fill_500_nonmath_audit.json"
)
DEFAULT_MATH_TEMPLATE = Path(
    "configs/synthetic_summary_generation/"
    "natural_language_teacher_deepseek_v4_pro_v15_math_ratio_fill_500.json"
)
DEFAULT_MATH_AUDIT_TEMPLATE = Path(
    "configs/synthetic_summary_generation/"
    "natural_language_teacher_deepseek_v4_pro_v15_math_ratio_fill_500_audit.json"
)

TARGET_SHARES = {
    "cs": 0.37,
    "physics": 0.28,
    "math": 0.16,
    "other": 0.19,
}

EXPECTED_STRICT_YIELDS = {
    "cs": 0.22,
    "physics": 0.20,
    "math": 0.55,
    "other": 0.21,
}

PHYSICS_PREFIXES = (
    "astro-ph",
    "cond-mat",
    "gr-qc",
    "hep-",
    "math-ph",
    "nucl-",
    "physics.",
    "quant-ph",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(Path(".").resolve()))
    except ValueError:
        return str(path)


def collapsed_umbrella(row: dict[str, Any]) -> str:
    primary = str(row.get("primary_category", "")).strip()
    umbrella = str(row.get("umbrella_category", "")).strip()
    if primary.startswith("cs."):
        return "cs"
    if primary.startswith("math."):
        return "math"
    if umbrella == "physics" or primary.startswith(PHYSICS_PREFIXES):
        return "physics"
    return "other"


def target_counts(target_size: int) -> dict[str, int]:
    raw = {bucket: TARGET_SHARES[bucket] * target_size for bucket in TARGET_SHARES}
    counts = {bucket: math.floor(value) for bucket, value in raw.items()}
    remainder = target_size - sum(counts.values())
    for bucket, _ in sorted(
        ((bucket, raw[bucket] - counts[bucket]) for bucket in TARGET_SHARES),
        key=lambda item: (-item[1], item[0]),
    )[:remainder]:
        counts[bucket] += 1
    return counts


def load_excluded_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        for row in load_jsonl(path):
            paper_id = str(row.get("paper_id", "")).strip()
            if paper_id:
                excluded.add(paper_id)
    return excluded


def row_sort_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("paper_id", "")), str(row.get("title", "")))


def select_bucket_rows(
    rows: list[dict[str, Any]],
    *,
    bucket: str,
    quota: int,
    seed: int,
    max_per_primary_category: int,
) -> list[dict[str, Any]]:
    if quota <= 0:
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if collapsed_umbrella(row) != bucket:
            continue
        grouped[str(row.get("primary_category") or "unknown")].append(row)

    rng = random.Random(seed)
    categories = sorted(grouped)
    rng.shuffle(categories)
    for category_rows in grouped.values():
        category_rows.sort(key=row_sort_key)
        rng.shuffle(category_rows)

    selected: list[dict[str, Any]] = []
    primary_counts: Counter[str] = Counter()
    category_offsets: Counter[str] = Counter()
    while len(selected) < quota:
        made_progress = False
        for category in categories:
            if primary_counts[category] >= max_per_primary_category:
                continue
            offset = category_offsets[category]
            if offset >= len(grouped[category]):
                continue
            selected.append(grouped[category][offset])
            category_offsets[category] += 1
            primary_counts[category] += 1
            made_progress = True
            if len(selected) >= quota:
                break
        if not made_progress:
            break
    return selected


def materialize_candidates(
    rows: list[dict[str, Any]],
    *,
    candidate_prefix: str,
    source_split: str,
    seed: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        paper_id = str(row.get("paper_id", "")).strip()
        candidates.append(
            {
                "candidate_id": f"{candidate_prefix}:{index:05d}:{paper_id}",
                "paper_id": paper_id,
                "title": row.get("title", ""),
                "abstract": row.get("abstract", ""),
                "primary_category": row.get("primary_category", ""),
                "umbrella_category": row.get("umbrella_category", ""),
                "categories": row.get("categories", []),
                "created_date": row.get("created_date", ""),
                "updated_date": row.get("updated_date", ""),
                "source_split": source_split,
                "selection_seed": seed,
                "selection_rank": index,
                "ratio_repair_bucket": collapsed_umbrella(row),
            }
        )
    return candidates


def selected_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(collapsed_umbrella(row) for row in rows).items()))


def selected_primary_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("primary_category") or "unknown") for row in rows).items()))


def build_normal_config(
    *,
    template_path: Path,
    input_path: Path,
    run_root: Path,
    release_dir: Path,
    candidate_prefix: str,
    source_split: str,
    workers: int,
    math: bool,
) -> dict[str, Any]:
    config = load_json(template_path)
    config["input"]["path"] = repo_rel(input_path)
    config["input"]["candidate_prefix"] = candidate_prefix
    config["input"]["count_fields"] = [
        "ratio_repair_bucket",
        "umbrella_category",
        "primary_category",
    ]
    config["output"]["run_root"] = repo_rel(run_root)
    config["output"]["release_dir"] = repo_rel(release_dir)
    config["writer"]["workers"] = workers
    config["reviewer"]["workers"] = workers
    config["materialize"]["source_split"] = source_split
    config["materialize"].setdefault("provenance_fields", [])
    for field in ("ratio_repair_bucket",):
        if field not in config["materialize"]["provenance_fields"]:
            config["materialize"]["provenance_fields"].append(field)
    config["reviewer"].pop("input_path", None)
    if math:
        config["writer"]["prompt_path"] = "prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_writer_v16_math.json"
        config["writer"]["writer_label"] = "deepseek_v4_pro_plain_language_v16_math"
        config["reviewer"]["prompt_path"] = "prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_verifier_v6_math.json"
        gates = config.setdefault("local_gates", {})
        gates["max_writer_technical_terms"] = max(
            int(gates.get("max_writer_technical_terms", 0) or 0), 4
        )
        gates["max_long_words"] = max(int(gates.get("max_long_words", 0) or 0), 3)
        gates["max_acronyms"] = max(int(gates.get("max_acronyms", 0) or 0), 2)
    return config


def build_audit_config(
    *,
    normal_config: dict[str, Any],
    audit_template_path: Path,
    normal_run_root: Path,
    audit_run_root: Path,
    audit_release_dir: Path,
    candidate_prefix: str,
    source_split: str,
    workers: int,
    math: bool,
) -> dict[str, Any]:
    audit_config = load_json(audit_template_path)
    audit_config.update(json.loads(json.dumps(normal_config)))
    audit_config["input"]["candidate_prefix"] = f"{candidate_prefix}_audit"
    audit_config["output"]["run_root"] = repo_rel(audit_run_root)
    audit_config["output"]["release_dir"] = repo_rel(audit_release_dir)
    audit_config["reviewer"]["workers"] = workers
    audit_config["reviewer"]["input_path"] = repo_rel(
        normal_run_root / "materialized_release" / "selected_review_rows.jsonl"
    )
    audit_config["materialize"]["source_split"] = f"{source_split}_audit"
    audit_config["reviewer"]["prompt_path"] = (
        "prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_verifier_v6_math.json"
        if math
        else "prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_verifier_v4.json"
    )
    return audit_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ratio-repair source pools and configs for natural-language gold generation."
    )
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--base-gold-jsonl", type=Path, default=DEFAULT_BASE_GOLD)
    parser.add_argument("--target-size", type=int, default=800)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--oversample-multiplier", type=float, default=1.25)
    parser.add_argument("--max-per-primary-category", type=int, default=18)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--only-bucket",
        action="append",
        choices=sorted(TARGET_SHARES),
        default=[],
        help="Build source candidates only for this collapsed bucket. Repeatable.",
    )
    parser.add_argument(
        "--source-quota",
        action="append",
        default=[],
        metavar="BUCKET=COUNT",
        help="Override the computed source quota for one bucket.",
    )
    parser.add_argument(
        "--exclude-jsonl",
        action="append",
        type=Path,
        default=[],
        help="Additional JSONL sources whose paper_id values should be skipped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus_rows = load_jsonl(args.input_jsonl)
    base_rows = load_jsonl(args.base_gold_jsonl)
    excluded_ids = load_excluded_ids([args.base_gold_jsonl, *args.exclude_jsonl])
    usable_rows = [
        row
        for row in corpus_rows
        if str(row.get("paper_id", "")).strip()
        and str(row.get("abstract", "")).strip()
        and str(row.get("paper_id", "")).strip() not in excluded_ids
    ]

    current_counts = Counter(collapsed_umbrella(row) for row in base_rows)
    targets = target_counts(args.target_size)
    needed = {
        bucket: max(targets[bucket] - current_counts.get(bucket, 0), 0)
        for bucket in TARGET_SHARES
    }
    source_quotas = {
        bucket: (
            0
            if needed[bucket] == 0
            else math.ceil(
                needed[bucket]
                / EXPECTED_STRICT_YIELDS[bucket]
                * args.oversample_multiplier
            )
        )
        for bucket in TARGET_SHARES
    }
    quota_overrides: dict[str, int] = {}
    for item in args.source_quota:
        if "=" not in item:
            raise ValueError(f"--source-quota must look like bucket=count, got {item!r}")
        bucket, count_text = item.split("=", 1)
        if bucket not in TARGET_SHARES:
            raise ValueError(f"Unknown bucket for --source-quota: {bucket}")
        quota_overrides[bucket] = int(count_text)
    source_quotas.update(quota_overrides)
    if args.only_bucket:
        allowed_buckets = set(args.only_bucket)
        source_quotas = {
            bucket: quota if bucket in allowed_buckets else 0
            for bucket, quota in source_quotas.items()
        }

    selected_by_bucket = {
        bucket: select_bucket_rows(
            usable_rows,
            bucket=bucket,
            quota=source_quotas[bucket],
            seed=args.seed + index * 997,
            max_per_primary_category=args.max_per_primary_category,
        )
        for index, bucket in enumerate(TARGET_SHARES)
    }

    run_base = Path("hf_bucket/experiments/natural_language_teacher")
    release_base = Path("hf_dataset/releases")
    config_base = Path("configs/synthetic_summary_generation")
    nonmath_name = f"{args.run_name}_nonmath"
    math_name = f"{args.run_name}_math"
    nonmath_run_root = run_base / nonmath_name
    math_run_root = run_base / math_name
    nonmath_release_dir = release_base / f"recent_3y_balanced_20000_natural_language_teacher_{nonmath_name}"
    math_release_dir = release_base / f"recent_3y_balanced_20000_natural_language_teacher_{math_name}"
    nonmath_audit_run_root = run_base / f"{nonmath_name}_audit"
    math_audit_run_root = run_base / f"{math_name}_audit"
    nonmath_audit_release_dir = release_base / f"recent_3y_balanced_20000_natural_language_teacher_{nonmath_name}_audit"
    math_audit_release_dir = release_base / f"recent_3y_balanced_20000_natural_language_teacher_{math_name}_audit"

    nonmath_rows = [
        row
        for bucket in ("cs", "physics", "other")
        for row in selected_by_bucket[bucket]
    ]
    math_rows = selected_by_bucket["math"]
    nonmath_candidates = materialize_candidates(
        nonmath_rows,
        candidate_prefix=f"natural_language_teacher_{nonmath_name}",
        source_split=f"recent_3y_balanced_natural_language_teacher_{nonmath_name}",
        seed=args.seed,
    )
    math_candidates = materialize_candidates(
        math_rows,
        candidate_prefix=f"natural_language_teacher_{math_name}",
        source_split=f"recent_3y_balanced_natural_language_teacher_{math_name}",
        seed=args.seed,
    )

    nonmath_source_path = nonmath_run_root / "source_candidates" / "natural_language_candidates.jsonl"
    math_source_path = math_run_root / "source_candidates" / "natural_language_candidates.jsonl"
    write_jsonl(nonmath_source_path, nonmath_candidates, overwrite=args.overwrite)
    write_jsonl(math_source_path, math_candidates, overwrite=args.overwrite)

    common_manifest = {
        "created_at": utc_now_iso(),
        "input_jsonl": str(args.input_jsonl),
        "base_gold_jsonl": str(args.base_gold_jsonl),
        "target_size": args.target_size,
        "target_counts": targets,
        "current_counts": dict(sorted(current_counts.items())),
        "needed_accepted_counts": needed,
        "expected_strict_yields": EXPECTED_STRICT_YIELDS,
        "oversample_multiplier": args.oversample_multiplier,
        "source_quotas": source_quotas,
        "source_quota_overrides": quota_overrides,
        "only_buckets": args.only_bucket,
        "corpus_rows": len(corpus_rows),
        "excluded_paper_ids": len(excluded_ids),
        "usable_rows": len(usable_rows),
        "max_per_primary_category": args.max_per_primary_category,
        "seed": args.seed,
    }
    nonmath_manifest = {
        **common_manifest,
        "output_jsonl": str(nonmath_source_path),
        "candidate_rows": len(nonmath_candidates),
        "collapsed_umbrella_counts": selected_counts(nonmath_candidates),
        "primary_category_counts": selected_primary_counts(nonmath_candidates),
        "paired_math_run_root": str(math_run_root),
    }
    math_manifest = {
        **common_manifest,
        "output_jsonl": str(math_source_path),
        "candidate_rows": len(math_candidates),
        "collapsed_umbrella_counts": selected_counts(math_candidates),
        "primary_category_counts": selected_primary_counts(math_candidates),
        "paired_nonmath_run_root": str(nonmath_run_root),
    }
    write_json(nonmath_source_path.with_name("run_manifest.json"), nonmath_manifest, overwrite=args.overwrite)
    write_json(math_source_path.with_name("run_manifest.json"), math_manifest, overwrite=args.overwrite)

    nonmath_config = build_normal_config(
        template_path=DEFAULT_NONMATH_TEMPLATE,
        input_path=nonmath_source_path,
        run_root=nonmath_run_root,
        release_dir=nonmath_release_dir,
        candidate_prefix=f"natural_language_teacher_{nonmath_name}",
        source_split=f"recent_3y_balanced_natural_language_teacher_{nonmath_name}",
        workers=args.workers,
        math=False,
    )
    math_config = build_normal_config(
        template_path=DEFAULT_MATH_TEMPLATE,
        input_path=math_source_path,
        run_root=math_run_root,
        release_dir=math_release_dir,
        candidate_prefix=f"natural_language_teacher_{math_name}",
        source_split=f"recent_3y_balanced_natural_language_teacher_{math_name}",
        workers=args.workers,
        math=True,
    )
    nonmath_audit_config = build_audit_config(
        normal_config=nonmath_config,
        audit_template_path=DEFAULT_NONMATH_AUDIT_TEMPLATE,
        normal_run_root=nonmath_run_root,
        audit_run_root=nonmath_audit_run_root,
        audit_release_dir=nonmath_audit_release_dir,
        candidate_prefix=f"natural_language_teacher_{nonmath_name}",
        source_split=f"recent_3y_balanced_natural_language_teacher_{nonmath_name}",
        workers=args.workers,
        math=False,
    )
    math_audit_config = build_audit_config(
        normal_config=math_config,
        audit_template_path=DEFAULT_MATH_AUDIT_TEMPLATE,
        normal_run_root=math_run_root,
        audit_run_root=math_audit_run_root,
        audit_release_dir=math_audit_release_dir,
        candidate_prefix=f"natural_language_teacher_{math_name}",
        source_split=f"recent_3y_balanced_natural_language_teacher_{math_name}",
        workers=args.workers,
        math=True,
    )

    nonmath_config_path = config_base / f"natural_language_teacher_{nonmath_name}.json"
    math_config_path = config_base / f"natural_language_teacher_{math_name}.json"
    nonmath_audit_config_path = config_base / f"natural_language_teacher_{nonmath_name}_audit.json"
    math_audit_config_path = config_base / f"natural_language_teacher_{math_name}_audit.json"
    write_json(nonmath_config_path, nonmath_config, overwrite=args.overwrite)
    write_json(math_config_path, math_config, overwrite=args.overwrite)
    write_json(nonmath_audit_config_path, nonmath_audit_config, overwrite=args.overwrite)
    write_json(math_audit_config_path, math_audit_config, overwrite=args.overwrite)

    plan = {
        **common_manifest,
        "run_name": args.run_name,
        "nonmath": {
            "config": str(nonmath_config_path),
            "audit_config": str(nonmath_audit_config_path),
            "run_root": str(nonmath_run_root),
            "audit_run_root": str(nonmath_audit_run_root),
            "release_dir": str(nonmath_release_dir),
            "audit_release_dir": str(nonmath_audit_release_dir),
            "candidate_rows": len(nonmath_candidates),
            "collapsed_umbrella_counts": selected_counts(nonmath_candidates),
        },
        "math": {
            "config": str(math_config_path),
            "audit_config": str(math_audit_config_path),
            "run_root": str(math_run_root),
            "audit_run_root": str(math_audit_run_root),
            "release_dir": str(math_release_dir),
            "audit_release_dir": str(math_audit_release_dir),
            "candidate_rows": len(math_candidates),
            "collapsed_umbrella_counts": selected_counts(math_candidates),
        },
    }
    plan_path = run_base / args.run_name / "ratio_repair_plan.json"
    write_json(plan_path, plan, overwrite=args.overwrite)
    print(f"[ratio-repair] wrote plan: {plan_path}")
    print(f"[ratio-repair] nonmath candidates: {len(nonmath_candidates)} {selected_counts(nonmath_candidates)}")
    print(f"[ratio-repair] math candidates: {len(math_candidates)} {selected_counts(math_candidates)}")
    print(f"[ratio-repair] needed accepted: {needed}")


if __name__ == "__main__":
    main()
