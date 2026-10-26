#!/usr/bin/env python3
"""Build a mixed natural-language plus low-weight anchor SFT dataset."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_NL_DATASET = Path(
    "hf_dataset/releases/recent_3y_balanced_20000_natural_language_teacher_deepseek_v4_pro_500_milestone/"
    "training_rows.jsonl"
)
DEFAULT_ANCHORS = Path(
    "hf_dataset/releases/recent_3y_balanced_10746_faithfulness_overlay_1400_v2/training_rows.jsonl"
)
DEFAULT_EVAL = Path(
    "hf_bucket/experiments/mlx_lora/gemma_3_270m_it_natural_language_500_lora_v1/"
    "review_samples/ckpt120_vs_ckpt220_fresh20/generations.jsonl"
)
DEFAULT_PROMPT_SPEC = Path("prompts/mlx_lora/gemma_270m_it_abstract_to_natural_language_summary_v1.json")
DEFAULT_OUTPUT_DIR = Path(
    "hf_bucket/experiments/sft/q5_promoted_q8_sidecar_natural_language_500_anchor_mix_v1/"
    "dataset_500nl_1000faithanchors_w025"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nl-dataset", type=Path, default=DEFAULT_NL_DATASET)
    parser.add_argument("--anchor-source", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--eval-jsonl", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--prompt-spec", type=Path, default=DEFAULT_PROMPT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anchor-count", type=int, default=1000)
    parser.add_argument("--anchor-weight", type=float, default=0.25)
    parser.add_argument("--nl-weight", type=float, default=1.0)
    parser.add_argument("--nl-component-name", default="natural_language_500")
    parser.add_argument("--seed", type=int, default=20260522)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def norm(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def load_prompt_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    required = ("prompt_version", "system_prompt", "user_prompt_templates")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"Prompt spec {path} missing keys: {missing}")
    if "title_abstract" not in spec["user_prompt_templates"]:
        raise ValueError(f"Prompt spec {path} is missing title_abstract template")
    return spec


def render_messages(row: dict[str, Any], summary: str, prompt_spec: dict[str, Any]) -> list[dict[str, str]]:
    title = norm(row.get("title"))
    abstract = norm(row.get("abstract"))
    if not title or not abstract or not summary:
        raise ValueError(f"Missing title, abstract, or summary for paper_id={row.get('paper_id')}")
    user = str(prompt_spec["user_prompt_templates"]["title_abstract"]).format(title=title, abstract=abstract)
    return [
        {"role": "system", "content": str(prompt_spec["system_prompt"])},
        {"role": "user", "content": user},
        {"role": "assistant", "content": summary},
    ]


def row_summary(row: dict[str, Any]) -> str:
    if row.get("summary"):
        return norm(row["summary"])
    messages = row.get("messages") or []
    if messages and isinstance(messages, list):
        return norm(messages[-1].get("content", ""))
    return ""


def convert_row(
    row: dict[str, Any],
    *,
    component: str,
    weight: float,
    prompt_spec: dict[str, Any],
    source_path: Path,
) -> dict[str, Any]:
    summary = row_summary(row)
    out = {
        "paper_id": str(row["paper_id"]),
        "title": norm(row.get("title")),
        "abstract": norm(row.get("abstract")),
        "summary": summary,
        "primary_category": row.get("primary_category", ""),
        "categories": row.get("categories", []),
        "created_date": row.get("created_date", ""),
        "source_split": row.get("source_split", ""),
        "teacher_model": row.get("teacher_model", ""),
        "generator_prompt_version": row.get("generator_prompt_version", ""),
        "review_prompt_version": row.get("review_prompt_version", ""),
        "dataset_component": component,
        "prompt_version": prompt_spec["prompt_version"],
        "input_mode": "title_abstract",
        "messages": render_messages(row, summary, prompt_spec),
        "loss_weight_hints": {
            "enabled": True,
            "base_weight": weight,
            "spans": [],
            "policy": f"q5_natural_language_anchor_mix_v1 {component} weight {weight:g}",
        },
        "provenance": {
            "source_path": str(source_path),
            "original_source_split": row.get("source_split", ""),
            "original_prompt_version": row.get("prompt_version", ""),
        },
    }
    if not out["paper_id"] or not out["title"] or not out["abstract"] or not out["summary"]:
        raise ValueError(f"Incomplete row after conversion from {source_path}: {row.get('paper_id')}")
    return out


def main() -> None:
    args = parse_args()
    if args.anchor_count <= 0:
        raise ValueError("--anchor-count must be positive")
    if args.anchor_weight <= 0 or args.nl_weight <= 0:
        raise ValueError("weights must be positive")

    prompt_spec = load_prompt_spec(args.prompt_spec)
    nl_rows = load_jsonl(args.nl_dataset)
    eval_ids = {str(row["paper_id"]) for row in load_jsonl(args.eval_jsonl)}
    nl_ids = {str(row["paper_id"]) for row in nl_rows}

    anchor_candidates = [
        row
        for row in load_jsonl(args.anchor_source)
        if str(row.get("paper_id")) not in nl_ids and str(row.get("paper_id")) not in eval_ids
    ]
    if len(anchor_candidates) < args.anchor_count:
        raise ValueError(f"Need {args.anchor_count} anchors, found {len(anchor_candidates)}")

    rng = random.Random(args.seed)
    rng.shuffle(anchor_candidates)
    anchor_rows = anchor_candidates[: args.anchor_count]

    converted = [
        convert_row(
            row,
            component=args.nl_component_name,
            weight=args.nl_weight,
            prompt_spec=prompt_spec,
            source_path=args.nl_dataset,
        )
        for row in nl_rows
    ]
    converted.extend(
        convert_row(
            row,
            component="faithfulness_anchor",
            weight=args.anchor_weight,
            prompt_spec=prompt_spec,
            source_path=args.anchor_source,
        )
        for row in anchor_rows
    )
    rng.shuffle(converted)

    output_path = args.output_dir / "train.jsonl"
    write_jsonl(output_path, converted)

    component_counts = Counter(row["dataset_component"] for row in converted)
    category_counts = Counter(norm(row.get("primary_category")) for row in converted)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": "scripts/build_q5_natural_language_anchor_mix_v1_dataset.py",
        "purpose": "Natural-language q5 sidecar SFT mix with low-weight faithfulness anchors.",
        "output_path": str(output_path),
        "prompt_spec": str(args.prompt_spec),
        "prompt_version": prompt_spec["prompt_version"],
        "input_mode": "title_abstract",
        "row_count": len(converted),
        "component_counts": dict(sorted(component_counts.items())),
        "weights": {
            args.nl_component_name: args.nl_weight,
            "faithfulness_anchor": args.anchor_weight,
        },
        "sources": {
            "natural_language": str(args.nl_dataset),
            "anchor": str(args.anchor_source),
            "eval_excluded": str(args.eval_jsonl),
        },
        "anchor_selection": {
            "seed": args.seed,
            "requested_count": args.anchor_count,
            "candidate_count_after_exclusions": len(anchor_candidates),
            "excluded_natural_language_ids": len(nl_ids),
            "excluded_eval_ids": len(eval_ids),
        },
        "primary_category_top20": dict(category_counts.most_common(20)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme = [
        "# q5 Natural-Language Anchor Mix v1",
        "",
        f"- Rows: {len(converted)}",
        f"- Natural-language rows: {component_counts[args.nl_component_name]} at weight {args.nl_weight:g}",
        f"- Faithfulness anchors: {component_counts['faithfulness_anchor']} at weight {args.anchor_weight:g}",
        f"- Prompt: `{prompt_spec['prompt_version']}` / `title_abstract`",
        "",
        "The anchor rows use the natural-language prompt but keep their older stable summaries.",
        "They are low-weight replay targets, not the desired final style.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"[anchor-mix] wrote {len(converted)} rows to {output_path}")
    print(f"[anchor-mix] components={dict(sorted(component_counts.items()))}")
    print(f"[anchor-mix] anchor candidates after exclusions={len(anchor_candidates)}")


if __name__ == "__main__":
    main()
