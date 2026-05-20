#!/usr/bin/env python3
"""Build DPO pairs from the q8-vs-promoted-q8-adapter-q5-sidecar blind review."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from dpo_utils import build_prompt_messages, load_prompt_spec, write_jsonl


DEFAULT_REVIEW = Path(
    "hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1/eval/"
    "blind_ab_q8_vs_promoted_q8_adapter_q5_sidecar_20260518_500/"
    "blind_review_merged_unblinded.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "hf_bucket/experiments/sft/q5_promoted_q8_adapter_sidecar_probe_v1/"
    "dpo_blind_q8_vs_sidecar_v1/dataset"
)
DEFAULT_PROMPT_SPEC = Path(
    "prompts/mlx_lora/gemma_270m_it_abstract_to_glanceable_summary_v1.json"
)
Q8_MODEL = "q8_mlx_python"
Q5_SIDECAR_MODEL = "q5_promoted_q8_adapter_q5_sidecar"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def norm(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def summary_for(row: dict[str, Any], model: str) -> str:
    if row.get("summary_a_model") == model:
        return norm(row.get("summary_a"))
    if row.get("summary_b_model") == model:
        return norm(row.get("summary_b"))
    return ""


def split_name(index: int) -> str:
    mod = index % 10
    if mod == 0:
        return "test"
    if mod == 1:
        return "valid"
    return "train"


def build_pair(
    row: dict[str, Any],
    *,
    chosen_model: str,
    rejected_model: str,
    prompt_spec: dict[str, Any],
) -> dict[str, Any]:
    chosen = summary_for(row, chosen_model)
    rejected = summary_for(row, rejected_model)
    if not chosen or not rejected:
        raise ValueError(f"Missing chosen/rejected text for {row.get('paper_id')}")
    return {
        "paper_id": str(row.get("paper_id", "")),
        "review_id": str(row.get("review_id", "")),
        "split": str(row.get("split", "")),
        "prompt": build_prompt_messages(row, prompt_spec, "title_abstract"),
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "chosen_model": chosen_model,
        "rejected_model": rejected_model,
        "preferred_model": row.get("preferred_model", ""),
        "more_accurate_model": row.get("more_accurate_model", ""),
        "review_notes": row.get("review_notes", ""),
        "summary_a_model": row.get("summary_a_model", ""),
        "summary_b_model": row.get("summary_b_model", ""),
        "source_kind": "blind_review_q8_vs_promoted_q8_adapter_q5_sidecar",
        "prompt_version": str(prompt_spec["prompt_version"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-jsonl", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-spec", type=Path, default=DEFAULT_PROMPT_SPEC)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} is not empty; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_spec = load_prompt_spec(args.prompt_spec)

    pairs: list[dict[str, Any]] = []
    skipped = collections.Counter()
    for row in load_jsonl(args.review_jsonl):
        preferred = str(row.get("preferred_model", ""))
        accurate = str(row.get("more_accurate_model", ""))
        if preferred != accurate:
            skipped["preference_accuracy_disagree"] += 1
            continue
        if preferred == Q8_MODEL:
            pairs.append(
                build_pair(
                    row,
                    chosen_model=Q8_MODEL,
                    rejected_model=Q5_SIDECAR_MODEL,
                    prompt_spec=prompt_spec,
                )
            )
        elif preferred == Q5_SIDECAR_MODEL:
            pairs.append(
                build_pair(
                    row,
                    chosen_model=Q5_SIDECAR_MODEL,
                    rejected_model=Q8_MODEL,
                    prompt_spec=prompt_spec,
                )
            )
        else:
            skipped[preferred or "missing_preferred"] += 1

    by_model: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for pair in pairs:
        by_model[pair["chosen_model"]].append(pair)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for chosen_model, model_pairs in sorted(by_model.items()):
        for index, pair in enumerate(
            sorted(model_pairs, key=lambda item: (str(item["split"]), str(item["paper_id"]))),
            start=1,
        ):
            splits[split_name(index)].append(pair)

    for split, rows in splits.items():
        rows.sort(key=lambda item: (item["chosen_model"], item["split"], item["paper_id"]))
        write_jsonl(args.output_dir / f"{split}.jsonl", rows)
    all_pairs = splits["train"] + splits["valid"] + splits["test"]
    write_jsonl(args.output_dir / "all_pairs.jsonl", all_pairs)

    manifest = {
        "created_by": "scripts/build_q5_promoted_q8_sidecar_blind_dpo_pairs_v1.py",
        "review_jsonl": str(args.review_jsonl),
        "output_dir": str(args.output_dir),
        "prompt_spec": str(args.prompt_spec),
        "prompt_version": str(prompt_spec["prompt_version"]),
        "pair_count": len(all_pairs),
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "chosen_model_counts": dict(collections.Counter(row["chosen_model"] for row in all_pairs)),
        "review_split_counts": dict(collections.Counter(row["split"] for row in all_pairs)),
        "skipped_counts": dict(skipped),
        "selection": (
            "Use only rows where preferred_model equals more_accurate_model and "
            "the winner is either q8_mlx_python or the promoted q8 adapter running "
            "as a q5 llama.cpp sidecar."
        ),
        "label_policy": (
            "No new written labels. Chosen/rejected responses are existing model "
            "outputs selected by blind review."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "# q8 vs promoted-q8-adapter q5-sidecar DPO pairs\n\n"
        "Preference pairs are derived from the blind q8-vs-q5-sidecar review. "
        "Rows are included only when preference and accuracy agreed on q8 or the "
        "q5 sidecar. No new written summaries are introduced by this dataset.\n",
        encoding="utf-8",
    )
    print(f"[q5-sidecar-dpo-pairs] wrote {len(all_pairs)} pairs to {args.output_dir}")
    print(f"[q5-sidecar-dpo-pairs] split counts: {manifest['split_counts']}")
    print(f"[q5-sidecar-dpo-pairs] chosen counts: {manifest['chosen_model_counts']}")
    print(f"[q5-sidecar-dpo-pairs] skipped: {manifest['skipped_counts']}")


if __name__ == "__main__":
    main()
