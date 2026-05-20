#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dpo_utils import build_prompt_messages, load_prompt_spec, write_jsonl


DEFAULT_INPUT = Path(
    "hf_dataset/releases/recent_3y_balanced_10746_q5_step2400_2000_mined_kimi_qwen_v1/"
    "training_rows.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "hf_dataset/releases/recent_3y_balanced_10746_q5_step2400_2000_mined_kimi_qwen_v1/"
    "dpo/title_abstract_preference_v1"
)
DEFAULT_PROMPT_SPEC = Path("prompts/mlx_lora/gemma_270m_it_abstract_to_glanceable_summary_v1.json")


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


def failure_axis(row: dict[str, Any]) -> str:
    tags = set(row.get("q5_issue_tags") or [])
    if tags & {"unsupported_number", "wrong_number_binding", "numeric_error"}:
        return "number_role_binding"
    if tags & {"polarity_flip", "polarity_risk", "overstrong_claim", "unsupported_causal_claim"}:
        return "polarity_mechanism"
    if tags & {"malformed_repetition", "unsupported_entity_label", "unsupported_entity"}:
        return "notation_entity"
    if tags & {"title_rehash", "too_generic", "style_drift"}:
        return "style_title_compression"
    return "other"


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows_by_axis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_axis[failure_axis(row)].append(row)

    result: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for axis in sorted(rows_by_axis):
        axis_rows = sorted(rows_by_axis[axis], key=lambda row: (str(row["paper_id"]), str(row.get("eval_id", ""))))
        n = len(axis_rows)
        valid = 1 if n >= 8 else 0
        test = max(1, round(n * 0.12)) if n >= 4 else 0
        train = n - valid - test
        if train <= 0:
            train = n
            valid = 0
            test = 0
        result["train"].extend(axis_rows[:train])
        result["valid"].extend(axis_rows[train : train + valid])
        result["test"].extend(axis_rows[train + valid :])
    return result


def build_dpo_row(row: dict[str, Any], prompt_spec: dict[str, Any]) -> dict[str, Any]:
    chosen = str(row.get("summary", "")).strip()
    rejected = str(row.get("q5_summary", "")).strip()
    if not chosen or not rejected:
        raise ValueError(f"Missing chosen/rejected summary for paper_id={row.get('paper_id')}")
    return {
        "paper_id": str(row["paper_id"]),
        "eval_id": str(row.get("eval_id", "")),
        "prompt": build_prompt_messages(row, prompt_spec, "title_abstract"),
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "failure_axis": failure_axis(row),
        "gold_source": "kimi_qwen_repair",
        "source_kind": "q5_step2400_runtime_failure_mined_2000",
        "candidate_origin": "q5_step2400_2000_mined_kimi_qwen_v1",
        "q5_issue_tags": row.get("q5_issue_tags", []),
        "heuristic_flags": row.get("heuristic_flags", []),
        "reviewer_note": row.get("reviewer_note", ""),
        "reviewer_reason_span": row.get("reviewer_reason_span", ""),
        "writer_evidence_span": row.get("writer_evidence_span", ""),
        "prompt_version": str(prompt_spec["prompt_version"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DPO chosen/rejected pairs from q5 step-2400 mined repairs."
    )
    parser.add_argument("--input-jsonl", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prompt-spec", default=str(DEFAULT_PROMPT_SPEC))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_spec = load_prompt_spec(Path(args.prompt_spec))
    source_rows = load_jsonl(input_path)
    split_source = split_rows(source_rows)
    split_dpo = {
        split: [build_dpo_row(row, prompt_spec) for row in rows]
        for split, rows in split_source.items()
    }

    for split in ("train", "valid", "test"):
        split_dpo[split].sort(key=lambda row: (row["failure_axis"], row["paper_id"]))
        write_jsonl(output_dir / f"{split}.jsonl", split_dpo[split])

    all_rows = split_dpo["train"] + split_dpo["valid"] + split_dpo["test"]
    write_jsonl(output_dir / "all_pairs.jsonl", all_rows)
    manifest = {
        "dataset_id": output_dir.name,
        "purpose": "DPO preference pairs from q5 step-2400 mined Kimi/Qwen repairs.",
        "input_jsonl": str(input_path),
        "prompt_spec": str(args.prompt_spec),
        "prompt_version": str(prompt_spec["prompt_version"]),
        "input_mode": "title_abstract",
        "pair_count": len(all_rows),
        "split_counts": {split: len(rows) for split, rows in split_dpo.items()},
        "axis_counts": dict(sorted(Counter(row["failure_axis"] for row in all_rows).items())),
        "q5_issue_tag_counts": dict(
            sorted(Counter(tag for row in all_rows for tag in row.get("q5_issue_tags", [])).items())
        ),
        "chosen_source": "accepted Kimi repair verified by strict Qwen",
        "rejected_source": "q5 step-2400 llama.cpp runtime summary",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# q5 step-2400 mined DPO preference pairs\n\n"
        "Chosen summaries are Kimi repairs accepted by strict Qwen verification. "
        "Rejected summaries are the original q5 step-2400 llama.cpp outputs from "
        "the 2000-row mining pass.\n",
        encoding="utf-8",
    )
    print(f"[q5-dpo-pairs] wrote {len(all_rows)} pairs to {output_dir}")
    print(f"[q5-dpo-pairs] split counts: {manifest['split_counts']}")
    print(f"[q5-dpo-pairs] axis counts: {manifest['axis_counts']}")


if __name__ == "__main__":
    main()
