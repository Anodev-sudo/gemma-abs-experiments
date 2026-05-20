#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from dpo_utils import build_prompt_messages, load_prompt_spec, write_jsonl


EXPERIMENT_ROOT = Path(
    "hf_bucket/experiments/mlx_lora/runtime_shift/"
    "llama_cpp_q5_clean_slate_broad_v1_20260508"
)
INPUT_JSONL = Path("hf_dataset/evals/q4_runtime_shift_200_v1/eval_inputs.jsonl")
AUDIT_JSON = EXPERIMENT_ROOT / "review_comparisons/step2400_flagged_rows_manual_audit.json"
OUTPUT_DIR = Path("hf_dataset/releases/recent_3y_balanced_10746_q5_step2400_dpo_smoke_v1")
PROMPT_SPEC = Path("prompts/mlx_lora/gemma_270m_it_abstract_to_glanceable_summary_v1.json")

CORRECTIONS: dict[str, tuple[str, str]] = {
    "entity_0017": (
        "polarity_mechanism",
        "Two shock episodes better explain GRB 240529A's plateau than magnetar energy injection.",
    ),
    "numeric_0007": (
        "number_role_binding",
        "Carbon-iridium measurements clarify incomplete fusion across low-energy heavy-ion reaction channels.",
    ),
    "numeric_0034": (
        "number_role_binding",
        "Conditional diffusion estimates oil-mixing intervals while reducing risky underestimation in pipelines.",
    ),
    "entity_0010": (
        "notation_paraphrase",
        "Characterizations link omega-base properties across free Abelian groups and hyperspaces.",
    ),
    "entity_0035": (
        "number_role_binding",
        "Ethereum rewards increased donation frequency far more than Bitcoin, while NFTs underperformed.",
    ),
    "long_notation_0017": (
        "title_compression",
        "Predictive control helps wastewater plants meet water-quality targets while offsetting energy costs.",
    ),
    "numeric_0050": (
        "polarity_mechanism",
        "Gaia data reveal an inactive nearby supercluster containing the new Duvia 1 cluster.",
    ),
    "numeric_0057": (
        "number_role_binding",
        "SEAQT models arsenic adsorption on graphene oxide within experimental isotherm error.",
    ),
    "polarity_0013": (
        "title_compression",
        "Free decomposition spaces unify deconcatenation-style comultiplications across many combinatorics examples.",
    ),
    "polarity_0017": (
        "polarity_mechanism",
        "Secret protection for labeled Petri nets is solvable, but ExpSpace-complete.",
    ),
    "polarity_0023": (
        "number_role_binding",
        "Compressed score encodings help predict piano difficulty, but accuracy remains modest.",
    ),
}

SPLITS = {
    "train": (
        "entity_0017",
        "numeric_0007",
        "numeric_0034",
        "entity_0035",
        "numeric_0050",
        "numeric_0057",
        "polarity_0017",
    ),
    "valid": (
        "entity_0010",
        "long_notation_0017",
    ),
    "test": (
        "polarity_0013",
        "polarity_0023",
    ),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_pair(
    *,
    source_row: dict[str, Any],
    audit_row: dict[str, Any],
    prompt_spec: dict[str, Any],
) -> dict[str, Any]:
    failure_axis, chosen_summary = CORRECTIONS[str(source_row["eval_id"])]
    return {
        "paper_id": str(source_row["paper_id"]),
        "eval_id": str(source_row["eval_id"]),
        "prompt": build_prompt_messages(source_row, prompt_spec, "title_abstract"),
        "chosen": [{"role": "assistant", "content": chosen_summary}],
        "rejected": [{"role": "assistant", "content": str(audit_row["summary"]).strip()}],
        "failure_axis": failure_axis,
        "gold_source": "hand_corrected_q5_step2400_smoke",
        "source_kind": "q5_step2400_runtime_failure",
        "candidate_origin": "llama_cpp_q5_step2400_runtime_shift_200",
        "manual_verdict": str(audit_row["manual_verdict"]),
        "audit_note": str(audit_row["audit_note"]),
        "heuristic_flags": list(audit_row.get("heuristic_flags", [])),
        "selection_bucket": str(source_row.get("selection_bucket", "")),
        "prompt_version": str(prompt_spec["prompt_version"]),
    }


def main() -> None:
    prompt_spec = load_prompt_spec(PROMPT_SPEC)
    input_rows = {str(row["eval_id"]): row for row in load_jsonl(INPUT_JSONL)}
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    audit_rows = {str(row["eval_id"]): row for row in audit["items"]}

    missing_inputs = sorted(set(CORRECTIONS) - set(input_rows))
    missing_audits = sorted(set(CORRECTIONS) - set(audit_rows))
    if missing_inputs or missing_audits:
        raise SystemExit(
            f"Missing inputs={missing_inputs or []}; missing audits={missing_audits or []}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    for split_name, eval_ids in SPLITS.items():
        rows = [
            build_pair(
                source_row=input_rows[eval_id],
                audit_row=audit_rows[eval_id],
                prompt_spec=prompt_spec,
            )
            for eval_id in eval_ids
        ]
        write_jsonl(OUTPUT_DIR / f"{split_name}.jsonl", rows)
        all_pairs.extend(rows)
        split_counts[split_name] = len(rows)

    write_jsonl(OUTPUT_DIR / "all_pairs.jsonl", all_pairs)

    manifest = {
        "dataset_id": OUTPUT_DIR.name,
        "purpose": (
            "Tiny DPO smoke dataset for checking the q5 step-2400 preference "
            "training path; not intended as a production correction dataset."
        ),
        "candidate": "q5_step2400",
        "runtime": "llama.cpp Q5_K_M base plus f16 GGUF LoRA sidecar",
        "input_jsonl": str(INPUT_JSONL),
        "source_audit": str(AUDIT_JSON),
        "prompt_spec": str(PROMPT_SPEC),
        "prompt_version": str(prompt_spec["prompt_version"]),
        "input_mode": "title_abstract",
        "pair_count": len(all_pairs),
        "split_counts": split_counts,
        "axis_counts": dict(sorted(Counter(row["failure_axis"] for row in all_pairs).items())),
        "gold_source": "hand_corrected_q5_step2400_smoke",
        "policy_start": str(EXPERIMENT_ROOT / "gguf_adapter/peft_2400"),
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "README.md").write_text(
        "# q5 step-2400 DPO smoke dataset\n\n"
        "This is a tiny preference dataset built from audited q5 step-2400 "
        "runtime failures. It exists only to check that the DPO stack can train "
        "from the q5 PEFT adapter and produce an adapter that can continue down "
        "the llama.cpp sidecar path.\n\n"
        "Do not treat this as a production correction dataset.\n",
        encoding="utf-8",
    )

    print(f"[q5-dpo-smoke] wrote {len(all_pairs)} pairs to {OUTPUT_DIR}")
    print(f"[q5-dpo-smoke] split counts: {split_counts}")
    print(f"[q5-dpo-smoke] axis counts: {manifest['axis_counts']}")


if __name__ == "__main__":
    main()
