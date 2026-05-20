#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REVIEW = Path("hf_bucket/experiments/sft/q5_step2400_sft700_clean_existing_targeted_weight_v1_lr4e7/eval/failure_shape_review_v1")
OUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_targeted_booster_clean_v1/kimi_qwen_label_run")

# Representative IDs named by worker diagnostics plus sticky remembered stress cases.
SELECTIONS: list[tuple[str, str, str]] = [
    ("stress", "1901.06548", "Sticky stress case: exact-number corruption changed into overlong tangle wording; needs compact complexity takeaway."),
    ("stress", "2111.10635", "Sticky stress case: throughput multiplier is repeatedly misbound to cost; preserve throughput/cost roles qualitatively."),
    ("stress", "2303.09722", "Sticky stress case: optical metrics dominate; needs mechanism-first fiber coupling summary without exact metrics."),
    ("stress", "2406.04373", "Mostly acceptable stress case; include as hardware-test coverage style anchor."),

    ("range_template", "1912.07751", "Range-template hallucination around axion limits; avoid invented mass ranges."),
    ("range_template", "2301.08235", "Range-template hallucination in distributed leader election; avoid bogus round ranges."),
    ("range_template", "2501.12848", "Formula complexity turns into numeric runaway; summarize approximation contribution without numbers."),
    ("range_template", "2505.03183", "Large invented timescale/range; preserve astrophysical association qualitatively."),
    ("range_template", "2509.15021", "Generic numeric exponent/range attractor; needs theorem-level qualitative abstraction."),

    ("role_binding", "2304.10778", "Code-generation benchmark numbers become code-quality scoreboard; avoid metric headline."),
    ("role_binding", "2506.11441", "Hardware architecture parameters become performance claims; preserve design contribution qualitatively."),
    ("role_binding", "2510.24750", "Clinical dataset size is promoted into result metric; keep screening task and signal source."),
    ("role_binding", "2409.18257", "Medical classifier scores become unsupported unseen-test claims; summarize diagnostic modeling qualitatively."),
    ("role_binding", "2411.17627", "Air-pollution notation splits into broken numeric tokens; preserve pollutant-source takeaway without exact values."),
    ("role_binding", "2502.19481", "Cosmology/redshift quantity is placed in wrong unit role; avoid exact distance-like claim."),
    ("role_binding", "2412.00469", "Nonsense numeric-level role; summarize algorithmic or modeling contribution without level count."),
    ("role_binding", "2512.14462", "Event counts and dates bundle into wrong measurement hierarchy; preserve phenomenon qualitatively."),

    ("single_metric", "2311.12249", "Single shiny scalar magnitude collapse; avoid exact star-formation-like count."),
    ("single_metric", "2410.10610", "Percent gain becomes generic exploration metric headline; preserve planning task and uncertainty."),
    ("single_metric", "2505.12292", "Speedup headline attractor; preserve architecture contribution without exact multiplier."),
    ("single_metric", "2506.19972", "Emissions-reduction percentage overcommitment; keep energy/carbon direction qualitative."),
    ("single_metric", "2510.08731", "Latency/token percentage overcommitment; keep efficiency direction without exact metric."),
    ("single_metric", "2305.06328", "Pull-request turnaround metric dominates; summarize developer workflow contribution qualitatively."),
    ("single_metric", "2503.02960", "Invented speedup multipliers; preserve distributed inference contribution qualitatively."),

    ("formula_residue", "2508.14890", "Scientific notation and mass comparison residue; summarize chemical-evolution inference without exact mass."),
    ("formula_residue", "1712.02758", "LaTeX meson/decay notation leaks; preserve exclusive-decay search or limits qualitatively."),
    ("formula_residue", "2406.11564", "Exponent/HTML residue in physics result; use qualitative constraint or measurement takeaway."),
    ("formula_residue", "2310.13416", "Repeated HTML exponent flare metrics; preserve stellar-flare physics qualitatively."),
    ("formula_residue", "2406.04635", "Pseudocode corpus counts are fabricated; summarize extraction/corpus contribution without counts."),
    ("formula_residue", "2307.13873", "Ordinal/formula powers echo; summarize mathematical structure without symbolic residue."),

    ("copy_round_drift", "2402.17496", "Emotion/audio dataset statistics become rounded metric claims; summarize annotation or modeling aim qualitatively."),
    ("copy_round_drift", "2411.00902", "Accuracy numbers copied/rounded from architecture search; summarize search contribution qualitatively."),
    ("copy_round_drift", "2209.13385", "Murmur grading metrics copied/rounded; keep clinical auscultation classification task."),
    ("copy_round_drift", "2601.07014", "Medical/vision scores copied into polished metric claim; keep diagnostic modeling qualitatively."),
    ("copy_round_drift", "2504.01597", "Medical segmentation scores become unsupported exact metric claim; preserve reconstruction task qualitatively."),

    ("dimension_object_swap", "2406.02036", "Molecular collision / dimensional notation corrupts object and count; keep interaction-modeling takeaway."),
    ("dimension_object_swap", "2509.25497", "MIMO/mobile radio numerals attract unsupported throughput claim; preserve reporting/procedure contribution."),
    ("dimension_object_swap", "2307.02365", "Resolution-like numeric role corrupts image framing; summarize imaging method without exact resolution."),
    ("dimension_object_swap", "2409.03808", "Optical wavelength survives with wrong object framing; preserve optical mechanism qualitatively."),

    ("normal_semantic_drift", "alg-geom/9403013", "Normal regression: dimension/formal geometry wording drifts; needs concise theorem-safe wording."),
    ("normal_semantic_drift", "2403.03907", "Graph/gonality numeric degeneracy; preserve graph-theoretic takeaway without exact values."),
    ("normal_semantic_drift", "2312.14094", "TESS bias summary overuses exact angular numbers; keep catalog-selection bias qualitatively."),
    ("normal_semantic_drift", "2411.13315", "Pollutant notation and locality/transboundary roles need qualitative preservation."),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    all_records: dict[str, dict[str, Any]] = {}
    for path in REVIEW.glob("*.jsonl"):
        for row in load_jsonl(path):
            all_records.setdefault(str(row.get("paper_id") or ""), row)
    missing = []
    candidates = []
    for idx, (family, pid, note) in enumerate(SELECTIONS, start=1):
        src = all_records.get(pid)
        if not src:
            missing.append(pid)
            continue
        tags = ["targeted_booster", family, *src.get("flags", []), *src.get("checkpoint700_numeric_tags", []), *src.get("checkpoint700_semantic_tags", [])]
        # For sticky stress cases, ask for two independent Kimi labels. Others get one.
        reps = 2 if family == "stress" else 1
        for rep in range(1, reps + 1):
            candidates.append({
                "candidate_id": f"targeted_booster:{len(candidates)+1:04d}",
                "paper_id": pid,
                "title": src.get("title", ""),
                "abstract": src.get("abstract", ""),
                "primary_category": "",
                "categories": [],
                "created_date": "",
                "bad_summary": src.get("output_summary", ""),
                "pre_dpo_summary": src.get("checkpoint700_bad_summary", ""),
                "triage_decision": "targeted_numeric_booster_clean_label",
                "triage_severity": "high",
                "failure_tags": tags,
                "failure_family": family,
                "short_reason": f"{note} Current failed output numbers: {', '.join(map(str, src.get('numbers_in_output') or [])) or 'none'}. Booster variant {rep}/{reps}.",
                "source_split": "q5_targeted_booster_clean_v1_candidate_pool",
                "checkpoint_reviewed": 200,
            })
    if missing:
        print("Missing paper_ids:", sorted(set(missing)))
    write_jsonl(OUT / "candidate_pool" / "candidates.jsonl", candidates)
    manifest = {
        "created_by": "scripts/build_targeted_booster_candidate_pack_v1.py",
        "source_review_dir": str(REVIEW),
        "candidate_rows": len(candidates),
        "unique_papers": len({c["paper_id"] for c in candidates}),
        "missing_paper_ids": sorted(set(missing)),
        "selection_policy": "tight representative pack from checkpoint-200 failure-shape review; no labels included",
    }
    (OUT / "candidate_pool" / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
