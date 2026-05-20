#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SRC = Path("hf_bucket/experiments/sft/q5_step2400_sft700_clean_existing_targeted_weight_v1_lr4e7/dataset_600normal_200numeric_targeted_weight_clean_v1/numeric_selected_200.jsonl")
OUT = Path("hf_bucket/experiments/sft/q5_step2400_sft700_numeric_ideal200_regen_v1/kimi_label_run")

HIGH_RISK = {
    "2411.17627": "Prior audit warning: avoid unsupported insulation-upgrade causal story; preserve sensor monitoring and air-quality scope.",
    "2507.01963": "Prior audit warning: avoid unsupported majority words like most; describe manipulation qualitatively without overclaiming prevalence.",
    "2304.12248": "Prior audit warning: avoid unsupported 'most test functions' or broad performance-win claims.",
    "2503.20132": "Prior audit warning: preserve tentative candidate and non-detection uncertainty; do not state a firm planet detection.",
    "2501.19292": "Prior audit warning: preserve abnormal trajectory behavior and phase comparison; avoid polarity flattening.",
    "2501.12848": "Prior audit warning: avoid absolute optimal-runtime claims; preserve conditional approximation/theorem strength qualitatively.",
    "2306.11756": "Prior audit warning: keep the specific visuospatial task scope; avoid broad human-ability claims.",
    "2304.13172": "Prior audit warning: preserve multimodal conditioned generator specifics; avoid generic title paraphrase.",
    "2401.16566": "Prior audit warning: preserve self-collision and tool-calibration constraints; avoid generic robotics phrasing.",
    "2311.09464": "Prior audit warning: avoid vacuous math abstraction; keep the actual contribution object if source supports it.",
}
MARGINAL = {
    "2505.03183": "Prior audit note: be cautious about star-formation and physical-association claims.",
    "2304.10778": "Prior audit note: avoid unsupported ranking/outperforms claims for code-generation tools.",
    "2301.08235": "Prior audit note: avoid title-only generic tradeoff phrasing; keep a concrete source-grounded relation.",
    "2507.10593": "Prior audit note: avoid product-blurb smoothing; keep protocol/tool-management specifics.",
    "2310.01016": "Prior audit note: avoid promotional future-tense certainty; preserve detection capability carefully.",
    "2306.12279": "Prior audit note: avoid generic real-time modeling phrasing; keep the motion-comfort task anchored.",
    "2409.18257": "Prior audit note: title-heavy classification label risk; add supported source detail without metrics.",
    "2309.10048": "Prior audit note: avoid awkward perceived-quality wording and generic smoothing.",
    "2508.05266": "Prior audit note: include RTL/RAG/rule-checking or debugging specifics if supported.",
    "2509.00203": "Prior audit note: avoid vague performance-fluffy phrases like strong accuracy gains.",
    "2501.00106": "Prior audit note: avoid bland corporate/title paraphrase.",
    "2410.11046": "Prior audit note: reduce title-copy phrasing; keep concrete mechanism or object.",
    "2407.00085": "Prior audit note: distinguish user-defined rules from manual rules.",
    "2401.11342": "Prior audit note: preserve prompt/nonprompt and decay-channel distinctions if relevant.",
    "2503.02964": "Prior audit note: avoid stronger causal verbs than the abstract supports.",
    "2507.17494": "Prior audit note: preserve calibration versus outage nuance.",
    "2602.13239": "Prior audit note: avoid generic multimodal-training smoothing.",
    "2406.05140": "Prior audit note: keep real-world/stability setup if supported; avoid template label.",
    "2410.21224": "Prior audit note: math result needs sharper construction/result object, not broad abstraction.",
    "2412.00469": "Prior audit note: avoid lab-note weakness; make the supported algorithm/model contribution clear.",
    "2506.04290": "Prior audit note: avoid taxonomy filler; keep distinctive contribution.",
    "2507.21623": "Prior audit note: avoid very generic safe wording.",
    "2510.04313": "Prior audit note: safe but weak; add source-specific anchor.",
    "2601.20706": "Prior audit note: safe but weak; add source-specific anchor.",
    "2102.01902": "Prior audit note: safe but weak; add source-specific anchor.",
    "2212.10759": "Prior audit note: safe but weak; add source-specific anchor.",
    "2308.10442": "Prior audit note: safe but weak; add source-specific anchor.",
    "2402.17164": "Prior audit note: safe but weak; add source-specific anchor.",
    "2404.19755": "Prior audit note: preserve codec caveat; do not make it a clean win if source says otherwise.",
    "2410.05759": "Prior audit note: avoid awkward matrix-based evolution wording and generic phrasing.",
    "2501.15861": "Prior audit note: grounded but abstract; add distinctive source object if possible.",
    "2510.24914": "Prior audit note: grounded but abstract; add distinctive source object if possible.",
    "2310.06776": "Prior audit note: grounded but abstract; add distinctive source object if possible.",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    rows = load_jsonl(SRC)
    candidates = []
    for idx, row in enumerate(rows, start=1):
        pid = str(row.get("paper_id"))
        audit_status = "high_risk" if pid in HIGH_RISK else "marginal" if pid in MARGINAL else "standard"
        guidance = HIGH_RISK.get(pid) or MARGINAL.get(pid) or "Regenerate from source using the ideal no-number abstraction policy; do not copy prior clean labels."
        candidates.append({
            "candidate_id": f"ideal200:{idx:04d}",
            "paper_id": pid,
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "primary_category": row.get("primary_category", ""),
            "categories": row.get("categories", []),
            "created_date": row.get("created_date", ""),
            "bad_summary": row.get("bad_summary", ""),
            "pre_dpo_summary": row.get("pre_dpo_summary", ""),
            "triage_decision": "regenerate_clean_numeric_label",
            "triage_severity": "high" if audit_status == "high_risk" else "medium" if audit_status == "marginal" else "standard",
            "failure_tags": ["ideal200_regen", audit_status, *row.get("failure_tags", [])],
            "short_reason": guidance,
            "audit_status": audit_status,
            "source_split": "q5_numeric_ideal200_regen_v1_candidate_pool",
            "old_repair_candidate_id": row.get("repair_candidate_id", ""),
        })
    out = OUT / "candidate_pool"
    write_jsonl(out / "candidates.jsonl", candidates)
    manifest = {
        "created_by": "scripts/setup_numeric_ideal200_regen_v1.py",
        "source": str(SRC),
        "candidate_rows": len(candidates),
        "high_risk_rows": sum(1 for c in candidates if c["audit_status"] == "high_risk"),
        "marginal_rows": sum(1 for c in candidates if c["audit_status"] == "marginal"),
        "standard_rows": sum(1 for c in candidates if c["audit_status"] == "standard"),
        "policy": "Regenerate full 200 from title+abstract; old clean labels are not included in prompts.",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
