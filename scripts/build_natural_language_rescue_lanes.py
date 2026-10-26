#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LANE_DEFAULTS = {
    "start-cleanup": {
        "selection": "local-rejected rewrites repaired by deterministic opening cleanup",
        "writer_prompt": "",
        "writer_label": "deterministic_start_cleanup",
        "review_tags": [],
        "reject_review_tags": [],
    },
    "too-long": {
        "selection": "local-rejected rewrites with only short too-long failures",
        "writer_prompt": "prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_writer_v11_length_rescue.json",
        "writer_label": "deepseek_v4_pro_plain_language_v11_length_rescue",
        "review_tags": [],
        "reject_review_tags": [],
    },
    "verifier-reason": {
        "selection": "normal-verifier rejects with fixable payoff or specificity issues",
        "writer_prompt": "prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_writer_v12_verifier_reason_rescue.json",
        "writer_label": "deepseek_v4_pro_plain_language_v12_verifier_reason_rescue",
        "review_tags": ["too_vague", "too_generic", "not_reason_to_open", "title_rehash"],
        "reject_review_tags": [],
    },
    "strict-jargon": {
        "selection": "strict-audit rejects with fixable jargon issues",
        "writer_prompt": "prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_writer_v13_jargon_repair.json",
        "writer_label": "deepseek_v4_pro_plain_language_v13_jargon_repair",
        "review_tags": ["too_technical", "jargon_stack"],
        "reject_review_tags": ["not_reason_to_open", "too_vague", "too_generic", "title_rehash"],
    },
    "math": {
        "selection": "math-category rows routed through the math natural-language writer",
        "writer_prompt": "prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_writer_v16_math.json",
        "writer_label": "deepseek_v4_pro_plain_language_v16_math",
        "review_tags": [],
        "reject_review_tags": [],
    },
}

UNSAFE_REVIEW_TAGS = {
    "unsupported_claim",
    "wrong_object",
    "wrong_method",
    "wrong_mechanism",
    "wrong_polarity",
    "false_value_attribution",
    "scope_upgrade",
}

OPENING_WORDS = ("a", "an", "the", "new")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create natural-language summary rescue-lane rows and configs."
    )
    parser.add_argument(
        "--lane",
        required=True,
        choices=sorted(LANE_DEFAULTS),
        help="Rescue lane to build.",
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--template-config",
        required=True,
        help="Primary natural-language config to copy.",
    )
    parser.add_argument(
        "--source-jsonl",
        required=True,
        help="Source JSONL for this lane, such as excluded local or reviewer rows.",
    )
    parser.add_argument("--run-root", required=True, help="Normal rescue run root.")
    parser.add_argument("--release-dir", required=True, help="Normal rescue release dir.")
    parser.add_argument("--config-out", required=True, help="Normal rescue config path.")
    parser.add_argument("--audit-config-out", required=True, help="Strict audit config path.")
    parser.add_argument("--audit-run-root", default="", help="Strict audit run root.")
    parser.add_argument("--audit-release-dir", default="", help="Strict audit release dir.")
    parser.add_argument("--writer-prompt", default="", help="Override writer prompt path.")
    parser.add_argument("--writer-label", default="", help="Override writer label.")
    parser.add_argument(
        "--math-reviewer-prompt",
        default="prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_verifier_v6_math.json",
        help="Math verifier prompt path for the math lane.",
    )
    parser.add_argument(
        "--strict-reviewer-prompt",
        default="prompts/synthetic_summary/deepseek_v4_pro_natural_language_summary_verifier_v4.json",
        help="Strict audit verifier prompt path.",
    )
    parser.add_argument(
        "--candidate-prefix",
        default="",
        help="Candidate prefix. Defaults to the run-root directory name.",
    )
    parser.add_argument("--source-split", default="", help="Materialized source split.")
    parser.add_argument(
        "--review-tags",
        nargs="*",
        default=None,
        help="Reviewer tags to select for verifier-reason or strict-jargon lanes.",
    )
    parser.add_argument(
        "--reject-review-tags",
        nargs="*",
        default=None,
        help="Reviewer tags that exclude rows from verifier-reason or strict-jargon lanes.",
    )
    parser.add_argument(
        "--allow-unsafe-review-tags",
        action="store_true",
        help="Do not drop reviewer rejects that also carry wrong/unsupported tags.",
    )
    parser.add_argument(
        "--allow-mixed-local-flags",
        action="store_true",
        help="For too-long, include rows where too_long appears with other local flags.",
    )
    parser.add_argument(
        "--max-previous-words",
        type=int,
        default=17,
        help="Maximum old-summary word count for too-long rescue rows.",
    )
    parser.add_argument(
        "--math-source",
        choices=("dropped", "all", "rewrites"),
        default="dropped",
        help="For the math lane, choose which math rows to select when writer_decision is present.",
    )
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated files.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def add_repo_imports(repo_root: Path) -> None:
    for import_path in (repo_root, repo_root / "scripts"):
        value = str(import_path)
        if value not in sys.path:
            sys.path.insert(0, value)


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)] if str(value).strip() else []


def review_decision(row: dict[str, Any]) -> str:
    return str(row.get("reviewer_decision", row.get("verifier_decision", ""))).strip().lower()


def review_tags(row: dict[str, Any]) -> list[str]:
    return string_list(row.get("reviewer_issue_tags", row.get("verifier_issue_tags", [])))


def summary_word_count(summary: str) -> int:
    return len([part for part in summary.split() if part])


def clean_opening(summary: str, normalize_text) -> str:
    text = normalize_text(summary)
    for _ in range(3):
        parts = text.split(" ", 1)
        if len(parts) != 2 or parts[0].casefold() not in OPENING_WORDS:
            break
        text = parts[1].strip()
        if text:
            text = text[:1].upper() + text[1:]
    return text


def base_rescue_row(
    row: dict[str, Any],
    *,
    lane: str,
    candidate_prefix: str,
    index: int,
    source_split: str,
) -> dict[str, Any]:
    previous_summary = str(row.get("generated_summary", row.get("summary", ""))).strip()
    rescue = dict(row)
    rescue["candidate_id"] = f"{candidate_prefix}:{index:04d}"
    rescue["original_candidate_id"] = row.get("candidate_id", "")
    rescue["rescue_lane"] = lane
    rescue["source_split"] = source_split
    rescue["previous_summary"] = previous_summary
    rescue["previous_summary_word_count"] = summary_word_count(previous_summary)
    rescue["previous_local_filter_flags"] = row.get("writer_local_filter_flags", [])
    rescue["previous_writer_prompt_version"] = row.get("writer_prompt_version", "")
    rescue["previous_writer_issue_tags"] = row.get("writer_issue_tags", [])
    rescue["previous_writer_plain_takeaway"] = row.get("writer_plain_takeaway", "")
    rescue["previous_writer_technical_terms_used"] = row.get(
        "writer_technical_terms_used", []
    )
    rescue["previous_reviewer_issue_tags"] = review_tags(row)
    rescue["previous_reviewer_evidence_span"] = row.get("reviewer_evidence_span", "")
    rescue["previous_reviewer_reason_span"] = row.get("reviewer_reason_span", "")
    rescue["previous_verifier_issue_tags"] = review_tags(row)
    rescue["previous_verifier_evidence_span"] = row.get("verifier_evidence_span", "")
    rescue["previous_verifier_reason_span"] = row.get("verifier_reason_span", "")
    return rescue


def select_start_cleanup(
    rows: list[dict[str, Any]],
    *,
    candidate_prefix: str,
    source_split: str,
    normalize_text,
    local_gate_flags,
    gate_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows: list[dict[str, Any]] = []
    eligible_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []

    for row in rows:
        if row.get("writer_decision") != "rewrite":
            continue
        old_summary = str(row.get("generated_summary", ""))
        fixed_summary = clean_opening(old_summary, normalize_text)
        if fixed_summary == normalize_text(old_summary):
            continue
        rescue = base_rescue_row(
            row,
            lane="start-cleanup",
            candidate_prefix=candidate_prefix,
            index=len(source_rows) + 1,
            source_split=source_split,
        )
        rescue["generated_summary"] = fixed_summary
        rescue["correction_summary"] = fixed_summary
        rescue["writer_prompt_version"] = "deterministic_start_cleanup"
        rescue["correction_prompt_version"] = "deterministic_start_cleanup"
        rescue["correction_writer"] = "deterministic_start_cleanup"
        rescue["writer_model"] = "deterministic_start_cleanup"
        rescue["correction_model"] = "deterministic_start_cleanup"
        rescue["writer_route_id"] = "deterministic_start_cleanup"
        rescue["writer_decision"] = "rewrite"
        rescue["correction_decision"] = "rewrite"
        rescue["writer_reported_word_count"] = summary_word_count(fixed_summary)
        rescue["writer_actual_word_count"] = summary_word_count(fixed_summary)
        rescue["writer_local_filter_flags"] = local_gate_flags(
            fixed_summary, rescue, gate_config
        )
        source_rows.append(rescue)
        if rescue["writer_local_filter_flags"]:
            excluded_rows.append(rescue)
        else:
            eligible_rows.append(rescue)
    return source_rows, eligible_rows, excluded_rows


def select_too_long(
    rows: list[dict[str, Any]],
    *,
    candidate_prefix: str,
    source_split: str,
    max_previous_words: int,
    allow_mixed_local_flags: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("writer_decision") != "rewrite":
            continue
        flags = set(string_list(row.get("writer_local_filter_flags")))
        if "too_long" not in flags:
            continue
        if not allow_mixed_local_flags and flags != {"too_long"}:
            continue
        summary = str(row.get("generated_summary", "")).strip()
        if summary_word_count(summary) > max_previous_words:
            continue
        selected.append(
            base_rescue_row(
                row,
                lane="too-long",
                candidate_prefix=candidate_prefix,
                index=len(selected) + 1,
                source_split=source_split,
            )
        )
    return selected


def select_reviewer_rows(
    rows: list[dict[str, Any]],
    *,
    lane: str,
    candidate_prefix: str,
    source_split: str,
    wanted_tags: set[str],
    rejected_tags: set[str],
    allow_unsafe_review_tags: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        tags = set(review_tags(row))
        if review_decision(row) != "reject":
            continue
        if not tags.intersection(wanted_tags):
            continue
        if tags.intersection(rejected_tags):
            continue
        if not allow_unsafe_review_tags and tags.intersection(UNSAFE_REVIEW_TAGS):
            continue
        if not str(row.get("generated_summary", "")).strip():
            continue
        selected.append(
            base_rescue_row(
                row,
                lane=lane,
                candidate_prefix=candidate_prefix,
                index=len(selected) + 1,
                source_split=source_split,
            )
        )
    return selected


def select_math_rows(
    rows: list[dict[str, Any]],
    *,
    candidate_prefix: str,
    source_split: str,
    math_source: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not str(row.get("primary_category", "")).startswith("math."):
            continue
        decision = str(row.get("writer_decision", "")).strip().lower()
        if decision:
            if math_source == "dropped" and decision == "rewrite":
                continue
            if math_source == "rewrites" and decision != "rewrite":
                continue
        selected.append(
            base_rescue_row(
                row,
                lane="math",
                candidate_prefix=candidate_prefix,
                index=len(selected) + 1,
                source_split=source_split,
            )
        )
    return selected


def add_provenance_fields(config: dict[str, Any]) -> None:
    materialize = config.setdefault("materialize", {})
    fields = list(materialize.get("provenance_fields", []))
    for field in (
        "rescue_lane",
        "original_candidate_id",
        "previous_summary",
        "previous_summary_word_count",
        "previous_local_filter_flags",
        "previous_writer_prompt_version",
        "previous_writer_issue_tags",
        "previous_writer_plain_takeaway",
        "previous_writer_technical_terms_used",
        "previous_reviewer_issue_tags",
        "previous_reviewer_evidence_span",
        "previous_reviewer_reason_span",
    ):
        if field not in fields:
            fields.append(field)
    materialize["provenance_fields"] = fields


def build_configs(
    *,
    template_config_path: Path,
    repo_root: Path,
    lane: str,
    source_path: Path,
    local_eligible_path: Path | None,
    run_root: Path,
    release_dir: Path,
    audit_run_root: Path,
    audit_release_dir: Path,
    config_out: Path,
    audit_config_out: Path,
    candidate_prefix: str,
    source_split: str,
    writer_prompt: str,
    writer_label: str,
    strict_reviewer_prompt: str,
    overwrite: bool,
) -> None:
    normal_config = load_json(template_config_path)
    normal_config["input"]["path"] = repo_rel(source_path, repo_root)
    normal_config["input"]["candidate_prefix"] = candidate_prefix
    normal_config["input"]["tag_fields"] = [
        "rescue_lane",
        "previous_local_filter_flags",
        "previous_reviewer_issue_tags",
    ]
    normal_config["output"]["run_root"] = str(run_root)
    normal_config["output"]["release_dir"] = str(release_dir)
    normal_config.setdefault("materialize", {})["source_split"] = source_split
    if lane == "start-cleanup":
        normal_config["reviewer"]["input_path"] = repo_rel(local_eligible_path, repo_root)
    else:
        normal_config["writer"]["prompt_path"] = writer_prompt
        normal_config["writer"]["writer_label"] = writer_label
        if lane == "math":
            normal_config["reviewer"]["prompt_path"] = strict_reviewer_prompt
            gates = normal_config.setdefault("local_gates", {})
            forbidden_prefixes = [
                str(prefix)
                for prefix in gates.get("forbidden_prefixes", [])
            ]
            for prefix in ("survey ", "review "):
                if prefix not in forbidden_prefixes:
                    forbidden_prefixes.append(prefix)
            gates["forbidden_prefixes"] = forbidden_prefixes
            gates["max_writer_technical_terms"] = max(
                int(gates.get("max_writer_technical_terms", 0) or 0), 4
            )
            gates["max_long_words"] = max(int(gates.get("max_long_words", 0) or 0), 3)
            gates["max_acronyms"] = max(int(gates.get("max_acronyms", 0) or 0), 2)
    add_provenance_fields(normal_config)
    write_json(config_out, normal_config, overwrite=overwrite)

    audit_config = json.loads(json.dumps(normal_config))
    audit_config["input"]["candidate_prefix"] = f"{candidate_prefix}_audit"
    audit_config["output"]["run_root"] = str(audit_run_root)
    audit_config["output"]["release_dir"] = str(audit_release_dir)
    audit_config["reviewer"]["prompt_path"] = strict_reviewer_prompt
    audit_config["reviewer"]["input_path"] = repo_rel(
        repo_root / run_root / "materialized_release" / "selected_review_rows.jsonl",
        repo_root,
    )
    audit_config["materialize"]["source_split"] = f"{source_split}_audit"
    write_json(audit_config_out, audit_config, overwrite=overwrite)


def main() -> None:
    args = parse_args()
    defaults = LANE_DEFAULTS[args.lane]
    repo_root = Path(args.repo_root).resolve()
    add_repo_imports(repo_root)

    from scripts.summary_generation.run_pipeline import load_config, local_gate_flags, normalize_text

    template_config_path = repo_root / args.template_config
    source_jsonl = repo_root / args.source_jsonl
    run_root = Path(args.run_root)
    release_dir = Path(args.release_dir)
    audit_run_root = Path(args.audit_run_root) if args.audit_run_root else Path(f"{args.run_root}_audit")
    audit_release_dir = (
        Path(args.audit_release_dir)
        if args.audit_release_dir
        else Path(f"{args.release_dir}_audit")
    )
    config_out = repo_root / args.config_out
    audit_config_out = repo_root / args.audit_config_out
    candidate_prefix = args.candidate_prefix or run_root.name
    source_split = args.source_split or candidate_prefix
    writer_prompt = args.writer_prompt or str(defaults["writer_prompt"])
    writer_label = args.writer_label or str(defaults["writer_label"])
    wanted_review_tags = set(args.review_tags if args.review_tags is not None else defaults["review_tags"])
    rejected_review_tags = set(
        args.reject_review_tags
        if args.reject_review_tags is not None
        else defaults["reject_review_tags"]
    )

    source_rows = load_jsonl(source_jsonl)
    gate_config = load_config(template_config_path)
    local_eligible_path: Path | None = None
    local_excluded_rows: list[dict[str, Any]] = []

    if args.lane == "start-cleanup":
        selected_rows, local_eligible_rows, local_excluded_rows = select_start_cleanup(
            source_rows,
            candidate_prefix=candidate_prefix,
            source_split=source_split,
            normalize_text=normalize_text,
            local_gate_flags=local_gate_flags,
            gate_config=gate_config,
        )
        if args.max_source_rows is not None:
            keep_ids = {row["candidate_id"] for row in selected_rows[: args.max_source_rows]}
            selected_rows = [row for row in selected_rows if row["candidate_id"] in keep_ids]
            local_eligible_rows = [
                row for row in local_eligible_rows if row["candidate_id"] in keep_ids
            ]
            local_excluded_rows = [
                row for row in local_excluded_rows if row["candidate_id"] in keep_ids
            ]
        local_dir = repo_root / run_root / "local_eligible_for_review"
        local_eligible_path = local_dir / "writer_outputs_ordered.jsonl"
        write_jsonl(local_eligible_path, local_eligible_rows, overwrite=args.overwrite)
        write_jsonl(local_dir / "excluded.jsonl", local_excluded_rows, overwrite=args.overwrite)
    elif args.lane == "too-long":
        selected_rows = select_too_long(
            source_rows,
            candidate_prefix=candidate_prefix,
            source_split=source_split,
            max_previous_words=args.max_previous_words,
            allow_mixed_local_flags=args.allow_mixed_local_flags,
        )
    elif args.lane == "math":
        selected_rows = select_math_rows(
            source_rows,
            candidate_prefix=candidate_prefix,
            source_split=source_split,
            math_source=args.math_source,
        )
    else:
        selected_rows = select_reviewer_rows(
            source_rows,
            lane=args.lane,
            candidate_prefix=candidate_prefix,
            source_split=source_split,
            wanted_tags=wanted_review_tags,
            rejected_tags=rejected_review_tags,
            allow_unsafe_review_tags=args.allow_unsafe_review_tags,
        )

    if args.max_source_rows is not None and args.lane != "start-cleanup":
        selected_rows = selected_rows[: args.max_source_rows]

    source_path = repo_root / run_root / "source_candidates" / "natural_language_candidates.jsonl"
    write_jsonl(source_path, selected_rows, overwrite=args.overwrite)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lane": args.lane,
        "selection": defaults["selection"],
        "input_jsonl": str(source_jsonl),
        "output_jsonl": str(source_path),
        "source_rows": len(source_rows),
        "selected_rows": len(selected_rows),
        "candidate_prefix": candidate_prefix,
        "source_split": source_split,
        "previous_local_filter_counts": dict(
            Counter(flag for row in selected_rows for flag in string_list(row.get("previous_local_filter_flags")))
        ),
        "previous_reviewer_issue_tag_counts": dict(
            Counter(tag for row in selected_rows for tag in string_list(row.get("previous_reviewer_issue_tags")))
        ),
        "wanted_review_tags": sorted(wanted_review_tags),
        "rejected_review_tags": sorted(rejected_review_tags),
    }
    if args.lane == "math":
        manifest["math_source"] = args.math_source
    if args.lane == "start-cleanup":
        manifest["local_eligible_rows"] = len(load_jsonl(local_eligible_path)) if local_eligible_path else 0
        manifest["local_excluded_rows"] = len(local_excluded_rows)
        manifest["excluded_local_filter_counts"] = dict(
            Counter(flag for row in local_excluded_rows for flag in string_list(row.get("writer_local_filter_flags")))
        )
        write_json(
            repo_root / run_root / "local_eligible_for_review" / "run_manifest.json",
            manifest,
            overwrite=args.overwrite,
        )
    write_json(source_path.parent / "run_manifest.json", manifest, overwrite=args.overwrite)

    build_configs(
        template_config_path=template_config_path,
        repo_root=repo_root,
        lane=args.lane,
        source_path=source_path,
        local_eligible_path=local_eligible_path,
        run_root=run_root,
        release_dir=release_dir,
        audit_run_root=audit_run_root,
        audit_release_dir=audit_release_dir,
        config_out=config_out,
        audit_config_out=audit_config_out,
        candidate_prefix=candidate_prefix,
        source_split=source_split,
        writer_prompt=writer_prompt,
        writer_label=writer_label,
        strict_reviewer_prompt=(
            args.math_reviewer_prompt if args.lane == "math" else args.strict_reviewer_prompt
        ),
        overwrite=args.overwrite,
    )

    print(f"[natural-language-rescue:{args.lane}] selected rows: {len(selected_rows)}")
    if args.lane == "start-cleanup":
        print(f"[natural-language-rescue:{args.lane}] local eligible: {manifest['local_eligible_rows']}")
    print(f"[natural-language-rescue:{args.lane}] wrote {config_out}")
    print(f"[natural-language-rescue:{args.lane}] wrote {audit_config_out}")


if __name__ == "__main__":
    main()
