import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from scripts.summary_generation import run_pipeline as pipeline
from scripts.synthetic_summary_runtime import extract_json_object_with_metadata


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_spec(path: Path, version: str, template: str) -> None:
    write_json(
        path,
        {
            "prompt_version": version,
            "role": version,
            "system_prompt": "Return JSON only.",
            "user_prompt_template": template,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": version,
                    "strict": True,
                    "schema": {"type": "object"},
                },
            },
        },
    )


def base_config(root: Path) -> dict:
    input_path = root / "input.jsonl"
    writer_prompt = root / "writer_prompt.json"
    reviewer_prompt = root / "reviewer_prompt.json"
    prompt_spec(
        writer_prompt,
        "writer_v1",
        "Title: {title}\nAbstract: {abstract}\nBad: {q5_summary}\nNote: {reviewer_note}",
    )
    prompt_spec(
        reviewer_prompt,
        "reviewer_v1",
        "Title: {title}\nAbstract: {abstract}\nSummary: {correction_summary}",
    )
    write_jsonl(
        input_path,
        [
            {
                "paper_id": "p1",
                "title": "Numeric drift",
                "abstract": "The method reduces numeric drift in summaries without changing supported claims.",
                "primary_category": "cs.CL",
                "categories": ["cs.CL"],
                "created_date": "2026-01-01",
                "q5_summary": "The method fixes 99 percent of unsupported numbers.",
                "q8_summary": "The method reduces numeric drift in summaries.",
                "note": "q5 invents an unsupported percentage.",
                "review_id": "review:1",
            },
            {
                "paper_id": "p2",
                "title": "Polarity drift",
                "abstract": "The study finds no consistent accuracy improvement under noisy data.",
                "primary_category": "cs.LG",
                "categories": ["cs.LG"],
                "created_date": "2026-01-02",
                "q5_summary": "The model consistently improves accuracy under noisy data.",
                "q8_summary": "No consistent accuracy improvement appears under noisy data.",
                "note": "q5 flips the result direction.",
                "review_id": "review:2",
            },
        ],
    )
    return {
        "input": {"path": str(input_path), "candidate_prefix": "test"},
        "output": {
            "run_root": str(root / "run"),
            "release_dir": str(root / "release"),
        },
        "writer": {
            "prompt_path": str(writer_prompt),
            "workers": 1,
            "max_output_tokens": 64,
            "max_retries": 0,
            "retry_backoff_seconds": 0.0,
            "accepted_decisions": ["rewrite"],
            "routes": [
                {
                    "route_id": "writer_primary",
                    "model": "writer-primary",
                    "response_format_mode": "schema",
                },
                {
                    "route_id": "writer_fallback",
                    "model": "writer-fallback",
                    "response_format_mode": "plain_json",
                },
            ],
        },
        "reviewer": {
            "prompt_path": str(reviewer_prompt),
            "workers": 1,
            "max_output_tokens": 64,
            "max_retries": 0,
            "retry_backoff_seconds": 0.0,
            "accepted_decisions": ["accept"],
            "routes": [
                {
                    "route_id": "reviewer_primary",
                    "model": "reviewer-primary",
                    "response_format_mode": "schema",
                },
                {
                    "route_id": "reviewer_fallback",
                    "model": "reviewer-fallback",
                    "response_format_mode": "plain_json",
                },
            ],
        },
        "local_gates": {
            "min_words": 8,
            "max_words": 15,
            "require_single_sentence": True,
            "max_title_overlap": 0.9,
            "reject_formula_shorthand": True,
            "reject_unsupported_summary_numbers": True,
        },
        "materialize": {
            "source_split": "test_split",
            "provenance_fields": ["q5_summary", "q8_summary", "reviewer_note", "q5_hardening_review_id"],
        },
    }


class SummaryGenerationPipelineTests(unittest.TestCase):
    def args(self) -> Namespace:
        return Namespace(
            limit=None,
            dry_run=False,
            continue_on_error=True,
            resume=True,
            no_route_probe=False,
            timeout=5,
            api_base="https://example.invalid/api/v1",
            api_key="test-key",
        )

    def test_route_probe_selects_first_working_route(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = base_config(Path(temp_dir))
            pipeline.command_prepare(config)

            def fake_request(**kwargs):
                if kwargs["model"] == "writer-primary":
                    raise RuntimeError("HTTP 503: route unavailable")
                return {"decision": "rewrite", "summary": "ok"}, "{}", {"id": "probe"}

            with mock.patch.object(pipeline, "request_json_with_retries", side_effect=fake_request):
                report = pipeline.command_probe_routes(
                    config,
                    api_base="https://example.invalid/api/v1",
                    api_key="test-key",
                    dry_run=False,
                    timeout=5,
                )

            self.assertEqual(
                report["selected_routes"]["writer"]["route_id"],
                "writer_fallback",
            )
            self.assertEqual(
                report["selected_routes"]["reviewer"]["route_id"],
                "reviewer_primary",
            )

    def test_mocked_end_to_end_uses_fallbacks_and_materializes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = base_config(root)
            args = self.args()
            pipeline.command_prepare(config)
            pipeline.command_probe_routes(
                config,
                api_base=args.api_base,
                api_key="",
                dry_run=True,
                timeout=args.timeout,
            )

            def fake_request(**kwargs):
                model = kwargs["model"]
                if model == "writer-primary":
                    raise RuntimeError("HTTP 503: writer route unavailable")
                if model == "reviewer-primary":
                    raise RuntimeError("HTTP 429: Too Many Requests")
                if model == "writer-fallback":
                    return (
                        {
                            "decision": "rewrite",
                            "issue_tags": ["none"],
                            "evidence_span": "reduces numeric drift",
                            "summary": "A careful method reduces numeric drift in supported paper-summary claims.",
                            "word_count": 9,
                            "reason_span": "",
                        },
                        "{}",
                        {"id": "writer-ok"},
                    )
                return (
                    {
                        "decision": "accept",
                        "issue_tags": ["none"],
                        "evidence_span": "reduces numeric drift",
                        "reason_span": "faithful",
                    },
                    "{}",
                    {"id": "reviewer-ok"},
                )

            with mock.patch.object(pipeline, "request_json_with_retries", side_effect=fake_request):
                pipeline.command_model_stage(config, role="writer", args=args)
                pipeline.command_local_filter(config)
                pipeline.command_model_stage(config, role="reviewer", args=args)
                training_rows = pipeline.command_materialize(config)
                report = pipeline.command_report(config)

            self.assertEqual(len(training_rows), 2)
            self.assertEqual(report["released_training_rows"], 2)
            self.assertEqual(training_rows[0]["teacher_model"], "writer-fallback")
            self.assertEqual(training_rows[0]["review_prompt_version"], "reviewer_v1")
            self.assertEqual(training_rows[0]["q5_hardening_review_id"], "review:1")

    def test_local_gate_rejects_unsupported_summary_number(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = base_config(Path(temp_dir))
            flags = pipeline.local_gate_flags(
                "The method improves accuracy by 99 percent for noisy data.",
                {
                    "title": "Noisy data",
                    "abstract": "The method is evaluated on noisy data without reporting a percentage.",
                },
                config,
            )
            self.assertIn("unsupported_number_string", flags)

    def test_runtime_json_repair_handles_missing_comma(self) -> None:
        parsed, _, repairs = extract_json_object_with_metadata(
            '{"decision":"rewrite"\n"summary":"A careful summary stays faithful."}'
        )
        self.assertEqual(parsed["decision"], "rewrite")
        self.assertIn("missing_json_commas", repairs)


if __name__ == "__main__":
    unittest.main()
