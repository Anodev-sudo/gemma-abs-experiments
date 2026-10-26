import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_natural_language_rescue_lanes as rescue


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def base_config(root: Path) -> Path:
    path = root / "base.json"
    write_json(
        path,
        {
            "input": {"path": "unused.jsonl", "candidate_prefix": "base"},
            "output": {
                "run_root": "unused_run",
                "release_dir": "unused_release",
            },
            "writer": {
                "prompt_path": "old_writer.json",
                "writer_label": "old_writer",
                "accepted_decisions": ["rewrite"],
            },
            "reviewer": {
                "prompt_path": "normal_reviewer.json",
                "accepted_decisions": ["accept"],
            },
            "local_gates": {
                "min_words": 3,
                "max_words": 14,
                "require_single_sentence": True,
                "require_final_period": True,
                "forbidden_prefixes": ["a ", "an ", "the ", "new "],
                "max_title_overlap": 1.0,
                "reject_formula_shorthand": True,
                "reject_unsupported_summary_numbers": False,
            },
            "materialize": {"source_split": "base_split", "provenance_fields": []},
        },
    )
    return path


class NaturalLanguageRescueLaneTests(unittest.TestCase):
    def run_builder(self, root: Path, *args: str) -> None:
        argv = ["build_natural_language_rescue_lanes.py", "--repo-root", str(root), *args]
        with mock.patch("sys.argv", argv):
            rescue.main()

    def test_too_long_lane_writes_source_and_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = base_config(root)
            source = root / "excluded.jsonl"
            write_jsonl(
                source,
                [
                    {
                        "candidate_id": "old:1",
                        "paper_id": "p1",
                        "title": "Robust routing",
                        "abstract": "Routing works better when schedules adapt to uncertain traffic.",
                        "generated_summary": "Adaptive routing schedules avoid congestion when traffic forecasts shift across busy networks today.",
                        "writer_decision": "rewrite",
                        "writer_local_filter_flags": ["too_long"],
                    },
                    {
                        "candidate_id": "old:2",
                        "paper_id": "p2",
                        "title": "Skipped",
                        "abstract": "Skipped.",
                        "generated_summary": "A skipped summary is both too long and badly opened for tests.",
                        "writer_decision": "rewrite",
                        "writer_local_filter_flags": ["too_long", "forbidden_prefix"],
                    },
                ],
            )

            self.run_builder(
                root,
                "--lane",
                "too-long",
                "--template-config",
                str(config.relative_to(root)),
                "--source-jsonl",
                str(source.relative_to(root)),
                "--run-root",
                "run_too_long",
                "--release-dir",
                "release_too_long",
                "--config-out",
                "configs/too_long.json",
                "--audit-config-out",
                "configs/too_long_audit.json",
                "--overwrite",
            )

            rows = load_jsonl(root / "run_too_long/source_candidates/natural_language_candidates.jsonl")
            normal_config = json.loads((root / "configs/too_long.json").read_text())
            audit_config = json.loads((root / "configs/too_long_audit.json").read_text())

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["previous_summary_word_count"], 13)
            self.assertEqual(normal_config["writer"]["writer_label"], "deepseek_v4_pro_plain_language_v11_length_rescue")
            self.assertIn("selected_review_rows.jsonl", audit_config["reviewer"]["input_path"])

    def test_start_cleanup_writes_local_eligible_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = base_config(root)
            source = root / "excluded.jsonl"
            write_jsonl(
                source,
                [
                    {
                        "candidate_id": "old:1",
                        "paper_id": "p1",
                        "title": "Routing",
                        "abstract": "Adaptive routing avoids congestion when traffic shifts.",
                        "generated_summary": "A routing schedule avoids congestion when traffic shifts.",
                        "writer_decision": "rewrite",
                        "writer_local_filter_flags": ["forbidden_prefix"],
                    }
                ],
            )

            self.run_builder(
                root,
                "--lane",
                "start-cleanup",
                "--template-config",
                str(config.relative_to(root)),
                "--source-jsonl",
                str(source.relative_to(root)),
                "--run-root",
                "run_start",
                "--release-dir",
                "release_start",
                "--config-out",
                "configs/start.json",
                "--audit-config-out",
                "configs/start_audit.json",
                "--overwrite",
            )

            rows = load_jsonl(root / "run_start/local_eligible_for_review/writer_outputs_ordered.jsonl")
            normal_config = json.loads((root / "configs/start.json").read_text())

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["generated_summary"], "Routing schedule avoids congestion when traffic shifts.")
            self.assertIn("local_eligible_for_review", normal_config["reviewer"]["input_path"])

    def test_math_lane_selects_math_drops_and_uses_math_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = base_config(root)
            source = root / "writer_outputs.jsonl"
            write_jsonl(
                source,
                [
                    {
                        "candidate_id": "old:1",
                        "paper_id": "m1",
                        "title": "Tilings",
                        "abstract": "We prove a condition under which multidimensional tilings must be periodic.",
                        "primary_category": "math.DS",
                        "writer_decision": "drop",
                        "writer_reason_span": "old prompt thought pure math was too technical",
                    },
                    {
                        "candidate_id": "old:2",
                        "paper_id": "m2",
                        "title": "Optimization",
                        "abstract": "Already rewritten.",
                        "primary_category": "math.OC",
                        "writer_decision": "rewrite",
                    },
                    {
                        "candidate_id": "old:3",
                        "paper_id": "c1",
                        "title": "Computer vision",
                        "abstract": "Not math.",
                        "primary_category": "cs.CV",
                        "writer_decision": "drop",
                    },
                ],
            )

            self.run_builder(
                root,
                "--lane",
                "math",
                "--template-config",
                str(config.relative_to(root)),
                "--source-jsonl",
                str(source.relative_to(root)),
                "--run-root",
                "run_math",
                "--release-dir",
                "release_math",
                "--config-out",
                "configs/math.json",
                "--audit-config-out",
                "configs/math_audit.json",
                "--overwrite",
            )

            rows = load_jsonl(root / "run_math/source_candidates/natural_language_candidates.jsonl")
            normal_config = json.loads((root / "configs/math.json").read_text())
            audit_config = json.loads((root / "configs/math_audit.json").read_text())

            self.assertEqual([row["paper_id"] for row in rows], ["m1"])
            self.assertIn("writer_v16_math", normal_config["writer"]["prompt_path"])
            self.assertIn("verifier_v6_math", normal_config["reviewer"]["prompt_path"])
            self.assertIn("verifier_v6_math", audit_config["reviewer"]["prompt_path"])
            self.assertEqual(normal_config["local_gates"]["max_writer_technical_terms"], 4)
            self.assertIn("survey ", normal_config["local_gates"]["forbidden_prefixes"])


if __name__ == "__main__":
    unittest.main()
