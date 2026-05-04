import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts import run_mlx_safety_eval_generations


class RunMLXSafetyEvalGenerationsTests(unittest.TestCase):
    def test_standalone_model_path_does_not_require_promoted_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            eval_jsonl = root / "eval.jsonl"
            prompt_spec = root / "prompt.json"
            output_dir = root / "out"
            eval_jsonl.write_text(
                json.dumps(
                    {
                        "eval_id": "normal_0001",
                        "split": "normal",
                        "paper_id": "1234.5678",
                        "title": "A title",
                        "abstract": "An abstract with enough detail.",
                        "primary_category": "cs.LG",
                        "umbrella_category": "cs",
                        "categories": ["cs.LG"],
                        "created_date": "2026-01-01",
                        "edge_topics": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            prompt_spec.write_text(
                json.dumps(
                    {
                        "prompt_version": "test_prompt",
                        "system_prompt": "System.",
                        "user_prompt_templates": {
                            "title_abstract": "Title: {title}\nAbstract: {abstract}",
                            "abstract_only": "Abstract: {abstract}",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                eval_jsonl=str(eval_jsonl),
                promoted_pointer=str(root / "missing_promoted.json"),
                label="q8_test",
                adapter_dir=None,
                checkpoint=None,
                prompt_spec=str(prompt_spec),
                input_mode="title_abstract",
                hf_home="unused",
                model_path="standalone/q8",
                output_root=str(root),
                output_dir=str(output_dir),
                batch_size=2,
                max_tokens=40,
                temp=0.0,
                top_p=0.0,
                max_examples=None,
            )

            original_parse_args = run_mlx_safety_eval_generations.parse_args
            original_load_promoted = run_mlx_safety_eval_generations.load_promoted_defaults
            original_generate_outputs = run_mlx_safety_eval_generations.generate_outputs
            try:
                run_mlx_safety_eval_generations.parse_args = lambda: args
                run_mlx_safety_eval_generations.load_promoted_defaults = (
                    lambda _: self.fail("promoted pointer should not be loaded")
                )

                def fake_generate_outputs(prepared_rows, *, model_path, variant, **kwargs):
                    self.assertEqual("standalone/q8", model_path)
                    self.assertEqual("standalone_model", variant.kind)
                    self.assertEqual("q8_test", variant.label)
                    self.assertIsNone(variant.adapter_load_path)
                    self.assertEqual(1, len(prepared_rows))
                    return [{"raw": "Generated summary.", "normalized": "Generated summary."}]

                run_mlx_safety_eval_generations.generate_outputs = fake_generate_outputs
                run_mlx_safety_eval_generations.main()
            finally:
                run_mlx_safety_eval_generations.parse_args = original_parse_args
                run_mlx_safety_eval_generations.load_promoted_defaults = original_load_promoted
                run_mlx_safety_eval_generations.generate_outputs = original_generate_outputs

            generations = [
                json.loads(line)
                for line in (output_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual("Generated summary.", generations[0]["summary"])
        self.assertEqual("q8_test", generations[0]["model_label"])


if __name__ == "__main__":
    unittest.main()
