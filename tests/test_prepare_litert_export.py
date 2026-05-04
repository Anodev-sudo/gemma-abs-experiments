import json
import tempfile
import unittest
from pathlib import Path

from scripts import prepare_litert_export


class PrepareLiteRTExportTests(unittest.TestCase):
    def test_load_promoted_pointer_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "adapters" / "0000600_adapters.safetensors"
            adapter_config = root / "adapters" / "adapter_config.json"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"adapter")
            adapter_config.write_text("{}", encoding="utf-8")
            pointer = root / "promoted.json"
            pointer.write_text(
                json.dumps(
                    {
                        "status": "promoted",
                        "label": "Faith Overlay 1400 v2 Final 600",
                        "base_model": "google/gemma-3-270m-it",
                        "checkpoint_path": "adapters/0000600_adapters.safetensors",
                        "adapter_config_path": "adapters/adapter_config.json",
                    }
                ),
                encoding="utf-8",
            )

            promoted = prepare_litert_export.load_promoted_pointer(pointer)

        self.assertEqual(promoted.label, "Faith Overlay 1400 v2 Final 600")
        self.assertEqual(promoted.base_model, "google/gemma-3-270m-it")
        self.assertEqual(promoted.checkpoint_path, checkpoint)
        self.assertEqual(promoted.adapter_config_path, adapter_config)

    def test_build_release_paths_uses_stable_slug(self) -> None:
        paths = prepare_litert_export.build_release_paths(
            Path("hf_bucket/experiments/litert_exports"),
            "Faith Overlay 1400 v2 Final 600",
        )

        self.assertEqual(
            paths.release_dir,
            Path("hf_bucket/experiments/litert_exports/faith_overlay_1400_v2_final_600"),
        )
        self.assertEqual(paths.peft_adapter_dir, paths.release_dir / "peft_adapter")
        self.assertEqual(paths.merged_model_dir, paths.release_dir / "merged_hf_model")
        self.assertEqual(paths.litert_output_dir, paths.release_dir / "litert")

    def test_litert_command_doc_uses_repo_relative_paths(self) -> None:
        paths = prepare_litert_export.build_release_paths(
            Path("hf_bucket/experiments/litert_exports"),
            "faith_overlay_1400_v2_final_600",
        )
        doc = prepare_litert_export.render_litert_command_doc(
            paths=paths,
            base_model="google/gemma-3-270m-it",
            label="faith_overlay_1400_v2_final_600",
            prompt_prefix="<start_of_turn>user\n",
            prompt_suffix="<end_of_turn>\n<start_of_turn>model\n",
            stop_tokens=["<end_of_turn>", "<eos>"],
        )

        self.assertIn("merged_hf_model", doc)
        self.assertIn("python -m litert_torch.generative.export_hf", doc)
        self.assertIn("docker build --platform linux/amd64", doc)
        self.assertIn("docker run --rm --platform linux/amd64", doc)
        self.assertIn("hf_bucket/experiments/litert_exports/faith_overlay_1400_v2_final_600/litert_prefill1024", doc)
        self.assertIn("--prefill_lengths=1024,", doc)
        self.assertIn("--quantization_recipe=dynamic_wi8_afp32", doc)
        self.assertNotIn("/Users/", doc)

    def test_manifest_records_artifact_paths(self) -> None:
        promoted = prepare_litert_export.PromotedAdapter(
            pointer_path=Path("hf_bucket/experiments/mlx_lora/promoted/current.json"),
            raw={"status": "promoted"},
            label="faith_overlay_1400_v2_final_600",
            base_model="google/gemma-3-270m-it",
            checkpoint_path=Path("run/adapters/0000600_adapters.safetensors"),
            adapter_config_path=Path("run/adapters/adapter_config.json"),
        )
        paths = prepare_litert_export.build_release_paths(
            Path("hf_bucket/experiments/litert_exports"),
            promoted.label,
        )

        manifest = prepare_litert_export.build_readiness_manifest(
            promoted=promoted,
            paths=paths,
            base_model_path_used="hf_bucket/experiments/mlx_lora/.hf-home/hub/snapshot",
            litert_torch_available=False,
            parity_check_requested=True,
            merged_model_written=True,
        )

        self.assertEqual(manifest["label"], promoted.label)
        self.assertEqual(manifest["base_model"], "google/gemma-3-270m-it")
        self.assertEqual(manifest["artifacts"]["peft_adapter_dir"], str(paths.peft_adapter_dir))
        self.assertEqual(
            manifest["artifacts"]["recommended_litert_output_dir"],
            "hf_bucket/experiments/litert_exports/faith_overlay_1400_v2_final_600/litert_prefill1024",
        )
        self.assertFalse(manifest["litert_torch_available"])
        self.assertTrue(manifest["merged_model_written"])

    def test_parse_args_keeps_mlx_parity_opt_in(self) -> None:
        args = prepare_litert_export.parse_args([])

        self.assertFalse(args.run_parity_check)


if __name__ == "__main__":
    unittest.main()
