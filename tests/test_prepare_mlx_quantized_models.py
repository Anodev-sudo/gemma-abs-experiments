import unittest
from pathlib import Path

from scripts import prepare_mlx_quantized_models


class PrepareMLXQuantizedModelsTests(unittest.TestCase):
    def test_build_output_dir_names_bit_width_and_group_size(self) -> None:
        path = prepare_mlx_quantized_models.build_output_dir(
            output_root=Path("hf_bucket/experiments/mlx_lora/quantized_models"),
            label="faith_overlay_1400_v2_final_600",
            bits=4,
            group_size=64,
        )

        self.assertEqual(
            path,
            Path(
                "hf_bucket/experiments/mlx_lora/quantized_models/"
                "faith_overlay_1400_v2_final_600/q4_g64"
            ),
        )

    def test_build_convert_command_uses_mlx_lm_convert(self) -> None:
        command = prepare_mlx_quantized_models.build_convert_command(
            runtime_python=Path("runtime/bin/python"),
            source_hf_model=Path("merged_hf_model"),
            output_dir=Path("quantized/q4_g64"),
            bits=4,
            group_size=64,
            trust_remote_code=True,
        )

        self.assertEqual(command[:4], ["runtime/bin/python", "-m", "mlx_lm", "convert"])
        self.assertIn("--quantize", command)
        self.assertIn("--q-bits", command)
        self.assertIn("4", command)
        self.assertIn("--q-group-size", command)
        self.assertIn("64", command)
        self.assertIn("--trust-remote-code", command)

    def test_parse_bits_defaults_to_8_and_4(self) -> None:
        self.assertEqual(
            prepare_mlx_quantized_models.parse_bits(None),
            [8, 4],
        )


if __name__ == "__main__":
    unittest.main()
