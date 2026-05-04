import json
import tempfile
import unittest
from pathlib import Path

from scripts import prepare_safety_allowed_review_packets


class PrepareSafetyAllowedReviewPacketsTests(unittest.TestCase):
    def test_packetizer_writes_jsonl_markdown_and_manifest(self) -> None:
        rows = [
            {
                "eval_id": f"normal_{index:04d}",
                "split": "normal",
                "paper_id": f"p{index}",
                "title": f"Title {index}",
                "abstract": "A harmless abstract about model evaluation.",
                "summary": "A harmless summary.",
                "primary_category": "cs.LG",
                "edge_topics": [],
                "filter_status": "allowed",
            }
            for index in range(1, 6)
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            allowed_path = root / "allowed.jsonl"
            output_dir = root / "packets"
            prepare_safety_allowed_review_packets.write_jsonl(allowed_path, rows)

            loaded = prepare_safety_allowed_review_packets.load_jsonl(allowed_path)
            manifest = prepare_safety_allowed_review_packets.write_packets(
                allowed_jsonl=allowed_path,
                output_dir=output_dir,
                rows=loaded,
                chunk_size=2,
            )

            packet_one = [
                json.loads(line)
                for line in (output_dir / "packet_01.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            packet_markdown = (output_dir / "packet_01.md").read_text(encoding="utf-8")
            manifest_file = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(3, manifest["packet_count"])
        self.assertEqual(2, len(packet_one))
        self.assertEqual("normal_0001", packet_one[0]["eval_id"])
        self.assertIn("Review only for launch-risk misses", packet_markdown)
        self.assertEqual(manifest["packet_count"], manifest_file["packet_count"])
        self.assertEqual({"normal": 5}, manifest_file["summary"]["by_split"])


if __name__ == "__main__":
    unittest.main()
