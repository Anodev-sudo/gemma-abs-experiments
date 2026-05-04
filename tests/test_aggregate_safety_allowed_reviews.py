import json
import tempfile
import unittest
from pathlib import Path

from scripts import aggregate_safety_allowed_reviews


class AggregateSafetyAllowedReviewsTests(unittest.TestCase):
    def test_aggregates_expected_packet_reviews_and_misses(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reviews_dir = root / "reviews"
            reviews_dir.mkdir()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "packets": [
                            {"jsonl": "packet_01.jsonl"},
                            {"jsonl": "packet_02.jsonl"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (reviews_dir / "packet_01_review.md").write_text(
                "No suspected launch-risk misses.\n",
                encoding="utf-8",
            )
            (reviews_dir / "packet_02_review.md").write_text(
                "- eval_id: adversarial_0001\n"
                "  risk: prompt_leakage\n"
                "  why: Echoed injected instructions.\n",
                encoding="utf-8",
            )

            summary = aggregate_safety_allowed_reviews.aggregate_reviews(
                reviews_dir,
                manifest_path,
            )

        self.assertEqual(2, summary["review_files_found"])
        self.assertEqual([], summary["missing_review_files"])
        self.assertEqual(1, summary["suspected_miss_count"])
        self.assertEqual("adversarial_0001", summary["suspected_misses"][0]["eval_id"])


if __name__ == "__main__":
    unittest.main()
