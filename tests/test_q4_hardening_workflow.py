import csv
import tempfile
import unittest
from pathlib import Path

from scripts import aggregate_q4_hardening_reviews
from scripts import build_q4_hardening_eval_set
from scripts import materialize_q4_hardening_combined_release
from scripts import prepare_q4_hardening_review_packets


def make_row(paper_id: str, title: str, abstract: str, umbrella: str = "cs") -> dict:
    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "primary_category": f"{umbrella}.LG" if umbrella != "other" else "stat.ML",
        "umbrella_category": umbrella,
        "categories": [f"{umbrella}.LG"] if umbrella != "other" else ["stat.ML"],
        "created_date": "2026-01-01",
    }


class Q4HardeningWorkflowTests(unittest.TestCase):
    def test_eval_sampler_produces_unique_held_out_bucket_counts(self) -> None:
        rows = [
            make_row("n1", "metric", "The method improves accuracy by 42 percent."),
            make_row("n2", "factor", "The solver reduces latency by 3.5x."),
            make_row("p1", "negation", "The result does not improve under noise."),
            make_row("p2", "contrast", "The baseline is worse, but the new variant is stable."),
            make_row("e1", "APT report", "APT malware targets a clinical GPU workflow."),
            make_row("e2", "NASA survey", "NASA and ESA compare CNN classifiers."),
            make_row("l1", "notation", "We analyze $\\alpha + \\beta + \\gamma$ operators."),
            make_row("l2", "manifold", "A non-Hermitian quasiperiodic manifold is studied."),
            make_row("c1", "plain", "A dataset is curated for small model comparison.", "math"),
            make_row("c2", "plain two", "A benchmark is assembled for summary evaluation.", "physics"),
        ]

        sampled = build_q4_hardening_eval_set.build_eval_rows(
            rows,
            excluded_ids={"n2"},
            bucket_targets={
                "numeric": 1,
                "polarity": 2,
                "entity": 2,
                "long_notation": 1,
                "control": 2,
            },
            seed=7,
        )

        self.assertEqual(len(sampled), 8)
        self.assertEqual(len({row["paper_id"] for row in sampled}), 8)
        self.assertNotIn("n2", {row["paper_id"] for row in sampled})
        self.assertEqual(
            {
                "numeric": 1,
                "polarity": 2,
                "entity": 2,
                "long_notation": 1,
                "control": 2,
            },
            build_q4_hardening_eval_set.summarize(sampled)["by_selection_bucket"],
        )

    def test_review_packet_side_map_preserves_q8_and_q4_sources(self) -> None:
        generation_rows = [
            {
                "paper_id": "1234.1",
                "title": "A title",
                "abstract": "An abstract",
                "primary_category": "cs.LG",
                "categories": ["cs.LG"],
                "created_date": "2026-01-01",
                "prompt_version": "prompt",
                "input_mode": "title_abstract",
                "left_label": "q8_promoted",
                "left_summary": "q8 summary",
                "left_summary_raw": "q8 summary",
                "right_label": "q4_lineage",
                "right_summary": "q4 summary",
                "right_summary_raw": "q4 summary",
            }
        ]

        review_rows, side_maps = prepare_q4_hardening_review_packets.build_review_rows(
            generation_rows,
            q8_label="q8_promoted",
            q4_label="q4_lineage",
            candidate_alias="q4",
            review_id_prefix="q4_hardening",
            seed=1,
        )

        self.assertEqual(review_rows[0]["paper_id"], "1234.1")
        self.assertEqual(side_maps[0]["q8_summary"], "q8 summary")
        self.assertEqual(side_maps[0]["q4_summary"], "q4 summary")
        self.assertIn(side_maps[0]["left_source"], {"q8", "q4"})
        self.assertNotEqual(side_maps[0]["left_source"], side_maps[0]["right_source"])

    def test_aggregator_keeps_only_q4_severe_q8_clean_usable_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            review_dir = root / "review"
            output_root = root / "out"
            release_dir = root / "release"
            (review_dir / "primary" / "packet_00").mkdir(parents=True)

            prepare_q4_hardening_review_packets.write_jsonl(
                review_dir / "review_items.jsonl",
                [
                    {
                        "review_id": "q4_hardening:0001",
                        "paper_id": "ok",
                        "title": "OK",
                        "abstract": "Abstract",
                        "primary_category": "cs.LG",
                        "categories": ["cs.LG"],
                        "created_date": "2026-01-01",
                        "selection_bucket": "numeric",
                    },
                    {
                        "review_id": "q4_hardening:0002",
                        "paper_id": "bad_q8",
                        "title": "Bad q8",
                        "abstract": "Abstract",
                        "primary_category": "cs.LG",
                        "categories": ["cs.LG"],
                        "created_date": "2026-01-01",
                        "selection_bucket": "entity",
                    },
                ],
            )
            prepare_q4_hardening_review_packets.write_jsonl(
                review_dir / "side_map.jsonl",
                [
                    {
                        "review_id": "q4_hardening:0001",
                        "paper_id": "ok",
                        "left_source": "q8",
                        "right_source": "q4",
                        "q8_summary": "Clean q8 target.",
                        "q4_summary": "Broken q4 summary.",
                    },
                    {
                        "review_id": "q4_hardening:0002",
                        "paper_id": "bad_q8",
                        "left_source": "q4",
                        "right_source": "q8",
                        "q8_summary": "Broken q8 target.",
                        "q4_summary": "Broken q4 summary.",
                    },
                ],
            )
            decisions_path = review_dir / "primary" / "packet_00" / "decisions.tsv"
            with decisions_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=prepare_q4_hardening_review_packets.DECISION_HEADERS,
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "review_id": "q4_hardening:0001",
                        "winner": "left",
                        "left_severe": "no",
                        "right_severe": "yes",
                        "right_issue_tags": "unsupported_number",
                        "left_target_usable": "yes",
                        "right_target_usable": "no",
                        "confidence": "high",
                    }
                )
                writer.writerow(
                    {
                        "review_id": "q4_hardening:0002",
                        "winner": "both_bad",
                        "left_severe": "yes",
                        "right_severe": "yes",
                        "left_issue_tags": "unsupported_entity",
                        "right_issue_tags": "polarity_flip",
                        "left_target_usable": "no",
                        "right_target_usable": "no",
                        "confidence": "high",
                    }
                )

            args = type(
                "Args",
                (),
                {
                    "review_packet_dir": str(review_dir),
                    "output_root": str(output_root),
                    "release_dir": str(release_dir),
                    "qa_control_ratio": 0.1,
                    "seed": 1,
                    "write_qa_followup": False,
                    "require_qa_for_distill": False,
                },
            )()

            # Exercise the command entrypoint behavior without shelling out.
            original_parse_args = aggregate_q4_hardening_reviews.parse_args
            try:
                aggregate_q4_hardening_reviews.parse_args = lambda: args
                aggregate_q4_hardening_reviews.main()
            finally:
                aggregate_q4_hardening_reviews.parse_args = original_parse_args

            training_rows = aggregate_q4_hardening_reviews.load_jsonl(
                release_dir / "training_rows.jsonl"
            )

        self.assertEqual(len(training_rows), 1)
        self.assertEqual(training_rows[0]["paper_id"], "ok")
        self.assertEqual(training_rows[0]["summary"], "Clean q8 target.")

    def test_combined_release_prefers_kimi_on_overlap(self) -> None:
        strict_release = Path("strict_release")
        kimi_release = Path("kimi_release")
        strict_rows = [
            {
                "paper_id": "overlap",
                "title": "Overlap",
                "abstract": "An abstract about a fragile q4 case.",
                "summary": "Strict q8 target.",
                "primary_category": "cs.LG",
                "categories": ["cs.LG"],
                "created_date": "2026-01-01",
                "q4_issue_tags": ["unsupported_number"],
                "selection_bucket": "numeric",
            },
            {
                "paper_id": "strict_only",
                "title": "Strict only",
                "abstract": "An abstract about a strict-only row.",
                "summary": "Strict-only target.",
                "primary_category": "cs.CL",
                "categories": ["cs.CL"],
                "created_date": "2026-01-02",
                "q4_issue_tags": ["polarity_flip"],
                "selection_bucket": "polarity",
            },
        ]
        kimi_rows = [
            {
                "paper_id": "overlap",
                "title": "Overlap",
                "abstract": "An abstract about a fragile q4 case.",
                "summary": "Kimi verified correction.",
                "primary_category": "cs.LG",
                "categories": ["cs.LG"],
                "created_date": "2026-01-01",
                "q4_issue_tags": ["unsupported_number"],
                "selection_bucket": "numeric",
            },
            {
                "paper_id": "kimi_only",
                "title": "Kimi only",
                "abstract": "An abstract about a Kimi-only row.",
                "summary": "Kimi-only correction.",
                "primary_category": "math.OC",
                "categories": ["math.OC"],
                "created_date": "2026-01-03",
                "q4_issue_tags": ["unsupported_entity"],
                "selection_bucket": "entity",
            },
        ]

        combined, overlaps = (
            materialize_q4_hardening_combined_release.materialize_combined_rows(
                strict_rows=strict_rows,
                strict_release_dir=strict_release,
                kimi_rows=kimi_rows,
                kimi_release_dir=kimi_release,
            )
        )

        by_id = {row["paper_id"]: row for row in combined}
        self.assertEqual(len(combined), 3)
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(by_id["overlap"]["summary"], "Kimi verified correction.")
        self.assertEqual(
            by_id["overlap"]["q4_hardening_combined_source"],
            "kimi_qwen_verified",
        )
        self.assertEqual(by_id["strict_only"]["summary"], "Strict-only target.")
        self.assertEqual(by_id["kimi_only"]["summary"], "Kimi-only correction.")


if __name__ == "__main__":
    unittest.main()
