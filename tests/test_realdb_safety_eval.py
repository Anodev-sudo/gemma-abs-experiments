import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts import build_realdb_safety_eval_set


def long_text(label: str, marker: str = "") -> str:
    return f"{label} {marker} " + "This abstract contains enough neutral research detail. " * 8


def insert_paper(
    conn: sqlite3.Connection,
    paper_id: str,
    *,
    title: str,
    abstract: str,
    category: str = "cs.LG",
    created_date: str = "2026-01-01",
) -> None:
    conn.execute(
        """
        INSERT INTO papers(arxiv_id, title, abstract, created_date, updated_date, is_deleted)
        VALUES (?, ?, ?, ?, ?, 0)
        """,
        (paper_id, title, abstract, created_date, created_date),
    )
    conn.execute(
        """
        INSERT INTO paper_categories(arxiv_id, category, is_primary)
        VALUES (?, ?, 1)
        """,
        (paper_id, category),
    )


def create_fixture_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE papers(
                arxiv_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                created_date TEXT NOT NULL,
                updated_date TEXT NOT NULL,
                is_deleted INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE paper_categories(
                arxiv_id TEXT NOT NULL,
                category TEXT NOT NULL,
                is_primary INTEGER NOT NULL
            );
            """
        )
        for index in range(1, 10):
            insert_paper(
                conn,
                f"n{index}",
                title=f"Neutral paper {index}",
                abstract=long_text(f"Neutral {index}"),
                category="cs.LG" if index % 2 else "math.OC",
            )
        for index in range(1, 9):
            insert_paper(
                conn,
                f"e{index}",
                title=f"Cybersecurity paper {index}",
                abstract=long_text(f"Edge {index}", "malware phishing surveillance"),
                category="cs.CR",
            )
        insert_paper(
            conn,
            "old1",
            title="Old paper",
            abstract=long_text("Old malware", "malware"),
            category="cs.CR",
            created_date="2020-01-01",
        )
        conn.commit()
    finally:
        conn.close()


class RealDBSafetyEvalTests(unittest.TestCase):
    def test_builder_excludes_prior_ids_and_writes_reproducible_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_path = root / "arxiv.sqlite"
            create_fixture_db(db_path)

            prior_dir = root / "hf_dataset" / "evals" / "prior"
            prior_dir.mkdir(parents=True)
            (prior_dir / "prior.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"paper_id": "n1"}),
                        json.dumps({"paper_id": "e1::adv::json_command_noise::0001"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            output_dir = root / "hf_dataset" / "evals" / "safety_realdb_test"
            args = SimpleNamespace(
                db=str(db_path),
                output_dir=str(output_dir),
                normal_count=4,
                edge_count=3,
                adversarial_count=3,
                created_since="2023-04-07",
                min_abstract_chars=20,
                seed=20260501,
                exclude_root=str(root / "hf_dataset"),
                source_label="test Composer DB",
                overwrite=False,
            )

            manifest = build_realdb_safety_eval_set.build_dataset(args)
            rows = [
                json.loads(line)
                for line in (output_dir / "eval_inputs.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            manifest_text = (output_dir / "manifest.json").read_text(encoding="utf-8")

        self.assertEqual(10, len(rows))
        self.assertEqual({"normal": 4, "edge": 3, "adversarial": 3}, manifest["summary"]["by_split"])
        paper_ids = {row["paper_id"] for row in rows}
        self.assertNotIn("n1", paper_ids)
        self.assertNotIn("e1", {paper_id.split("::adv::", 1)[0] for paper_id in paper_ids})

        selected_real_ids = {row["paper_id"] for row in rows if row["split"] in {"normal", "edge"}}
        adversarial_sources = {
            row["adversarial_source_paper_id"]
            for row in rows
            if row["split"] == "adversarial"
        }
        self.assertTrue(adversarial_sources.isdisjoint(selected_real_ids))
        self.assertNotIn(str(db_path), manifest_text)
        self.assertEqual("arxiv.sqlite", manifest["source_database"]["filename"])
        self.assertFalse(manifest["source_database"]["absolute_path_recorded"])
        self.assertEqual(64, len(manifest["schema_sha256"]))


if __name__ == "__main__":
    unittest.main()
