from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from clouda_data.pipeline.cli import main


class IngestionCliTests(unittest.TestCase):
    def test_inspect_source_and_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (8, 8), "white").save(root / "page.png")
            (root / "gt.txt").write_text("نص", encoding="utf-8")
            manifest = {
                "documents": [
                    {
                        "document_id": "doc",
                        "source_path": "page.png",
                        "source_type": "image",
                    }
                ],
                "pages": [
                    {
                        "document_id": "doc",
                        "page_id": "p1",
                        "page_number": 1,
                        "source_path": "page.png",
                        "source_type": "image",
                        "ground_truth_path": "gt.txt",
                    }
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
            )
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                inspect_code = main(["inspect-source", str(root / "page.png")])
            self.assertEqual(inspect_code, 0)
            self.assertIn("image", out.getvalue())
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                ingest_code = main(["ingest", str(manifest_path), "--dry-run"])
            self.assertEqual(ingest_code, 0)
            self.assertIn('"dry_run": true', out.getvalue())
            report_path = root / "report.json"
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                report_code = main(
                    [
                        "generate-ingestion-report",
                        "--manifest",
                        str(manifest_path),
                        "--output",
                        str(report_path),
                    ]
                )
            self.assertEqual(report_code, 0)
            self.assertTrue(report_path.exists())

    def test_list_and_duplicate_commands_are_safe_without_ingested_data(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            list_code = main(["list-ingested"])
        self.assertEqual(list_code, 0)
        self.assertIn('"pages"', out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            duplicate_code = main(["find-duplicates"])
        self.assertEqual(duplicate_code, 0)
        self.assertIn("{", out.getvalue())


if __name__ == "__main__":
    unittest.main()
