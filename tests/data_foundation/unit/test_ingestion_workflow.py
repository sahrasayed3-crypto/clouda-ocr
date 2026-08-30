from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from clouda_data.ingestion.file_inspection import inspect_file, text_from_ground_truth
from clouda_data.ingestion.workflow import (
    ingest_source_manifest,
    validate_source_manifest_file,
)


def create_tiny_docx(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types></Types>")
        archive.writestr("word/document.xml", "<w:document>tiny</w:document>")


def create_manifest(
    root: Path, source_name: str = "page.png", gt_name: str = "gt.txt"
) -> Path:
    manifest = {
        "documents": [
            {
                "document_id": "doc_001",
                "source_path": source_name,
                "source_type": "image",
                "language": "ar",
                "source_license": "synthetic-test-only",
            }
        ],
        "pages": [
            {
                "document_id": "doc_001",
                "page_id": "doc_001_p001",
                "page_number": 1,
                "source_path": source_name,
                "source_type": "image",
                "language": "ar",
                "ground_truth_path": gt_name,
                "source_license": "synthetic-test-only",
            }
        ],
    }
    path = root / "source_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return path


class IngestionWorkflowTests(unittest.TestCase):
    def test_inspects_supported_tiny_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tiny.pdf").write_bytes(b"%PDF-1.4\n%tiny\n")
            create_tiny_docx(root / "tiny.docx")
            Image.new("RGB", (8, 8), "white").save(root / "tiny.png")
            (root / "tiny.txt").write_text("نص", encoding="utf-8")
            (root / "gt.json").write_text(
                json.dumps({"clean_text": "نص"}, ensure_ascii=False), encoding="utf-8"
            )
            (root / "page.xml").write_text(
                "<PcGts><TextLine><Unicode>نص</Unicode></TextLine></PcGts>",
                encoding="utf-8",
            )
            (root / "alto.xml").write_text(
                '<alto><String CONTENT="نص" /></alto>', encoding="utf-8"
            )
            for name in [
                "tiny.pdf",
                "tiny.docx",
                "tiny.png",
                "tiny.txt",
                "gt.json",
                "page.xml",
                "alto.xml",
            ]:
                self.assertTrue(inspect_file(root / name).ok, name)

    def test_extracts_ground_truth_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "gt.txt").write_text("نص عربي", encoding="utf-8")
            (root / "gt.json").write_text(
                json.dumps({"clean_text": "نص JSON"}, ensure_ascii=False),
                encoding="utf-8",
            )
            (root / "page.xml").write_text(
                "<PcGts><TextLine><Unicode>نص XML</Unicode></TextLine></PcGts>",
                encoding="utf-8",
            )
            self.assertEqual(text_from_ground_truth(root / "gt.txt"), "نص عربي")
            self.assertEqual(text_from_ground_truth(root / "gt.json"), "نص JSON")
            self.assertIn("نص XML", text_from_ground_truth(root / "page.xml"))

    def test_validates_and_ingests_copy_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            project = Path(tmp) / "project"
            source.mkdir()
            for folder in [
                "data/raw/documents",
                "data/raw/pages",
                "data/raw/ground_truth",
                "data/raw/layout_annotations",
                "data/manifests",
                "outputs/reports",
            ]:
                (project / folder).mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), "white").save(source / "page.png")
            (source / "gt.txt").write_text("نص عربي", encoding="utf-8")
            manifest = create_manifest(source)
            dry_run = ingest_source_manifest(manifest, project, dry_run=True)
            self.assertTrue(dry_run.ok)
            self.assertFalse((project / "data/manifests/page_manifest.json").exists())
            result = ingest_source_manifest(manifest, project, dry_run=False)
            self.assertTrue(result.ok)
            self.assertTrue((project / "data/raw/pages/doc_001_p001.png").exists())
            self.assertTrue((project / "data/manifests/page_manifest.json").exists())
            page_manifest = json.loads(
                (project / "data/manifests/page_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                page_manifest[0]["source_path"], "data/raw/pages/doc_001_p001.png"
            )
            self.assertTrue((source / "page.png").exists())

    def test_rejects_missing_ground_truth_empty_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (8, 8), "white").save(root / "a.png")
            Image.new("RGB", (8, 8), "white").save(root / "b.png")
            payload = {
                "documents": [
                    {
                        "document_id": "doc",
                        "source_path": "a.png",
                        "source_type": "image",
                    }
                ],
                "pages": [
                    {
                        "document_id": "doc",
                        "page_id": "p1",
                        "page_number": 1,
                        "source_path": "a.png",
                        "source_type": "image",
                        "clean_text": "",
                    },
                    {
                        "document_id": "doc",
                        "page_id": "p2",
                        "page_number": 2,
                        "source_path": "b.png",
                        "source_type": "image",
                    },
                ],
            }
            path = root / "bad_manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            plan = validate_source_manifest_file(path, Path.cwd())
            self.assertFalse(plan.ok)
            codes = {issue.code for issue in plan.issues}
            self.assertTrue(
                {"invalid_manifest", "empty_text", "duplicate_file"}.intersection(codes)
            )

    def test_rejects_ground_truth_checksum_mismatch_and_bad_reading_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (8, 8), "white").save(root / "page.png")
            payload = {
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
                        "clean_text": "نص",
                        "text_checksum": "0" * 64,
                        "reading_order": ["missing_region"],
                    }
                ],
            }
            path = root / "bad_checksum_manifest.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            plan = validate_source_manifest_file(path, Path.cwd())
            self.assertFalse(plan.ok)
            codes = {issue.code for issue in plan.issues}
            self.assertIn("ground_truth_checksum_mismatch", codes)
            self.assertIn("invalid_page_record", codes)


if __name__ == "__main__":
    unittest.main()
