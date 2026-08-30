from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from clouda_data.datasets.rasam_batch import (
    RasamBatchPlan,
    RasamPageCandidate,
    verify_rasam_first_batch,
)
from clouda_data.ingestion.file_inspection import text_from_ground_truth


def write_page_xml(path: Path, page_id: str, text: str) -> None:
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
  <Metadata><Creator>unit test metadata</Creator></Metadata>
  <Page imageFilename="{page_id}.jpg" imageWidth="120" imageHeight="90">
    <TextRegion id="r1" type="body">
      <TextLine id="l1"><TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>
    </TextRegion>
  </Page>
</PcGts>
""",
        encoding="utf-8",
    )


def make_candidate(page_id: str) -> RasamPageCandidate:
    return RasamPageCandidate(
        page_id=page_id,
        subset="rasam1",
        page_xml_url=f"https://example.test/{page_id}.xml",
        page_xml_size_bytes=100,
        github_sha="abc",
        image_id="1",
        image_url=f"https://example.test/{page_id}.jpg",
        image_width=120,
        image_height=90,
        iiif_manifest_url="https://example.test/manifest",
        bibliographical_record_url="https://example.test/item",
        manuscript="MS.TEST",
        image_rights="Domaine public",
        image_license="Public Domain Mark 1.0",
        estimated_image_size_bytes=100,
    )


class RasamBatchTests(unittest.TestCase):
    def test_page_xml_text_ignores_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.xml"
            write_page_xml(path, "page", "نص مضبوط")
            self.assertEqual(text_from_ground_truth(path, "page_xml"), "نص مضبوط")

    def test_verify_batch_accepts_valid_pages_and_records_empty_ground_truth(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_root = root / "data/downloads/rasam_dataset/first_batch"
            (batch_root / "page_xml").mkdir(parents=True)
            (batch_root / "images").mkdir()
            (batch_root / "metadata").mkdir()
            for name in ["LICENSE", "README.md", "contributing.md", "list-images.tsv"]:
                (batch_root / f"metadata/{name}").write_text(
                    "metadata", encoding="utf-8"
                )
            for page_id, text, color in [
                ("valid_page", "نص", "white"),
                ("empty_page", "", "black"),
            ]:
                write_page_xml(batch_root / f"page_xml/{page_id}.xml", page_id, text)
                Image.new("RGB", (120, 90), color=color).save(
                    batch_root / f"images/{page_id}.jpg"
                )
            plan = RasamBatchPlan(
                source_id="rasam_dataset",
                batch_id="rasam_first_batch",
                ok=True,
                created_at="2026-07-23T00:00:00+00:00",
                page_count=2,
                file_count=8,
                estimated_download_size_bytes=1000,
                estimated_extracted_size_bytes=1000,
                available_disk_bytes=1000000,
                directories_to_create=[],
                files_to_create=[],
                license_summary={},
                pages=[make_candidate("valid_page"), make_candidate("empty_page")],
            )
            result = verify_rasam_first_batch(root, plan=plan)
            self.assertTrue(result.ok, result.issues)
            self.assertEqual(result.valid_pages, 1)
            self.assertEqual(result.rejected_pages, 1)
            manifest = json.loads(
                (root / "data/manifests/rasam_first_batch_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["ground_truth_coverage"], "1/2")
            rejections = (
                root / "data/manifests/rasam_first_batch_rejections.jsonl"
            ).read_text(encoding="utf-8")
            self.assertIn("empty_ground_truth", rejections)


if __name__ == "__main__":
    unittest.main()
