from __future__ import annotations

import unittest

from clouda_data.datasets.registry import get_source, list_sources, verify_license


class DatasetRegistryTests(unittest.TestCase):
    def test_registry_contains_classified_sources(self) -> None:
        sources = list_sources()
        self.assertGreaterEqual(len(sources), 10)
        classifications = {source["classification"] for source in sources}
        self.assertTrue(
            {
                "approved_with_conditions",
                "research_only",
                "unclear_license",
                "rejected",
            }.issubset(classifications)
        )

    def test_license_verification_blocks_unclear_and_noncommercial(self) -> None:
        rasam = verify_license(get_source("rasam_dataset"))
        self.assertTrue(rasam["commercial_use_allowed"])
        self.assertTrue(rasam["sample_download_allowed"])
        openiti = verify_license(get_source("openiti_makhzan"))
        self.assertFalse(openiti["commercial_use_allowed"])
        self.assertFalse(openiti["sample_download_allowed"])
        qnl = verify_license(get_source("qnl_arabic_ocr_corpus_v2"))
        self.assertFalse(qnl["sample_download_allowed"])


if __name__ == "__main__":
    unittest.main()
