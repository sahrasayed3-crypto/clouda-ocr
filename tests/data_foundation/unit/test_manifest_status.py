from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clouda_data.core.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from clouda_data.core.manifest import GeneratedPageManifestEntry
from clouda_data.core.status import PageStatus, assert_transition
from clouda_data.storage.manifest_store import JsonlManifestStore
from clouda_data.validation.manifest_validation import validate_generated_manifest


def sample_entry() -> GeneratedPageManifestEntry:
    return GeneratedPageManifestEntry(
        source_document_id="doc",
        source_page_id="p1",
        generated_page_id="g1",
        source_image_path="data/raw/pages/p1.png",
        output_image_path="outputs/distorted_pages/g1.png",
        ground_truth_path="data/raw/ground_truth/p1.txt",
        source_checksum="a" * 64,
        output_checksum="b" * 64,
        text_checksum="c" * 64,
        distortion_profile="clean_control",
        distortion_operations=["clean_control"],
        operation_order=["clean_control"],
        parameters={},
        random_seed=1,
        render_dpi=300,
        image_width=100,
        image_height=100,
        source_language="ar",
    )


class ManifestStatusTests(unittest.TestCase):
    def test_status_transition(self) -> None:
        assert_transition(PageStatus.QUEUED, PageStatus.RENDERING)
        with self.assertRaises(ValueError):
            assert_transition(PageStatus.DONE, PageStatus.FAILED)

    def test_manifest_store_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JsonlManifestStore(Path(tmp) / "manifest.jsonl")
            store.append(sample_entry())
            self.assertEqual(store.generated_ids(), {"g1"})

    def test_manifest_validation(self) -> None:
        validate_generated_manifest(sample_entry())

    def test_checkpoint_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            checkpoint = Checkpoint("job")
            checkpoint.mark("p1", PageStatus.FAILED)
            checkpoint.increment_attempt("p1")
            save_checkpoint(checkpoint, path)
            loaded = load_checkpoint(path)
            self.assertEqual(loaded.statuses["p1"], PageStatus.FAILED)
            self.assertEqual(loaded.attempts["p1"], 1)


if __name__ == "__main__":
    unittest.main()
