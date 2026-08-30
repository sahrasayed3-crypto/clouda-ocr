from __future__ import annotations

import unittest

from clouda_data.distortion.base import DistortionSpec
from clouda_data.distortion.pipeline import DistortionPipeline
from clouda_data.distortion.randomness import derive_seed
from clouda_data.pipeline.profiles import (
    list_profile_paths,
    load_profile,
    profile_to_specs,
)


class DistortionProfileTests(unittest.TestCase):
    def test_seed_reproducibility(self) -> None:
        self.assertEqual(derive_seed(7, "page", "blur"), derive_seed(7, "page", "blur"))
        self.assertNotEqual(
            derive_seed(7, "page", "blur"), derive_seed(8, "page", "blur")
        )

    def test_pipeline_metadata_only(self) -> None:
        pipeline = DistortionPipeline(
            "test", [DistortionSpec("blur", probability=1.0, severity="light")], 42
        )
        image, metadata = pipeline.run_metadata_only({"page": "tiny"}, "p1")
        self.assertEqual(image["page"], "tiny")
        self.assertEqual(metadata[0].name, "blur")

    def test_all_profiles_validate(self) -> None:
        paths = list_profile_paths()
        self.assertGreaterEqual(len(paths), 16)
        for path in paths:
            profile = load_profile(path)
            self.assertTrue(profile_to_specs(profile))


if __name__ == "__main__":
    unittest.main()
