from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clouda_data.config.models import ConfigError, config_from_mapping
from clouda_data.utils.paths import assert_inside


class ConfigTests(unittest.TestCase):
    def test_valid_config_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config_from_mapping({}, Path(tmp))
        self.assertEqual(cfg.runtime.render_dpi, 300)
        self.assertTrue(
            str(cfg.paths.raw_documents).endswith("data\\raw\\documents")
            or str(cfg.paths.raw_documents).endswith("data/raw/documents")
        )

    def test_invalid_dpi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                config_from_mapping({"runtime": {"render_dpi": 12}}, Path(tmp))

    def test_path_safety(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(assert_inside(root, root / "data").name, "data")
            with self.assertRaises(ValueError):
                assert_inside(root, root.parent)


if __name__ == "__main__":
    unittest.main()
