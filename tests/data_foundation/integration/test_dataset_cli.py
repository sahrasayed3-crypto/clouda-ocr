from __future__ import annotations

import contextlib
import io
import unittest

from clouda_data.pipeline.cli import main


class DatasetCliTests(unittest.TestCase):
    def test_dataset_registry_commands(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["list-dataset-sources"])
        self.assertEqual(code, 0)
        self.assertIn("rasam_dataset", out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            describe_code = main(["describe-dataset-source", "rasam_dataset"])
        self.assertEqual(describe_code, 0)
        self.assertIn("Apache-2.0", out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            estimate_code = main(["estimate-download", "rasam_dataset"])
        self.assertEqual(estimate_code, 0)
        self.assertIn("estimated_bytes", out.getvalue())

    def test_license_command_blocks_unclear_source(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["verify-dataset-license", "pats_a01"])
        self.assertEqual(code, 1)
        self.assertIn("sample_download_allowed", out.getvalue())


if __name__ == "__main__":
    unittest.main()
