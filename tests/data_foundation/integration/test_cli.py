from __future__ import annotations

import contextlib
import io
import unittest

from clouda_data.pipeline.cli import main


class CliTests(unittest.TestCase):
    def test_list_profiles(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["list-profiles"])
        self.assertEqual(code, 0)
        self.assertIn("modern_light", out.getvalue())

    def test_describe_profile(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["describe-profile", "clean_control"])
        self.assertEqual(code, 0)
        self.assertIn("clean_control", out.getvalue())


if __name__ == "__main__":
    unittest.main()
