import os
import tempfile
import unittest

from pdfword.self_learning import SelfLearningEngine


class TestSelfLearningRuntime(unittest.TestCase):
    def test_runtime_error_profile_adapts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "learning.json")
            engine = SelfLearningEngine(path=path)

            for _ in range(3):
                engine.record_runtime_error(
                    quality_label="CLEAR",
                    model="openai/gpt-4o-mini",
                    dpi=900,
                    aggressive=False,
                    error_text="429 Rate Limit from provider",
                    page_no=1,
                )
            for _ in range(2):
                engine.record_runtime_error(
                    quality_label="CLEAR",
                    model="anthropic/claude-sonnet-4.5",
                    dpi=1000,
                    aggressive=True,
                    error_text="no endpoints found for model",
                    page_no=2,
                )

            profile = engine.get_adaptive_profile(
                "CLEAR",
                allowed_models={"openai/gpt-4o-mini", "anthropic/claude-sonnet-4.5"},
            )

            self.assertLess(profile["max_width_scale"], 1.0)
            self.assertLess(profile["jpeg_quality_delta"], 0)
            self.assertIn("anthropic/claude-sonnet-4.5", profile["avoid_models"])


if __name__ == "__main__":
    unittest.main()
