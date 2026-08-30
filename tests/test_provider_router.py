import os
import tempfile
import unittest

from pdfword.provider_router import ProviderRouter


class TestProviderRouter(unittest.TestCase):
    def test_choose_best_provider_from_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = os.path.join(td, "provider.json")
            router = ProviderRouter(store_path=store)
            for _ in range(5):
                router.record_success(
                    "openrouter",
                    latency_ms=1400,
                    quality_score=71.0,
                    signature="books|medium|ar|p10",
                )
            for _ in range(6):
                router.record_success(
                    "together",
                    latency_ms=420,
                    quality_score=85.0,
                    signature="books|medium|ar|p10",
                )
            choice = router.choose_provider(
                ["openrouter", "together"], signature="books|medium|ar|p10"
            )
            self.assertEqual(choice.name, "together")

    def test_choose_backup_provider(self) -> None:
        router = ProviderRouter(
            store_path=os.path.join(tempfile.gettempdir(), "provider_router_tmp.json")
        )
        backup = router.choose_backup_provider(
            current="openrouter",
            candidates=["openrouter", "fireworks"],
            signature="default",
        )
        self.assertEqual(backup, "fireworks")


if __name__ == "__main__":
    unittest.main()
