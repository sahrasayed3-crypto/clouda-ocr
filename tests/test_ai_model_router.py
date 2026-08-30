import unittest

from pdfword.ai_model_router import AIModelRouter, PageSignals
from pdfword.constants import (
    MODEL_ROUTER_CLAUDE_SONNET,
    MODEL_ROUTER_DEEPSEEK_V3,
    MODEL_ROUTER_GPT5,
    MODEL_ROUTER_GPT5_MINI,
)


class TestAIModelRouter(unittest.TestCase):
    def test_simple_page_never_routes_images_to_deepseek(self) -> None:
        router = AIModelRouter()
        signals = PageSignals(
            page_no=1,
            total_pages=5,
            born_digital=True,
            scanned_page=False,
            complexity=0.2,
            layout_complexity=0.2,
            arabic_ratio=0.15,
            english_ratio=0.8,
            has_tables=False,
            has_images=False,
            has_equations=False,
            ocr_confidence_hint=80.0,
        )
        attempts = router.build_attempts(signals, doc_signature="doc-a", mode="turbo")
        self.assertEqual(attempts[0].model, MODEL_ROUTER_GPT5_MINI)
        self.assertNotIn(
            MODEL_ROUTER_DEEPSEEK_V3, [attempt.model for attempt in attempts]
        )

    def test_arabic_difficult_prefers_gpt5(self) -> None:
        router = AIModelRouter()
        signals = PageSignals(
            page_no=2,
            total_pages=60,
            born_digital=False,
            scanned_page=True,
            complexity=0.45,
            layout_complexity=0.48,
            arabic_ratio=0.7,
            english_ratio=0.2,
            has_tables=False,
            has_images=False,
            has_equations=False,
            ocr_confidence_hint=61.0,
        )
        attempts = router.build_attempts(
            signals, doc_signature="doc-b", mode="balanced"
        )
        self.assertEqual(attempts[0].model, MODEL_ROUTER_GPT5)

    def test_visual_table_complex_uses_text_fidelity_routing(self) -> None:
        router = AIModelRouter()
        signals = PageSignals(
            page_no=3,
            total_pages=10,
            born_digital=False,
            scanned_page=True,
            complexity=0.86,
            layout_complexity=0.81,
            arabic_ratio=0.1,
            english_ratio=0.85,
            has_tables=True,
            has_images=True,
            has_equations=False,
            ocr_confidence_hint=48.0,
        )
        attempts = router.build_attempts(signals, doc_signature="doc-c", mode="hyper")
        self.assertEqual(attempts[0].model, MODEL_ROUTER_CLAUDE_SONNET)
        self.assertNotEqual(attempts[0].reason, "complex_table_or_visual_structure")

    def test_layout_extreme_prefers_claude(self) -> None:
        router = AIModelRouter()
        signals = PageSignals(
            page_no=4,
            total_pages=8,
            born_digital=False,
            scanned_page=True,
            complexity=0.6,
            layout_complexity=0.92,
            arabic_ratio=0.2,
            english_ratio=0.6,
            has_tables=False,
            has_images=True,
            has_equations=False,
            ocr_confidence_hint=67.0,
        )
        attempts = router.build_attempts(
            signals, doc_signature="doc-d", mode="max_accuracy"
        )
        self.assertEqual(attempts[0].model, MODEL_ROUTER_CLAUDE_SONNET)

    def test_configured_order_does_not_override_quality_routing(self) -> None:
        router = AIModelRouter(
            preferred_models=[MODEL_ROUTER_CLAUDE_SONNET, MODEL_ROUTER_GPT5_MINI]
        )
        signals = PageSignals(
            page_no=1,
            total_pages=1,
            born_digital=True,
            scanned_page=False,
            complexity=0.1,
            layout_complexity=0.1,
            arabic_ratio=0.1,
            english_ratio=0.9,
            has_tables=False,
            has_images=False,
            has_equations=False,
            ocr_confidence_hint=95.0,
        )
        attempts = router.build_attempts(
            signals, doc_signature="doc-order", mode="turbo"
        )
        self.assertEqual(attempts[0].model, MODEL_ROUTER_GPT5_MINI)


if __name__ == "__main__":
    unittest.main()
