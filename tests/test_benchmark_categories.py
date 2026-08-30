import unittest

from tools.generate_test_pdfs import (
    MIN_SAMPLES_PER_CATEGORY,
    REQUIRED_CATEGORIES,
    validate_manifest_cases,
)
from tools.run_full_benchmark import (
    _build_category_summary,
    _category_execution_plan,
    _performance_comparison,
    _validate_selected_categories,
    _weakest_categories,
    passes_text_acceptance,
)


def _mk_case(case_id: str, category: str, status: str = "ok") -> dict:
    return {
        "id": case_id,
        "category": category,
        "status": status,
        "word_accuracy": 90.0,
        "char_accuracy": 92.0,
        "visual_score": 88.0,
        "elapsed_sec": 1.0,
    }


class TestBenchmarkCategories(unittest.TestCase):
    def test_acceptance_rejects_wer_above_ten_percent(self) -> None:
        row = {
            "pages_below_90": 0,
            "reference_quality": "valid",
            "char_accuracy": 96.0,
            "word_accuracy": 89.9,
        }
        self.assertFalse(passes_text_acceptance(row))
        row["word_accuracy"] = 90.0
        self.assertTrue(passes_text_acceptance(row))

    def test_validate_manifest_cases_requires_all_categories(self) -> None:
        cases = [_mk_case("a", "born_digital_modern")]
        errors = validate_manifest_cases(cases)
        self.assertTrue(any("missing category" in e for e in errors))

    def test_validate_selected_categories_enforces_min_samples(self) -> None:
        cases = []
        for cat in REQUIRED_CATEGORIES:
            for i in range(MIN_SAMPLES_PER_CATEGORY):
                cases.append(_mk_case(f"{cat}_{i}", cat))
        errors = _validate_selected_categories(cases, require_all_categories=True)
        self.assertEqual(errors, [])

        too_few = [
            c
            for c in cases
            if c["category"] != "encrypted_pdfs" or c["id"].endswith("_0")
        ]
        errors = _validate_selected_categories(too_few, require_all_categories=True)
        self.assertTrue(any("too few selected samples" in e for e in errors))

    def test_report_aggregations(self) -> None:
        cases = [
            _mk_case("c1", "clean_scans"),
            _mk_case("c2", "clean_scans"),
            {
                **_mk_case("c3", "blurry_docs"),
                "word_accuracy": 70.0,
                "elapsed_sec": 2.5,
            },
            {
                **_mk_case("c4", "blurry_docs"),
                "word_accuracy": 72.0,
                "elapsed_sec": 2.0,
            },
        ]
        summary = _build_category_summary(cases)
        self.assertEqual(len(summary), 2)

        weakest = _weakest_categories(summary, limit=1)
        self.assertEqual(weakest[0]["category"], "blurry_docs")

        perf = _performance_comparison(summary)
        self.assertEqual(perf["fastest_category"], "clean_scans")
        self.assertEqual(perf["slowest_category"], "blurry_docs")
        self.assertGreater(perf["time_gap_sec"], 0.0)

    def test_category_execution_plan_switches_models_by_type(self) -> None:
        hard = _category_execution_plan(
            "blurry_docs",
            default_mode="hyper",
            fast_model="fast-x",
            accurate_model="acc-y",
        )
        self.assertEqual(hard["mode"], "max_accuracy")
        self.assertEqual(hard["fast_model"], "acc-y")
        self.assertEqual(hard["accurate_model"], "acc-y")

        easy = _category_execution_plan(
            "born_digital_modern",
            default_mode="hyper",
            fast_model="fast-x",
            accurate_model="acc-y",
        )
        self.assertEqual(easy["mode"], "turbo")
        self.assertEqual(easy["fast_model"], "fast-x")


if __name__ == "__main__":
    unittest.main()
