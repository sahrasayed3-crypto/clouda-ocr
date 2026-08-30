from __future__ import annotations

import pytest

from clouda_data.evaluation.aggregations import PageMetric, group_by_dimension
from clouda_data.evaluation.cer import cer
from clouda_data.evaluation.normalization import normalize_ocr_text
from pdfword.accuracy import compute_accuracy_metrics


@pytest.mark.parametrize(
    "reference,hypothesis",
    [
        ("مرحباً، Clouda 2026!", "مرحباً، Clouda 2026!"),
        ("نص عربي؛ English.", "نص عربي؛ English."),
    ],
)
def test_both_policies_are_zero_for_identical_mixed_text(
    reference: str,
    hypothesis: str,
) -> None:
    assert compute_accuracy_metrics(reference, hypothesis)["cer"] == 0
    assert cer(normalize_ocr_text(reference), normalize_ocr_text(hypothesis)) == 0


def test_runtime_preserves_diacritics_while_data_policy_folds_them() -> None:
    reference = "كِتاب"
    hypothesis = "كتاب"
    assert compute_accuracy_metrics(reference, hypothesis)["cer"] > 0
    assert cer(normalize_ocr_text(reference), normalize_ocr_text(hypothesis)) == 0


def test_punctuation_remains_significant_in_both_policies() -> None:
    reference = "مرحبا، world!"
    hypothesis = "مرحبا world"
    assert compute_accuracy_metrics(reference, hypothesis)["cer"] > 0
    assert cer(normalize_ocr_text(reference), normalize_ocr_text(hypothesis)) > 0


def test_metric_scale_difference_is_explicit() -> None:
    runtime = compute_accuracy_metrics("abc", "axc")["cer"]
    foundation = cer("abc", "axc")
    assert runtime == pytest.approx(foundation * 100)


def test_metrics_group_by_required_dimensions() -> None:
    rows = [
        PageMetric(
            "p1",
            "clean",
            "low",
            0.1,
            0.2,
            dataset_id="rasam",
            document_type="manuscript",
            quality_class="clean",
            language="ar",
        ),
        PageMetric(
            "p2",
            "clean",
            "low",
            0.3,
            0.4,
            dataset_id="rasam",
            document_type="manuscript",
            quality_class="clean",
            language="ar",
        ),
    ]
    assert group_by_dimension(rows, "dataset_id")["rasam"]["pages"] == 2
    assert group_by_dimension(rows, "document_type")["manuscript"][
        "cer_mean"
    ] == pytest.approx(0.2)
