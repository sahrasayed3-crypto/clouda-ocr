"""Ground truth, prediction, and holdout safety tests."""

from __future__ import annotations

import pytest

from clouda_data.results.ground_truth import (
    build_ground_truth_record,
    normalized_view,
    verify_ground_truth,
)
from clouda_data.results.models import (
    GroundTruthRecord,
    OCRPrediction,
    PageRecord,
    ProtectionInfo,
)
from tests.results.fixtures_e2e import build_fixture_pages

ARABIC_RAW = "كِتَابٌ قَدِيمٌ فِي مَكْتَبَةِ القَاهِرَة"


def _page(split: str = "train") -> PageRecord:
    return PageRecord(
        page_id="pg1",
        document_id="doc1",
        dataset_id="ds",
        split=split,
    )


class TestGroundTruth:
    def test_arabic_preserved_exactly(self) -> None:
        record = build_ground_truth_record(page=_page(), raw_text=ARABIC_RAW)
        # Diacritics and alef variant must survive byte-for-byte.
        assert record.raw_text == ARABIC_RAW
        assert "\u0650" in record.raw_text  # kasra diacritic retained
        assert "القَاهِرَة" in record.raw_text  # alef variant retained

    def test_hash_stable(self) -> None:
        first = build_ground_truth_record(page=_page(), raw_text=ARABIC_RAW)
        second = build_ground_truth_record(page=_page(), raw_text=ARABIC_RAW)
        assert first.raw_text_sha256 == second.raw_text_sha256
        assert verify_ground_truth(first)

    def test_hash_detects_mutation(self) -> None:
        record = build_ground_truth_record(page=_page(), raw_text=ARABIC_RAW)
        mutated = GroundTruthRecord(
            page_id=record.page_id,
            raw_text=record.raw_text + "x",
            raw_text_sha256=record.raw_text_sha256,
        )
        assert not verify_ground_truth(mutated)

    def test_no_silent_normalization(self) -> None:
        record = build_ground_truth_record(page=_page(), raw_text=ARABIC_RAW)
        # Raw view is untouched; normalized view is separate and explicit.
        assert record.raw_text != normalized_view(record)
        assert normalized_view(record) == "كتاب قديم في مكتبة القاهرة"

    def test_round_trip_preserves_arabic(self) -> None:
        record = build_ground_truth_record(page=_page(), raw_text=ARABIC_RAW)
        restored = GroundTruthRecord.from_dict(record.to_dict())
        assert restored.raw_text == ARABIC_RAW
        assert restored.raw_text_sha256 == record.raw_text_sha256


class TestProtection:
    def test_holdout_not_training_eligible(self) -> None:
        page = _page(split="holdout")
        assert page.protection.is_training_eligible is False
        assert page.is_training_eligible is False

    def test_protection_survives_serialization(self) -> None:
        page = _page(split="protected_holdout")
        restored = PageRecord.from_dict(page.to_dict())
        assert restored.protection.protected is True
        assert restored.is_training_eligible is False

    def test_protection_fails_closed_on_unknown_split(self) -> None:
        protection = ProtectionInfo(split="mystery-split")
        assert protection.is_training_eligible is False

    @pytest.mark.parametrize(
        "split",
        ["holdout", "protected_holdout", "benchmark_holdout", "private_holdout"],
    )
    def test_all_protected_markers(self, split: str) -> None:
        assert ProtectionInfo(split=split).is_training_eligible is False

    def test_train_is_eligible(self) -> None:
        assert ProtectionInfo(split="train").is_training_eligible is True
        assert ProtectionInfo(split="validation").is_training_eligible is True
        assert ProtectionInfo(split="test").is_training_eligible is True

    def test_protected_requires_reason(self) -> None:
        with pytest.raises(ValueError):
            ProtectionInfo(protected=True)


class TestPredictions:
    def test_multiple_models_same_page(self) -> None:
        from clouda_data.results.identity import prediction_identity

        first = OCRPrediction(
            prediction_id=prediction_identity(run_id="r1", page_id="pg1"),
            run_id="r1",
            page_id="pg1",
            model_id="fake-model-a",
            model_revision="1.0",
            text="أ",
            text_sha256="0" * 64,
        )
        second = OCRPrediction(
            prediction_id=prediction_identity(run_id="r2", page_id="pg1"),
            run_id="r2",
            page_id="pg1",
            model_id="fake-model-b",
            model_revision="0.9",
            text="ب",
            text_sha256="1" * 64,
        )
        assert first.prediction_id != second.prediction_id
        assert first.page_id == second.page_id

    def test_round_trip(self) -> None:
        from clouda_data.results.identity import prediction_identity

        prediction = OCRPrediction(
            prediction_id=prediction_identity(run_id="r", page_id="p"),
            run_id="r",
            page_id="p",
            model_id="m",
            model_revision="rev",
            text=ARABIC_RAW,
            text_sha256="0" * 64,
            metrics={"cer": 0.5},
        )
        restored = OCRPrediction.from_dict(prediction.to_dict())
        assert restored.text == ARABIC_RAW
        assert restored.metrics == {"cer": 0.5}

    def test_rejects_non_numeric_metric(self) -> None:
        from clouda_data.results.identity import prediction_identity

        with pytest.raises(ValueError):
            OCRPrediction(
                prediction_id=prediction_identity(run_id="r", page_id="p"),
                run_id="r",
                page_id="p",
                model_id="m",
                model_revision="rev",
                text="t",
                text_sha256="0" * 64,
                metrics={"cer": "high"},
            )


def test_fixture_pages_holdout_marker() -> None:
    pages = build_fixture_pages()
    holdout = [page for page in pages if page.split == "holdout"]
    assert len(holdout) == 1
    assert holdout[0].is_training_eligible is False
