from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from clouda_contracts.adapters import (
    dataset_page_reference_from_record,
    model_observation_from_page_result,
    ocr_observation_from_page_result,
    ocr_observation_from_result,
    page_identity_from_record,
)
from clouda_contracts.checksums import Checksum
from clouda_contracts.ocr_observation import OCRObservation
from clouda_contracts.page_identity import PageIdentity
from clouda_contracts.statuses import ObservationStatus
from clouda_data.ingestion.schema import PageRecord
from pdfword.engines import OCRResult
from pdfword.models import PageResult


def test_page_identity_is_immutable_and_backward_compatible() -> None:
    page = PageIdentity.from_dict(
        {"document": "doc", "page_no": 2, "id": "page-2", "source_path": "x"}
    )
    assert PageIdentity.from_dict(page.to_dict()) == page
    with pytest.raises(FrozenInstanceError):
        page.page_id = "changed"  # type: ignore[misc]


def test_checksum_validation() -> None:
    checksum = Checksum.from_dict("a" * 64)
    assert checksum.to_dict()["algorithm"] == "sha256"
    with pytest.raises(ValueError):
        Checksum.from_dict("not-a-checksum")


def test_page_record_and_engine_adapters() -> None:
    record = PageRecord(
        document_id="doc",
        page_id="page-1",
        source_path="dataset://data/raw/page.png",
        source_type="image",
        language="ar",
        page_number=1,
        clean_text="نص",
        text_checksum="a" * 64,
    )
    page = page_identity_from_record(record)
    reference = dataset_page_reference_from_record(record, dataset_id="synthetic")
    assert reference.page == page
    result = OCRResult(
        engine_name="test",
        status="pending_ocr_model",
        processing_time=0.25,
    )
    observation = ocr_observation_from_result(result, page=page)
    assert observation.status is ObservationStatus.PENDING
    assert OCRObservation.from_dict(observation.to_dict()) == observation


def test_page_result_adapter_preserves_quality() -> None:
    result = PageResult(
        page_no=1,
        model_used="model",
        markdown="text",
        text_quality_score=0.8,
    )
    observation = model_observation_from_page_result(result, document_id="doc")
    ocr_observation = ocr_observation_from_page_result(result, document_id="doc")
    assert observation.quality_score == 0.8
    assert ocr_observation.confidence == 0.8
    assert observation.to_dict()["page"]["page_number"] == 1


def test_contract_validation_rejects_invalid_ranges() -> None:
    page = PageIdentity("doc", 1, "page")
    with pytest.raises(ValueError, match="Confidence"):
        OCRObservation(
            page=page,
            engine_name="engine",
            status=ObservationStatus.SUCCEEDED,
            confidence=1.1,
        )
