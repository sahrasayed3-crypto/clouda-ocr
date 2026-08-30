from __future__ import annotations

from typing import TYPE_CHECKING

from clouda_contracts.dataset_identity import DatasetIdentity
from clouda_contracts.model_observation import ModelObservation
from clouda_contracts.observations import EvaluationObservation
from clouda_contracts.ocr_observation import OCRObservation
from clouda_contracts.page_identity import PageIdentity
from clouda_contracts.references import DatasetPageReference
from clouda_contracts.statuses import ObservationStatus

if TYPE_CHECKING:
    from clouda_data.ingestion.schema import PageRecord
    from pdfword.engines import OCRResult
    from pdfword.models import PageResult


def page_identity_from_record(record: "PageRecord") -> PageIdentity:
    return PageIdentity(
        document_id=record.document_id,
        page_number=record.page_number,
        page_id=record.page_id,
        source_uri=record.source_path,
    )


def dataset_page_reference_from_record(
    record: "PageRecord",
    *,
    dataset_id: str,
    dataset_version: str = "1",
    split: str = "unspecified",
) -> DatasetPageReference:
    return DatasetPageReference(
        dataset=DatasetIdentity(
            dataset_id=dataset_id,
            version=dataset_version,
            split=split,
            record_id=record.page_id,
        ),
        page=page_identity_from_record(record),
        text_checksum=record.text_checksum,
        image_checksum=record.image_checksum,
    )


def ocr_observation_from_result(
    result: "OCRResult",
    *,
    page: PageIdentity,
) -> OCRObservation:
    status = {
        "succeeded": ObservationStatus.SUCCEEDED,
        "failed": ObservationStatus.FAILED,
        "pending_ocr_model": ObservationStatus.PENDING,
    }.get(result.status, ObservationStatus.FAILED)
    return OCRObservation(
        page=page,
        engine_name=result.engine_name,
        model_name=result.model_name,
        status=status,
        text=result.text,
        confidence=result.confidence,
        processing_seconds=result.processing_time,
        error_code=result.error_message,
    )


def ocr_observation_from_page_result(
    result: "PageResult",
    *,
    document_id: str,
) -> OCRObservation:
    page = PageIdentity(
        document_id=document_id,
        page_number=result.page_no,
        page_id=f"{document_id}:page:{result.page_no}",
    )
    status = (
        ObservationStatus.MANUAL_REVIEW
        if result.requires_manual_review
        else ObservationStatus.SUCCEEDED
    )
    return OCRObservation(
        page=page,
        engine_name=result.model_used,
        model_name=result.model_used,
        status=status,
        text=result.markdown,
        confidence=(
            result.final_score / 100
            if result.final_score is not None and result.final_score > 1
            else result.final_score
        ),
        processing_seconds=result.elapsed_time,
        error_code=result.review_reason,
    )


def model_observation_from_page_result(
    result: "PageResult",
    *,
    document_id: str,
) -> ModelObservation:
    ocr = ocr_observation_from_page_result(result, document_id=document_id)
    return ModelObservation(
        page=ocr.page,
        model_id=result.model_used,
        status=ocr.status,
        output_text=result.markdown,
        quality_score=ocr.confidence,
        elapsed_seconds=result.elapsed_time,
    )


def evaluation_observation_from_ground_truth(
    *,
    page: PageIdentity,
    policy_id: str,
    metrics: dict[str, float],
) -> EvaluationObservation:
    return EvaluationObservation(
        page=page,
        policy_id=policy_id,
        metrics=tuple(sorted(metrics.items())),
    )
