from __future__ import annotations

import pytest

from pdfword.engines import (
    DirectPdfTextEngine,
    EngineRegistry,
    FeatureFlaggedLocalModelEngine,
    FutureOcrEngine,
    OCRResult,
    OCR_STATUS_FAILED,
    OCR_STATUS_PENDING_MODEL,
    OCR_STATUS_SUCCEEDED,
    available_engine_status,
)


class RaisingProvider:
    def available(self) -> bool:
        return True

    def extract_page(self, **_kwargs) -> OCRResult:
        raise RuntimeError("provider failed")


def test_registry_validation_and_status_reporting() -> None:
    registry = EngineRegistry()
    unnamed = type("Unnamed", (), {"name": ""})()
    with pytest.raises(ValueError):
        registry.register(unnamed)  # type: ignore[arg-type]
    engine = DirectPdfTextEngine()
    registry.register(engine)
    with pytest.raises(ValueError):
        registry.register(engine)
    with pytest.raises(KeyError):
        registry.get("missing")
    assert available_engine_status(registry)[0]["status"] == "ready"
    assert registry.all() == [engine]


def test_direct_and_future_engine_failure_contracts() -> None:
    direct = DirectPdfTextEngine()
    missing = direct.extract_page()
    invalid = direct.extract_page(pdf_bytes=b"not-a-pdf", page_no=1)
    pending = FutureOcrEngine().extract_page()
    assert missing.status == OCR_STATUS_FAILED
    assert invalid.status == OCR_STATUS_FAILED
    assert pending.status == OCR_STATUS_PENDING_MODEL
    assert pending.requires_future_ocr is True


def test_local_engine_limits_and_provider_errors() -> None:
    engine = FeatureFlaggedLocalModelEngine(
        RaisingProvider(),
        enabled=True,
        model_revision="pinned",
        max_image_bytes=4,
        retry_count=1,
    )
    missing = engine.extract_page(page_no=1)
    oversized = engine.extract_page(image_bytes=b"12345", page_no=1)
    failed = engine.extract_page(image_bytes=b"1234", page_no=1)
    assert missing.status == OCR_STATUS_FAILED
    assert oversized.status == OCR_STATUS_FAILED
    assert failed.status == OCR_STATUS_FAILED
    assert "RuntimeError" in (failed.error_message or "")


def test_unresolved_local_engine_is_unavailable() -> None:
    engine = FeatureFlaggedLocalModelEngine(
        RaisingProvider(),
        enabled=True,
        model_revision="latest",
    )
    assert engine.available() is False
    result = engine.extract_page(image_bytes=b"x", page_no=1)
    assert result.status == OCR_STATUS_PENDING_MODEL


def test_ocr_result_properties() -> None:
    success = OCRResult(
        engine_name="engine",
        status=OCR_STATUS_SUCCEEDED,
        text="text",
    )
    failed = OCRResult(
        engine_name="engine",
        status=OCR_STATUS_FAILED,
        error_message="failure",
    )
    assert success.success is True
    assert failed.failure_reason == "failure"
