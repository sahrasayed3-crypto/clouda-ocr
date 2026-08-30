import io
from pathlib import Path

from pypdf import PdfWriter

from pdfword.engines import (
    EngineRegistry,
    FeatureFlaggedLocalModelEngine,
    OCRBox,
    OCRResult,
    OCR_STATUS_FAILED,
    OCR_STATUS_PENDING_MODEL,
    OCR_STATUS_SUCCEEDED,
    get_engine_registry,
)
from pdfword.ocr_pipeline import process_pdf
from pdfword.settings import validate_setting


def _blank_pdf(page_count: int = 1) -> bytes:
    out = io.BytesIO()
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)
    writer.write(out)
    return out.getvalue()


class MockOcrEngine:
    name = "test_mock_ocr_engine"
    engine_type = "digital_text"
    model_name: str | None = "test-only-model"

    def __init__(self, result: OCRResult | None = None) -> None:
        self.result = result

    def available(self) -> bool:
        return True

    def extract_page(self, **_kwargs) -> OCRResult:
        return self.result or OCRResult(
            engine_name=self.name,
            model_name=self.model_name,
            status=OCR_STATUS_SUCCEEDED,
            text="mock text",
            processing_time=0.01,
            confidence=0.91,
        )


class MockLocalProvider:
    def __init__(self, result: OCRResult) -> None:
        self.result = result
        self.calls = 0

    def available(self) -> bool:
        return True

    def extract_page(self, **_kwargs) -> OCRResult:
        self.calls += 1
        return self.result


def test_engine_registry_can_register_new_engine() -> None:
    registry = EngineRegistry()
    engine = MockOcrEngine()

    registry.register(engine)

    assert registry.get(engine.name) is engine
    assert registry.available_names() == [engine.name]


def test_engine_can_be_selected_from_settings_without_router_change() -> None:
    registry = get_engine_registry()
    engine = MockOcrEngine()
    registry.register(engine, replace=True)

    validate_setting("enabled_engines", [engine.name])
    rows, text = process_pdf(
        pdf_bytes=_blank_pdf(),
        from_page=1,
        to_page=1,
        progress_bar=None,
        status_placeholder=None,
        enabled_engines=[engine.name],
    )

    assert text == "mock text"
    assert rows[0].route_used == engine.name
    assert rows[0].model_used == f"local:{engine.name}"


def test_ocr_result_schema_has_optional_layout_and_confidence() -> None:
    result = OCRResult(
        engine_name="schema_engine",
        model_name="optional-model-name",
        status=OCR_STATUS_SUCCEEDED,
        text="A B",
        confidence=0.87,
        processing_time=1.25,
        boxes=(
            OCRBox(
                text="A", bbox=(0.0, 0.0, 10.0, 10.0), confidence=0.9, reading_order=1
            ),
            OCRBox(text="B", bbox=(12.0, 0.0, 20.0, 10.0), reading_order=2),
        ),
        reading_order=(1, 2),
    )

    payload = result.as_dict()

    assert payload["status"] == OCR_STATUS_SUCCEEDED
    assert payload["confidence"] == 0.87
    assert payload["boxes"][0]["bbox"] == (0.0, 0.0, 10.0, 10.0)
    assert payload["reading_order"] == (1, 2)


def test_engine_error_is_reported_as_pending_model_page() -> None:
    registry = get_engine_registry()
    engine = MockOcrEngine(
        OCRResult(
            engine_name="test_mock_error_engine",
            model_name="test-only-model",
            status=OCR_STATUS_FAILED,
            text="",
            error_message="engine unavailable",
            processing_time=0.02,
        )
    )
    engine.name = "test_mock_error_engine"
    registry.register(engine, replace=True)

    rows, text = process_pdf(
        pdf_bytes=_blank_pdf(),
        from_page=1,
        to_page=1,
        progress_bar=None,
        status_placeholder=None,
        enabled_engines=[engine.name],
    )

    assert rows[0].route_used == OCR_STATUS_PENDING_MODEL
    assert rows[0].review_reason == OCR_STATUS_PENDING_MODEL
    assert "engine unavailable" in text


def test_configuration_is_cpu_gpu_neutral() -> None:
    result = OCRResult(engine_name="neutral", status=OCR_STATUS_PENDING_MODEL)
    payload = result.as_dict()

    assert "cuda" not in str(payload).lower()
    assert "rocm" not in str(payload).lower()
    assert "device" not in payload


def test_local_model_boundary_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("CLOUDA_LOCAL_OCR_ENABLED", raising=False)
    engine = FeatureFlaggedLocalModelEngine()
    result = engine.extract_page(image_bytes=b"image", page_no=1)
    assert engine.available() is False
    assert result.status == OCR_STATUS_PENDING_MODEL


def test_local_model_requires_text_quality_and_pinned_revision() -> None:
    provider = MockLocalProvider(
        OCRResult(
            engine_name="provider",
            status=OCR_STATUS_SUCCEEDED,
            text="نص صالح",
            confidence=0.9,
        )
    )
    engine = FeatureFlaggedLocalModelEngine(
        provider,
        enabled=True,
        model_name="test-model",
        model_revision="abc123",
    )
    result = engine.extract_page(image_bytes=b"image", page_no=1)
    assert result.status == OCR_STATUS_SUCCEEDED
    assert result.metadata["model_revision"] == "abc123"


def test_local_model_rejects_success_without_quality_metadata() -> None:
    provider = MockLocalProvider(
        OCRResult(
            engine_name="provider",
            status=OCR_STATUS_SUCCEEDED,
            text="نص بلا ثقة",
        )
    )
    engine = FeatureFlaggedLocalModelEngine(
        provider,
        enabled=True,
        model_revision="abc123",
        retry_count=1,
    )
    result = engine.extract_page(image_bytes=b"image", page_no=1)
    assert result.status == OCR_STATUS_FAILED
    assert provider.calls == 2


def test_feature_flagged_local_model_can_process_scanned_fixture() -> None:
    provider = MockLocalProvider(
        OCRResult(
            engine_name="provider",
            status=OCR_STATUS_SUCCEEDED,
            text="نص عربي مستخرج من صورة ممسوحة ضوئياً",
            confidence=0.97,
        )
    )
    engine = FeatureFlaggedLocalModelEngine(
        provider,
        enabled=True,
        model_name="test-model",
        model_revision="abc123",
    )
    registry = get_engine_registry()
    original = registry.get(engine.name)
    registry.register(engine, replace=True)
    pdf_bytes = (Path(__file__).parent / "fixtures" / "scanned.pdf").read_bytes()
    try:
        rows, text = process_pdf(
            pdf_bytes=pdf_bytes,
            from_page=1,
            to_page=1,
            progress_bar=None,
            status_placeholder=None,
            enabled_engines=["direct_pdf_text", engine.name],
        )
    finally:
        registry.register(original, replace=True)
    assert text
    assert rows[0].route_used == engine.name
    assert rows[0].metadata is not None
    assert rows[0].metadata["model_revision"] == "abc123"


def test_blank_page_is_reported_without_requesting_an_ocr_model() -> None:
    rows, text = process_pdf(
        pdf_bytes=_blank_pdf(),
        from_page=1,
        to_page=1,
        progress_bar=None,
        status_placeholder=None,
    )

    assert rows[0].route_used == "blank_page"
    assert rows[0].accepted is False
    assert text == ""
