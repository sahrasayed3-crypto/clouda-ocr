from __future__ import annotations

import base64
import io
import os
import re
import tempfile
import threading

import pypdfium2 as pdfium
from pypdf import PdfReader

from .accuracy import estimate_quality_components, final_acceptance_decision
from .engines import (
    DIRECT_TEXT_ENGINE,
    FUTURE_OCR_ENGINE,
    OCR_STATUS_FAILED,
    OCR_STATUS_PENDING_MODEL,
    get_engine_registry,
)
from .job_queue import JobCancelled
from .models import PageResult

TARGET_QUALITY_SCORE = 97.0
MIN_ACCEPT_QUALITY_SCORE = 90.0
_PDFIUM_RENDER_LOCK = threading.Lock()
BLANK_PAGE_ROUTE = "blank_page"
NEAR_BLANK_PAGE_ROUTE = "near_blank"
_NEAR_BLANK_TEXT_LIMIT = 12
_SMALL_IMAGE_DENSITY_LIMIT = 0.10


def encode_image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def render_pdf_page_to_png_bytes(
    pdf_bytes: bytes, page_no: int, dpi: int = 220
) -> bytes:
    """Render a page for diagnostics or future engines; the active pipeline does not OCR it."""
    tmp_path = None
    page = None
    bitmap = None
    doc = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        with _PDFIUM_RENDER_LOCK:
            doc = pdfium.PdfDocument(tmp_path)
            page = doc.get_page(page_no - 1)
            bitmap = page.render(scale=max(72, int(dpi)) / 72.0)
            pil_image = bitmap.to_pil()
        out = io.BytesIO()
        pil_image.save(out, format="PNG")
        return out.getvalue()
    finally:
        if bitmap is not None and hasattr(bitmap, "close"):
            bitmap.close()
        if page is not None:
            page.close()
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _clean_markdown_output(text: str) -> str:
    cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.replace("\u200f", "").replace("\u200e", "")
    lines: list[str] = []
    previous = None
    for line in cleaned.splitlines():
        normalized = re.sub(r"\s+", " ", line).strip()
        if normalized and normalized == previous:
            continue
        lines.append(line.rstrip())
        if normalized:
            previous = normalized
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _requires_manual_review(
    score: float | None, threshold: float = MIN_ACCEPT_QUALITY_SCORE
) -> bool:
    if score is None:
        return True
    try:
        return float(score) < float(threshold)
    except (TypeError, ValueError):
        return True


def _page_visual_metadata(page) -> dict:
    """Return conservative visual signals without rendering or running OCR."""
    image_sizes: list[tuple[int, int]] = []
    try:
        for image in page.images:
            size = getattr(getattr(image, "image", None), "size", None)
            if size and len(size) == 2:
                image_sizes.append((int(size[0]), int(size[1])))
    except Exception:
        # A malformed image resource must not turn a page into a false blank.
        return {"embedded_image_count": None, "visual_signal_unavailable": True}

    page_area = max(1.0, float(page.mediabox.width) * float(page.mediabox.height))
    largest_image_pixels = max(
        (width * height for width, height in image_sizes), default=0
    )
    return {
        "embedded_image_count": len(image_sizes),
        "largest_image_pixels": largest_image_pixels,
        "largest_image_density_estimate": largest_image_pixels / page_area,
        "visual_signal_unavailable": False,
    }


def _classify_page_without_digital_text(page, text: str) -> tuple[str, dict]:
    """Classify a page safely without inferring text from an OCR model."""
    normalized = (text or "").strip()
    visual = _page_visual_metadata(page)
    metadata = {"embedded_text_chars": len(normalized), **visual}

    if normalized and len(normalized) <= _NEAR_BLANK_TEXT_LIMIT:
        return NEAR_BLANK_PAGE_ROUTE, metadata

    image_count = visual.get("embedded_image_count")
    density = visual.get("largest_image_density_estimate", 0.0)
    if image_count is None:
        return OCR_STATUS_PENDING_MODEL, metadata
    if image_count == 0 and not normalized:
        return BLANK_PAGE_ROUTE, metadata
    if image_count > 0 and density <= _SMALL_IMAGE_DENSITY_LIMIT:
        return NEAR_BLANK_PAGE_ROUTE, metadata
    return OCR_STATUS_PENDING_MODEL, metadata


def _classified_non_text_page(
    page_no: int,
    route: str,
    metadata: dict,
    reason: str | None = None,
    engines_attempted: tuple[str, ...] | None = None,
    text: str = "",
) -> PageResult:
    """Build a non-text result while preserving its explicit routing reason."""
    if route == OCR_STATUS_PENDING_MODEL:
        return _future_ocr_page(page_no, reason, engines_attempted, metadata=metadata)

    return PageResult(
        page_no=page_no,
        model_used=f"system:{route}",
        markdown=text,
        quality_score=None,
        text_quality_score=None,
        layout_quality_score=None,
        direction_quality_score=None,
        completeness_score=None,
        requires_manual_review=route == NEAR_BLANK_PAGE_ROUTE,
        review_reason=route,
        engines_attempted=engines_attempted or (DIRECT_TEXT_ENGINE.name,),
        route_used=route,
        accepted=False,
        attempts_count=1,
        selection_reason=route,
        metadata={"page_state": route, **metadata},
    )


def _future_ocr_page(
    page_no: int,
    reason: str | None = None,
    engines_attempted: tuple[str, ...] | None = None,
    metadata: dict | None = None,
) -> PageResult:
    message = (
        "## Requires future OCR model\n\n"
        "This page does not contain an extractable digital text layer. "
        "Scanned-page OCR is intentionally disabled because no approved trainable OCR model "
        "has been selected for this project yet. The generic engine interface is ready for "
        "a future AMD-compatible OCR model after validation."
    )
    if reason:
        message += f"\n\nReason: {reason}"
    quality_parts = estimate_quality_components("", base_text_score=0.0)
    return PageResult(
        page_no=page_no,
        model_used=f"pending:{FUTURE_OCR_ENGINE.name}",
        markdown=message,
        quality_score=quality_parts["final_quality"],
        text_quality_score=0.0,
        layout_quality_score=quality_parts["layout_quality"],
        direction_quality_score=quality_parts["direction_quality"],
        completeness_score=0.0,
        requires_manual_review=True,
        review_reason=OCR_STATUS_PENDING_MODEL,
        engines_attempted=engines_attempted
        or (DIRECT_TEXT_ENGINE.name, FUTURE_OCR_ENGINE.name),
        route_used=OCR_STATUS_PENDING_MODEL,
        accepted=False,
        attempts_count=1,
        selection_reason="scanned_or_image_only_page_pending_model_selection",
        metadata={"page_state": OCR_STATUS_PENDING_MODEL, **(metadata or {})},
    )


def process_pdf(
    pdf_bytes: bytes,
    from_page: int | None,
    to_page: int | None,
    api_key: str = "",
    fast_model: str = "",
    accurate_model: str = "",
    progress_bar=None,
    status_placeholder=None,
    speed_mode: str = "direct",
    page_numbers: list[int] | None = None,
    max_parallel_pages: int = 1,
    doc_category: str | None = None,
    cancellation_check=None,
    checkpoint_callback=None,
    existing_results: dict[int, PageResult] | None = None,
    acceptance_threshold: float = MIN_ACCEPT_QUALITY_SCORE,
    max_cloud_attempts: int | None = None,
    scan_dpi: int = 300,
    enabled_engines: list[str] | None = None,
    enabled_models: list[str] | None = None,
    batch_size: int = 10,
    max_dpi: int = 400,
    cloud_attempt_allowed=None,
    cloud_attempt_callback=None,
) -> tuple[list[PageResult], str]:
    del api_key, fast_model, accurate_model, speed_mode, max_parallel_pages
    del doc_category, max_cloud_attempts, scan_dpi, enabled_models
    del batch_size, max_dpi, cloud_attempt_allowed, cloud_attempt_callback

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pdf_page_count = len(reader.pages)
    if page_numbers is not None:
        pages = list(page_numbers)
    else:
        if from_page is None or to_page is None:
            raise ValueError("from_page/to_page or page_numbers are required")
        pages = list(range(from_page, to_page + 1))
    if not pages:
        raise ValueError("Page selection cannot be empty")
    if any(not isinstance(page, int) or isinstance(page, bool) for page in pages):
        raise ValueError("Page numbers must be integers")
    if any(page < 1 or page > pdf_page_count for page in pages):
        raise ValueError(f"Page selection must be between 1 and {pdf_page_count}")

    acceptance_threshold = max(MIN_ACCEPT_QUALITY_SCORE, float(acceptance_threshold))
    results_by_page: dict[int, PageResult] = dict(existing_results or {})
    total_pages = len(pages)
    completed = len(results_by_page)

    def _cancelled() -> bool:
        return bool(cancellation_check and cancellation_check())

    for page_no in pages:
        if page_no in results_by_page:
            continue
        if _cancelled():
            raise JobCancelled("Conversion was cancelled by the user")
        if status_placeholder is not None:
            status_placeholder.info(f"Extracting digital text from page {page_no}...")

        registry = get_engine_registry()
        configured_engines = list(
            enabled_engines or [DIRECT_TEXT_ENGINE.name, FUTURE_OCR_ENGINE.name]
        )
        active_engines = [
            registry.get(name)
            for name in configured_engines
            if name in registry.names()
        ]
        direct_engines = [
            engine
            for engine in active_engines
            if engine.engine_type == DIRECT_TEXT_ENGINE.engine_type
        ]
        direct_engine = direct_engines[0] if direct_engines else DIRECT_TEXT_ENGINE
        local_model_engine = next(
            (
                engine
                for engine in active_engines
                if engine.engine_type == "local_model" and engine.available()
            ),
            None,
        )
        extraction = direct_engine.extract_page(pdf_bytes=pdf_bytes, page_no=page_no)
        text = _clean_markdown_output(extraction.text)
        if extraction.status == OCR_STATUS_FAILED:
            attempted = tuple(engine.name for engine in active_engines) or (
                direct_engine.name,
                FUTURE_OCR_ENGINE.name,
            )
            route = OCR_STATUS_PENDING_MODEL
            metadata = {
                "page_state": route,
                "embedded_text_chars": len((extraction.text or "").strip()),
                "engine_status": extraction.status,
                "engine_error": extraction.failure_reason,
            }
            results_by_page[page_no] = _classified_non_text_page(
                page_no,
                route,
                metadata,
                extraction.failure_reason,
                attempted,
            )
        elif direct_engine.name == DIRECT_TEXT_ENGINE.name and not text:
            attempted = tuple(engine.name for engine in active_engines) or (
                direct_engine.name,
                FUTURE_OCR_ENGINE.name,
            )
            route, metadata = _classify_page_without_digital_text(
                reader.pages[page_no - 1], extraction.text or ""
            )
            model_extraction = None
            if route == OCR_STATUS_PENDING_MODEL and local_model_engine is not None:
                try:
                    image_bytes = render_pdf_page_to_png_bytes(pdf_bytes, page_no)
                    model_extraction = local_model_engine.extract_page(
                        image_bytes=image_bytes,
                        page_no=page_no,
                    )
                except Exception as exc:
                    metadata["local_model_error"] = f"{type(exc).__name__}: {exc}"
            model_text = _clean_markdown_output(
                model_extraction.text if model_extraction is not None else ""
            )
            valid_model_result = bool(
                model_extraction is not None
                and model_extraction.success
                and model_text
                and model_extraction.confidence is not None
            )
            if valid_model_result and model_extraction is not None:
                assert model_extraction.confidence is not None
                quality_parts = estimate_quality_components(
                    model_text,
                    base_text_score=float(model_extraction.confidence) * 100,
                )
                decision = final_acceptance_decision(
                    model_text,
                    estimated_text_quality=quality_parts["text_quality"],
                    threshold=acceptance_threshold,
                    expected_non_empty=True,
                )
                results_by_page[page_no] = PageResult(
                    page_no=page_no,
                    model_used=model_extraction.model_name
                    or f"local:{model_extraction.engine_name}",
                    markdown=model_text,
                    quality_score=quality_parts["final_quality"],
                    text_quality_score=decision["estimated_text_quality"],
                    layout_quality_score=quality_parts["layout_quality"],
                    direction_quality_score=quality_parts["direction_quality"],
                    completeness_score=quality_parts["completeness"],
                    requires_manual_review=bool(decision["requires_manual_review"]),
                    review_reason=decision["review_reason"],
                    engines_attempted=attempted,
                    route_used=model_extraction.engine_name,
                    accepted=bool(decision["accepted"]),
                    attempts_count=1,
                    elapsed_time=model_extraction.processing_time,
                    corruption_diagnostics=decision["diagnostics"],
                    selection_reason="feature_flagged_local_model",
                    metadata={
                        **metadata,
                        **model_extraction.metadata,
                        "engine_status": model_extraction.status,
                        "page_state": "local_model_ocr",
                    },
                )
            else:
                reason = (
                    model_extraction.failure_reason
                    if model_extraction is not None
                    else extraction.failure_reason
                )
                results_by_page[page_no] = _classified_non_text_page(
                    page_no,
                    route,
                    metadata,
                    reason,
                    attempted,
                )
        elif direct_engine.name == DIRECT_TEXT_ENGINE.name:
            route, metadata = _classify_page_without_digital_text(
                reader.pages[page_no - 1], extraction.text or ""
            )
            if route == NEAR_BLANK_PAGE_ROUTE:
                attempted = tuple(engine.name for engine in active_engines) or (
                    direct_engine.name,
                )
                results_by_page[page_no] = _classified_non_text_page(
                    page_no,
                    route,
                    metadata,
                    "Only a short embedded text fragment was found; source content was preserved.",
                    attempted,
                    text,
                )
                completed += 1
                if checkpoint_callback:
                    checkpoint_callback(dict(results_by_page))
                if progress_bar is not None:
                    progress_bar.progress(
                        completed / total_pages,
                        text=f"Processed {completed} of {total_pages} pages",
                    )
                continue
            quality_parts = estimate_quality_components(text, base_text_score=100.0)
            decision = final_acceptance_decision(
                text,
                estimated_text_quality=quality_parts["text_quality"],
                threshold=acceptance_threshold,
                expected_non_empty=True,
            )
            results_by_page[page_no] = PageResult(
                page_no=page_no,
                model_used=f"local:{extraction.engine_name}",
                markdown=text,
                quality_score=quality_parts["final_quality"],
                text_quality_score=decision["estimated_text_quality"],
                layout_quality_score=quality_parts["layout_quality"],
                direction_quality_score=quality_parts["direction_quality"],
                completeness_score=quality_parts["completeness"],
                requires_manual_review=bool(decision["requires_manual_review"]),
                review_reason=decision["review_reason"],
                engines_attempted=(extraction.engine_name,),
                route_used=extraction.engine_name,
                accepted=bool(decision["accepted"]),
                attempts_count=1,
                elapsed_time=extraction.processing_time,
                corruption_diagnostics=decision["diagnostics"],
                selection_reason="digital_text_layer_extracted",
                metadata={
                    "page_state": "digital_text",
                    "embedded_text_chars": len(text),
                    "engine_status": extraction.status,
                },
            )
        elif not text:
            attempted = tuple(engine.name for engine in active_engines) or (
                direct_engine.name,
                FUTURE_OCR_ENGINE.name,
            )
            results_by_page[page_no] = _future_ocr_page(
                page_no,
                extraction.failure_reason,
                attempted,
                metadata={
                    "engine_status": extraction.status,
                    "embedded_text_chars": 0,
                    "page_state": OCR_STATUS_PENDING_MODEL,
                },
            )
        else:
            quality_parts = estimate_quality_components(text, base_text_score=100.0)
            decision = final_acceptance_decision(
                text,
                estimated_text_quality=quality_parts["text_quality"],
                threshold=acceptance_threshold,
                expected_non_empty=True,
            )
            results_by_page[page_no] = PageResult(
                page_no=page_no,
                model_used=f"local:{extraction.engine_name}",
                markdown=text,
                quality_score=quality_parts["final_quality"],
                text_quality_score=decision["estimated_text_quality"],
                layout_quality_score=quality_parts["layout_quality"],
                direction_quality_score=quality_parts["direction_quality"],
                completeness_score=quality_parts["completeness"],
                requires_manual_review=bool(decision["requires_manual_review"]),
                review_reason=decision["review_reason"],
                engines_attempted=(extraction.engine_name,),
                route_used=extraction.engine_name,
                accepted=bool(decision["accepted"]),
                attempts_count=1,
                elapsed_time=extraction.processing_time,
                corruption_diagnostics=decision["diagnostics"],
                selection_reason="engine_text_extracted",
                metadata={
                    "page_state": "digital_text",
                    "embedded_text_chars": len(text),
                    "engine_status": extraction.status,
                },
            )
        completed += 1
        if checkpoint_callback:
            checkpoint_callback(dict(results_by_page))
        if progress_bar is not None:
            progress_bar.progress(
                completed / total_pages,
                text=f"Processed {completed} of {total_pages} pages",
            )

    ordered_results = [results_by_page[p] for p in pages]
    full_markdown = "\n\n".join(result.markdown for result in ordered_results)
    if status_placeholder is not None:
        pending = sum(
            1
            for result in ordered_results
            if result.route_used == OCR_STATUS_PENDING_MODEL
        )
        if pending:
            status_placeholder.warning(
                f"Completed digital extraction; {pending} page(s) require a future OCR model."
            )
        else:
            status_placeholder.success("Completed direct PDF text extraction.")
    return ordered_results, full_markdown
