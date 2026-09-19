from __future__ import annotations

import base64
import io
import os
import re
import tempfile
import threading

import pypdfium2 as pdfium
from pypdf import PdfReader

from .engines import (
    DIRECT_TEXT_ENGINE,
    FUTURE_OCR_ENGINE,
    OCR_STATUS_PENDING_MODEL,
    get_engine_registry,
)
from .ocr_self_review import (
    OCRIssueCode,
    OCRPageState,
    ReReadResult,
    crop_review_region,
    make_rendered_page_context,
    reconcile_ocr_results,
    review_first_pass,
)
from .job_queue import JobCancelled
from .models import PageResult
from .page_routing import (
    DocumentRoutingContext,
    PageAnalysis,
    PageDecisionResult,
    PageNextPath,
    analyze_pdf_page,
    decide_page_route,
    evaluate_digital_text_trust,
    validate_trusted_context_before_extraction,
    validate_trusted_text_after_extraction,
)

_PDFIUM_RENDER_LOCK = threading.Lock()
TARGET_QUALITY_SCORE = 97.0  # legacy compatibility only; not OCR acceptance policy
MIN_ACCEPT_QUALITY_SCORE = 90.0  # legacy compatibility only; not OCR acceptance policy
BLANK_PAGE_ROUTE = "blank_page"
NEAR_BLANK_PAGE_ROUTE = "near_blank"


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


def _routing_metadata(
    analysis: PageAnalysis,
    decision: PageDecisionResult,
    gate_diagnostics: dict[str, object],
) -> dict[str, object]:
    return {
        "page_state": decision.next_path.value,
        "embedded_text_chars": analysis.normalized_character_count,
        "embedded_image_count": analysis.image_count,
        "document_intelligence": {
            **decision.to_diagnostics(),
            "gate": gate_diagnostics,
            "analysis": analysis.to_diagnostics(),
        },
    }


def _blank_or_near_blank_page(
    analysis: PageAnalysis,
    decision: PageDecisionResult,
    metadata: dict[str, object],
    engines_attempted: tuple[str, ...],
) -> PageResult:
    near_blank = analysis.near_blank_evidence
    route = NEAR_BLANK_PAGE_ROUTE if near_blank else BLANK_PAGE_ROUTE
    text = _clean_markdown_output(analysis.embedded_text) if near_blank else ""
    return PageResult(
        page_no=analysis.page_number,
        model_used=f"system:{route}",
        markdown=text,
        quality_score=None,
        text_quality_score=None,
        layout_quality_score=None,
        direction_quality_score=None,
        completeness_score=None,
        requires_manual_review=near_blank,
        review_reason=route,
        engines_attempted=engines_attempted,
        route_used=route,
        accepted=False,
        attempts_count=1,
        selection_reason=(
            "near_blank_structural_evidence" if near_blank else "blank_content_stream"
        ),
        metadata={**metadata, "page_state": route},
    )


def _review_page(
    analysis: PageAnalysis,
    decision: PageDecisionResult,
    metadata: dict[str, object],
    engines_attempted: tuple[str, ...],
    *,
    reason: str | None = None,
) -> PageResult:
    reason_codes = (reason,) if reason else decision.reason_codes or ("gate_uncertain",)
    placeholder = (
        f"[PAGE {analysis.page_number} REQUIRES REVIEW - embedded text was not "
        f"used; reasons: {', '.join(reason_codes)}]"
    )
    return PageResult(
        page_no=analysis.page_number,
        model_used="system:review_required",
        markdown=placeholder,
        quality_score=None,
        text_quality_score=None,
        layout_quality_score=None,
        direction_quality_score=None,
        completeness_score=None,
        requires_manual_review=True,
        review_reason=reason or "review_required",
        engines_attempted=engines_attempted,
        route_used="review_required",
        accepted=False,
        attempts_count=1,
        selection_reason="document_intelligence_review_required",
        metadata={**metadata, "page_state": "review_required"},
    )


def _future_ocr_page(
    page_no: int,
    reason: str | None = None,
    engines_attempted: tuple[str, ...] | None = None,
    metadata: dict | None = None,
) -> PageResult:
    message = (
        "## Requires future OCR model\n\n"
        "This page requires visual OCR because its embedded text is absent or "
        "not trusted. "
        "Scanned-page OCR is intentionally disabled because no approved trainable OCR model "
        "has been selected for this project yet. The generic engine interface is ready for "
        "a future AMD-compatible OCR model after validation."
    )
    if reason:
        message += f"\n\nReason: {reason}"
    return PageResult(
        page_no=page_no,
        model_used=f"pending:{FUTURE_OCR_ENGINE.name}",
        markdown=message,
        quality_score=None,
        text_quality_score=None,
        layout_quality_score=None,
        direction_quality_score=None,
        completeness_score=None,
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
    acceptance_threshold: float = 90.0,
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

    document_context = DocumentRoutingContext.from_pdf(pdf_bytes)
    reader = PdfReader(io.BytesIO(document_context.pdf_bytes))
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

    del acceptance_threshold
    results_by_page: dict[int, PageResult] = dict(existing_results or {})
    total_pages = len(pages)
    completed = len(results_by_page)
    registry = get_engine_registry()
    configured_engines = list(
        enabled_engines or [DIRECT_TEXT_ENGINE.name, FUTURE_OCR_ENGINE.name]
    )
    active_engines = [
        registry.get(name) for name in configured_engines if name in registry.names()
    ]
    direct_engine = next(
        (engine for engine in active_engines if engine.name == DIRECT_TEXT_ENGINE.name),
        DIRECT_TEXT_ENGINE,
    )
    ocr_engine = next(
        (
            engine
            for engine in active_engines
            if engine.name not in {DIRECT_TEXT_ENGINE.name, FUTURE_OCR_ENGINE.name}
            and engine.available()
        ),
        None,
    )
    attempted = tuple(engine.name for engine in active_engines) or (
        DIRECT_TEXT_ENGINE.name,
        FUTURE_OCR_ENGINE.name,
    )

    def _cancelled() -> bool:
        return bool(cancellation_check and cancellation_check())

    for page_no in pages:
        if page_no in results_by_page:
            continue
        if _cancelled():
            raise JobCancelled("Conversion was cancelled by the user")
        if status_placeholder is not None:
            status_placeholder.info(f"Analyzing page {page_no}...")

        analysis = analyze_pdf_page(
            reader.pages[page_no - 1], page_no, document_context
        )
        gate = evaluate_digital_text_trust(analysis)
        page_decision = decide_page_route(
            analysis,
            gate,
            ocr_available=ocr_engine is not None,
        )
        metadata = _routing_metadata(analysis, page_decision, gate.to_diagnostics())

        if page_decision.next_path is PageNextPath.NO_EXTRACTION:
            page_result = _blank_or_near_blank_page(
                analysis, page_decision, metadata, attempted
            )
        elif page_decision.next_path is PageNextPath.MANUAL_REVIEW:
            page_result = _review_page(analysis, page_decision, metadata, attempted)
        elif page_decision.next_path is PageNextPath.PENDING_OCR_MODEL:
            page_result = _future_ocr_page(
                page_no,
                "Embedded text did not pass the trusted digital text gate.",
                attempted,
                metadata=metadata,
            )
        elif page_decision.next_path is PageNextPath.DIRECT_PDF_TEXT:
            trusted_context = page_decision.trusted_context
            context_valid = bool(
                trusted_context
                and validate_trusted_context_before_extraction(
                    trusted_context,
                    document_sha256=document_context.pdf_sha256,
                    page_number=page_no,
                    gate=gate,
                )
            )
            extraction = (
                direct_engine.extract_page(
                    pdf_bytes=document_context.pdf_bytes,
                    page_no=page_no,
                )
                if context_valid
                else None
            )
            extraction_valid = bool(
                extraction is not None
                and extraction.success
                and trusted_context is not None
                and validate_trusted_text_after_extraction(
                    trusted_context, extraction.text
                )
            )
            if not extraction_valid or extraction is None:
                mismatch_metadata = dict(metadata)
                raw_diagnostics = mismatch_metadata.get("document_intelligence")
                diagnostics: dict[str, object] = (
                    dict(raw_diagnostics) if isinstance(raw_diagnostics, dict) else {}
                )
                diagnostics.update(
                    {
                        "decision": "review_required",
                        "next_path": "manual_review",
                        "review_required": True,
                        "reason_codes": [
                            *page_decision.reason_codes,
                            "trusted_context_mismatch",
                        ],
                    }
                )
                mismatch_metadata["document_intelligence"] = diagnostics
                page_result = _review_page(
                    analysis,
                    page_decision,
                    mismatch_metadata,
                    attempted,
                    reason="trusted_context_mismatch",
                )
            else:
                text = _clean_markdown_output(extraction.text)
                page_result = PageResult(
                    page_no=page_no,
                    model_used=f"local:{extraction.engine_name}",
                    markdown=text,
                    quality_score=None,
                    text_quality_score=None,
                    layout_quality_score=None,
                    direction_quality_score=None,
                    completeness_score=None,
                    requires_manual_review=False,
                    review_reason=None,
                    engines_attempted=(extraction.engine_name,),
                    route_used=extraction.engine_name,
                    accepted=True,
                    attempts_count=1,
                    elapsed_time=extraction.processing_time,
                    selection_reason="trusted_digital_text_gate",
                    metadata={
                        **metadata,
                        "page_state": "digital_text",
                        "engine_status": extraction.status,
                    },
                )
        else:
            assert page_decision.next_path is PageNextPath.LOCAL_OCR
            model_extraction = None
            local_metadata = dict(metadata)
            image_bytes = b""
            try:
                image_bytes = render_pdf_page_to_png_bytes(
                    document_context.pdf_bytes, page_no
                )
                assert ocr_engine is not None
                model_extraction = ocr_engine.extract_page(
                    image_bytes=image_bytes,
                    pdf_bytes=document_context.pdf_bytes,
                    page_no=page_no,
                )
            except Exception as exc:
                local_metadata["local_model_error"] = f"{type(exc).__name__}: {exc}"
            if model_extraction is None:
                page_result = _review_page(
                    analysis,
                    page_decision,
                    local_metadata,
                    attempted,
                    reason=OCRIssueCode.ENGINE_ERROR.value,
                )
                page_result.route_used = OCRPageState.OCR_FAILED.value
                page_result.review_reason = OCRIssueCode.ENGINE_ERROR.value
                page_result.metadata = {
                    **(page_result.metadata or {}),
                    "page_state": OCRPageState.OCR_FAILED.value,
                }
            else:
                render_context = make_rendered_page_context(page_no, image_bytes)
                review = review_first_pass(model_extraction, analysis, render_context)
                rereads: list[ReReadResult] = []
                if review.selective_reread_justified:
                    assert ocr_engine is not None
                    reread_bytes = 0
                    reread_attempts = 0
                    for region in review.suspicious_regions:
                        if _cancelled():
                            raise JobCancelled("Conversion was cancelled by the user")
                        try:
                            if (
                                reread_attempts
                                >= review.reread_budget.max_total_attempts
                            ):
                                raise ValueError(
                                    "Selective reread attempt budget exceeded"
                                )
                            crop_bytes = crop_review_region(
                                image_bytes, region, render_context
                            )
                            if (
                                reread_bytes + len(crop_bytes)
                                > review.reread_budget.max_total_bytes
                            ):
                                raise ValueError(
                                    "Selective reread byte budget exceeded"
                                )
                            reread = ocr_engine.extract_page(
                                image_bytes=crop_bytes, page_no=page_no
                            )
                            reread_bytes += len(crop_bytes)
                            reread_attempts += 1
                            rereads.append(
                                ReReadResult(
                                    region.region_id,
                                    reread.engine_name,
                                    reread.success,
                                    reread.text,
                                    reread.metadata.get("model_revision"),
                                )
                            )
                        except Exception:
                            rereads.append(
                                ReReadResult(
                                    region.region_id, ocr_engine.name, False, ""
                                )
                            )
                reconciliation = reconcile_ocr_results(
                    model_extraction.text, review, tuple(rereads)
                )
                review_metadata = {
                    **local_metadata,
                    **model_extraction.metadata,
                    "engine_status": model_extraction.status,
                    "ocr_self_review": {
                        **review.to_diagnostics(),
                        **reconciliation.to_diagnostics(),
                        "reread_attempt_count": len(rereads),
                    },
                    "page_state": reconciliation.state.value,
                }
                if review.verdict.value == "failed":
                    page_result = _review_page(
                        analysis,
                        page_decision,
                        review_metadata,
                        attempted,
                        reason=OCRIssueCode.ENGINE_ERROR.value,
                    )
                    page_result.route_used = OCRPageState.OCR_FAILED.value
                    page_result.review_reason = OCRIssueCode.ENGINE_ERROR.value
                    page_result.metadata = {
                        **(page_result.metadata or {}),
                        "page_state": OCRPageState.OCR_FAILED.value,
                    }
                elif reconciliation.review_required:
                    reason = (
                        reconciliation.reason_codes[0].value
                        if reconciliation.reason_codes
                        else "review_required"
                    )
                    page_result = _review_page(
                        analysis,
                        page_decision,
                        review_metadata,
                        attempted,
                        reason=reason,
                    )
                else:
                    text = _clean_markdown_output(reconciliation.text)
                    page_result = PageResult(
                        page_no=page_no,
                        model_used=(
                            model_extraction.model_name
                            if ocr_engine is not None
                            and ocr_engine.engine_type == "local_model"
                            and model_extraction.model_name
                            else f"local:{model_extraction.engine_name}"
                        ),
                        markdown=text,
                        quality_score=None,
                        text_quality_score=None,
                        layout_quality_score=None,
                        direction_quality_score=None,
                        completeness_score=None,
                        requires_manual_review=False,
                        review_reason=None,
                        engines_attempted=attempted,
                        route_used=reconciliation.state.value,
                        accepted=True,
                        attempts_count=1 + len(rereads),
                        elapsed_time=model_extraction.processing_time,
                        selection_reason="ocr_self_review_reconciliation",
                        metadata=review_metadata,
                    )

        results_by_page[page_no] = page_result
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
        review_count = sum(
            1 for result in ordered_results if result.route_used == "review_required"
        )
        if pending:
            status_placeholder.warning(
                f"Completed page routing; {pending} page(s) require a future OCR model."
            )
        elif review_count:
            status_placeholder.warning(
                f"Completed page routing; {review} page(s) require review."
            )
        else:
            status_placeholder.success("Completed page analysis and routing.")
    return ordered_results, full_markdown
