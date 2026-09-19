from __future__ import annotations

import io
from typing import Any

from pypdf import PdfReader

from pdfword.engines import get_engine_registry
from pdfword.page_routing import (
    DocumentRoutingContext,
    analyze_pdf_page,
    decide_page_route,
    evaluate_digital_text_trust,
)

from .security import browser_safe
from .settings import LabSettings

MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 25
DOCUMENT_INTELLIGENCE_SCHEMA = "clouda.lab.document-intelligence.v1"


class DocumentIntelligenceService:
    """Bounded, in-memory inspection of canonical page-routing decisions."""

    def __init__(self, settings: LabSettings) -> None:
        self.settings = settings

    @staticmethod
    def _validate_pdf(pdf_bytes: bytes) -> PdfReader:
        if len(pdf_bytes) > MAX_PDF_BYTES:
            raise ValueError("Document Intelligence accepts PDFs up to 10 MiB")
        if not pdf_bytes.startswith(b"%PDF-"):
            raise ValueError("Uploaded content does not have a PDF signature")
        reader = PdfReader(io.BytesIO(pdf_bytes))
        page_count = len(reader.pages)
        if page_count < 1:
            raise ValueError("PDF must contain at least one page")
        if page_count > MAX_PDF_PAGES:
            raise ValueError("Document Intelligence accepts at most 25 pages")
        return reader

    @staticmethod
    def _ocr_available() -> bool:
        return any(
            engine.engine_type == "local_model" and engine.available()
            for engine in get_engine_registry().all()
        )

    def analyze(self, pdf_bytes: bytes) -> dict[str, Any]:
        reader = self._validate_pdf(pdf_bytes)
        document = DocumentRoutingContext.from_pdf(pdf_bytes)
        ocr_available = self._ocr_available()
        pages: list[dict[str, Any]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            analysis = analyze_pdf_page(page, page_number, document)
            gate = evaluate_digital_text_trust(analysis)
            decision = decide_page_route(
                analysis,
                gate,
                ocr_available=ocr_available,
            )
            pages.append(
                {
                    **decision.to_diagnostics(),
                    "page_number": page_number,
                    "analysis": analysis.to_diagnostics(),
                    "gate_checks": gate.to_diagnostics()["checks"],
                }
            )
        return browser_safe(
            {
                "schema_version": DOCUMENT_INTELLIGENCE_SCHEMA,
                "page_count": len(pages),
                "pages": pages,
            },
            (self.settings.repo_root,),
        )


__all__ = [
    "DOCUMENT_INTELLIGENCE_SCHEMA",
    "MAX_PDF_BYTES",
    "MAX_PDF_PAGES",
    "DocumentIntelligenceService",
]
