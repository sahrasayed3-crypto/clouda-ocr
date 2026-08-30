from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from pypdf import PdfReader


@dataclass(frozen=True)
class ProcessingLimits:
    max_upload_bytes: int = 100 * 1024 * 1024
    max_result_bytes: int = 100 * 1024 * 1024
    max_pdf_bytes: int = 100 * 1024 * 1024
    max_pdf_pages: int = 500
    max_image_pixels: int = 40_000_000
    max_archive_members: int = 10_000
    max_decompressed_bytes: int = 1024 * 1024 * 1024
    max_dpi: int = 400
    max_parallel_pages: int = 2
    max_ocr_attempts: int = 6
    page_timeout_seconds: int = 300
    file_timeout_seconds: int = 7200


DEFAULT_LIMITS = ProcessingLimits()


def limits_from_settings(settings: dict[str, Any]) -> ProcessingLimits:
    return ProcessingLimits(
        max_upload_bytes=max(
            1024, int(settings.get("max_upload_bytes", 100 * 1024 * 1024))
        ),
        max_result_bytes=max(
            1024, int(settings.get("max_result_bytes", 100 * 1024 * 1024))
        ),
        max_pdf_bytes=max(1024, int(settings.get("max_pdf_bytes", 100 * 1024 * 1024))),
        max_pdf_pages=max(1, int(settings.get("max_pdf_pages", 500))),
        max_image_pixels=max(
            1_000_000, int(settings.get("max_image_pixels", 40_000_000))
        ),
        max_archive_members=max(10, int(settings.get("max_archive_members", 10_000))),
        max_decompressed_bytes=max(
            1024 * 1024,
            int(settings.get("max_decompressed_bytes", 1024 * 1024 * 1024)),
        ),
        max_dpi=max(100, int(settings.get("max_dpi", 400))),
        max_parallel_pages=max(1, min(2, int(settings.get("max_parallel_pages", 2)))),
        max_ocr_attempts=max(1, int(settings.get("max_ocr_attempts", 6))),
        page_timeout_seconds=max(30, int(settings.get("page_timeout_seconds", 300))),
        file_timeout_seconds=max(60, int(settings.get("file_timeout_seconds", 7200))),
    )


def limits_from_env() -> ProcessingLimits:
    return limits_from_settings(
        {
            "max_upload_bytes": os.getenv("CLOUDA_MAX_UPLOAD_BYTES", 100 * 1024 * 1024),
            "max_result_bytes": os.getenv("CLOUDA_MAX_RESULT_BYTES", 100 * 1024 * 1024),
            "max_pdf_bytes": os.getenv("CLOUDA_MAX_PDF_BYTES", 100 * 1024 * 1024),
            "max_pdf_pages": os.getenv("CLOUDA_MAX_PDF_PAGES", 500),
            "max_image_pixels": os.getenv("CLOUDA_MAX_IMAGE_PIXELS", 40_000_000),
            "max_archive_members": os.getenv("CLOUDA_MAX_ARCHIVE_MEMBERS", 10_000),
            "max_decompressed_bytes": os.getenv(
                "CLOUDA_MAX_DECOMPRESSED_BYTES", 1024 * 1024 * 1024
            ),
        }
    )


def validate_pdf_limits(
    pdf_size_bytes: int,
    total_pages: int,
    *,
    limits: ProcessingLimits = DEFAULT_LIMITS,
) -> None:
    if pdf_size_bytes <= 0:
        raise ValueError("PDF is empty.")
    if total_pages <= 0:
        raise ValueError("PDF has no processable pages.")
    if pdf_size_bytes > min(limits.max_upload_bytes, limits.max_pdf_bytes):
        raise ValueError("PDF exceeds the configured byte limit.")
    if total_pages > limits.max_pdf_pages:
        raise ValueError(
            f"PDF page count exceeds the configured limit ({limits.max_pdf_pages})."
        )


def inspect_pdf_upload(
    uploaded_file,
    *,
    limits: ProcessingLimits | None = None,
) -> tuple[int, int]:
    """Read an uploaded PDF defensively and return its actual byte/page counts."""
    active_limits = limits or limits_from_env()
    byte_limit = min(active_limits.max_upload_bytes, active_limits.max_pdf_bytes)
    total = 0
    try:
        uploaded_file.seek(0)
        while chunk := uploaded_file.read(1024 * 1024):
            total += len(chunk)
            if total > byte_limit:
                raise ValueError("يتجاوز حجم الملف الحد المسموح.")
        if total <= 0:
            raise ValueError("الملف فارغ.")
        uploaded_file.seek(0)
        try:
            reader = PdfReader(uploaded_file)
            if reader.is_encrypted:
                raise ValueError("لا يمكن معالجة ملف PDF محمي بكلمة مرور.")
            page_count = len(reader.pages)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("ملف PDF غير صالح أو تالف.") from exc
        if page_count <= 0:
            raise ValueError("لا يحتوي ملف PDF على صفحات قابلة للمعالجة.")
        if page_count > active_limits.max_pdf_pages:
            raise ValueError(
                "يتجاوز عدد صفحات الملف الحد المسموح "
                f"({active_limits.max_pdf_pages} صفحة)."
            )
        return total, page_count
    finally:
        uploaded_file.seek(0)
