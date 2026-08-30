import io
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass(frozen=True)
class DocumentAnalysis:
    page_count: int
    selected_pages: int
    avg_width_pt: float
    avg_height_pt: float
    landscape_ratio: float
    avg_text_chars: float
    arabic_ratio: float
    has_extractable_text: bool
    category: str
    signature: str


def _safe_page_dims(page) -> tuple[float, float]:
    try:
        box = page.mediabox
        return float(box.width), float(box.height)
    except Exception:
        return 0.0, 0.0


def _estimate_category(
    avg_text_chars: float, arabic_ratio: float, landscape_ratio: float, has_text: bool
) -> str:
    if not has_text or avg_text_chars < 60:
        return "low_quality_scans"
    if avg_text_chars >= 900 and arabic_ratio >= 0.35:
        return "born_digital_modern"
    if landscape_ratio >= 0.45:
        return "mixed_multi_page_docs"
    if avg_text_chars < 220:
        return "blurry_docs"
    return "books"


def analyze_pdf_document(
    pdf_bytes: bytes, page_numbers: list[int] | None = None
) -> DocumentAnalysis:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    if total_pages <= 0:
        raise ValueError("PDF has no pages")

    pages = page_numbers or list(range(1, total_pages + 1))
    pages = [p for p in pages if 1 <= int(p) <= total_pages]
    if not pages:
        pages = list(range(1, total_pages + 1))

    widths: list[float] = []
    heights: list[float] = []
    text_chars = 0
    arabic_chars = 0
    alpha_chars = 0
    landscape_pages = 0

    for page_no in pages:
        page = reader.pages[page_no - 1]
        w, h = _safe_page_dims(page)
        widths.append(w)
        heights.append(h)
        if w > h:
            landscape_pages += 1

        txt = page.extract_text() or ""
        text_chars += len(txt)
        for ch in txt:
            code = ord(ch)
            if 0x0600 <= code <= 0x06FF:
                arabic_chars += 1
            if ch.isalpha() or (0x0600 <= code <= 0x06FF):
                alpha_chars += 1

    selected = len(pages)
    avg_w = sum(widths) / max(1, selected)
    avg_h = sum(heights) / max(1, selected)
    avg_txt = text_chars / max(1, selected)
    arabic_ratio = arabic_chars / max(1, alpha_chars)
    landscape_ratio = landscape_pages / max(1, selected)
    has_text = text_chars > 0

    category = _estimate_category(
        avg_text_chars=avg_txt,
        arabic_ratio=arabic_ratio,
        landscape_ratio=landscape_ratio,
        has_text=has_text,
    )
    text_bucket = (
        "dense" if avg_txt >= 900 else ("medium" if avg_txt >= 240 else "light")
    )
    arabic_bucket = "ar" if arabic_ratio >= 0.25 else "mixed"
    signature = f"{category}|{text_bucket}|{arabic_bucket}|p{selected}"

    return DocumentAnalysis(
        page_count=total_pages,
        selected_pages=selected,
        avg_width_pt=avg_w,
        avg_height_pt=avg_h,
        landscape_ratio=landscape_ratio,
        avg_text_chars=avg_txt,
        arabic_ratio=arabic_ratio,
        has_extractable_text=has_text,
        category=category,
        signature=signature,
    )
