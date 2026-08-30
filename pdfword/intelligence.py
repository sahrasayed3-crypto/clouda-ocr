import io
import re
from dataclasses import dataclass

from PIL import Image, ImageFilter
from pypdf import PdfReader


@dataclass(frozen=True)
class DocumentIntelligence:
    page_count: int
    extractable_ratio: float
    arabic_ratio: float
    english_ratio: float
    image_ratio: float
    table_ratio: float
    equation_ratio: float
    scan_quality_hint: float
    signature: str


@dataclass(frozen=True)
class LayoutRegionSummary:
    titles: int
    paragraphs: int
    tables: int
    figures: int
    headers: int
    footers: int
    side_notes: int
    captions: int


@dataclass(frozen=True)
class PageIntelligence:
    blur: float
    contrast: float
    brightness: float
    complexity: float
    columns_hint: int
    table_density: float
    image_density: float
    rtl_probability: float
    layout_complexity: float


def _ratios(text: str) -> tuple[float, float]:
    t = text or ""
    alpha = max(1, len(re.findall(r"[A-Za-z\u0600-\u06FF]", t)))
    ar = len(re.findall(r"[\u0600-\u06FF]", t)) / alpha
    en = len(re.findall(r"[A-Za-z]", t)) / alpha
    return ar, en


def analyze_document_intelligence(
    pdf_bytes: bytes, pages: list[int]
) -> DocumentIntelligence:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    extractable = 0
    ar_sum = 0.0
    en_sum = 0.0
    image_pages = 0
    table_pages = 0
    eq_pages = 0
    quality_sum = 0.0

    for p in pages:
        page = reader.pages[p - 1]
        txt = (page.extract_text() or "").strip()
        has_text = len(txt) >= 20
        if has_text:
            extractable += 1
        ar, en = _ratios(txt)
        ar_sum += ar
        en_sum += en
        if "|" in txt or "\t" in txt:
            table_pages += 1
        if re.search(r"[=+\-*/^∑√≈≤≥]", txt):
            eq_pages += 1
        try:
            images = getattr(page, "images", None)
            if images and len(images) > 0:
                image_pages += 1
        except Exception:
            pass
        quality_sum += 0.95 if has_text else 0.45

    total = max(1, len(pages))
    extractable_ratio = extractable / total
    ar_ratio = ar_sum / total
    en_ratio = en_sum / total
    signature = f"p{total}|ex{int(extractable_ratio*10)}|ar{int(ar_ratio*10)}|en{int(en_ratio*10)}|img{int((image_pages/total)*10)}"
    return DocumentIntelligence(
        page_count=len(reader.pages),
        extractable_ratio=extractable_ratio,
        arabic_ratio=ar_ratio,
        english_ratio=en_ratio,
        image_ratio=image_pages / total,
        table_ratio=table_pages / total,
        equation_ratio=eq_pages / total,
        scan_quality_hint=max(0.0, min(1.0, quality_sum / total)),
        signature=signature,
    )


def analyze_layout_intelligence(page_text: str) -> LayoutRegionSummary:
    text = page_text or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    titles = sum(
        1 for ln in lines if len(ln) < 90 and (ln.endswith(":") or ln.isupper())
    )
    tables = sum(1 for ln in lines if "|" in ln)
    figures = sum(
        1
        for ln in lines
        if re.search(r"\b(figure|شكل|img|image)\b", ln, flags=re.IGNORECASE)
    )
    captions = sum(
        1
        for ln in lines
        if re.search(r"\b(جدول|table|caption)\b", ln, flags=re.IGNORECASE)
    )
    headers = 1 if lines and len(lines[0]) < 80 else 0
    footers = 1 if len(lines) > 1 and len(lines[-1]) < 80 else 0
    side_notes = sum(1 for ln in lines if len(ln) < 35 and not re.search(r"[.!؟]$", ln))
    paragraphs = max(0, len(lines) - titles - tables - side_notes)
    return LayoutRegionSummary(
        titles=titles,
        paragraphs=paragraphs,
        tables=tables,
        figures=figures,
        headers=headers,
        footers=footers,
        side_notes=side_notes,
        captions=captions,
    )


def analyze_page_intelligence(
    image_bytes: bytes, page_text: str, has_image_objects: bool
) -> PageIntelligence:
    with Image.open(io.BytesIO(image_bytes)) as source:
        img = source.convert("L").resize((300, 300), Image.Resampling.LANCZOS)
        hist = img.histogram()
        total = float(sum(hist)) or 1.0
        mean = sum(i * c for i, c in enumerate(hist)) / total
        var = sum(((i - mean) ** 2) * c for i, c in enumerate(hist)) / total
        contrast = min(1.0, (var**0.5) / 64.0)
        brightness = max(0.0, min(1.0, mean / 255.0))

        edges = img.filter(ImageFilter.FIND_EDGES)
        e_hist = edges.histogram()
        edge_ratio = sum(e_hist[170:]) / max(1.0, float(sum(e_hist)))
        blur = max(0.0, min(1.0, 1.0 - (edge_ratio * 2.1)))

    ar, en = _ratios(page_text or "")
    table_density = 1.0 if ("|" in (page_text or "")) else 0.0
    image_density = 0.8 if has_image_objects else 0.2
    rtl_probability = min(1.0, ar * 1.4)
    columns_hint = 2 if re.search(r"\s{6,}", page_text or "") else 1
    complexity = max(
        0.0,
        min(
            1.0,
            (edge_ratio * 1.1)
            + (table_density * 0.35)
            + (0.25 if columns_hint > 1 else 0.0),
        ),
    )
    layout_complexity = max(complexity, table_density * 0.6, image_density * 0.55)
    return PageIntelligence(
        blur=blur,
        contrast=contrast,
        brightness=brightness,
        complexity=complexity,
        columns_hint=columns_hint,
        table_density=table_density,
        image_density=image_density,
        rtl_probability=rtl_probability,
        layout_complexity=layout_complexity,
    )
