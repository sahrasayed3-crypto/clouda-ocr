"""Create deterministic, copyright-free PDF fixtures for readiness tests."""

from __future__ import annotations

import io
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
PAGE_RECT = fitz.Rect(0, 0, 595, 842)


def _png_bytes(size: tuple[int, int], label: str) -> bytes:
    image = Image.new("RGB", size, "white")
    drawer = ImageDraw.Draw(image)
    drawer.rectangle((2, 2, size[0] - 3, size[1] - 3), outline="black", width=2)
    drawer.text((max(5, size[0] // 10), max(5, size[1] // 2 - 8)), label, fill="black")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _save(document: fitz.Document, name: str) -> None:
    document.set_metadata(
        {"title": f"Clouda test fixture: {name}", "creator": "Clouda tests"}
    )
    document.save(ROOT / name, garbage=4, deflate=True)
    document.close()


def _digital_page(document: fitz.Document, text: str) -> None:
    page = document.new_page(width=PAGE_RECT.width, height=PAGE_RECT.height)
    page.insert_textbox(fitz.Rect(72, 72, 523, 300), text, fontsize=14, fontname="helv")


def _scanned_page(document: fitz.Document, label: str = "SCANNED PAGE") -> None:
    page = document.new_page(width=PAGE_RECT.width, height=PAGE_RECT.height)
    page.insert_image(PAGE_RECT, stream=_png_bytes((1190, 1684), label))


def generate() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)

    document = fitz.open()
    _digital_page(document, "Digital PDF text is selectable and extracted without OCR.")
    _save(document, "digital_text.pdf")

    document = fitz.open()
    _scanned_page(document)
    _save(document, "scanned.pdf")

    document = fitz.open()
    document.new_page(width=PAGE_RECT.width, height=PAGE_RECT.height)
    _save(document, "blank.pdf")

    document = fitz.open()
    _digital_page(document, "12")
    _save(document, "near_blank_page_number.pdf")

    document = fitz.open()
    page = document.new_page(width=PAGE_RECT.width, height=PAGE_RECT.height)
    page.insert_image(fitz.Rect(550, 800, 570, 820), stream=_png_bytes((20, 20), "X"))
    _save(document, "near_blank_stamp.pdf")

    document = fitz.open()
    _digital_page(document, "First page has a digital text layer.")
    _scanned_page(document, "SECOND PAGE IS A SCAN")
    _save(document, "mixed.pdf")

    (ROOT / "empty.pdf").write_bytes(b"")
    (ROOT / "corrupt.pdf").write_bytes(b"%PDF-1.7\nnot a valid PDF\n")
    (ROOT / "not_a_pdf.txt").write_text("This is not a PDF.", encoding="utf-8")


if __name__ == "__main__":
    generate()
