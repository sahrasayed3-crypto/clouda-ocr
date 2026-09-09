"""Exporters: clean/distorted PDFs and page images.

- pdf_scan: System B's img2pdf assembly verbatim (fixed DPI layout, pinned
  2000-01-01 metadata dates for byte determinism, atomic write).
- png: simple atomic PNG page writer (System A's direct-PNG behavior).
- clean PDFs are written by the weasyprint render backend itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from ..provenance.integrity import atomic_target

_STABLE_DATE = datetime(2000, 1, 1, tzinfo=timezone.utc)


def jpeg_images_to_pdf(images: list[Path], output: Path, dpi: int = 300) -> None:
    """System B verbatim: embed page JPEGs into an image-only PDF."""
    import img2pdf

    layout = img2pdf.get_fixed_dpi_layout_fun((int(dpi), int(dpi)))
    with atomic_target(output) as temporary:
        with open(temporary, "wb") as destination:
            destination.write(
                img2pdf.convert(
                    [str(Path(path)) for path in images],
                    layout_fun=layout,
                    creationdate=_STABLE_DATE,
                    moddate=_STABLE_DATE,
                    engine=img2pdf.Engine.internal,
                )
            )


def save_page_png(image: Image.Image, output: Path) -> Path:
    """Atomic PNG page write."""
    with atomic_target(output) as tmp:
        image.save(tmp, format="PNG")
    return output
