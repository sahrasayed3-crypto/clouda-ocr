"""WeasyPrint/Pango/HarfBuzz render backend (from System B).

Vendored/adapted from arabic-scan-factory typeset.py + effects.rasterize:
RTL HTML/CSS typesetting through Pango/HarfBuzz produces a searchable
multi-page clean PDF; pages are rasterized with Poppler pdftoppm when page
images are requested. Font/size selection stays seed-derived exactly as in
System B so `legacy_seed_mode: arabic_scan_factory` renders identically.

Requires: weasyprint (with native Pango/HarfBuzz) and poppler-utils.
"""

from __future__ import annotations

import html
import subprocess
from pathlib import Path
from typing import Any

from ..provenance.integrity import atomic_target
from .base import RenderBackend, RenderResult

try:  # pragma: no cover - native dependency availability varies by host
    import contextlib
    import io as _io

    # WeasyPrint prints native-library troubleshooting text to stdout/stderr
    # when its import fails; swallow it so CLI JSON output stays clean.
    with (
        contextlib.redirect_stdout(_io.StringIO()),
        contextlib.redirect_stderr(_io.StringIO()),
    ):
        from weasyprint import HTML
    _HAS_WEASYPRINT = True
except Exception as _exc:  # pragma: no cover
    HTML = None  # type: ignore[assignment]
    _IMPORT_ERROR = _exc
    _HAS_WEASYPRINT = False

FONTS = ["Amiri", "Noto Naskh Arabic", "Scheherazade", "Lateef", "Noto Sans Arabic"]
FONT_SIZES = [15, 16, 17, 18]


def _require() -> None:
    if not _HAS_WEASYPRINT:
        raise RuntimeError(
            "weasyprint is not available on this host "
            f"({_IMPORT_ERROR!r}); install Pango/HarfBuzz or use --backend raqm"
        )


def typeset_text(text: str, output: Path, style_seed: int = 0) -> dict[str, Any]:
    """System B typesetting (byte-compatible with the original tool)."""
    _require()
    font = FONTS[style_seed % len(FONTS)]
    size = FONT_SIZES[style_seed % 4]
    body = html.escape(text)
    css = f"""
    @page {{ size: A5; margin: 18mm 17mm 20mm;
      @bottom-center {{ content: counter(page); font: 9pt sans-serif; }} }}
    html {{ direction: rtl; }}
    body {{ font-family: "{font}"; font-size: {size}pt; line-height: 1.75;
      text-align: start; direction: rtl; }}
    .source {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
    """
    document = (
        '<html lang="ar" dir="rtl"><meta charset="utf-8"><style>'
        + css
        + '</style><body><div class="source">'
        + body
        + "</div></body></html>"
    )
    with atomic_target(output) as temporary:
        HTML(string=document).write_pdf(temporary)
    return {
        "font": font,
        "font_size_pt": size,
        "direction": "rtl",
        "engine": "WeasyPrint/Pango/HarfBuzz",
        "text_normalization": "none",
    }


def rasterize(pdf: Path, dpi: int, output_dir: Path) -> list[Path]:
    """System B rasterization via Poppler pdftoppm (JPEG q96 intermediates)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(Path(output_dir) / "page")
    subprocess.run(
        [
            "pdftoppm",
            "-jpeg",
            "-jpegopt",
            "quality=96",
            "-r",
            str(dpi),
            str(pdf),
            prefix,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return sorted(Path(output_dir).glob("page-*.jpg"))


class WeasyPrintBackend(RenderBackend):
    name = "weasyprint"
    searchable = True

    def __init__(self, poppler_dpi: int = 150) -> None:
        self.poppler_dpi = int(poppler_dpi)

    def render(
        self,
        text: str | None,
        image_path: Path | None,
        out_dir: Path,
        style_seed: int,
        dpi: int = 150,
    ) -> RenderResult:
        if image_path is not None:
            raise ValueError("weasyprint backend renders text documents only")
        _require()
        out_dir.mkdir(parents=True, exist_ok=True)
        clean_pdf = out_dir / "clean.pdf"
        layout = typeset_text(text or "", clean_pdf, style_seed)
        return RenderResult(
            clean_pdf=clean_pdf,
            pages=[],
            searchable=True,
            renderer=self.name,
            layout=layout,
        )

    def page_images(
        self, clean_pdf: Path, out_dir: Path, dpi: int | None = None
    ) -> list[Path]:
        effective = int(dpi or self.poppler_dpi)
        pages = rasterize(clean_pdf, effective, out_dir)
        renamed = []
        for index, page in enumerate(pages):
            target = out_dir / f"page_{index:06d}.png"
            from PIL import Image

            with Image.open(page) as img:
                img.save(target)
            page.unlink()
            renamed.append(target)
        return renamed
