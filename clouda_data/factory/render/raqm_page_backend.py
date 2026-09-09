"""RAQM page backend (from System A): randomized benchmark-style Arabic pages.

Wraps the vendored ocrbench render stack (Pillow + RAQM layout engine —
real HarfBuzz/FriBiDi shaping/bidi) to compose and draw clean page images
exactly as System A did: A4 @ 150 dpi grayscale (page geometry from
configs/render_profiles/layout.yaml), weighted-random layout selection
(columns, margins, running header/footer, header rules, Arabic-Indic page
numerals) driven deterministically by the global seed, using the bundled
ocr_benchmark fonts (Amiri, Scheherazade, Noto Naskh, Cairo).

The clean PDF for this backend is an image-only img2pdf assembly (System A
rendered PNGs; its text layer is not searchable). Searchable documents
require the weasyprint backend.
"""

from __future__ import annotations

import tempfile
from dataclasses import fields as dataclass_fields
from pathlib import Path


from ..provenance.integrity import atomic_target
from .base import RenderBackend, RenderResult

try:  # pragma: no cover
    from PIL import ImageFont

    from . import _raqm as _v

    _HAS_RAQM = hasattr(ImageFont, "Layout") and hasattr(ImageFont.Layout, "RAQM")
except Exception as _exc:  # pragma: no cover
    _v = None  # type: ignore
    _HAS_RAQM = False
    _IMPORT_ERROR = _exc

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LAYOUT_YAML = _REPO_ROOT / "configs" / "data_factory" / "render_layout.yaml"
# Bundled Arabic fonts live in the canonical assets tree.
_FONT_ROOT = _REPO_ROOT / "assets" / "fonts"


def _require() -> None:
    if not _HAS_RAQM:
        raise RuntimeError(
            "RAQM layout engine unavailable on this host "
            f"({_IMPORT_ERROR!r}); Pillow needs libraqm, or use --backend weasyprint"
        )


def _layout_cfg() -> dict:
    import yaml

    if not _LAYOUT_YAML.is_file():
        raise RuntimeError(f"layout config not found: {_LAYOUT_YAML}")
    return yaml.safe_load(_LAYOUT_YAML.read_text(encoding="utf-8"))


def _spec_from_dict(d: dict):
    from ._raqm.layouts import LayoutSpec

    names = {f.name for f in dataclass_fields(LayoutSpec)}
    kwargs = {k: v for k, v in d.items() if k in names}
    if "margins_mm" in kwargs and isinstance(kwargs["margins_mm"], list):
        kwargs["margins_mm"] = tuple(kwargs["margins_mm"])
    return LayoutSpec(**kwargs)


class RqmPageBackend(RenderBackend):
    name = "raqm"
    searchable = False

    def __init__(self, max_pages: int = 8, allow_cycle: bool = True) -> None:
        _require()
        self.max_pages = int(max_pages)
        self.allow_cycle = bool(allow_cycle)

    def render(
        self,
        text: str | None,
        image_path: Path | None,
        out_dir: Path,
        style_seed: int,
        dpi: int = 150,
    ) -> RenderResult:
        if image_path is not None:
            raise ValueError("raqm backend renders text documents only")
        _require()

        from ._raqm import paginate as v_paginate, render as v_render

        out_dir.mkdir(parents=True, exist_ok=True)
        layout_cfg = _layout_cfg()
        # The bundled font directory from layout.yaml resolves against the
        # canonical repository assets tree, not the working directory.
        layout_cfg.setdefault("fonts", {})
        layout_cfg["fonts"]["dir"] = str(_FONT_ROOT)
        fontbook = v_render.FontBook(layout_cfg, _REPO_ROOT)
        measurer = v_render.TextMeasurer()

        doc = _document_from_text(text or "", style_seed)
        view = _View(layout_cfg)
        paginator = v_paginate.Paginator(view, fontbook, measurer, int(style_seed))  # type: ignore
        pages = paginator.paginate([doc], self.max_pages, allow_cycle=self.allow_cycle)
        if not pages:
            raise RuntimeError("raqm backend produced no pages (input too short?)")

        page_renderer = v_render.PageRenderer(layout_cfg, fontbook)
        page_paths: list[Path] = []
        layout_records: list[dict] = []
        for page in pages:
            spec = _spec_from_dict(page.meta["layout"])
            img = page_renderer.render(page, spec)
            target = out_dir / f"page_{page.index:06d}.png"
            with atomic_target(target) as tmp:
                img.save(tmp)
            page_paths.append(target)
            layout_records.append(page.meta["layout"])

        return RenderResult(
            clean_pdf=None,
            pages=page_paths,
            searchable=False,
            renderer=self.name,
            layout={
                "pages": layout_records,
                "engine": "Pillow/RAQM (HarfBuzz+FriBiDi)",
            },
        )


class _View:
    """Minimal stand-in for BenchmarkConfig: Paginator only reads `.layout`."""

    def __init__(self, layout: dict) -> None:
        self.layout = layout


def _document_from_text(text: str, style_seed: int):
    """Build a SourceDocument from an in-memory string via the vendored
    corpus module (writes a temp file, since load_document reads from disk)."""
    from ._raqm import corpus as v_corpus

    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", encoding="utf-8", delete=False, prefix="cdf-doc-"
    ) as handle:
        handle.write(text)
        path = Path(handle.name)
    try:
        return v_corpus.load_document(path, markup="auto")
    finally:
        path.unlink(missing_ok=True)
