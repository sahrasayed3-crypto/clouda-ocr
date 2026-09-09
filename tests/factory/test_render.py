"""Render backends (vendored System A + System B stacks).

Backend render tests skip on hosts without libraqm (Pillow) or native
Pango/HarfBuzz (WeasyPrint); they run on the Linux/GPU host.
Migrated from the standalone repository; imports retargeted.
"""

from __future__ import annotations

import pytest

from clouda_data.factory.render import available_backends, get_backend

try:
    from PIL import features as _pil_features

    _HAS_RAQM = _pil_features.check("raqm")
except Exception:  # pragma: no cover
    _HAS_RAQM = False

try:
    import weasyprint  # noqa: F401

    _HAS_WEASYPRINT = True
except Exception:  # pragma: no cover
    _HAS_WEASYPRINT = False

ARABIC = (
    "السَّلَامُ عَلَيْكُمْ وَرَحْمَةُ اللهِ وَبَرَكَاتُهُ\n\n"
    "هذا نص عربي تجريبي لإنشاء ملف PDF نظيف ونسخة مسح ضوئي اصطناعية.\n"
    "Mixed Arabic and English: الإصدار 1.2 يعمل مع UTF-8.\n"
)


@pytest.mark.skipif(not _HAS_RAQM, reason="Pillow built without libraqm")
def test_raqm_backend_renders_deterministic_pages(tmp_path):
    backend = get_backend("raqm")
    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    r1 = backend.render(ARABIC, None, out1, 20260831)
    r2 = backend.render(ARABIC, None, out2, 20260831)
    assert r1.pages and r2.pages
    assert r1.renderer == "raqm"
    assert not r1.searchable
    from clouda_data.factory.provenance.hashing import sha256_file

    assert sha256_file(r1.pages[0]) == sha256_file(r2.pages[0])


@pytest.mark.skipif(
    not _HAS_WEASYPRINT, reason="WeasyPrint/Pango native libs unavailable"
)
def test_weasyprint_backend_searchable_pdf(tmp_path):
    import subprocess

    backend = get_backend("weasyprint")
    out = tmp_path / "wp"
    result = backend.render(ARABIC, None, out, 20260831)
    assert result.searchable and result.clean_pdf.is_file()
    text = subprocess.run(
        ["pdftotext", str(result.clean_pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "PDF" in text  # searchable text layer contains the Latin run
    pages = backend.page_images(result.clean_pdf, out / "pages")
    assert pages


def test_backends_reject_wrong_input_type(tmp_path):
    for name in available_backends():
        backend = get_backend(name)
        with pytest.raises(ValueError):
            backend.render(None, tmp_path / "x.png", tmp_path / "out", 1)


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_backend("nope")


def test_layout_config_and_fonts_resolve_in_canonical_tree():
    """The integrated package reads layout/fonts from the canonical repo tree."""
    from clouda_data.factory.render.raqm_page_backend import _FONT_ROOT, _LAYOUT_YAML

    assert _LAYOUT_YAML.is_file(), _LAYOUT_YAML
    assert _FONT_ROOT.is_dir(), _FONT_ROOT
    families = {
        "Amiri-Regular.ttf",
        "Amiri-Bold.ttf",
        "Cairo.ttf",
        "NotoNaskhArabic.ttf",
        "ScheherazadeNew-Regular.ttf",
        "ScheherazadeNew-Bold.ttf",
    }
    present = {p.name for p in _FONT_ROOT.glob("*.ttf")}
    assert families <= present
