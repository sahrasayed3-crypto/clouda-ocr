"""Data Factory (Phase 6), renderer (Phases 7-8), PDF/image (Phase 9) checks.

Import/resource checks only — the doctor never generates datasets. Renderer
readiness is deliberately stricter than "the Python package imports":

- WeasyPrint: module import with swallowed stdout/stderr (mirroring
  ``clouda_data.factory.render.weasyprint_backend``) plus a tiny in-memory
  HTML->PDF smoke when the module is present.
- RAQM: the vendored backend's ``_HAS_RAQM`` flag is **not trusted** — it is
  True whenever the Pillow ``ImageFont.Layout.RAQM`` enum exists, even when
  the native library is missing. Real readiness requires
  ``PIL._imagingft.HAVE_RAQM`` and a smoke draw that does not emit Pillow's
  "Falling back to basic layout" warning.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import warnings
from pathlib import Path
from typing import Any

from .environment import installed_version
from .models import DoctorCheck, DoctorSection, DoctorStatus

FACTORY_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "resources"

FACTORY_IMPORT_MODULES: tuple[str, ...] = (
    "clouda_data.factory",
    "clouda_data.factory.factory",
    "clouda_data.factory.cli",
    "clouda_data.factory.ingest.text_file",
    "clouda_data.factory.ingest.image_file",
    "clouda_data.factory.distort.atomic",
    "clouda_data.factory.distort.qc",
    "clouda_data.factory.distort.scan_composite",
    "clouda_data.factory.export",
    "clouda_data.factory.manifest",
    "clouda_data.factory.profiles",
    "clouda_data.factory.provenance.integrity",
    "clouda_data.factory.seed.derive",
)

# Renderer smoke text: real Arabic so shaping/bidi matter.
_SMOKE_ARABIC = "مرحبا"


def check_factory(
    repo_root: Path | None,
    *,
    cv2_available: bool | None = None,
    yaml_available: bool | None = None,
    numpy_available: bool | None = None,
) -> DoctorSection:
    """Lightweight Data Factory readiness (imports + packaged resources)."""
    checks: list[DoctorCheck] = []

    if cv2_available is None:
        cv2_available = _module_importable("cv2")
    if yaml_available is None:
        yaml_available = _module_importable("yaml")
    if numpy_available is None:
        numpy_available = _module_importable("numpy")

    failed_imports: list[str] = []
    for module_name in FACTORY_IMPORT_MODULES:
        if not _module_importable(module_name):
            failed_imports.append(module_name)
    engine_ok = cv2_available and numpy_available and yaml_available
    if failed_imports or not engine_ok:
        missing = [
            name
            for name, ok in (
                ("opencv (cv2)", cv2_available),
                ("numpy", numpy_available),
                ("PyYAML", yaml_available),
            )
            if not ok
        ]
        checks.append(
            DoctorCheck(
                id="factory.package",
                name="Data Factory package",
                subsystem="data-factory",
                status=DoctorStatus.FAIL,
                message=(
                    "Factory imports failed: " + ", ".join(failed_imports)
                    if failed_imports
                    else "Factory engine dependencies missing: " + ", ".join(missing)
                ),
                details={
                    "failed_imports": failed_imports,
                    "missing_engine_deps": missing,
                },
                remediation='pip install -e ".[factory,data]"',
            )
        )
    else:
        checks.append(
            DoctorCheck(
                id="factory.package",
                name="Data Factory package",
                subsystem="data-factory",
                status=DoctorStatus.PASS,
                message="Factory package and engine imports OK",
                details={"modules": len(FACTORY_IMPORT_MODULES)},
            )
        )

    # configs / resources -------------------------------------------------
    configs_dir = FACTORY_PACKAGE_ROOT / "data_factory"
    fonts_dir = FACTORY_PACKAGE_ROOT / "fonts"
    distortions_dir = FACTORY_PACKAGE_ROOT / "distortions"
    expected_files = {
        "ocr_benchmark.yaml": configs_dir,
        "render_layout.yaml": configs_dir,
        "foundation_sources_v1.json": FACTORY_PACKAGE_ROOT,
    }
    missing_resources = [
        f"{parent.name}/{name}"
        for name, parent in expected_files.items()
        if not (parent / name).is_file()
    ]
    checks.append(
        DoctorCheck(
            id="factory.configs",
            name="Factory configs and resources",
            subsystem="data-factory",
            status=DoctorStatus.PASS if not missing_resources else DoctorStatus.FAIL,
            message=(
                "Factory configs present (ocr_benchmark.yaml, render_layout.yaml)"
                if not missing_resources
                else "Missing factory resources: " + ", ".join(missing_resources)
            ),
            details={"missing": missing_resources},
            remediation=(
                None
                if not missing_resources
                else "Restore packaged resources or reinstall the package."
            ),
        )
    )

    fonts = sorted(fonts_dir.glob("*.ttf")) if fonts_dir.is_dir() else []
    readable = [f for f in fonts if f.is_file() and f.stat().st_size > 0]
    checks.append(
        DoctorCheck(
            id="factory.fonts",
            name="Packaged fonts",
            subsystem="data-factory",
            status=DoctorStatus.PASS if readable else DoctorStatus.WARN,
            message=(
                f"{len(readable)} packaged Arabic TTF fonts readable"
                if readable
                else "No packaged TTF fonts found in resources/fonts"
            ),
            details={"fonts": [f.name for f in readable], "count": len(readable)},
            remediation=(
                None if readable else "Reinstall package data (resources/fonts/*.ttf)."
            ),
        )
    )

    profile_count = (
        len(list(distortions_dir.glob("*.yaml")))
        + len(list(distortions_dir.glob("*.yml")))
        if distortions_dir.is_dir()
        else 0
    )
    checks.append(
        DoctorCheck(
            id="factory.profiles",
            name="Distortion profiles",
            subsystem="data-factory",
            status=DoctorStatus.PASS if profile_count else DoctorStatus.WARN,
            message=(
                f"{profile_count} distortion profile YAMLs available"
                if profile_count
                else "No distortion profile YAMLs found"
            ),
            details={"count": profile_count},
            remediation=(
                None
                if profile_count
                else "Reinstall package data (resources/distortions)."
            ),
        )
    )

    return DoctorSection(id="data-factory", name="Data Factory", checks=checks)


# ---------------------------------------------------------------------------
# Phase 7 — WeasyPrint renderer
# ---------------------------------------------------------------------------


def check_weasyprint() -> DoctorSection:
    checks: list[DoctorCheck] = []
    version = installed_version("WeasyPrint")

    # Import exactly the way the vendored backend does: swallow native
    # library troubleshooting text printed to stdout/stderr. Routing through
    # _module_importable keeps the probe monkeypatchable in tests.
    error: str | None = None
    import_ok = _module_importable("weasyprint")
    if not import_ok:
        try:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                importlib.import_module("weasyprint")
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = None

    if import_ok and version:
        smoke_status, smoke_msg = _weasyprint_smoke()
        checks.append(
            DoctorCheck(
                id="render.weasyprint",
                name="WeasyPrint renderer",
                subsystem="rendering",
                status=smoke_status,
                message=smoke_msg,
                required=False,
                details={"version": version, "smoke": smoke_status.value},
                remediation=(
                    None
                    if smoke_status is DoctorStatus.PASS
                    else 'pip install -e ".[factory-render]" (native Pango/HarfBuzz/GObject required)'
                ),
            )
        )
    else:
        checks.append(
            DoctorCheck(
                id="render.weasyprint",
                name="WeasyPrint renderer",
                subsystem="rendering",
                status=DoctorStatus.FAIL,
                message=f"WeasyPrint not available: {error}",
                required=False,
                details={"version": None, "import_error": error},
                remediation=(
                    'pip install -e ".[factory-render]" '
                    "(Windows needs GTK/Pango runtime; render with --backend raqm meanwhile)"
                ),
            )
        )
    return DoctorSection(
        id="rendering-weasyprint", name="WeasyPrint Renderer", checks=checks
    )


def _weasyprint_smoke() -> tuple[DoctorStatus, str]:
    """Tiny in-memory render; never writes to disk."""
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            from weasyprint import HTML  # type: ignore[import-not-found]

            html_doc = HTML(
                string="<html><body><p>clouda doctor</p></body></html>",
                base_url=".",
            )
            pdf_bytes = html_doc.write_pdf()
        if pdf_bytes and len(pdf_bytes) > 100:
            return DoctorStatus.PASS, "WeasyPrint in-memory PDF smoke render succeeded"
        return (
            DoctorStatus.WARN,
            "WeasyPrint imported but smoke render produced no PDF bytes",
        )
    except Exception as exc:  # noqa: BLE001
        return (
            DoctorStatus.FAIL,
            f"WeasyPrint imported but render failed (native libraries?): {type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Phase 8 — RAQM renderer (native-aware)
# ---------------------------------------------------------------------------


def native_raqm_available() -> bool:
    """True only when Pillow's C layer actually ships RAQM.

    ``PIL._imagingft.HAVE_RAQM`` is the authoritative flag; the public
    ``ImageFont.Layout.RAQM`` enum merely means "this Pillow *can* be built
    with RAQM" and is True even when the native lib is absent.
    """
    try:
        from PIL import _imagingft  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return False
    return bool(getattr(_imagingft, "HAVE_RAQM", False))


def raqm_fallback_warning(
    text: str = _SMOKE_ARABIC, font_path: Path | None = None
) -> str | None:
    """Draw Arabic text at layout_engine=RAQM and capture Pillow's fallback.

    Returns the fallback warning text when Pillow downgraded to basic layout,
    ``None`` when the draw used real RAQM shaping (or no RAQM-capable font
    was supplied).
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:  # noqa: BLE001
        return "Pillow unavailable"
    try:
        if font_path is None:
            font_path = FACTORY_PACKAGE_ROOT / "fonts" / "Amiri-Regular.ttf"
        if not font_path.is_file():
            return None
        font = ImageFont.truetype(
            str(font_path), 24, layout_engine=ImageFont.Layout.RAQM
        )
        image = Image.new("L", (200, 60), 255)
        draw = ImageDraw.Draw(image)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            draw.text((10, 10), text, font=font, fill=0)
        for item in caught:
            if "Falling back to basic layout" in str(item.message):
                return str(item.message)
    except Exception:  # noqa: BLE001
        return None
    return None


def check_raqm() -> DoctorSection:
    checks: list[DoctorCheck] = []
    native = native_raqm_available()
    details: dict[str, Any] = {
        "native_raqm_flag": native,
        "module_flag_is_unreliable": True,
    }
    if not native:
        checks.append(
            DoctorCheck(
                id="render.raqm-native",
                name="RAQM renderer (native)",
                subsystem="rendering",
                status=DoctorStatus.FAIL,
                message=(
                    "Native RAQM/libraqm not available; Pillow falls back to basic "
                    "layout (no Arabic shaping/bidi). Vendored backend's _HAS_RAQM "
                    "flag is a false positive here."
                ),
                required=False,
                details=details,
                remediation=(
                    "Install Pillow built with libraqm "
                    "(e.g. 'pip install --upgrade pillow' wheel with raqm, or system "
                    "libraqm: apt install libraqm0 / brew install libraqm); "
                    "render with --backend weasyprint meanwhile"
                ),
            )
        )
        return DoctorSection(id="rendering-raqm", name="RAQM Renderer", checks=checks)

    fallback = raqm_fallback_warning()
    details["fallback_warning"] = fallback
    if fallback:
        checks.append(
            DoctorCheck(
                id="render.raqm-native",
                name="RAQM renderer (native)",
                subsystem="rendering",
                status=DoctorStatus.FAIL,
                message=f"RAQM requested but Pillow downgraded: {fallback}",
                required=False,
                details=details,
                remediation="Reinstall a Pillow wheel with libraqm support.",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                id="render.raqm-native",
                name="RAQM renderer (native)",
                subsystem="rendering",
                status=DoctorStatus.PASS,
                message="Native RAQM available; Arabic shaping smoke draw succeeded",
                required=False,
                details=details,
            )
        )
    return DoctorSection(id="rendering-raqm", name="RAQM Renderer", checks=checks)


def _module_importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001
        return False
