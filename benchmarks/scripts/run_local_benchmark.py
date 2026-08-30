from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fitz  # noqa: E402
from PIL import (  # noqa: E402
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageFont,
    ImageOps,
)

from calculate_metrics import (  # noqa: E402
    calculate_text_metrics,
    normalize_reference_text,
    quality_gate_decision,
)
from compare_local_engines import summarize_results  # noqa: E402
from pdfword.accuracy import estimate_quality_components  # noqa: E402
from pdfword.docx_export import markdown_to_docx  # noqa: E402
from pdfword.models import PageResult  # noqa: E402
from pdfword.ocr_pipeline import process_pdf, render_pdf_page_to_png_bytes  # noqa: E402
from validate_docx import validate_docx_bytes  # noqa: E402

ACCEPTANCE_THRESHOLD = 90.0
SAMPLES_DIR = PROJECT_ROOT / "benchmarks" / "samples"
GROUND_TRUTH_DIR = PROJECT_ROOT / "benchmarks" / "ground_truth"
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
REPORT_DIR = PROJECT_ROOT / "reports" / "local_benchmark"
MANIFEST_PATH = PROJECT_ROOT / "benchmarks" / "manifest.json"


@dataclass(frozen=True)
class Engine:
    name: str
    kind: str
    available: bool
    reason: str
    runner: Callable[[Path, int, str], tuple[str, dict]] | None = None


def _rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = (
        normalize_reference_text(text).replace("\r\n", "\n").replace("\r", "\n")
    )
    path.write_text(normalized, encoding="utf-8", newline="\n")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "tahoma.ttf",
    ):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def _page_image(
    lines: list[str],
    *,
    size: tuple[int, int] = (1240, 1754),
    quality: str = "clear",
    two_pages: bool = False,
) -> Image.Image:
    background = (250, 248, 235) if quality == "old_book" else (255, 255, 255)
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    margin = 90
    if two_pages:
        draw.line(
            (size[0] // 2, 40, size[0] // 2, size[1] - 40), fill=(80, 80, 80), width=3
        )
    draw.rectangle((40, 40, size[0] - 40, size[1] - 40), outline=(80, 80, 80), width=2)
    font = _font(34 if quality != "weak" else 30)
    small = _font(24)
    y = margin
    for index, line in enumerate(lines):
        x = (
            margin
            if not two_pages or index < len(lines) // 2
            else size[0] // 2 + margin
        )
        if two_pages and index == len(lines) // 2:
            y = margin
        draw.text((x, y), line, fill=(20, 20, 20), font=font)
        y += 58
    draw.text(
        (margin, size[1] - 95), "Benchmark footer 2026", fill=(70, 70, 70), font=small
    )
    if quality in {"medium", "weak", "old_book"}:
        image = image.rotate(
            2.2 if quality != "weak" else 6.0, expand=False, fillcolor=background
        )
    if quality in {"medium", "weak"}:
        image = image.filter(
            ImageFilter.GaussianBlur(0.7 if quality == "medium" else 1.8)
        )
        image = ImageEnhance.Contrast(image).enhance(
            0.78 if quality == "medium" else 0.52
        )
    if quality == "old_book":
        image = ImageEnhance.Contrast(image).enhance(0.72)
    if quality == "weak":
        overlay = Image.new("RGB", size, (225, 225, 225))
        image = Image.blend(image, overlay, 0.18)
    return image


def _image_pdf(path: Path, pages: list[Image.Image]) -> None:
    doc = fitz.open()
    rect = fitz.Rect(0, 0, 595, 842)
    for image in pages:
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=78)
        page = doc.new_page(width=rect.width, height=rect.height)
        page.insert_image(rect, stream=out.getvalue())
    doc.save(path, garbage=4, deflate=True)
    doc.close()


def _digital_pdf(path: Path, pages: list[list[str]]) -> None:
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page(width=595, height=842)
        y = 72
        for line in lines:
            page.insert_text((72, y), line, fontsize=12, fontname="helv")
            y += 22
    doc.save(path, garbage=4, deflate=True)
    doc.close()


def _ensure_dataset() -> list[dict]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    def add(
        sample_id: str,
        pdf: Path,
        page: int,
        category: str,
        language: str,
        quality_level: str,
        truth: str,
        expected_route: str,
        notes: str,
    ) -> None:
        truth_path = GROUND_TRUTH_DIR / f"{sample_id}_p{page}.txt"
        _write_text(truth_path, truth)
        manifest.append(
            {
                "sample_id": sample_id,
                "file_path": _rel(pdf),
                "page_number": page,
                "category": category,
                "language": language,
                "quality_level": quality_level,
                "ground_truth_path": _rel(truth_path),
                "expected_local_route": expected_route,
                "is_real_sample": False,
                "notes": notes,
            }
        )

    existing = [
        (
            "repo_digital_ar",
            PROJECT_ROOT / "samples" / "sample_clear_ar.pdf",
            PROJECT_ROOT / "samples" / "sample_clear_ar_ref.txt",
            "digital_pdf",
            "ar",
        ),
        (
            "repo_digital_en",
            PROJECT_ROOT / "samples" / "sample_clear_en.pdf",
            PROJECT_ROOT / "samples" / "sample_clear_en_ref.txt",
            "digital_pdf",
            "en",
        ),
        (
            "repo_complex_ar",
            PROJECT_ROOT / "samples" / "sample_complex_ar.pdf",
            PROJECT_ROOT / "samples" / "sample_complex_ar_ref.txt",
            "digital_pdf",
            "ar",
        ),
        (
            "repo_complex_en",
            PROJECT_ROOT / "samples" / "sample_complex_en.pdf",
            PROJECT_ROOT / "samples" / "sample_complex_en_ref.txt",
            "digital_pdf",
            "en",
        ),
        (
            "repo_clear_ar_scan",
            PROJECT_ROOT / "samples" / "generated" / "clear_ar_clean_scans.pdf",
            PROJECT_ROOT / "samples" / "generated" / "clear_ar_clean_scans_ref.txt",
            "clear_scan",
            "ar",
        ),
        (
            "repo_weak_ar_scan",
            PROJECT_ROOT / "samples" / "generated" / "clear_ar_low_quality_scans.pdf",
            PROJECT_ROOT
            / "samples"
            / "generated"
            / "clear_ar_low_quality_scans_ref.txt",
            "weak_scan",
            "ar",
        ),
    ]
    for sample_id, pdf, truth, category, language in existing:
        if pdf.is_file() and truth.is_file():
            add(
                sample_id,
                pdf,
                1,
                category,
                language,
                "repository_fixture",
                truth.read_text(encoding="utf-8", errors="replace"),
                "digital_text" if category == "digital_pdf" else "local_ocr",
                "Existing repository benchmark fixture; treated as synthetic/provenance-limited.",
            )

    synthetic_specs = [
        (
            "synthetic_digital_mixed",
            "digital_pdf",
            "mixed",
            "clear",
            [
                "Local benchmark mixed document.",
                "Arabic phrase: مرحبا بالعالم.",
                "English phrase: reliable OCR pipeline.",
                "List: one, two, three.",
            ],
        ),
        (
            "synthetic_digital_multicolumn",
            "digital_pdf",
            "en",
            "multi_column",
            [
                "Column A heading      Column B heading",
                "First paragraph text  Second paragraph text",
                "Repeated header       Repeated header",
                "Page footer marker    42",
            ],
        ),
    ]
    for sample_id, category, language, quality, lines in synthetic_specs:
        pdf = SAMPLES_DIR / f"{sample_id}.pdf"
        if not pdf.is_file():
            _digital_pdf(pdf, [lines])
        add(
            sample_id,
            pdf,
            1,
            category,
            language,
            quality,
            "\n".join(lines),
            "digital_text",
            "Generated before engine execution.",
        )

    multi_pdf = SAMPLES_DIR / "synthetic_digital_multipage.pdf"
    multi_pages = [
        ["First synthetic page.", "The order of pages must stay stable."],
        ["Second synthetic page.", "DOCX export must add a page break."],
    ]
    if not multi_pdf.is_file():
        _digital_pdf(multi_pdf, multi_pages)
    for page_no, lines in enumerate(multi_pages, start=1):
        add(
            "synthetic_digital_multipage",
            multi_pdf,
            page_no,
            "digital_pdf",
            "en",
            "multi_page",
            "\n".join(lines),
            "digital_text",
            "Generated multipage digital fixture.",
        )

    scan_cases = [
        (
            "synthetic_clear_en_scan",
            "clear_scan",
            "en",
            "clear",
            [
                "Clear OCR scan sample.",
                "Printed English text.",
                "Font size changes 123.",
            ],
        ),
        (
            "synthetic_medium_en_scan",
            "medium_scan",
            "en",
            "medium",
            [
                "Medium scan sample.",
                "Light noise and mild skew.",
                "JPEG compression is present.",
            ],
        ),
        (
            "synthetic_weak_en_scan",
            "weak_scan",
            "en",
            "weak",
            [
                "Weak scan sample.",
                "Blur, low contrast and skew.",
                "Manual review may be required.",
            ],
        ),
        (
            "synthetic_old_ar_book",
            "old_arabic_book",
            "ar",
            "old_book",
            ["صفحة عربية قديمة", "هذا نص عربي للتقييم.", "حاشية ورقم صفحة ١٢"],
        ),
        (
            "synthetic_two_pages_in_image",
            "special_case",
            "en",
            "medium",
            [
                "Left virtual page.",
                "Line two on left.",
                "Right virtual page.",
                "Line two on right.",
            ],
        ),
        ("synthetic_blank_page", "special_case", "mixed", "blank", [""]),
    ]
    for sample_id, category, language, quality, lines in scan_cases:
        pdf = SAMPLES_DIR / f"{sample_id}.pdf"
        if not pdf.is_file():
            if quality == "blank":
                _image_pdf(pdf, [Image.new("RGB", (1240, 1754), "white")])
            else:
                _image_pdf(
                    pdf,
                    [
                        _page_image(
                            lines,
                            quality=quality,
                            two_pages=sample_id.endswith("image"),
                        )
                    ],
                )
        expected = "" if quality == "blank" else "\n".join(lines)
        route = "blank_page" if quality == "blank" else "local_ocr"
        add(
            sample_id,
            pdf,
            1,
            category,
            language,
            quality,
            expected,
            route,
            "Generated image-only PDF fixture.",
        )

    mixed_pdf = SAMPLES_DIR / "synthetic_mixed_quality_document.pdf"
    mixed_pages = [
        ("clear", ["Mixed quality document.", "Page one is clean."]),
        ("weak", ["Mixed quality document.", "Page two is weak."]),
    ]
    if not mixed_pdf.is_file():
        _image_pdf(
            mixed_pdf,
            [_page_image(lines, quality=quality) for quality, lines in mixed_pages],
        )
    for page_no, (quality, lines) in enumerate(mixed_pages, start=1):
        add(
            "synthetic_mixed_quality_document",
            mixed_pdf,
            page_no,
            "special_case",
            "en",
            quality,
            "\n".join(lines),
            "local_ocr",
            "Generated file with pages of different quality.",
        )

    corrupt_pdf = PROJECT_ROOT / "tests" / "fixtures" / "corrupt.pdf"
    if corrupt_pdf.is_file():
        add(
            "fixture_corrupt_pdf",
            corrupt_pdf,
            1,
            "special_case",
            "mixed",
            "corrupt",
            "",
            "manual_review",
            "Corrupt PDF fixture; expected to fail safely.",
        )

    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _module_available(module: str) -> tuple[bool, str]:
    import importlib.util

    return (
        importlib.util.find_spec(module) is not None,
        (
            "importable"
            if importlib.util.find_spec(module) is not None
            else f"Python module {module!r} is not installed"
        ),
    )


def _windows_short_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    try:
        import ctypes

        buffer = ctypes.create_unicode_buffer(32768)
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return path
        result = windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))
        if result:
            return Path(buffer.value)
    except Exception:
        pass
    return path


def _tesseract_info() -> tuple[bool, str, dict]:
    exe = shutil.which("tesseract")
    if not exe:
        bundled = (
            Path.home()
            / "AppData"
            / "Local"
            / "Programs"
            / "Tesseract-OCR"
            / "tesseract.EXE"
        )
        exe = str(bundled) if bundled.is_file() else ""
    if not exe:
        return (
            False,
            "tesseract executable was not found on PATH or the common local install path",
            {},
        )
    tessdata = Path(exe).parent / "tessdata"
    if not tessdata.is_dir():
        return (
            False,
            f"tessdata directory was not found beside {Path(exe).name}",
            {"exe": exe},
        )
    langs = sorted(path.stem for path in tessdata.glob("*.traineddata"))
    required = {"eng", "ara"}
    missing = sorted(required - set(langs))
    if missing:
        return (
            False,
            f"missing traineddata languages: {', '.join(missing)}",
            {"exe": exe, "tessdata": str(tessdata), "languages": langs},
        )
    return (
        True,
        "available",
        {
            "exe": exe,
            "tessdata": str(_windows_short_path(tessdata)),
            "languages": langs,
        },
    )


def _extract_pypdf(path: Path, page_no: int, _language: str) -> tuple[str, dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return (reader.pages[page_no - 1].extract_text() or "").strip(), {}


def _extract_pymupdf(path: Path, page_no: int, _language: str) -> tuple[str, dict]:
    with fitz.open(path) as document:
        return document.load_page(page_no - 1).get_text("text").strip(), {}


def _extract_pdfminer(path: Path, page_no: int, _language: str) -> tuple[str, dict]:
    from pdfminer.high_level import extract_text

    return (extract_text(str(path), page_numbers=[page_no - 1]) or "").strip(), {}


def _extract_pdfplumber(path: Path, page_no: int, _language: str) -> tuple[str, dict]:
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        return (pdf.pages[page_no - 1].extract_text() or "").strip(), {}


def _preprocess(image: Image.Image, mode: str) -> Image.Image:
    image = image.convert("RGB")
    if mode == "none":
        return image
    if mode == "contrast":
        return ImageEnhance.Contrast(image).enhance(1.65)
    if mode == "denoise":
        return image.filter(ImageFilter.MedianFilter(size=3))
    if mode == "binarize":
        gray = ImageOps.grayscale(image)
        return gray.point(lambda value: 255 if value > 165 else 0, mode="1").convert(
            "RGB"
        )
    if mode == "deskew_basic":
        return image.rotate(-2.0, expand=False, fillcolor="white")
    if mode == "project_best":
        gray = ImageOps.grayscale(image.filter(ImageFilter.MedianFilter(size=3)))
        enhanced = ImageEnhance.Contrast(gray).enhance(1.45)
        return enhanced.point(
            lambda value: 255 if value > 170 else 0, mode="1"
        ).convert("RGB")
    raise ValueError(f"unknown preprocessing mode: {mode}")


def _render_for_ocr(path: Path, page_no: int) -> Image.Image:
    png = render_pdf_page_to_png_bytes(path.read_bytes(), page_no, dpi=220)
    return Image.open(io.BytesIO(png)).convert("RGB")


def _extract_tesseract(
    path: Path, page_no: int, language: str, preprocessing: str, info: dict
) -> tuple[str, dict]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        in {"PATH", "PATHEXT", "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "TMP", "WINDIR"}
    }
    env["TESSDATA_PREFIX"] = info["tessdata"]
    lang = "ara+eng" if language in {"ar", "mixed"} else "eng"
    started_render = time.perf_counter()
    image = _render_for_ocr(path, page_no)
    render_seconds = time.perf_counter() - started_render
    started_preprocess = time.perf_counter()
    processed = _preprocess(image, preprocessing)
    preprocess_seconds = time.perf_counter() - started_preprocess
    with tempfile.TemporaryDirectory(prefix="clouda-local-benchmark-ocr-") as directory:
        image_path = Path(directory) / "page.png"
        processed.save(image_path, format="PNG")
        started_ocr = time.perf_counter()
        completed = subprocess.run(
            [info["exe"], str(image_path), "stdout", "-l", lang, "--psm", "6"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
            shell=False,
            env=env,
        )
        ocr_seconds = time.perf_counter() - started_ocr
    if completed.returncode != 0:
        raise RuntimeError(
            (completed.stderr or "").strip()
            or f"tesseract exited {completed.returncode}"
        )
    return completed.stdout.strip(), {
        "render_seconds": render_seconds,
        "preprocess_seconds": preprocess_seconds,
        "ocr_seconds": ocr_seconds,
        "tesseract_language": lang,
    }


def discover_engines() -> tuple[list[Engine], list[dict]]:
    engines: list[Engine] = []
    availability: list[dict] = []
    for name, module, kind, extractor in [
        ("pypdf", "pypdf", "digital", _extract_pypdf),
        ("pymupdf", "fitz", "digital", _extract_pymupdf),
        ("pdfminer", "pdfminer", "digital", _extract_pdfminer),
        ("pdfplumber", "pdfplumber", "digital", _extract_pdfplumber),
    ]:
        available, reason = _module_available(module)
        engines.append(
            Engine(name, kind, available, reason, extractor if available else None)
        )
        availability.append(
            {
                "engine": name,
                "kind": kind,
                "status": "available" if available else "unavailable",
                "reason": reason,
            }
        )

    tesseract_available, tesseract_reason, tesseract_meta = _tesseract_info()
    if tesseract_available:

        def tesseract_runner(
            path: Path, page_no: int, language: str
        ) -> tuple[str, dict]:
            return _extract_tesseract(path, page_no, language, "none", tesseract_meta)

        engines.append(
            Engine("tesseract_cli", "ocr", True, tesseract_reason, tesseract_runner)
        )
    else:
        engines.append(Engine("tesseract_cli", "ocr", False, tesseract_reason, None))
    availability.append(
        {
            "engine": "tesseract_cli",
            "kind": "ocr",
            "status": "available" if tesseract_available else "unavailable",
            "reason": tesseract_reason,
        }
    )

    for name, module in [
        ("pytesseract", "pytesseract"),
        ("paddleocr", "paddleocr"),
        ("kraken", "kraken"),
        ("surya_ocr", "surya"),
        ("arabic_nougat", "nougat"),
    ]:
        available, reason = _module_available(module)
        engines.append(
            Engine(
                name,
                "ocr",
                False,
                (
                    reason
                    if not available
                    else "module importable but no project-local runner is configured"
                ),
                None,
            )
        )
        availability.append(
            {
                "engine": name,
                "kind": "ocr",
                "status": "unavailable",
                "reason": engines[-1].reason,
            }
        )
    return engines, availability


def _timed_call(
    fn: Callable[[], tuple[str, dict]],
) -> tuple[str, dict, float, float, str]:
    tracemalloc.start()
    started = time.perf_counter()
    try:
        text, metadata = fn()
        error = ""
    except Exception as exc:
        text, metadata = "", {}
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return text, metadata, elapsed, peak / (1024 * 1024), error


def _estimated_quality(text: str) -> float | None:
    if not text.strip():
        return None
    return float(estimate_quality_components(text)["text_quality"])


def run_engine_benchmark(manifest: list[dict], engines: list[Engine]) -> list[dict]:
    rows: list[dict] = []
    for sample in manifest:
        path = PROJECT_ROOT / sample["file_path"]
        truth_path = PROJECT_ROOT / sample["ground_truth_path"]
        reference = (
            truth_path.read_text(encoding="utf-8") if truth_path.is_file() else ""
        )
        has_ground_truth = truth_path.is_file()
        for engine in engines:
            modes = ["not_applicable"]
            if engine.kind == "ocr":
                modes = [
                    "none",
                    "contrast",
                    "denoise",
                    "deskew_basic",
                    "binarize",
                    "project_best",
                ]
            for mode in modes:
                metadata: dict = {}
                text = ""
                elapsed = 0.0
                peak = 0.0
                error = ""
                if not engine.available or engine.runner is None:
                    error = engine.reason
                else:
                    extractor = engine.runner
                    page_number = int(sample["page_number"])
                    language = str(sample["language"])
                    if engine.kind == "ocr":

                        def call_ocr() -> tuple[str, dict]:
                            if engine.name == "tesseract_cli":
                                return _extract_tesseract(
                                    path,
                                    page_number,
                                    language,
                                    mode,
                                    _tesseract_info()[2],
                                )
                            return extractor(path, page_number, language)

                        text, metadata, elapsed, peak, error = _timed_call(call_ocr)
                    else:

                        def call_digital() -> tuple[str, dict]:
                            return extractor(path, page_number, language)

                        text, metadata, elapsed, peak, error = _timed_call(call_digital)
                metrics = (
                    calculate_text_metrics(reference, text)
                    if has_ground_truth and not error
                    else {}
                )
                estimated = _estimated_quality(text) if text and not error else None
                local_exhausted = engine.kind == "ocr"
                decision, reason = quality_gate_decision(
                    character_accuracy=metrics.get("character_accuracy"),
                    has_ground_truth=has_ground_truth,
                    estimated_quality=estimated,
                    local_routes_exhausted=local_exhausted,
                )
                if error:
                    decision = (
                        "unavailable" if not engine.available else "manual_review"
                    )
                    reason = error
                rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "page_number": sample["page_number"],
                        "category": sample["category"],
                        "language": sample["language"],
                        "quality_level": sample["quality_level"],
                        "engine": engine.name,
                        "engine_kind": engine.kind,
                        "preprocessing": mode,
                        "extracted_text": text,
                        "cer": metrics.get("cer"),
                        "wer": metrics.get("wer"),
                        "character_accuracy": metrics.get("character_accuracy"),
                        "word_accuracy": metrics.get("word_accuracy"),
                        "char_substitutions": metrics.get("char_substitutions"),
                        "char_insertions": metrics.get("char_insertions"),
                        "char_deletions": metrics.get("char_deletions"),
                        "word_substitutions": metrics.get("word_substitutions"),
                        "word_insertions": metrics.get("word_insertions"),
                        "word_deletions": metrics.get("word_deletions"),
                        "reference_characters": metrics.get("reference_characters"),
                        "reference_words": metrics.get("reference_words"),
                        "estimated_quality": estimated,
                        "decision": decision,
                        "decision_reason": reason,
                        "elapsed_seconds": elapsed,
                        "render_seconds": metadata.get("render_seconds"),
                        "preprocess_seconds": metadata.get("preprocess_seconds"),
                        "ocr_seconds": metadata.get("ocr_seconds"),
                        "evaluation_seconds": None,
                        "docx_seconds": None,
                        "memory_peak_mb": peak,
                        "cpu_usage": None,
                        "retry_count": (
                            0
                            if decision == "accepted"
                            else (1 if decision == "retry" else 0)
                        ),
                        "error": error,
                    }
                )
    return rows


def evaluate_router(manifest: list[dict], rows: list[dict]) -> dict:
    by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("error") or row.get("cer") is None:
            continue
        by_page[(row["sample_id"], int(row["page_number"]))].append(row)

    evaluations = []
    for sample in manifest:
        path = PROJECT_ROOT / sample["file_path"]
        page_no = int(sample["page_number"])
        candidates = by_page.get((sample["sample_id"], page_no), [])
        best = (
            min(
                candidates,
                key=lambda row: (row["cer"], row["wer"], row["elapsed_seconds"]),
            )
            if candidates
            else None
        )
        fastest_accepted = min(
            [row for row in candidates if row["decision"] == "accepted"],
            key=lambda row: row["elapsed_seconds"],
            default=None,
        )
        try:
            result, _ = process_pdf(
                path.read_bytes(),
                page_no,
                page_no,
                enabled_engines=["direct_pdf_text", "future_ocr_engine"],
            )
            route = result[0].route_used or ""
            chosen = "pypdf" if route == "direct_pdf_text" else route
            accepted = bool(result[0].accepted)
            reason = result[0].review_reason
            error = ""
        except Exception as exc:
            route = "error"
            chosen = "error"
            accepted = False
            reason = f"{type(exc).__name__}: {exc}"
            error = reason
        router_correct = bool(best and chosen in {best["engine"], best.get("engine")})
        acceptable_not_best = bool(fastest_accepted and not router_correct and accepted)
        evaluations.append(
            {
                "sample_id": sample["sample_id"],
                "page_number": page_no,
                "category": sample["category"],
                "expected_local_route": sample["expected_local_route"],
                "router_route": route,
                "router_engine": chosen,
                "router_accepted": accepted,
                "router_reason": reason,
                "best_engine": best["engine"] if best else None,
                "best_cer": best["cer"] if best else None,
                "best_wer": best["wer"] if best else None,
                "fastest_accepted_engine": (
                    fastest_accepted["engine"] if fastest_accepted else None
                ),
                "router_selected_best": router_correct,
                "router_acceptable_not_best": acceptable_not_best,
                "router_wrong": bool(
                    best and not router_correct and not acceptable_not_best
                ),
                "error": error,
            }
        )
    total = len(evaluations)
    return {
        "pages": evaluations,
        "summary": {
            "page_count": total,
            "selection_accuracy": (
                sum(1 for row in evaluations if row["router_selected_best"]) / total
                if total
                else 0.0
            ),
            "best_engine_rate": (
                sum(1 for row in evaluations if row["router_selected_best"]) / total
                if total
                else 0.0
            ),
            "acceptable_not_best_rate": (
                sum(1 for row in evaluations if row["router_acceptable_not_best"])
                / total
                if total
                else 0.0
            ),
            "wrong_rate": (
                sum(1 for row in evaluations if row["router_wrong"]) / total
                if total
                else 0.0
            ),
            "local_under_90_pages": sum(
                1
                for row in evaluations
                if row["best_cer"] is None or float(row["best_cer"]) > 10.0
            ),
        },
    }


def validate_docx_outputs(manifest: list[dict], rows: list[dict]) -> dict:
    by_file: dict[str, list[dict]] = defaultdict(list)
    best_by_page: dict[tuple[str, int], dict] = {}
    for row in rows:
        if row.get("error") or row.get("cer") is None:
            continue
        key = (row["sample_id"], int(row["page_number"]))
        current = best_by_page.get(key)
        if current is None or (row["cer"], row["wer"], row["elapsed_seconds"]) < (
            current["cer"],
            current["wer"],
            current["elapsed_seconds"],
        ):
            best_by_page[key] = row
    for sample in manifest:
        by_file[sample["file_path"]].append(sample)

    outputs = []
    for file_path, samples in by_file.items():
        pages: list[PageResult] = []
        started = time.perf_counter()
        for sample in sorted(samples, key=lambda item: int(item["page_number"])):
            best = best_by_page.get((sample["sample_id"], int(sample["page_number"])))
            text = ""
            accepted = False
            score: float | None = None
            reason: str | None = "local_routes_exhausted"
            model = "none"
            if best:
                text = str(best.get("extracted_text") or "")
                accepted = best.get("decision") == "accepted"
                score = best.get("character_accuracy")
                reason = None if accepted else str(best.get("decision_reason") or "")
                model = str(best.get("engine", "unknown"))
            pages.append(
                PageResult(
                    page_no=int(sample["page_number"]),
                    model_used=model,
                    markdown=text,
                    text_quality_score=score,
                    quality_score=score,
                    requires_manual_review=not accepted,
                    review_reason=reason,
                    accepted=accepted,
                    route_used=model,
                )
            )
        try:
            payload = markdown_to_docx(pages)
            docx_seconds = time.perf_counter() - started
            validation = validate_docx_bytes(payload, expected_pages=len(pages))
            output_path = RESULTS_DIR / "docx" / (Path(file_path).stem + ".docx")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(payload)
        except Exception as exc:
            docx_seconds = time.perf_counter() - started
            validation = {"valid": False, "errors": [f"{type(exc).__name__}: {exc}"]}
            output_path = RESULTS_DIR / "docx" / (Path(file_path).stem + ".docx")
        outputs.append(
            {
                "file_path": file_path,
                "output_docx": _rel(output_path),
                "page_count": len(pages),
                "accepted_pages": sum(1 for page in pages if page.accepted),
                "manual_review_pages": sum(
                    1 for page in pages if page.requires_manual_review
                ),
                "docx_seconds": docx_seconds,
                **validation,
            }
        )
    return {
        "files": outputs,
        "summary": {
            "files": len(outputs),
            "valid": sum(1 for row in outputs if row.get("valid")),
        },
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _overall_stats(rows: list[dict]) -> dict:
    measured = [
        row for row in rows if row.get("cer") is not None and not row.get("error")
    ]
    accepted = [row for row in measured if row.get("decision") == "accepted"]
    manual = [row for row in rows if row.get("decision") == "manual_review"]
    cer = sorted(float(row["cer"]) for row in measured)
    wer = sorted(float(row["wer"]) for row in measured)

    def median(values: list[float]) -> float | None:
        if not values:
            return None
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2

    def p90(values: list[float]) -> float | None:
        if not values:
            return None
        index = min(len(values) - 1, int(round((len(values) - 1) * 0.9)))
        return values[index]

    return {
        "total_records": len(rows),
        "measured_records": len(measured),
        "accepted_rate": len(accepted) / len(measured) if measured else 0.0,
        "manual_review_rate": len(manual) / len(rows) if rows else 0.0,
        "avg_cer": sum(cer) / len(cer) if cer else None,
        "median_cer": median(cer),
        "p90_cer": p90(cer),
        "avg_wer": sum(wer) / len(wer) if wer else None,
        "median_wer": median(wer),
        "p90_wer": p90(wer),
        "fastest_path": min(
            measured, key=lambda row: row["elapsed_seconds"], default=None
        ),
        "slowest_path": max(
            measured, key=lambda row: row["elapsed_seconds"], default=None
        ),
    }


def write_reports(
    manifest: list[dict],
    availability: list[dict],
    rows: list[dict],
    engine_summary: dict,
    router: dict,
    docx: dict,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    overall = _overall_stats(rows)
    files = sorted({row["file_path"] for row in manifest})
    synthetic = sum(1 for row in manifest if not row["is_real_sample"])
    real = sum(1 for row in manifest if row["is_real_sample"])
    unavailable = [row for row in availability if row["status"] == "unavailable"]
    available = [row for row in availability if row["status"] == "available"]

    reports = {
        "01_environment.md": [
            "# Local Benchmark Environment",
            f"- Project root: `{PROJECT_ROOT}`",
            f"- Python: `{sys.executable}`",
            "- External API requests: not used",
            "- Cloud route: future phase only, not implemented or benchmarked",
            "## Engines",
            *[
                f"- {row['engine']}: {row['status']} ({row['reason']})"
                for row in availability
            ],
        ],
        "02_dataset.md": [
            "# Dataset",
            f"- Files: {len(files)}",
            f"- Pages: {len(manifest)}",
            f"- Real samples: {real}",
            f"- Synthetic/provenance-limited samples: {synthetic}",
            "- Ground truth is UTF-8 NFC with LF newlines and is written before engine execution.",
        ],
        "03_digital_extraction.md": [
            "# Digital Extraction",
            "Digital PDF extraction is reported separately from OCR.",
            *[
                f"- {item['category']} / {item['engine']}: avg CER={item['avg_cer']:.2f}, median CER={item['median_cer']:.2f}, accepted={item['accepted_rate']:.2%}"
                for item in engine_summary["engine_summary"]
                if item["engine"] in {"pypdf", "pymupdf", "pdfminer", "pdfplumber"}
            ],
        ],
        "04_ocr_accuracy.md": [
            "# OCR Accuracy",
            *[
                f"- {item['category']} / {item['engine']}: avg CER={item['avg_cer']:.2f}, median CER={item['median_cer']:.2f}, accepted={item['accepted_rate']:.2%}"
                for item in engine_summary["engine_summary"]
                if item["engine"] not in {"pypdf", "pymupdf", "pdfminer", "pdfplumber"}
            ],
        ],
        "05_preprocessing_comparison.md": [
            "# Preprocessing Comparison",
            "Modes tested for local OCR: none, contrast, denoise, deskew_basic, binarize, project_best.",
            "Effect is measured by CER/WER deltas, not by heuristic estimated quality.",
        ],
        "06_engine_comparison.md": [
            "# Engine Comparison",
            *[
                f"- {category}: best engine `{data['engine']}` with CER={data['cer']:.2f}, WER={data['wer']:.2f}"
                for category, data in sorted(engine_summary["best_by_category"].items())
            ],
        ],
        "07_local_router_evaluation.md": [
            "# Local Router Evaluation",
            f"- Pages evaluated: {router['summary']['page_count']}",
            f"- Router selection accuracy: {router['summary']['selection_accuracy']:.2%}",
            f"- Best-engine selection rate: {router['summary']['best_engine_rate']:.2%}",
            f"- Acceptable but not best rate: {router['summary']['acceptable_not_best_rate']:.2%}",
            f"- Wrong route rate: {router['summary']['wrong_rate']:.2%}",
            f"- Pages not reaching 90% locally: {router['summary']['local_under_90_pages']}",
        ],
        "08_docx_validation.md": [
            "# DOCX Validation",
            f"- Files generated: {docx['summary']['files']}",
            f"- Valid DOCX files: {docx['summary']['valid']}",
            "- Validation checks python-docx opening, page breaks, RTL XML, and absence of media/tables.",
        ],
        "09_failures.md": [
            "# Failures",
            *[f"- {row['engine']}: {row['reason']}" for row in unavailable],
            *[
                f"- {row['sample_id']} p{row['page_number']} {row['engine']} {row['preprocessing']}: {row['error']}"
                for row in rows
                if row.get("error") and row["engine"] == "tesseract_cli"
            ][:20],
        ],
        "10_recommendations.md": [
            "# Recommendations",
            "- Keep the 90% gate based on CER/WER when ground truth exists.",
            "- Use digital extraction first for born-digital PDFs; it is faster and avoids unnecessary OCR.",
            "- Treat local OCR pages under 90% as Manual Review after local routes are exhausted.",
            "- Cloud routing is a future phase only and was not implemented, mocked, or tested here.",
        ],
        "11_final_summary.md": [
            "# Final Summary",
            f"- Files/pages tested: {len(files)} files / {len(manifest)} pages",
            f"- Available engines: {', '.join(row['engine'] for row in available) or 'none'}",
            f"- Unavailable engines: {', '.join(row['engine'] for row in unavailable) or 'none'}",
            f"- Avg/Median/P90 CER: {overall['avg_cer']}, {overall['median_cer']}, {overall['p90_cer']}",
            f"- Avg/Median/P90 WER: {overall['avg_wer']}, {overall['median_wer']}, {overall['p90_wer']}",
            f"- Accepted rate: {overall['accepted_rate']:.2%}",
            f"- Manual Review rate: {overall['manual_review_rate']:.2%}",
            "- No cloud provider was added or benchmarked.",
        ],
    }
    for name, lines in reports.items():
        (REPORT_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict:
    for directory in (SAMPLES_DIR, GROUND_TRUTH_DIR, RESULTS_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    manifest = _ensure_dataset()
    engines, availability = discover_engines()
    rows = run_engine_benchmark(manifest, engines)
    for row in rows:
        row["file_path"] = next(
            item["file_path"]
            for item in manifest
            if item["sample_id"] == row["sample_id"]
            and item["page_number"] == row["page_number"]
        )
    results_json = RESULTS_DIR / "local_benchmark_results.json"
    results_csv = RESULTS_DIR / "local_benchmark_results.csv"
    results_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(results_csv, rows)
    engine_summary = summarize_results(rows)
    (RESULTS_DIR / "engine_summary.json").write_text(
        json.dumps(engine_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(RESULTS_DIR / "engine_summary.csv", engine_summary["engine_summary"])
    router = evaluate_router(manifest, rows)
    (RESULTS_DIR / "router_summary.json").write_text(
        json.dumps(router, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    docx = validate_docx_outputs(manifest, rows)
    (RESULTS_DIR / "docx_validation.json").write_text(
        json.dumps(docx, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "files": len({item["file_path"] for item in manifest}),
        "pages": len(manifest),
        "real_samples": sum(1 for item in manifest if item["is_real_sample"]),
        "synthetic_samples": sum(1 for item in manifest if not item["is_real_sample"]),
        "availability": availability,
        "overall": _overall_stats(rows),
        "engine_summary": engine_summary,
        "router": router["summary"],
        "docx": docx["summary"],
        "results_json": _rel(results_json),
        "results_csv": _rel(results_csv),
        "reports": _rel(REPORT_DIR),
        "cloud_route": "future_phase_not_implemented_or_tested",
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    write_reports(manifest, availability, rows, engine_summary, router, docx)
    return summary


if __name__ == "__main__":
    result = run()
    print(
        json.dumps(
            {
                "files": result["files"],
                "pages": result["pages"],
                "real_samples": result["real_samples"],
                "synthetic_samples": result["synthetic_samples"],
                "accepted_rate": result["overall"]["accepted_rate"],
                "manual_review_rate": result["overall"]["manual_review_rate"],
                "results_json": result["results_json"],
                "results_csv": result["results_csv"],
                "reports": result["reports"],
                "cloud_route": result["cloud_route"],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
