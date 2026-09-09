"""Zero-configuration one-command runner (`clouda-data-factory run`).

Wraps the existing factory pipeline with automatic, sensible defaults:
- input-type detection (text / image / PDF / mixed directory);
- best available text-render backend (weasyprint > raqm > none);
- PDF ingestion via Poppler when available (each page becomes an image
  document); PDFs are skipped with clear per-document errors when it is not;
- deterministic run id derived from input content + configuration, so
  re-running the same command automatically resumes the same run;
- automatic worker count from the host CPU count;
- automatic post-run verification of every recorded output hash;
- a concise final summary (counts, elapsed time, output size, manifest
  paths, unavailable dependencies).

All heavy lifting (ingestion, rendering, distortion, exports, manifests,
isolation, seed modes) stays in `factory.py`; nothing here duplicates it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from .distort import scan_composite  # noqa: F401  (ensures import graph is warm)
from .factory import _expand_inputs, generate_run
from .manifest import read_manifest, write_manifest_csv, write_manifest_jsonl
from .provenance.hashing import sha256_file
from .render.weasyprint_backend import rasterize as _poppler_rasterize

DEFAULT_PROFILES = ["03_normal_office_scan", "05_old_book_medium"]
DEFAULT_VARIANTS = 2
_PDF_SUFFIXES = {".pdf"}


def detect_environment() -> dict[str, Any]:
    """Which optional native dependencies/backends work on this host."""
    text_backend = None
    try:
        from PIL import features

        if features.check("raqm"):
            text_backend = "raqm"
    except Exception:  # pragma: no cover
        pass
    if text_backend is None:
        try:
            import contextlib
            import io as _io

            with (
                contextlib.redirect_stdout(_io.StringIO()),
                contextlib.redirect_stderr(_io.StringIO()),
            ):
                import weasyprint  # noqa: F401

            text_backend = "weasyprint"
        except Exception:  # pragma: no cover
            text_backend = None
    import shutil as _shutil

    return {
        "raqm": text_backend == "raqm",
        "weasyprint": text_backend == "weasyprint",
        "poppler": _shutil.which("pdftoppm") is not None,
        "text_backend": text_backend,
        "unavailable": [
            name
            for name, ok in (
                ("raqm (Pillow Arabic text shaping)", text_backend == "raqm"),
                (
                    "weasyprint/Pango (searchable Arabic PDF)",
                    text_backend == "weasyprint",
                ),
                (
                    "poppler pdftoppm (PDF ingestion/page images)",
                    _shutil.which("pdftoppm") is not None,
                ),
            )
            if not ok
        ],
    }


def auto_workers() -> int:
    """Sensible worker count: up to 4, bounded by host CPUs."""
    return max(1, min(4, os.cpu_count() or 1))


def _classify(files: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    text, images, pdfs = [], [], []
    for f in files:
        suffix = f.suffix.lower()
        if suffix == ".txt":
            text.append(f)
        elif suffix in _PDF_SUFFIXES:
            pdfs.append(f)
        else:
            from .ingest.image_file import IMAGE_SUFFIXES

            if suffix in IMAGE_SUFFIXES:
                images.append(f)
            else:
                images.append(f)  # unknown: let the factory record an error row
    return text, images, pdfs


def _prepare_pdf_pages(pdf: Path, sources_dir: Path, dpi: int = 150) -> list[Path]:
    """Rasterize a PDF into per-page images (Poppler)."""
    out_dir = sources_dir / f"{pdf.stem}_pages"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = _poppler_rasterize(pdf, dpi, out_dir)
    return sorted(pages)


def _content_run_id(files: list[Path], config: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for f in sorted(files, key=lambda p: str(p).lower()):
        digest.update(str(f.name).lower().encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(sha256_file(f).encode("utf-8"))
        digest.update(b"\x1e")
    digest.update(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    return "run_" + digest.hexdigest()[:16]


def auto_run(
    input_path: Path,
    output_root: Path,
    variants: int = DEFAULT_VARIANTS,
    profile_names: list[str] | None = None,
    base_seed: int = 20260831,
    seed_mode: str = "v1",
    backend: str = "auto",
    workers: int | None = None,
    export_pdf: bool = True,
    export_png: bool = True,
    max_pages: int = 8,
    fresh: bool = False,
) -> dict[str, Any]:
    """One-command run. Returns the final summary dictionary."""
    started = time.monotonic()
    env = detect_environment()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    profile_names = list(profile_names or DEFAULT_PROFILES)
    workers = int(workers or auto_workers())
    if backend == "auto":
        backend = (
            env["text_backend"] or "raqm"
        )  # name only; text docs fail soft if unusable

    files = _expand_inputs([Path(input_path)])
    if not files:
        raise ValueError(f"no input files found under {input_path}")

    text_files, image_files, pdf_files = _classify(files)

    # PDFs become per-page image documents when Poppler is available;
    # otherwise they are recorded as failed documents and everything else proceeds.
    pdf_pages: list[Path] = []
    pdf_failures: list[dict[str, Any]] = []
    pdf_page_source: dict[Path, Path] = {}
    staging = output_root / ".pdf_staging"
    if pdf_files:
        if env["poppler"]:
            staging.mkdir(parents=True, exist_ok=True)
            for pdf in pdf_files:
                try:
                    pages = _prepare_pdf_pages(pdf, staging / pdf.stem)
                    pdf_pages.extend(pages)
                    for page in pages:
                        pdf_page_source[page.resolve()] = pdf
                except Exception as exc:
                    pdf_failures.append(
                        _error_row(pdf, f"PDF rasterization failed: {exc!r}")
                    )
        else:
            for pdf in pdf_files:
                pdf_failures.append(
                    _error_row(
                        pdf,
                        "PDF input skipped: poppler pdftoppm is not available on this host",
                    )
                )

    run_inputs = text_files + image_files + pdf_pages
    if not run_inputs:
        raise ValueError(
            "no processable inputs (text rendering and PDF ingestion unavailable "
            "and no image files found)"
        )

    config = {
        "profiles": profile_names,
        "variants": variants,
        "base_seed": base_seed,
        "seed_mode": seed_mode,
        "backend": backend,
        "export_pdf": export_pdf,
        "export_png": export_png,
        "max_pages": max_pages,
        "factory": "auto_run",
    }
    run_id = _content_run_id(run_inputs, config)
    if fresh:
        run_id = f"{run_id}_{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir = output_root / run_id
    resume = run_dir.exists() and (run_dir / "manifest.jsonl").is_file()
    rows_before = read_manifest(run_dir / "manifest.jsonl") if resume else []
    skipped = sum(1 for r in rows_before if r.get("status") == "ok")

    metadata = generate_run(
        inputs=run_inputs,
        runs_root=output_root,
        profile_names=profile_names,
        variants=variants,
        base_seed=base_seed,
        seed_mode=seed_mode,
        backend=backend,
        workers=workers,
        export_pdf=export_pdf,
        export_png=export_png,
        max_pages=max_pages,
        resume=resume,
        run_id=run_id,
    )

    # fold PDF-skip error rows into the manifest so failures are recorded
    extra_rows: list[dict[str, Any]] = list(pdf_failures)
    if extra_rows:
        rows = read_manifest(run_dir / "manifest.jsonl")
        rows.extend(extra_rows)
        write_manifest_jsonl(rows, run_dir / "manifest.jsonl")
        write_manifest_csv(rows, run_dir / "manifest.csv")

    # automatic post-run verification of every recorded successful output
    rows_after = read_manifest(run_dir / "manifest.jsonl")
    mismatches: list[str] = []
    for row in rows_after:
        rel = row.get("output_path")
        if row.get("status") == "ok" and rel:
            out = run_dir / rel
            if not out.is_file() or sha256_file(out) != row.get("output_sha256"):
                mismatches.append(rel)

    if staging.exists() and not any(staging.iterdir()):
        shutil.rmtree(staging, ignore_errors=True)

    ok = sum(1 for r in rows_after if r.get("status") == "ok")
    failed = sum(1 for r in rows_after if r.get("status") == "error")
    total_size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
    elapsed = time.monotonic() - started
    documents = metadata.get("counts", {}).get(
        "documents", len(text_files) + len(image_files)
    )
    page_count = sum(
        1
        for r in rows_after
        if r.get("status") == "ok" and r.get("variant_id") == _first_variant(rows_after)
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "inputs": {
            "text": len(text_files),
            "image": len(image_files),
            "pdf": len(pdf_files),
            "total": len(text_files) + len(image_files) + len(pdf_files),
            "documents_processed": documents,
        },
        "pages": page_count,
        "outputs_ok": ok,
        "outputs_failed": failed,
        "skipped_resumed": skipped,
        "new_this_run": max(0, ok - skipped),
        "verification": {
            "checked": ok,
            "mismatches": mismatches,
            "passed": not mismatches,
        },
        "elapsed_seconds": round(elapsed, 2),
        "output_size_bytes": total_size,
        "workers": workers,
        "profiles": profile_names,
        "seed_mode": seed_mode,
        "backend": backend,
        "environment": env,
        "manifests": {
            "jsonl": str(run_dir / "manifest.jsonl"),
            "csv": str(run_dir / "manifest.csv"),
            "metadata": str(run_dir / "metadata.json"),
        },
    }


def _first_variant(rows: list[dict[str, Any]]) -> str | None:
    for r in rows:
        if r.get("status") == "ok" and r.get("variant_id"):
            return str(r["variant_id"])
    return None


def _error_row(source: Path, message: str) -> dict[str, Any]:
    from datetime import datetime, timezone

    return {
        "run_id": "",
        "document_id": "",
        "source_ref": str(source),
        "source_type": "pdf",
        "status": "error",
        "error": message,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
