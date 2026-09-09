"""Run orchestration: the unified data factory.

Ties ingest -> render -> distort -> export -> manifest together per
MERGE_PLAN.md phase 6-7. Every (document, variant) job is independent and
seeded from source content + explicit fields, so output is identical for
1 worker or N workers, and interrupted runs resume from the manifest.

Seed modes:
  v1                    unified BLAKE2b per-stage scheme (seed.derive)
  ocr_benchmark         System A legacy derivation + A engine (per-sample seed)
  arabic_scan_factory   System B legacy derivation + B composite backend
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import __version__
from .distort import atomic as atomic_engine
from .distort import qc as qc_engine
from .distort import scan_composite
from .export import jpeg_images_to_pdf, save_page_png
from .ingest.image_file import ImageSource, is_image_file, load_image_source
from .ingest.text_file import TextSource, load_text_source
from .manifest import read_manifest, write_manifest_csv, write_manifest_jsonl
from .profiles import ProfileBook, load_profile_book
from .provenance.hashing import config_hash, sha256_file
from .provenance.integrity import atomic_target
from .seed import derive as seed_v1
from .seed import legacy as seed_legacy

SEED_MODES = ("v1", "ocr_benchmark", "arabic_scan_factory")


@dataclass
class RunConfig:
    run_id: str
    runs_root: Path
    run_dir: Path
    profiles: list[str]
    variants: int
    base_seed: int
    seed_mode: str
    backend: str
    workers: int
    export_pdf: bool
    export_png: bool
    max_pages: int
    resume: bool
    config_hash: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() == ".txt"


def _load_profile_book_for(seed_mode: str):
    return load_profile_book(include_scan_factory=True)


def plan_run(
    inputs: list[Path],
    runs_root: Path,
    profile_names: list[str],
    variants: int,
    base_seed: int,
    seed_mode: str,
    backend: str,
    workers: int,
    export_pdf: bool,
    export_png: bool,
    max_pages: int,
    resume: bool = False,
    run_id: str | None = None,
) -> RunConfig:
    """Validate arguments and create the run directory (new or resumable)."""
    if seed_mode not in SEED_MODES:
        raise ValueError(f"unknown seed mode '{seed_mode}' (have: {SEED_MODES})")
    if variants < 1:
        raise ValueError("--variants must be positive")
    book = _load_profile_book_for(seed_mode)
    for name in profile_names:
        book.profile(name)

    config = {
        "inputs": [str(p) for p in inputs],
        "profiles": profile_names,
        "variants": variants,
        "base_seed": base_seed,
        "seed_mode": seed_mode,
        "backend": backend,
        "export_pdf": export_pdf,
        "export_png": export_png,
        "max_pages": max_pages,
        "factory_version": __version__,
    }
    config_hash_full = config_hash(config)
    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}_{config_hash_full[:8]}"
    runs_root = Path(runs_root)
    run_dir = runs_root / run_id
    if run_dir.exists():
        if not resume:
            raise FileExistsError(
                f"run directory already exists: {run_dir} (pass --resume to continue it)"
            )
        if not (run_dir / "manifest.jsonl").is_file():
            raise FileExistsError(
                f"run directory has no manifest.jsonl to resume: {run_dir}"
            )
    else:
        if resume:
            raise FileNotFoundError(f"cannot resume missing run directory: {run_dir}")
        (run_dir / "scans").mkdir(parents=True)
    run_config = RunConfig(
        run_id=run_id,
        runs_root=runs_root,
        run_dir=run_dir,
        profiles=list(profile_names),
        variants=variants,
        base_seed=int(base_seed),
        seed_mode=seed_mode,
        backend=backend,
        workers=int(workers),
        export_pdf=export_pdf,
        export_png=export_png,
        max_pages=int(max_pages),
        resume=resume,
        config_hash=config_hash_full,
    )
    with atomic_target(run_dir / "run_config.json") as tmp:
        Path(tmp).write_text(
            json.dumps(
                {"run_id": run_id, "config": config, "config_hash": config_hash_full},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return run_config


def _expand_inputs(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        item = Path(item)
        if item.is_dir():
            files.extend(sorted(p for p in item.iterdir() if p.is_file()))
        elif item.is_file():
            files.append(item)
        else:
            raise FileNotFoundError(f"input not found: {item}")
    return files


def _job_seeds(
    rc: RunConfig,
    source_sha: str,
    doc_id: str,
    page_index: int,
    variant_index: int,
    profile_name: str,
    stage: str = "",
    severity: str = "",
) -> int:
    if rc.seed_mode == "v1":
        return seed_v1.derive_seed(
            rc.base_seed,
            source_sha,
            document_id=doc_id,
            page_index=page_index,
            variant_index=variant_index,
            profile=profile_name,
            distortion_stage=stage,
            severity=severity,
        )
    if rc.seed_mode == "arabic_scan_factory":
        return seed_legacy.page_seed_arabic_scan_factory(
            source_sha, variant_index, profile_name, page_index, base_seed=rc.base_seed
        )
    raise ValueError(f"seed mode '{rc.seed_mode}' is handled by the legacy path")


def _render_clean(rc: RunConfig, doc_dir: Path, source: Any, style_seed: int):
    """Render clean artifacts for one document. Returns (clean_pdf, page_pngs, layout, renderer, searchable)."""
    from .render import get_backend

    backend = get_backend(rc.backend)
    clean_dir = doc_dir / "clean"
    if isinstance(source, TextSource):
        result = backend.render(source.text, None, clean_dir, style_seed, dpi=150)
        clean_pdf = result.clean_pdf
        pages = list(result.pages)
        if clean_pdf is None and pages:
            # raqm backend: image-only clean PDF from rendered pages
            clean_pdf = clean_dir / "clean.pdf"
            _pages_to_pdf(pages, clean_pdf, dpi=150)
        if not pages and clean_pdf is not None:
            # weasyprint backend: rasterize the clean PDF into page images
            if hasattr(backend, "page_images"):
                pages = backend.page_images(clean_pdf, clean_dir / "pages")
        return clean_pdf, pages, result.layout, result.renderer, result.searchable
    # image source: pristine bytes are the clean page
    clean_dir.mkdir(parents=True, exist_ok=True)
    page_png = clean_dir / "pages" / "page_000000.png"
    page_png.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source.path) as img:
        img.save(page_png)
    clean_pdf = None
    if rc.export_pdf or True:
        clean_pdf = clean_dir / "clean.pdf"
        with tempfile.TemporaryDirectory() as tmp:
            buf = Path(tmp) / "page.jpg"
            with Image.open(page_png) as img:
                img.convert("RGB").save(buf, "JPEG", quality=96)
            jpeg_images_to_pdf([buf], clean_pdf, dpi=150)
    return clean_pdf, [page_png], {"source": "image"}, "passthrough", False


def _distort_page(
    rc: RunConfig,
    book: ProfileBook,
    profile_name: str,
    variant_index: int,
    img: Image.Image,
    source_sha: str,
    doc_id: str,
    page_index: int,
    verso: np.ndarray | None,
) -> tuple[Image.Image, list[dict], int, dict | None]:
    profile = book.profile(profile_name)
    if profile.composite or rc.seed_mode == "arabic_scan_factory":
        seed = _job_seeds(
            rc, source_sha, doc_id, page_index, variant_index, profile_name
        )
        params = {
            "damage": profile.damage,
            "paper": profile.paper,
            "photocopy_generation": profile.photocopy_generations,
            "color": profile.color,
        }
        out = scan_composite.degrade(img, params, seed)
        return out, [{"stage": "scan_composite", "profile": profile_name}], seed, None
    # atomic path (System A engine)
    if rc.seed_mode == "ocr_benchmark":
        seed = seed_legacy.derive_seed_ocr_benchmark(
            rc.base_seed,
            source_sha,
            distortion="",
            severity="",
            profile=profile_name,
            variant=variant_index,
        )
        rng = np.random.default_rng(seed)
        steps = profile.steps
        out_arr = np.asarray(img)
        applied: list[dict[str, Any]] = []
        for step in steps:
            spec = book.spec(step.distortion)
            out_arr = atomic_engine.apply_distortion(
                step.distortion,
                out_arr,
                spec.params(step.severity),
                rng,
                {"verso": verso},
            )
            applied.append({"distortion": step.distortion, "severity": step.severity})
        return Image.fromarray(out_arr), applied, seed, profile.qc or None
    # unified v1: independent per-stage seeds
    out_arr = np.asarray(img)
    applied: list[dict[str, Any]] = []  # type: ignore
    first_seed = 0
    for index, step in enumerate(profile.steps):
        stage_seed = _job_seeds(
            rc,
            source_sha,
            doc_id,
            page_index,
            variant_index,
            profile_name,
            stage=step.distortion,
            severity=step.severity,
        )
        first_seed = first_seed or stage_seed
        rng = np.random.default_rng(stage_seed)
        spec = book.spec(step.distortion)
        out_arr = atomic_engine.apply_distortion(
            step.distortion, out_arr, spec.params(step.severity), rng, {"verso": verso}
        )
        applied.append(
            {
                "distortion": step.distortion,
                "severity": step.severity,
                "seed": stage_seed,
            }
        )
    return Image.fromarray(out_arr), applied, first_seed, profile.qc or None


def _gt_for(
    doc_id: str, source: Any, run_dir: Path
) -> tuple[str, str] | tuple[None, None]:
    gt_dir = run_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(source, TextSource):
        target = gt_dir / f"{doc_id}.txt"
        if not target.exists():
            shutil.copyfile(source.path, target)
        return str(target), sha256_file(target)
    sibling = source.path.with_suffix(".txt")
    if sibling.is_file():
        target = gt_dir / f"{doc_id}.txt"
        if not target.exists():
            shutil.copyfile(sibling, target)
        return str(target), sha256_file(target)
    return None, None


def _process_document(
    rc: RunConfig, book: ProfileBook, source_path: Path
) -> list[dict]:
    """All variants for one source document. Fully self-contained so it can
    run inside a worker process."""
    run_dir = rc.run_dir
    rows: list[dict] = []
    created = _utc_now()

    source: TextSource | ImageSource
    if _is_text_file(source_path):
        source = load_text_source(source_path)
        source_type = "text"
    elif is_image_file(source_path):
        source = load_image_source(source_path)
        source_type = "image"
    else:
        return [
            {
                "run_id": rc.run_id,
                "document_id": "",
                "source_ref": str(source_path),
                "source_type": "unknown",
                "status": "error",
                "error": "unsupported input type (want .txt or an image file)",
                "created_utc": created,
            }
        ]

    doc_id = source.document_id
    doc_dir = run_dir / "documents" / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    # pristine source copy
    sources_dir = run_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    source_copy = sources_dir / f"{doc_id}__{source_path.name}"
    if not source_copy.exists():
        shutil.copyfile(source_path, source_copy)

    gt_path, gt_sha = _gt_for(doc_id, source, run_dir)

    try:
        clean_pdf, clean_pages, layout, renderer, searchable = _render_clean(
            rc, doc_dir, source, style_seed=rc.base_seed
        )
    except Exception as exc:  # renderer unavailability / bad document: isolate
        return [
            {
                "run_id": rc.run_id,
                "document_id": doc_id,
                "source_type": source_type,
                "source_ref": str(source_path),
                "source_sha256": source.byte_sha256,
                "renderer": rc.backend,
                "status": "error",
                "error": repr(exc),
                "created_utc": created,
            }
        ]
    clean_sha = sha256_file(clean_pdf) if clean_pdf else ""
    completed: set[tuple[str, int]] = set()
    if rc.resume:
        manifest_file = run_dir / "manifest.jsonl"
        if manifest_file.exists():
            for row in read_manifest(manifest_file):
                if row.get("document_id") == doc_id and row.get("status") == "ok":
                    completed.add(
                        (str(row.get("variant_id")), int(row.get("page_index", -1)))
                    )
    page_index = 0
    for page in clean_pages:
        for variant_index in range(rc.variants):
            profile_name = rc.profiles[variant_index % len(rc.profiles)]
            variant_id = f"v{variant_index:02d}__{profile_name}"
            if (variant_id, page_index) in completed:
                continue
            profile = book.profile(profile_name)
            with Image.open(page) as img:
                clean_img = img.convert("RGB").copy()
            if profile.composite or rc.seed_mode == "arabic_scan_factory":
                work_img = scan_composite.image_at_dpi(
                    clean_img, _png_dpi(clean_img), profile.dpi_target
                )
            else:
                work_img = clean_img
            verso = None
            if len(clean_pages) > 1 and page_index + 1 < len(clean_pages):
                with Image.open(clean_pages[page_index + 1]) as nxt:
                    verso = np.asarray(nxt.convert("RGB"))
            try:
                degraded, steps_applied, seed, qc_thresholds = _distort_page(
                    rc,
                    book,
                    profile_name,
                    variant_index,
                    work_img,
                    source.byte_sha256,
                    doc_id,
                    page_index,
                    verso,
                )
            except Exception as exc:  # per-page failure isolation
                rows.append(
                    {
                        "run_id": rc.run_id,
                        "document_id": doc_id,
                        "page_index": page_index,
                        "variant_id": variant_id,
                        "source_type": source_type,
                        "source_ref": str(source_path),
                        "source_sha256": source.byte_sha256,
                        "profile": profile_name,
                        "status": "error",
                        "error": repr(exc),
                        "created_utc": created,
                    }
                )
                continue
            qc_record = None
            if qc_thresholds:
                clean_arr = np.asarray(clean_img)
                distorted_arr = (
                    np.asarray(degraded.convert("RGB"))
                    if degraded.mode != "L"
                    else np.asarray(degraded)
                )
                try:
                    readability = qc_engine.assess(
                        (
                            distorted_arr
                            if distorted_arr.ndim == 3
                            else np.asarray(degraded)
                        ),
                        clean_arr,
                        min_contrast_ratio=float(
                            qc_thresholds.get("min_contrast_ratio", 0.0)
                        ),
                        min_sharpness_ratio=float(
                            qc_thresholds.get("min_sharpness_ratio", 0.0)
                        ),
                        min_ink_ratio=float(qc_thresholds.get("min_ink_ratio", 0.0)),
                        max_ink_ratio=float(qc_thresholds.get("max_ink_ratio", 1e9)),
                    )
                    qc_record = readability.as_dict()
                except Exception:
                    qc_record = None

            scans_dir = run_dir / "scans"
            variant_dir = scans_dir / f"{doc_id}__{variant_id}"
            variant_dir.mkdir(parents=True, exist_ok=True)
            out_png: Path | None = None
            if rc.export_png:
                out_png = save_page_png(
                    degraded, variant_dir / f"page_{page_index:06d}.png"
                )
            output_sha = ""
            output_path = ""
            if rc.export_pdf or out_png is None:
                page_pdf = variant_dir / f"page_{page_index:06d}.pdf"
                with tempfile.TemporaryDirectory() as tmp:
                    buf = Path(tmp) / "page.jpg"
                    degraded.convert("RGB").save(
                        buf, "JPEG", quality=int(profile.jpeg_quality)
                    )
                    jpeg_images_to_pdf([buf], page_pdf, dpi=int(profile.dpi_target))
                output_sha = sha256_file(page_pdf)
                output_path = str(page_pdf.relative_to(run_dir))
            elif out_png is not None:
                output_sha = sha256_file(out_png)
                output_path = str(out_png.relative_to(run_dir))

            rows.append(
                {
                    "run_id": rc.run_id,
                    "document_id": doc_id,
                    "page_index": page_index,
                    "variant_id": variant_id,
                    "source_type": source_type,
                    "source_ref": str(source_path),
                    "source_sha256": source.byte_sha256,
                    "clean_sha256": clean_sha,
                    "output_sha256": output_sha,
                    "renderer": renderer,
                    "profile": profile_name,
                    "profile_schema": profile.schema,
                    "seed": seed,
                    "seed_mode": rc.seed_mode,
                    "base_seed": rc.base_seed,
                    "transform_steps": steps_applied,
                    "dpi": int(profile.dpi_target),
                    "color": profile.color,
                    "jpeg_quality": int(profile.jpeg_quality),
                    "gt_path": gt_path or "",
                    "gt_sha256": gt_sha or "",
                    "status": "ok",
                    "error": "",
                    "qc_passed": bool(qc_record["passed"]) if qc_record else None,
                    "qc": qc_record,
                    "created_utc": created,
                    "output_path": output_path,
                }
            )
        page_index += 1
    return rows


def _pages_to_pdf(pages: list[Path], output: Path, dpi: int = 150) -> None:
    """Assemble rendered page images into an image-only clean PDF."""
    with tempfile.TemporaryDirectory() as tmp:
        jpegs = []
        for page in pages:
            buf = Path(tmp) / (Path(page).stem + ".jpg")
            with Image.open(page) as img:
                img.convert("RGB").save(buf, "JPEG", quality=96)
            jpegs.append(buf)
        jpeg_images_to_pdf(jpegs, output, dpi=dpi)


def _png_dpi(img: Image.Image) -> int:
    dpiinfo = img.info.get("dpi", (150, 150))
    try:
        return int(round(float(dpiinfo[0]))) or 150
    except Exception:
        return 150


def _worker(payload: dict[str, object]) -> list[dict]:
    """Multiprocessing entry point: rebuild RunConfig/ProfileBook per job."""
    rc = RunConfig(**payload["run_config"])  # type: ignore
    book = load_profile_book(include_scan_factory=True)
    return _process_document(rc, book, Path(str(payload["source_path"])))


def generate_run(
    inputs: list[Path],
    runs_root: Path,
    profile_names: list[str] | None = None,
    variants: int = 1,
    base_seed: int = 20260831,
    seed_mode: str = "v1",
    backend: str = "weasyprint",
    workers: int = 1,
    export_pdf: bool = True,
    export_png: bool = True,
    max_pages: int = 8,
    resume: bool = False,
    run_id: str | None = None,
) -> dict:
    """Execute a full run and write run manifest/metadata. Returns metadata."""
    if profile_names is None:
        profile_names = ["05_old_book_medium"]
    rc = plan_run(
        inputs,
        runs_root,
        profile_names,
        variants,
        base_seed,
        seed_mode,
        backend,
        workers,
        export_pdf,
        export_png,
        max_pages,
        resume,
        run_id,
    )
    files = _expand_inputs(inputs)
    if not files:
        raise ValueError("no input files found")

    payloads = [{"run_config": rc.__dict__, "source_path": str(f)} for f in files]
    previous_rows: list[dict] = []
    manifest_file = rc.run_dir / "manifest.jsonl"
    if rc.resume and manifest_file.exists():
        previous_rows = read_manifest(manifest_file)
    all_rows: list[dict] = list(previous_rows)
    new_rows: list[dict] = []
    if rc.workers > 1:
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor(max_workers=rc.workers) as pool:
            for rows in pool.map(_worker, payloads):
                new_rows.extend(rows)
    else:
        for payload in payloads:
            new_rows.extend(
                _process_document(
                    rc,
                    load_profile_book(include_scan_factory=True),
                    Path(str(payload["source_path"])),
                )
            )

    # merge, de-duplicating against previous rows by stable identity
    # (document, source, variant, page) — first occurrence wins, so resumed
    # runs never duplicate ok or error rows.
    def _identity(row: dict) -> tuple:
        return (
            row.get("document_id"),
            row.get("source_ref"),
            row.get("variant_id"),
            row.get("page_index"),
        )

    seen = {_identity(r) for r in all_rows}
    for row in new_rows:
        key = _identity(row)
        if key in seen:
            continue
        seen.add(key)
        all_rows.append(row)

    write_manifest_jsonl(all_rows, rc.run_dir / "manifest.jsonl")
    write_manifest_csv(all_rows, rc.run_dir / "manifest.csv")
    ok = sum(1 for r in all_rows if r.get("status") == "ok")
    errors = [r for r in all_rows if r.get("status") == "error"]
    metadata = {
        "schema_version": 1,
        "factory_version": __version__,
        "run_id": rc.run_id,
        "config_hash": rc.config_hash,
        "seed_mode": rc.seed_mode,
        "base_seed": rc.base_seed,
        "backend": rc.backend,
        "profiles": rc.profiles,
        "variants": rc.variants,
        "counts": {
            "documents": len(files),
            "rows": len(all_rows),
            "ok": ok,
            "errors": len(errors),
        },
        "errors": [
            {"document_id": e.get("document_id"), "error": e.get("error")}
            for e in errors
        ][:50],
        "created_utc": _utc_now(),
    }
    with atomic_target(rc.run_dir / "metadata.json") as tmp:
        Path(tmp).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return metadata
