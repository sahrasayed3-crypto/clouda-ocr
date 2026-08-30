from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError

from clouda_contracts.storage import StorageRoots

SUPPORTED_IMAGES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
SUPPORTED_FORMATS = {"png": "PNG", "jpeg": "JPEG", "tiff": "TIFF", "webp": "WEBP"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class RenderConfig:
    dpi: int = 200
    output_format: Literal["png", "jpeg", "tiff", "webp"] = "png"
    color_mode: Literal["color", "grayscale", "binary"] = "color"
    max_dimension: int = 8000
    max_pixels: int = 40_000_000
    start_page: int = 1
    end_page: int | None = None
    max_pages: int = 1000
    dry_run: bool = False
    resume: bool = False

    def validate(self) -> None:
        if not 72 <= self.dpi <= 600:
            raise ValueError("dpi must be between 72 and 600")
        if self.output_format not in SUPPORTED_FORMATS:
            raise ValueError("unsupported output format")
        if self.color_mode not in {"color", "grayscale", "binary"}:
            raise ValueError("unsupported color mode")
        if not 256 <= self.max_dimension <= 20_000:
            raise ValueError("max_dimension is outside safe limits")
        if not 1_000_000 <= self.max_pixels <= 100_000_000:
            raise ValueError("max_pixels is outside safe limits")
        if self.start_page < 1 or (self.end_page and self.end_page < self.start_page):
            raise ValueError("invalid page range")
        if not 1 <= self.max_pages <= 10_000:
            raise ValueError("max_pages is outside safe limits")
        if self.end_page and self.end_page - self.start_page + 1 > self.max_pages:
            raise ValueError("requested page range exceeds max_pages")


def _normalized(image: Image.Image, config: RenderConfig) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.width * image.height > config.max_pixels:
        ratio = (config.max_pixels / (image.width * image.height)) ** 0.5
        image = image.resize(
            (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
            Image.Resampling.LANCZOS,
        )
    if max(image.size) > config.max_dimension:
        image.thumbnail(
            (config.max_dimension, config.max_dimension), Image.Resampling.LANCZOS
        )
    if config.color_mode == "grayscale":
        return image.convert("L")
    if config.color_mode == "binary":
        return image.convert("L").point(lambda pixel: 255 if pixel >= 128 else 0, "1")
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, "white")
        canvas.alpha_composite(image)
        return canvas.convert("RGB")
    return image.convert("RGB")


def _iter_pages(
    path: Path, config: RenderConfig
) -> Iterator[tuple[int, Image.Image, str, str]]:
    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(path))
        try:
            end = min(len(document), config.end_page or len(document))
            end = min(end, config.start_page + config.max_pages - 1)
            for index in range(config.start_page - 1, end):
                page = document[index]
                bitmap = None
                try:
                    width, height = page.get_size()
                    scale = config.dpi / 72
                    if width * scale * height * scale > config.max_pixels:
                        raise ValueError(
                            f"PDF page {index + 1} exceeds the configured pixel limit"
                        )
                    bitmap = page.render(scale=config.dpi / 72)
                    yield index + 1, bitmap.to_pil().copy(), "pypdfium2", str(
                        getattr(pdfium, "__version__", "unknown")
                    )
                finally:
                    if bitmap is not None and hasattr(bitmap, "close"):
                        bitmap.close()
                    page.close()
        finally:
            document.close()
        return
    if path.suffix.lower() not in SUPPORTED_IMAGES:
        raise ValueError(f"Unsupported source type: {path.suffix}")
    try:
        with Image.open(path) as source:
            frame_count = getattr(source, "n_frames", 1)
            end = min(frame_count, config.end_page or frame_count)
            end = min(end, config.start_page + config.max_pages - 1)
            for index, frame in enumerate(ImageSequence.Iterator(source), start=1):
                if index < config.start_page:
                    continue
                if index > end:
                    break
                frame.load()
                yield index, frame.copy(), "Pillow", Image.__version__
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Source image cannot be decoded: {exc}") from exc


def _atomic_image_save(image: Image.Image, target: Path, fmt: str, dpi: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.stem}-", suffix=target.suffix, dir=target.parent
    )
    os.close(fd)
    temp = Path(temp_name)
    try:
        if fmt in {"JPEG", "WEBP"}:
            image.save(temp, format=fmt, dpi=(dpi, dpi), quality=92)
        else:
            image.save(temp, format=fmt, dpi=(dpi, dpi))
        with temp.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def _write_manifest_atomic(manifest: Path, records: list[dict[str, object]]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temp = manifest.with_suffix(manifest.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(manifest)


def _matches_existing_image(path: Path, expected: Image.Image) -> bool:
    try:
        with Image.open(path) as existing:
            existing.load()
            return (
                existing.size == expected.size
                and existing.convert("RGBA").tobytes()
                == expected.convert("RGBA").tobytes()
            )
    except Exception:
        return False


def render_document(
    source: str | Path,
    *,
    output_root: str | Path | None = None,
    config: RenderConfig | None = None,
    run_id: str | None = None,
) -> Path:
    config = config or RenderConfig()
    config.validate()
    roots = StorageRoots.from_env()
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not _inside(source_path, roots.dataset_root):
        raise PermissionError(
            "Render source must be inside the configured dataset root"
        )
    root = (
        Path(output_root).expanduser().resolve()
        if output_root
        else roots.dataset_root / "rendered"
    )
    if not _inside(root, roots.dataset_root):
        raise PermissionError(
            "Render output must be inside the configured dataset root"
        )
    source_checksum = _sha256(source_path)
    document_id = hashlib.sha256(
        f"{source_checksum}:{source_path.name}".encode()
    ).hexdigest()[:24]
    config_hash = hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True).encode()
    ).hexdigest()
    run = (
        run_id
        or hashlib.sha256(f"{document_id}:{config_hash}".encode()).hexdigest()[:20]
    )
    run_root = root / run
    manifest = run_root / "render_manifest.v1.jsonl"
    completed: dict[int, dict[str, object]] = {}
    if config.resume and manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            loaded_record = json.loads(line)
            if loaded_record.get("status") == "complete":
                uri = str(loaded_record.get("output_uri", ""))
                if not uri.startswith("dataset://"):
                    raise ValueError("Resume manifest contains an unsafe output URI")
                output = roots.dataset_root / uri.removeprefix("dataset://")
                checksum = str(loaded_record.get("output_checksum") or "")
                if (
                    not _inside(output, roots.dataset_root)
                    or not output.is_file()
                    or not checksum
                    or _sha256(output) != checksum
                ):
                    raise ValueError(
                        "Resume manifest output is missing or has a checksum mismatch"
                    )
                completed[int(loaded_record["source_page_number"])] = loaded_record
    records: list[dict[str, object]] = list(completed.values())
    for page_number, raw, renderer, version in _iter_pages(source_path, config):
        if page_number in completed:
            continue
        output = run_root / f"{document_id}-p{page_number:05d}.{config.output_format}"
        record: dict[str, object] = {
            "schema_version": 1,
            "run_id": run,
            "source_document_id": document_id,
            "source_page_number": page_number,
            "source_uri": f"dataset://{source_path.relative_to(roots.dataset_root).as_posix()}",
            "source_checksum": source_checksum,
            "output_uri": f"dataset://{output.relative_to(roots.dataset_root).as_posix()}",
            "output_checksum": None,
            "width": 0,
            "height": 0,
            "dpi": config.dpi,
            "color_mode": config.color_mode,
            "output_format": config.output_format,
            "renderer_name": renderer,
            "renderer_version": version,
            "render_config_hash": config_hash,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "planned" if config.dry_run else "complete",
            "error": None,
        }
        image = _normalized(raw, config)
        record["width"], record["height"] = image.size
        if not config.dry_run:
            if output.exists():
                if not config.resume or not _matches_existing_image(output, image):
                    raise FileExistsError(
                        f"Existing resume output does not match {output}"
                    )
            else:
                _atomic_image_save(
                    image, output, SUPPORTED_FORMATS[config.output_format], config.dpi
                )
            record["output_checksum"] = _sha256(output)
        records.append(record)
        _write_manifest_atomic(manifest, records)
    completion = run_root / "COMPLETE.v1.json"
    if not config.dry_run:
        completion.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": run,
                    "pages": len(records),
                    "host": platform.system(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return manifest


def validate_render_manifest(path: str | Path) -> dict[str, object]:
    manifest = Path(path).expanduser().resolve()
    roots = StorageRoots.from_env()
    failures: list[dict[str, object]] = []
    total = 0
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), 1
    ):
        total += 1
        record = json.loads(line)
        uri = str(record.get("output_uri", ""))
        if not uri.startswith("dataset://"):
            failures.append({"line": line_number, "error": "unsafe output URI"})
            continue
        target = roots.dataset_root / uri.removeprefix("dataset://")
        if not _inside(target, roots.dataset_root) or not target.is_file():
            failures.append({"line": line_number, "error": "missing or escaped output"})
            continue
        if _sha256(target) != record.get("output_checksum"):
            failures.append({"line": line_number, "error": "checksum mismatch"})
            continue
        try:
            with Image.open(target) as image:
                image.verify()
        except Exception as exc:
            failures.append({"line": line_number, "error": f"decode failed: {exc}"})
    return {
        "schema_version": 1,
        "records": total,
        "failures": failures,
        "passed": not failures,
    }
