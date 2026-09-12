"""Shared helpers for tests/quality (Wave2-Q, Agent Q).

Everything here is CPU-only, offline and deterministic: the same
``(seed, layout, text)`` always renders to identical pixels, and
``make_manifest`` writes the canonical pretraining manifest byte-for-byte
identically for identical inputs.
"""

from __future__ import annotations

import hashlib
import random
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
from PIL import Image, ImageDraw, ImageFont

from clouda_data.pretraining.manifest import write_manifest
from clouda_data.pretraining.schema import DatasetSample, SplitName

__all__ = [
    "render_arabic_page",
    "recompress",
    "resize_img",
    "brightness",
    "add_noise",
    "blank_like",
    "make_manifest",
    "make_row",
    "save_png",
    "sha256_bytes",
]

# Optional real-font path: probe once at import; deterministic either way
# (bars only when the probe fails).
try:  # pragma: no cover - environment dependent
    _FONT: ImageFont.FreeTypeFont | None = ImageFont.truetype("arial.ttf", 10)
except Exception:
    _FONT = None

DEFAULT_SIZE = (128, 96)


def _layout_rng(seed: int, layout: str, text: str) -> random.Random:
    """Deterministic RNG: ints only, no python hash()."""
    layout_text_key = zlib.crc32(f"{layout}|{text}".encode("utf-8")) & 0xFFFFFFFF
    return random.Random((int(seed) & 0xFFFFFFFF) ^ layout_text_key)


def render_arabic_page(
    seed: int,
    layout: str,
    text: str,
    size: tuple[int, int] = DEFAULT_SIZE,
) -> Image.Image:
    """Render a deterministic bar-layout page as a grayscale PIL image."""
    width, height = size
    img = Image.new("L", size, 255)
    draw = ImageDraw.Draw(img)
    rng = _layout_rng(seed, layout, text)
    bar_height = max(2, height // 12)
    for _ in range(3 + rng.randrange(5)):
        y = rng.randrange(0, max(1, height - bar_height))
        x0 = rng.randrange(0, max(1, width // 4))
        x1 = width - rng.randrange(0, max(1, width // 4))
        draw.rectangle([x0, y, x1, y + bar_height], fill=rng.randrange(20, 120))
    if _FONT is not None:
        draw.text((4, 4), text, fill=0, font=_FONT)
    return img


def recompress(img: Image.Image, quality: int = 75) -> Image.Image:
    """JPEG round-trip (lossy) simulating recompression artifacts."""
    import io

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert(img.mode)


def resize_img(img: Image.Image, scale: float) -> Image.Image:
    """Deterministic down/upscale."""
    width, height = img.size
    return img.resize((max(1, int(width * scale)), max(1, int(height * scale))))


def brightness(img: Image.Image, delta: int) -> Image.Image:
    """Shift every pixel by ``delta`` (clamped)."""
    return img.point(lambda value: max(0, min(255, value + delta)))


def add_noise(img: Image.Image, seed: int, count: int) -> Image.Image:
    """Flip ``count`` pseudo-random pixels; deterministic per (seed, count)."""
    out = img.copy()
    rng = random.Random(int(seed))
    width, height = out.size
    for _ in range(max(0, int(count))):
        x = rng.randrange(width)
        y = rng.randrange(height)
        out.putpixel((x, y), rng.randrange(0, 256))
    return out


def blank_like(img: Image.Image) -> Image.Image:
    """Blank (white) image with the same size and mode."""
    return Image.new(img.mode, img.size, 255 if img.mode != "1" else 1)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_png(img: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path


def make_row(
    sample_id: str,
    *,
    source_id: str = "src1",
    image_path: str | None = None,
    text: str | None = None,
    width: int | None = DEFAULT_SIZE[0],
    height: int | None = DEFAULT_SIZE[1],
    target_split: str | SplitName | None = None,
    **overrides: Any,
) -> DatasetSample:
    """Build a :class:`DatasetSample` with sensible defaults for tests."""
    kwargs: dict[str, Any] = {
        "sample_id": sample_id,
        "source_id": source_id,
        "document_id": f"doc-{source_id}",
        "page_id": f"{sample_id}-page",
    }
    if image_path is not None:
        kwargs["image_path"] = image_path
    if text is not None:
        kwargs["text"] = text
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height
    if target_split is not None:
        kwargs["target_split"] = (
            target_split
            if isinstance(target_split, SplitName)
            else SplitName(target_split)
        )
    kwargs.update(overrides)
    return DatasetSample(**kwargs)


def make_manifest(
    root: Path, records: Sequence[DatasetSample | Mapping[str, Any]]
) -> Path:
    """Write the canonical pretraining manifest JSONL under ``root``.

    Returns the manifest path. Rows go through ``DatasetSample.to_dict()``
    and the canonical writer (header ``_schema_version``
    ``clouda.pretraining.manifest.v1``).
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record, DatasetSample):
            rows.append(record.to_dict())
        else:
            rows.append(make_row(**dict(record)).to_dict())
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return write_manifest(root / "manifest.jsonl", rows)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: long-running scale-tier tests (deselect with -m 'not slow')"
    )
