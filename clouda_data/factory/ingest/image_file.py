"""Image source ingestion (System A semantics).

An image source is read as raw bytes and hashed; it is decoded only for
processing. Sources are never modified — processing writes re-encoded copies
into the run's own output tree (sources/ keeps pristine byte copies).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..provenance.hashing import sha256_bytes

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".bin"}


class ImageDecodeError(ValueError):
    """Raised when an image source cannot be decoded."""


@dataclass(frozen=True)
class ImageSource:
    path: Path
    byte_sha256: str
    byte_length: int
    width: int
    height: int
    mode: str

    @property
    def document_id(self) -> str:
        return self.byte_sha256[:16]


def is_image_file(path: Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def load_image_source(path: Path) -> ImageSource:
    path = Path(path)
    raw = path.read_bytes()
    try:
        with Image.open(io.BytesIO(raw)) as img:
            width, height = img.size
            mode = img.mode
    except Exception as exc:
        raise ImageDecodeError(f"{path} is not a readable image: {exc}") from exc
    return ImageSource(
        path=path,
        byte_sha256=sha256_bytes(raw),
        byte_length=len(raw),
        width=width,
        height=height,
        mode=mode,
    )
