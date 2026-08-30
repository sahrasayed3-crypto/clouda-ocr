from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RenderRequest:
    source_path: str
    output_path: str
    page_number: int = 1
    dpi: int = 200
    output_format: Literal["png", "jpeg", "tiff", "webp"] = "png"
    color_mode: Literal["color", "grayscale", "binary"] = "color"
    max_dimension: int = 8000
    max_pixels: int = 40_000_000
    dry_run: bool = False
    resume: bool = False


@dataclass(frozen=True)
class RenderResult:
    image_path: str
    width: int
    height: int
    dpi: int
    checksum: str
    source_checksum: str = ""
    status: str = "complete"
    error: str | None = None
