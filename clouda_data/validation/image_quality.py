from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageQualityResult:
    ok: bool
    reason: str = ""
    metrics: dict[str, float] | None = None


def validate_image_file(path: str | Path, *, min_bytes: int = 32) -> ImageQualityResult:
    image_path = Path(path)
    if not image_path.exists():
        return ImageQualityResult(False, "missing_image")
    if image_path.stat().st_size < min_bytes:
        return ImageQualityResult(False, "too_small_or_corrupt")
    return ImageQualityResult(True, metrics={"bytes": float(image_path.stat().st_size)})


def detect_blank_from_pixels(
    pixels: list[int], *, almost_blank_threshold: float = 0.995
) -> ImageQualityResult:
    if not pixels:
        return ImageQualityResult(False, "no_pixels")
    whiteish = sum(1 for value in pixels if value >= 250) / len(pixels)
    if whiteish == 1.0:
        return ImageQualityResult(
            False, "completely_blank", {"whiteish_ratio": whiteish}
        )
    if whiteish >= almost_blank_threshold:
        return ImageQualityResult(False, "almost_blank", {"whiteish_ratio": whiteish})
    return ImageQualityResult(True, metrics={"whiteish_ratio": whiteish})


def validate_dimensions(width: int, height: int) -> ImageQualityResult:
    if width <= 0 or height <= 0:
        return ImageQualityResult(False, "invalid_dimensions")
    return ImageQualityResult(
        True, metrics={"width": float(width), "height": float(height)}
    )
