"""System B composite scan-simulation backend (legacy scan-family profiles).

Vendored math from arabic-scan-factory effects.degrade / image_at_dpi so that
`legacy_seed_mode: arabic_scan_factory` reproduces historical output bytes.
The degradation is a fixed-order chain parameterized by a single `damage`
scalar plus profile fields (paper, photocopy_generation, color) and a page
seed — exactly as in the original tool.
"""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

__all__ = ["image_at_dpi", "degrade"]


def image_at_dpi(image: Image.Image, source_dpi: int, target_dpi: int) -> Image.Image:
    """Resample a page from its source dpi to the profile dpi (B: LANCZOS)."""
    if int(source_dpi) == int(target_dpi):
        return image
    scale = float(target_dpi) / float(source_dpi)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def degrade(image: Image.Image, params: dict, seed: int) -> Image.Image:
    """B's fixed-order composite degradation. `params` mirrors the original
    profile dict: damage, paper ('white'|'aged'), photocopy_generation (0-3),
    color ('rgb'|'grayscale')."""
    rng = np.random.default_rng(seed)
    damage = params["damage"]
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = pixels.shape[:2]
    x = (np.arange(width, dtype=np.float32) - np.float32(width * 0.48)) / np.float32(
        width
    )
    y = (np.arange(height, dtype=np.float32) - np.float32(height * 0.5)) / np.float32(
        height
    )
    illumination = np.float32(1.0) - np.float32(damage * 0.08) * x * x
    illumination = illumination[None, :] - np.float32(damage * 0.07) * (y * y)[:, None]  # type: ignore
    pixels *= illumination[..., None]
    pixels += rng.standard_normal(pixels.shape, dtype=np.float32) * np.float32(
        1.2 + 5 * damage
    )
    np.clip(pixels, 0, 255, out=pixels)
    gray = cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    ghost = cv2.flip(gray, 1)
    pixels -= ((255 - ghost)[..., None]) * np.float32(0.015 + 0.05 * damage)
    pixels *= np.float32(rng.uniform(1 - 0.06 * damage, 1 + 0.03 * damage))
    for _ in range(max(1, int(damage * 8))):
        xpos = int(rng.integers(0, width))
        pixels[:, max(0, xpos - 1) : xpos + 1] -= np.float32(
            rng.uniform(2, 10) * damage
        )
    np.clip(pixels, 0, 255, out=pixels)
    result = Image.fromarray(pixels.astype(np.uint8))
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    if params["paper"] == "aged":
        overlay = Image.new("RGBA", result.size, (238, 220, 175, int(20 + 35 * damage)))
        draw = ImageDraw.Draw(overlay)
        for _ in range(int(8 + 45 * damage)):
            xpos = int(rng.integers(0, width))
            ypos = int(rng.integers(0, height))
            radius = int(rng.integers(1, max(2, int(4 + 14 * damage))))
            draw.ellipse(
                (xpos - radius, ypos - radius, xpos + radius, ypos + radius),
                fill=(100, 65, 25, int(rng.integers(3, 18))),
            )
    draw.rectangle(
        (0, 0, int(width * 0.025), height), fill=(35, 25, 15, int(20 + 60 * damage))
    )
    draw.rectangle(
        (0, 0, width, int(height * 0.008)), fill=(40, 30, 20, int(8 + 25 * damage))
    )
    overlay_array = cv2.GaussianBlur(
        np.asarray(overlay, dtype=np.uint8),
        (0, 0),
        sigmaX=max(1, width / 900),
        borderType=cv2.BORDER_REPLICATE,
    )
    result = Image.alpha_composite(
        result.convert("RGBA"), Image.fromarray(overlay_array, "RGBA")
    ).convert("RGB")
    pixels = np.asarray(result)
    if damage > 0.12:
        pixels = cv2.GaussianBlur(  # type: ignore
            pixels, (0, 0), sigmaX=0.15 + damage, borderType=cv2.BORDER_REPLICATE
        )
    angle = float(rng.uniform(-0.18 - 0.7 * damage, 0.18 + 0.7 * damage))
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    pixels = cv2.warpAffine(  # type: ignore
        pixels,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(245, 241, 230),
    )
    result = Image.fromarray(pixels)
    for _ in range(params["photocopy_generation"]):
        result = (
            ImageEnhance.Contrast(result.convert("L"))
            .enhance(1.08 + damage * 0.2)
            .filter(ImageFilter.UnsharpMask(1, 80, 2))
            .convert("RGB")
        )
    if params["color"] == "grayscale":
        result = result.convert("L")
    return result
