"""Image quality metrics used by the readability guard and the QC report.

The point of these is narrow: catch samples that are degraded past the point a
human could read them, so the benchmark measures OCR difficulty rather than
impossibility. Each metric is expressed as a *ratio against the clean page*, so
thresholds are independent of font, layout and page density.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def contrast(img: np.ndarray) -> float:
    """Ink-to-paper separation, 0..1.

    Uses robust percentiles rather than min/max so a single speck of salt noise
    does not report a perfect-contrast page.
    """
    g = _gray(img).astype(np.float32)
    lo, hi = np.percentile(g, (2.0, 98.0))
    return float(max(0.0, hi - lo) / 255.0)


def sharpness(img: np.ndarray) -> float:
    """Denoised variance of the Laplacian - high for crisp glyph edges.

    The light prefilter prevents film grain and scanner noise in an original
    from becoming the sharpness baseline. Without it, a still-readable
    downsampled historical page can falsely appear to retain <1% of detail.
    """
    g = _gray(img).astype(np.float32) / 255.0
    g = cv2.GaussianBlur(g, (0, 0), 0.8)  # type: ignore
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def _flat_field(img: np.ndarray) -> np.ndarray:
    """Divide out the low-frequency illumination, leaving ink on flat paper.

    Without this, any illumination effect (shadow, vignette, aged paper) shifts
    a *global* ink threshold and gets misreported as text loss. Flat-fielding is
    what a real adaptive binariser does first, so measuring after it asks the
    question we actually care about: is there ink here, relative to the paper
    immediately around it?
    """
    g = _gray(img).astype(np.float32)
    h, w = g.shape
    # Kernel wide enough to span glyphs and interline gaps but not page-scale
    # lighting, so the background estimate contains no text.
    k = max(3, int(min(h, w) * 0.05) | 1)
    background = cv2.GaussianBlur(g, (k, k), 0)
    background = np.maximum(background, 1.0)
    return np.clip(g / background * 255.0, 0, 255)


def ink_ratio(img: np.ndarray) -> float:
    """Fraction of pixels that read as ink after flat-field correction.

    Illumination-invariant: a shadowed or tinted page reports the same ink
    coverage as the clean original, while genuinely dissolving or flooding
    glyphs moves the number.
    """
    flat = _flat_field(img).astype(np.uint8)
    # Remove isolated sensor speckles before counting foreground. This keeps
    # configured Gaussian noise from being mistaken for a flooded page while
    # preserving connected glyph strokes.
    flat = cv2.medianBlur(flat, 3)  # type: ignore
    # A percentile-derived dark point becomes paper when ink occupies <1% of
    # a sparse line image. Adaptive Gaussian thresholding instead asks whether
    # each pixel is dark relative to its own neighbourhood, preserving sparse
    # HTR lines and remaining invariant to cast shadows and tinted paper.
    block = max(15, (int(min(flat.shape) * 0.08) | 1))
    block = min(block, 151)
    mask = cv2.adaptiveThreshold(
        flat,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        6,
    )
    return float((mask > 0).mean())


@dataclass(frozen=True)
class Readability:
    """Metrics for one distorted page, relative to its clean original."""

    contrast: float
    sharpness: float
    ink_ratio: float
    contrast_ratio: float
    sharpness_ratio: float
    ink_change_ratio: float
    passed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        return d


def _safe_ratio(value: float, base: float) -> float:
    if base <= 1e-9:
        return 1.0
    return float(value / base)


def assess(
    distorted: np.ndarray,
    clean: np.ndarray,
    *,
    min_contrast_ratio: float,
    min_sharpness_ratio: float,
    min_ink_ratio: float,
    max_ink_ratio: float,
) -> Readability:
    """Compare a distorted page against its clean source.

    Ratios below the configured floors mean the page has lost so much contrast
    or edge detail that the text is likely unreadable to a human; an ink ratio
    far from 1.0 means glyphs have either dissolved into the paper or flooded
    it. Thresholds come from `readability:` in configs/distortions.yaml.
    """
    dc, sc, ic = contrast(distorted), sharpness(distorted), ink_ratio(distorted)
    bc, bs, bi = contrast(clean), sharpness(clean), ink_ratio(clean)

    cr = _safe_ratio(dc, bc)
    sr = _safe_ratio(sc, bs)
    ir = _safe_ratio(ic, bi)

    reasons: list[str] = []
    if cr < min_contrast_ratio:
        reasons.append(f"contrast_ratio {cr:.3f} < {min_contrast_ratio}")
    if sr < min_sharpness_ratio:
        reasons.append(f"sharpness_ratio {sr:.4f} < {min_sharpness_ratio}")
    if ir < min_ink_ratio:
        reasons.append(
            f"ink_change_ratio {ir:.3f} < {min_ink_ratio} (glyphs dissolving)"
        )
    if ir > max_ink_ratio:
        reasons.append(f"ink_change_ratio {ir:.3f} > {max_ink_ratio} (page flooding)")

    return Readability(
        contrast=dc,
        sharpness=sc,
        ink_ratio=ic,
        contrast_ratio=cr,
        sharpness_ratio=sr,
        ink_change_ratio=ir,
        passed=not reasons,
        reasons=tuple(reasons),
    )
