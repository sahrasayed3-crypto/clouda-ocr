"""Modular CPU-only distortion engine (canonical atomic registry).

Vendored from System A (ocrbench/distortions.py) with the logging import
adapted; the registered distortion functions and their parameter contracts
are unchanged so historical ocr_benchmark outputs remain reproducible.
Three additional ops are ported from System B (arabic-scan-factory):
vertical_streaks, ink_density and photocopy_generation - their math matches
the original composite backend when given B-derived parameters.

Every distortion is a standalone, reusable function with the signature

    f(img: np.ndarray, rng: np.random.Generator, **params) -> np.ndarray

`params` always comes from configuration - no tunable value is
hard-coded here. All randomness is drawn from the supplied `rng`, which the
pipeline seeds deterministically per sample, so the same
(image, distortion, severity, seed) always yields byte-identical output.

Images are uint8 numpy arrays, either 2-D greyscale or 3-D RGB. Functions that
need colour promote greyscale internally and return RGB; the rest preserve the
input shape. Geometric transforms keep the output the same size as the input so
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

import logging

log = logging.getLogger("clouda_data_factory.distort.atomic")

Distortion = Callable[..., np.ndarray]
_REGISTRY: dict[str, Distortion] = {}

_INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


class DistortionError(RuntimeError):
    """Raised when a distortion cannot be applied."""


def register(name: str) -> Callable[[Distortion], Distortion]:
    def decorator(fn: Distortion) -> Distortion:
        if name in _REGISTRY:
            raise DistortionError(f"distortion '{name}' registered twice")
        _REGISTRY[name] = fn
        return fn

    return decorator


def available() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def _as_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _clip(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def paper_value(img: np.ndarray) -> Sequence[float]:
    """Estimate the page background colour from its border pixels.

    Used as the fill for geometric transforms so rotated pages show paper at
    the corners rather than black.
    """
    rgb = _as_rgb(img)
    border = np.concatenate(
        [
            rgb[:8].reshape(-1, 3),
            rgb[-8:].reshape(-1, 3),
            rgb[:, :8].reshape(-1, 3),
            rgb[:, -8:].reshape(-1, 3),
        ]
    )
    med = np.median(border, axis=0)
    if img.ndim == 2:
        return (float(med[0]),)
    return tuple(float(v) for v in med)


def _uniform(rng: np.random.Generator, span: Sequence[float]) -> float:
    lo, hi = float(span[0]), float(span[1])
    return float(rng.uniform(lo, hi))


def _odd(n: int) -> int:
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


def _warp(img: np.ndarray, matrix: np.ndarray, perspective: bool = False) -> np.ndarray:
    h, w = img.shape[:2]
    fill = paper_value(img)
    border = tuple(fill) if img.ndim == 3 else float(fill[0])
    fn = cv2.warpPerspective if perspective else cv2.warpAffine
    return fn(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def _smooth_field(
    rng: np.random.Generator,
    shape: tuple[int, int],
    sigma_frac: float,
    low_res: int = 24,
) -> np.ndarray:
    """A smooth random field in [0, 1], generated at low resolution then blurred.

    Building the field small and upscaling keeps this fast on CPU and makes the
    result resolution-independent.
    """
    h, w = shape
    small = rng.random((low_res, low_res)).astype(np.float32)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    k = _odd(int(max(3, min(h, w) * sigma_frac * 0.5)))
    field = cv2.GaussianBlur(field, (k, k), 0)
    lo, hi = float(field.min()), float(field.max())
    if hi - lo < 1e-6:
        return np.zeros((h, w), np.float32)
    return (field - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# blur
# ---------------------------------------------------------------------------
@register("gaussian_blur")
def gaussian_blur(
    img: np.ndarray, rng: np.random.Generator, *, sigma: float
) -> np.ndarray:
    """Out-of-focus optics."""
    sigma = float(sigma)
    if sigma <= 0:
        return img.copy()
    k = _odd(int(round(sigma * 6)))
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)


@register("motion_blur")
def motion_blur(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    kernel: int,
    angle_range: Sequence[float],
) -> np.ndarray:
    """Linear camera/platen movement during exposure."""
    size = _odd(int(kernel))
    if size <= 1:
        return img.copy()
    angle = _uniform(rng, angle_range)
    k = np.zeros((size, size), np.float32)
    k[size // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, rot, (size, size))  # type: ignore
    total = k.sum()
    if total <= 0:  # pragma: no cover - degenerate kernel
        return img.copy()
    return cv2.filter2D(img, -1, k / total, borderType=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# codec
# ---------------------------------------------------------------------------
@register("jpeg_compression")
def jpeg_compression(
    img: np.ndarray, rng: np.random.Generator, *, quality: int
) -> np.ndarray:
    """Lossy JPEG round-trip: blocking and ringing around glyph edges."""
    quality = int(np.clip(int(quality), 1, 100))
    buf = _as_rgb(img)[:, :, ::-1] if img.ndim == 3 else img  # RGB -> BGR for OpenCV
    ok, encoded = cv2.imencode(".jpg", buf, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise DistortionError("JPEG encoding failed")
    flag = cv2.IMREAD_COLOR if img.ndim == 3 else cv2.IMREAD_GRAYSCALE
    out = cv2.imdecode(encoded, flag)
    if out is None:
        raise DistortionError("JPEG decoding failed")
    return out[:, :, ::-1] if img.ndim == 3 else out


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------
@register("gaussian_noise")
def gaussian_noise(
    img: np.ndarray, rng: np.random.Generator, *, sigma: float
) -> np.ndarray:
    """Additive sensor noise."""
    noise = rng.normal(0.0, float(sigma), img.shape).astype(np.float32)
    return _clip(img.astype(np.float32) + noise)


@register("salt_pepper_noise")
def salt_pepper_noise(
    img: np.ndarray, rng: np.random.Generator, *, amount: float
) -> np.ndarray:
    """Impulse noise - dust, dead pixels, binarisation speckle."""
    out = img.copy()
    h, w = img.shape[:2]
    n = int(round(float(amount) * h * w))
    if n <= 0:
        return out
    ys = rng.integers(0, h, n * 2)
    xs = rng.integers(0, w, n * 2)
    half = n
    out[ys[:half], xs[:half]] = 0  # pepper
    out[ys[half:], xs[half:]] = 255  # salt
    return out


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
@register("low_resolution")
def low_resolution(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    scale: float,
    down_interp: str = "area",
    up_interp: str = "cubic",
) -> np.ndarray:
    """Downscale then upscale - the detail loss that merges dots and diacritics."""
    scale = float(scale)
    if not 0 < scale < 1:
        raise DistortionError(f"low_resolution scale must be in (0,1), got {scale}")
    h, w = img.shape[:2]
    small = cv2.resize(
        img,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=_INTERP[down_interp],
    )
    return cv2.resize(small, (w, h), interpolation=_INTERP[up_interp])


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
@register("rotation")
def rotation(
    img: np.ndarray, rng: np.random.Generator, *, angle_range: Sequence[float]
) -> np.ndarray:
    """Whole-page rotation - the sheet was not square on the platen."""
    angle = _uniform(rng, angle_range)
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return _warp(img, matrix)


@register("skew")
def skew(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    shear_x_range: Sequence[float],
    shear_y_range: Sequence[float],
) -> np.ndarray:
    """Shear distortion from a misaligned feed."""
    sx = _uniform(rng, shear_x_range)
    sy = _uniform(rng, shear_y_range)
    h, w = img.shape[:2]
    matrix = np.array(
        [[1.0, sx, -sx * h / 2.0], [sy, 1.0, -sy * w / 2.0]], dtype=np.float32
    )
    return _warp(img, matrix)


@register("perspective")
def perspective(
    img: np.ndarray, rng: np.random.Generator, *, jitter: float
) -> np.ndarray:
    """Keystone distortion from photographing a page off-axis."""
    h, w = img.shape[:2]
    j = float(jitter)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])  # type: ignore
    offsets = rng.uniform(-j, j, (4, 2)) * np.float32([w, h])  # type: ignore
    dst = (src + offsets).astype(np.float32)
    return _warp(img, cv2.getPerspectiveTransform(src, dst), perspective=True)  # type: ignore


# ---------------------------------------------------------------------------
# tone
# ---------------------------------------------------------------------------
@register("low_contrast")
def low_contrast(
    img: np.ndarray, rng: np.random.Generator, *, alpha: float, beta: float
) -> np.ndarray:
    """Compress the tonal range - washed-out scan."""
    return _clip(img.astype(np.float32) * float(alpha) + float(beta))


@register("brightness")
def brightness(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    dark_range: Sequence[float],
    bright_range: Sequence[float],
    bright_probability: float = 0.5,
) -> np.ndarray:
    """Global exposure error, modelled as a multiplicative gain.

    Gain rather than an additive offset: page paper already sits at 255, so
    adding a positive constant clips to no visible change. A gain above 1.0
    lifts the *ink* toward the paper (over-exposure), a gain below 1.0 pulls
    the whole page down (under-exposure). The direction is chosen from `rng`,
    so it is part of the sample's deterministic seed.
    """
    brighten = rng.random() < float(bright_probability)
    factor = _uniform(rng, bright_range if brighten else dark_range)
    return _clip(img.astype(np.float32) * factor)


@register("uneven_lighting")
def uneven_lighting(
    img: np.ndarray, rng: np.random.Generator, *, strength: float, sigma_frac: float
) -> np.ndarray:
    """Smooth illumination gradient across the sheet."""
    h, w = img.shape[:2]
    field = _smooth_field(rng, (h, w), float(sigma_frac))
    gain = 1.0 - float(strength) * field
    if img.ndim == 3:
        gain = gain[:, :, None]
    return _clip(img.astype(np.float32) * gain)


@register("shadow")
def shadow(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    strength: float,
    width_frac: Sequence[float],
    softness_frac: float,
) -> np.ndarray:
    """A soft-edged cast shadow along one edge - book gutter or a hand."""
    h, w = img.shape[:2]
    side = int(rng.integers(0, 4))
    frac = _uniform(rng, width_frac)
    mask = np.zeros((h, w), np.float32)
    if side == 0:
        mask[:, : max(1, int(w * frac))] = 1.0
    elif side == 1:
        mask[:, w - max(1, int(w * frac)) :] = 1.0
    elif side == 2:
        mask[: max(1, int(h * frac)), :] = 1.0
    else:
        mask[h - max(1, int(h * frac)) :, :] = 1.0
    k = _odd(int(max(3, min(h, w) * float(softness_frac))))
    mask = cv2.GaussianBlur(mask, (k, k), 0)  # type: ignore
    gain = 1.0 - float(strength) * mask
    if img.ndim == 3:
        gain = gain[:, :, None]
    return _clip(img.astype(np.float32) * gain)


# ---------------------------------------------------------------------------
# media / ageing
# ---------------------------------------------------------------------------
@register("faded_scan")
def faded_scan(
    img: np.ndarray, rng: np.random.Generator, *, ink_lift: float, gamma: float
) -> np.ndarray:
    """Lift the ink toward the paper value - weak toner or aged writing."""
    arr = img.astype(np.float32) / 255.0
    lifted = 1.0 - (1.0 - arr) * (1.0 - float(ink_lift))
    return _clip(np.power(np.clip(lifted, 0, 1), 1.0 / float(gamma)) * 255.0)


@register("paper_degradation")
def paper_degradation(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    tint: Sequence[int],
    grain: float,
    blotch_count: Sequence[int],
    blotch_strength: float,
) -> np.ndarray:
    """Aged paper: warm tint, fibre grain and diffuse foxing blotches."""
    rgb = _as_rgb(img).astype(np.float32)
    h, w = rgb.shape[:2]

    tint_arr = np.asarray(tint, dtype=np.float32) / 255.0
    rgb *= tint_arr[None, None, :]

    if grain > 0:
        rgb += rng.normal(0.0, float(grain), (h, w, 1)).astype(np.float32)

    n = int(rng.integers(int(blotch_count[0]), int(blotch_count[1]) + 1))
    if n > 0 and blotch_strength > 0:
        blotches = np.zeros((h, w), np.float32)
        for _ in range(n):
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            radius = int(rng.integers(int(min(h, w) * 0.02), int(min(h, w) * 0.09) + 1))
            cv2.circle(blotches, (cx, cy), radius, 1.0, -1)
        k = _odd(int(min(h, w) * 0.05))
        blotches = cv2.GaussianBlur(blotches, (k, k), 0)  # type: ignore
        rgb *= (1.0 - float(blotch_strength) * blotches)[:, :, None]

    return _clip(rgb)


@register("stains")
def stains(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    count: Sequence[int],
    radius_frac: Sequence[float],
    strength: float,
) -> np.ndarray:
    """Light liquid stains and scanner-glass artefacts."""
    rgb = _as_rgb(img).astype(np.float32)
    h, w = rgb.shape[:2]
    n = int(rng.integers(int(count[0]), int(count[1]) + 1))
    if n <= 0:
        return _clip(rgb)

    mask = np.zeros((h, w), np.float32)
    for _ in range(n):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        rad = int(max(3, _uniform(rng, radius_frac) * min(h, w)))
        axes = (rad, int(rad * _uniform(rng, (0.6, 1.4))))
        cv2.ellipse(mask, (cx, cy), axes, float(rng.uniform(0, 180)), 0, 360, 1.0, -1)
    k = _odd(int(min(h, w) * 0.02))
    mask = cv2.GaussianBlur(mask, (k, k), 0)  # type: ignore
    # Stains darken and warm the paper slightly.
    tint = np.array([1.0, 0.96, 0.88], np.float32)
    factor = 1.0 - float(strength) * mask
    rgb *= factor[:, :, None]
    rgb *= (1.0 - float(strength) * 0.5 * mask)[:, :, None] * tint[None, None, :] + (
        1.0 - (1.0 - float(strength) * 0.5 * mask)[:, :, None]
    )
    return _clip(rgb)


@register("bleedthrough")
def bleedthrough(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    alpha: float,
    blur_sigma: float,
    offset_frac: Sequence[float],
    verso: np.ndarray | None = None,
) -> np.ndarray:
    """Text from the reverse of the sheet showing through thin paper.

    `verso` is a real page image supplied by the pipeline; it is mirrored
    horizontally (as the back of a sheet appears from the front), blurred and
    subtracted at low opacity. Falls back to mirroring the page itself if no
    verso is provided.
    """
    h, w = img.shape[:2]
    back = _as_gray(img) if verso is None else _as_gray(verso)
    if back.shape[:2] != (h, w):
        back = cv2.resize(back, (w, h), interpolation=cv2.INTER_AREA)
    back = np.fliplr(back)

    sigma = float(blur_sigma)
    if sigma > 0:
        k = _odd(int(round(sigma * 6)))
        back = cv2.GaussianBlur(back, (k, k), sigmaX=sigma, sigmaY=sigma)

    dx = int(round(float(offset_frac[0]) * w))
    dy = int(round(float(offset_frac[1]) * h))
    if dx or dy:
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])  # type: ignore
        back = cv2.warpAffine(back, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)  # type: ignore

    ink = (255.0 - back.astype(np.float32)) * float(alpha)
    if img.ndim == 3:
        ink = ink[:, :, None]
    return _clip(img.astype(np.float32) - ink)


# ---------------------------------------------------------------------------
# ported from System B (arabic-scan-factory effects.degrade)
# ---------------------------------------------------------------------------
@register("vertical_streaks")
def vertical_streaks(
    img: np.ndarray, rng: np.random.Generator, *, count: int, strength: float
) -> np.ndarray:
    """Dark vertical scanner streaks (B: 1px columns, count scaled by damage).

    Matches effects.degrade: for each streak, a random x column is darkened by
    ``uniform(2, 10) * strength`` over a 2px band.
    """
    out = img.astype(np.float32)
    h, w = out.shape[:2]
    for _ in range(max(1, int(count))):
        xpos = int(rng.integers(0, w))
        out[:, max(0, xpos - 1) : xpos + 1] -= np.float32(
            rng.uniform(2, 10) * float(strength)
        )
    return _clip(out)


@register("ink_density")
def ink_density(
    img: np.ndarray, rng: np.random.Generator, *, low_gain: float, high_gain: float
) -> np.ndarray:
    """Random global ink-density gain (B: uniform(1 - 0.06*d, 1 + 0.03*d))."""
    factor = np.float32(rng.uniform(float(low_gain), float(high_gain)))
    return _clip(img.astype(np.float32) * factor)


@register("photocopy_generation")
def photocopy_generation(
    img: np.ndarray,
    rng: np.random.Generator,
    *,
    generations: int,
    contrast: float,
    unsharp: int,
) -> np.ndarray:
    """Generational photocopy loss (B: contrast enhance + unsharp, repeated).

    Matches effects.degrade's loop: convert to L, ImageEnhance.Contrast
    (1.08 + damage*0.2), UnsharpMask(1, 80, 2), back to RGB — repeated
    `generations` times.
    """
    from PIL import Image, ImageEnhance, ImageFilter

    result = Image.fromarray(_as_rgb(img))
    for _ in range(max(0, int(generations))):
        result = (
            ImageEnhance.Contrast(result.convert("L"))
            .enhance(float(contrast))
            .filter(ImageFilter.UnsharpMask(1, int(unsharp), 2))
            .convert("RGB")
        )
    return np.asarray(result, dtype=np.uint8)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def apply_distortion(
    name: str,
    img: np.ndarray,
    params: Mapping[str, Any],
    rng: np.random.Generator,
    context: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Apply one registered distortion by name.

    `context` carries extras a distortion may need that are not configuration -
    currently only `verso` for bleedthrough.
    """
    if name not in _REGISTRY:
        raise DistortionError(f"unknown distortion '{name}' (have: {available()})")
    kwargs = dict(params)
    if name == "bleedthrough" and context and "verso" in context:
        kwargs["verso"] = context["verso"]
    try:
        out = _REGISTRY[name](img, rng, **kwargs)
    except TypeError as exc:
        raise DistortionError(
            f"distortion '{name}' rejected its configured parameters {sorted(kwargs)}: {exc}"
        ) from exc
    if out.dtype != np.uint8:  # pragma: no cover - defensive
        raise DistortionError(
            f"distortion '{name}' returned dtype {out.dtype}, expected uint8"
        )
    if out.shape[:2] != img.shape[:2]:
        raise DistortionError(
            f"distortion '{name}' changed page size {img.shape[:2]} -> {out.shape[:2]}"
        )
    return out


def apply_profile(
    steps: Sequence[Any],
    img: np.ndarray,
    config,
    rng: np.random.Generator,
    context: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Apply an ordered list of ProfileStep objects in sequence."""
    out = img
    for step in steps:
        spec = config.distortion(step.distortion)
        out = apply_distortion(
            step.distortion, out, spec.params(step.severity), rng, context
        )
    return out
