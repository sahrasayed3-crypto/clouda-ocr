"""Deterministic image fingerprinting for the quality gate (Agent D).

Integer-only aHash / dHash / pHash over Pillow images plus a shared
single-decode safe-load helper. No numpy: every reduction is plain
Python integer arithmetic so hashes are bit-stable across processes
and platforms (no floating-point accumulation anywhere).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

__all__ = [
    "AHASH_HEX_DIGITS",
    "CONFIRMED_A_MAX",
    "CONFIRMED_D_MAX",
    "CONFIRMED_P_MAX",
    "DCT_TABLE_32",
    "DHASH_HEX_DIGITS",
    "HASH_SIZE",
    "PHASH_HEX_DIGITS",
    "ASPECT_ANCHORS",
    "ASPECT_TOLERANCE",
    "BLANK_STD_MAX",
    "BLANK_MODAL_FRACTION_NUM",
    "BLANK_MODAL_FRACTION_DEN",
    "CONTENT_CROP_THRESHOLD",
    "CONTENT_CROP_MARGIN_VALUE",
    "CONTENT_DOWNSAMPLE_WIDTH",
    "CONTENT_CROP_PAD_NUM",
    "CONTENT_CROP_PAD_DEN",
    "FINGERPRINT_VERSION",
    "FingerprintResult",
    "aspect_bucket",
    "blankish_stats",
    "fingerprint_image",
    "fingerprint_path",
    "fingerprint_path_cached",
    "hamming_distance",
    "ahash_hex",
    "dhash_hex",
    "phash_hex",
    "safe_load_image",
    "safe_load_image_cached",
    "get_max_image_pixels",
    "reset_decode_cache",
    "DECODE_COUNT",
]

# ---------------------------------------------------------------------------
# Public constants (tests and sibling modules import these)
# ---------------------------------------------------------------------------

HASH_SIZE = 8
#: CONFIRMED thresholds (Agent F triple conjunction: d_p<=8 AND d_d<=10 AND d_a<=10)
CONFIRMED_P_MAX = 8
CONFIRMED_D_MAX = 10
CONFIRMED_A_MAX = 10

AHASH_HEX_DIGITS = 16  # 64 bits
DHASH_HEX_DIGITS = 16  # 64 bits
PHASH_HEX_DIGITS = 16  # 64 bits (63 signed coefficients + 1 sign-magnitude bit)

#: Aspect-ratio anchors (w/h), 2% tolerance band, else round(r * 100)
ASPECT_ANCHORS: tuple[float, ...] = (0.707, 0.773, 1.0, 1.294, 1.414)
ASPECT_TOLERANCE = 0.02

#: Blankish detection thresholds (32x32, integer math)
BLANK_STD_MAX = 3
BLANK_MODAL_FRACTION_NUM = 995
BLANK_MODAL_FRACTION_DEN = 1000

#: Content-crop preprocessing parameters
CONTENT_CROP_THRESHOLD = 12
CONTENT_CROP_MARGIN_VALUE = 251
CONTENT_DOWNSAMPLE_WIDTH = 256
CONTENT_CROP_PAD_NUM = 10
CONTENT_CROP_PAD_DEN = 100

_PILLOW_VERSION = Image.__version__
FINGERPRINT_VERSION = f"clouda.quality.imgfp.v1:pillow=={_PILLOW_VERSION}"

# ---------------------------------------------------------------------------
# Committed fixed-point DCT cosine table (Agent D spec)
# ---------------------------------------------------------------------------
# DCT_TABLE_32[u][x] = round(2048 * c(u) * cos((2x + 1) * u * pi / 32)) with
# c(0) = 1/sqrt(2) and c(u>0) = 1. The exact mathematical values were rounded
# once at generation time and committed as integer literals so the pHash never
# depends on libm rounding or float repr. Rows are symmetric around the centre
# (T[u][x] == T[u][31 - x]), so only the first half of each row is committed;
# the accessor below mirrors indices x in [16, 32) to 31 - x.

DCT_TABLE_32: tuple[tuple[int, ...], ...] = (
    (
        1448,
        1448,
        1448,
        1448,
        1448,
        1448,
        1448,
        1448,
        1448,
        1448,
        1448,
        1448,
        1448,
        1448,
        1448,
        1448,
    ),
    (
        2038,
        1960,
        1806,
        1583,
        1299,
        965,
        595,
        201,
        -201,
        -595,
        -965,
        -1299,
        -1583,
        -1806,
        -1960,
        -2038,
    ),
    (
        2009,
        1703,
        1138,
        400,
        -400,
        -1138,
        -1703,
        -2009,
        -2009,
        -1703,
        -1138,
        -400,
        400,
        1138,
        1703,
        2009,
    ),
    (
        1960,
        1299,
        201,
        -965,
        -1806,
        -2038,
        -1583,
        -595,
        595,
        1583,
        2038,
        1806,
        965,
        -201,
        -1299,
        -1960,
    ),
    (
        1892,
        784,
        -784,
        -1892,
        -1892,
        -784,
        784,
        1892,
        1892,
        784,
        -784,
        -1892,
        -1892,
        -784,
        784,
        1892,
    ),
    (
        1806,
        201,
        -1583,
        -1960,
        -595,
        1299,
        2038,
        965,
        -965,
        -2038,
        -1299,
        595,
        1960,
        1583,
        -201,
        -1806,
    ),
    (
        1703,
        -400,
        -2009,
        -1138,
        1138,
        2009,
        400,
        -1703,
        -1703,
        400,
        2009,
        1138,
        -1138,
        -2009,
        -400,
        1703,
    ),
    (
        1583,
        -965,
        -1960,
        201,
        2038,
        595,
        -1806,
        -1299,
        1299,
        1806,
        -595,
        -2038,
        -201,
        1960,
        965,
        -1583,
    ),
    (
        1448,
        -1448,
        -1448,
        1448,
        1448,
        -1448,
        -1448,
        1448,
        1448,
        -1448,
        -1448,
        1448,
        1448,
        -1448,
        -1448,
        1448,
    ),
    (
        1299,
        -1806,
        -595,
        2038,
        -201,
        -1960,
        965,
        1583,
        -1583,
        -965,
        1960,
        201,
        -2038,
        595,
        1806,
        -1299,
    ),
    (
        1138,
        -2009,
        400,
        1703,
        -1703,
        -400,
        2009,
        -1138,
        -1138,
        2009,
        -400,
        -1703,
        1703,
        400,
        -2009,
        1138,
    ),
    (
        965,
        -2038,
        1299,
        595,
        -1960,
        1583,
        201,
        -1806,
        1806,
        -201,
        -1583,
        1960,
        -595,
        -1299,
        2038,
        -965,
    ),
    (
        784,
        -1892,
        1892,
        -784,
        -784,
        1892,
        -1892,
        784,
        784,
        -1892,
        1892,
        -784,
        -784,
        1892,
        -1892,
        784,
    ),
    (
        595,
        -1583,
        2038,
        -1806,
        965,
        201,
        -1299,
        1960,
        -1960,
        1299,
        -201,
        -965,
        1806,
        -2038,
        1583,
        -595,
    ),
    (
        400,
        -1138,
        1703,
        -2009,
        2009,
        -1703,
        1138,
        -400,
        -400,
        1138,
        -1703,
        2009,
        -2009,
        1703,
        -1138,
        400,
    ),
    (
        201,
        -595,
        965,
        -1299,
        1583,
        -1806,
        1960,
        -2038,
        2038,
        -1960,
        1806,
        -1583,
        1299,
        -965,
        595,
        -201,
    ),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (
        -201,
        595,
        -965,
        1299,
        -1583,
        1806,
        -1960,
        2038,
        -2038,
        1960,
        -1806,
        1583,
        -1299,
        965,
        -595,
        201,
    ),
    (
        -400,
        1138,
        -1703,
        2009,
        -2009,
        1703,
        -1138,
        400,
        400,
        -1138,
        1703,
        -2009,
        2009,
        -1703,
        1138,
        -400,
    ),
    (
        -595,
        1583,
        -2038,
        1806,
        -965,
        -201,
        1299,
        -1960,
        1960,
        -1299,
        201,
        965,
        -1806,
        2038,
        -1583,
        595,
    ),
    (
        -784,
        1892,
        -1892,
        784,
        784,
        -1892,
        1892,
        -784,
        -784,
        1892,
        -1892,
        784,
        784,
        -1892,
        1892,
        -784,
    ),
    (
        -965,
        2038,
        -1299,
        -595,
        1960,
        -1583,
        -201,
        1806,
        -1806,
        201,
        1583,
        -1960,
        595,
        1299,
        -2038,
        965,
    ),
    (
        -1138,
        2009,
        -400,
        -1703,
        1703,
        400,
        -2009,
        1138,
        1138,
        -2009,
        400,
        1703,
        -1703,
        -400,
        2009,
        -1138,
    ),
    (
        -1299,
        1806,
        595,
        -2038,
        201,
        1960,
        -965,
        -1583,
        1583,
        965,
        -1960,
        -201,
        2038,
        -595,
        -1806,
        1299,
    ),
    (
        -1448,
        1448,
        1448,
        -1448,
        -1448,
        1448,
        1448,
        -1448,
        -1448,
        1448,
        1448,
        -1448,
        -1448,
        1448,
        1448,
        -1448,
    ),
    (
        -1583,
        965,
        1960,
        -201,
        -2038,
        -595,
        1806,
        1299,
        -1299,
        -1806,
        595,
        2038,
        201,
        -1960,
        -965,
        1583,
    ),
    (
        -1703,
        400,
        2009,
        1138,
        -1138,
        -2009,
        -400,
        1703,
        1703,
        -400,
        -2009,
        -1138,
        1138,
        2009,
        400,
        -1703,
    ),
    (
        -1806,
        -201,
        1583,
        1960,
        595,
        -1299,
        -2038,
        -965,
        965,
        2038,
        1299,
        -595,
        -1960,
        -1583,
        201,
        1806,
    ),
    (
        -1892,
        -784,
        784,
        1892,
        1892,
        784,
        -784,
        -1892,
        -1892,
        -784,
        784,
        1892,
        1892,
        784,
        -784,
        -1892,
    ),
    (
        -1960,
        -1299,
        -201,
        965,
        1806,
        2038,
        1583,
        595,
        -595,
        -1583,
        -2038,
        -1806,
        -965,
        201,
        1299,
        1960,
    ),
    (
        -2009,
        -1703,
        -1138,
        -400,
        400,
        1138,
        1703,
        2009,
        2009,
        1703,
        1138,
        400,
        -400,
        -1138,
        -1703,
        -2009,
    ),
    (
        -2038,
        -1960,
        -1806,
        -1583,
        -1299,
        -965,
        -595,
        -201,
        201,
        595,
        965,
        1299,
        1583,
        1806,
        1960,
        2038,
    ),
)


def _dct_t(u: int, x: int) -> int:
    """Committed cosine literal with mirrored right half."""

    if u < 0 or u > 31 or x < 0 or x > 31:
        raise ValueError(f"DCT index out of range: u={u}, x={x}")
    return DCT_TABLE_32[u][x if x < 16 else 31 - x]


# ---------------------------------------------------------------------------
# Integer pixel grid helpers (pure Pillow, no numpy)
# ---------------------------------------------------------------------------


def _gray_32(image: Image.Image) -> list[list[int]]:
    """Grayscale 32x32 integer pixel grid via deterministic Pillow ops."""

    small = image.convert("L").resize((32, 32), Image.Resampling.BOX)
    flat = list(small.tobytes())
    return [flat[row * 32 : (row + 1) * 32] for row in range(32)]


def ahash_hex(image: Image.Image) -> str:
    """64-bit average hash. Mean via ``// 64`` with ties hashing to 0."""

    small = image.convert("L").resize((HASH_SIZE, HASH_SIZE), Image.Resampling.BOX)
    pixels = list(small.tobytes())
    mean = sum(pixels) // 64
    value = 0
    for index, pixel in enumerate(pixels):
        if pixel > mean:  # ties (== mean) contribute 0
            value |= 1 << index
    return f"{value:016x}"


def dhash_hex(image: Image.Image) -> str:
    """64-bit horizontal-gradient hash over a 9x8 grid (matches manifests.py)."""

    small = image.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.Resampling.BOX)
    pixels = list(small.tobytes())
    value = 0
    for index in range(64):
        row = index // HASH_SIZE
        column = index % HASH_SIZE
        if pixels[row * 9 + column] > pixels[row * 9 + column + 1]:
            value |= 1 << index
    return f"{value:016x}"


def phash_hex(image: Image.Image) -> str:
    """pHash via committed fixed-point DCT on a 32x32 grayscale grid.

    2D DCT-II evaluated with the committed integer cosine table (pure
    integer arithmetic). Coefficients F[0:8][0:8] are kept, DC dropped,
    and the remaining 63 coefficients are thresholded at their
    lower-median (sorted index 31): bit=1 iff coefficient > threshold
    (ties contribute 0). The sign bit of F[0][0] rides in the freed bit
    slot at index 0 so the DC energy still contributes to the distance.
    """

    grid = _gray_32(image)
    # Separable transform, coefficients u/v in 0..7 only:
    # rows pass: R[y][u] = sum_x px[y][x] * T[u][x]
    rows: list[list[int]] = []
    for y in range(32):
        row_pixels = grid[y]
        rows.append(
            [sum(row_pixels[x] * _dct_t(u, x) for x in range(32)) for u in range(8)]
        )
    # columns pass: F[v][u] = sum_y R[y][u] * T[v][y]
    freq: list[list[int]] = []
    for v in range(8):
        freq.append(
            [sum(rows[y][u] * _dct_t(v, y) for y in range(32)) for u in range(8)]
        )
    coeffs = [freq[v][u] for v in range(8) for u in range(8) if not (v == 0 and u == 0)]
    ordered = sorted(coeffs)
    threshold = ordered[31]  # lower-median of 63, ties -> 0 bit
    value = 1 if freq[0][0] < 0 else 0
    bit = 1
    for v in range(8):
        for u in range(8):
            if v == 0 and u == 0:
                continue
            if freq[v][u] > threshold:
                value |= 1 << bit
            bit += 1
    return f"{value:016x}"


def hamming_distance(first: str, second: str) -> int:
    """Bit flips between two hex fingerprints (same width required)."""

    if len(first) != len(second):
        raise ValueError(f"hash width mismatch: {len(first)} vs {len(second)}")
    return (int(first, 16) ^ int(second, 16)).bit_count()


# ---------------------------------------------------------------------------
# Aspect bucket key
# ---------------------------------------------------------------------------


def aspect_bucket(width: int, height: int) -> str:
    """Stable aspect-ratio bucket string (no float hashing)."""

    if width <= 0 or height <= 0:
        raise ValueError(f"invalid dimensions for aspect bucket: {width}x{height}")
    ratio = width / height
    for anchor in ASPECT_ANCHORS:
        if abs(ratio - anchor) / anchor <= ASPECT_TOLERANCE:
            return f"a{round(anchor * 1000):04d}"
    return f"r{round(ratio * 100):04d}"


# ---------------------------------------------------------------------------
# Blankish detection (32x32 integer statistics)
# ---------------------------------------------------------------------------


def blankish_stats(image: Image.Image) -> dict[str, int | bool]:
    """Integer-only statistics used for blank/near-blank detection."""

    grid = _gray_32(image)
    flat = [pixel for row in grid for pixel in row]
    total = len(flat)  # 1024
    mean = sum(flat) // total
    # Integer std proxy: sqrt(var) via sum of squared deviations, integer floor
    var = sum((pixel - mean) ** 2 for pixel in flat) // total
    std = 0
    while (std + 1) * (std + 1) <= var:
        std += 1
    # Modal value within +-6 window
    histogram = [0] * 256
    for pixel in flat:
        histogram[pixel] += 1
    modal = max(range(256), key=histogram.__getitem__)
    modal_count = sum(
        histogram[pixel] for pixel in range(256) if abs(pixel - modal) <= 6
    )
    # >= 0.995 * 1024 pixels within the window -> blankish
    blankish = std <= BLANK_STD_MAX or (
        modal_count * BLANK_MODAL_FRACTION_DEN >= total * BLANK_MODAL_FRACTION_NUM
    )
    return {
        "std": std,
        "modal_value": modal,
        "modal_count": modal_count,
        "total": total,
        "blankish": blankish,
    }


# ---------------------------------------------------------------------------
# Content-crop preprocessing (white-margin removal before fingerprinting)
# ---------------------------------------------------------------------------


def _content_crop(image: Image.Image) -> Image.Image:
    """Crop white margins, pad 10%, fall back to the full page.

    The generous pad keeps enough white margin inside the cropped frame
    that the 32x32 downsampled DCT stays robust to sparse scanner noise
    (a tight crop moves the median-thresholded pHash bits too close to
    the decision boundary; see tests asserting noise robustness).
    """

    gray = image.convert("L")
    width, height = gray.size
    if width <= 0 or height <= 0:
        return image
    down_height = max(1, height * CONTENT_DOWNSAMPLE_WIDTH // width)
    down = gray.resize((CONTENT_DOWNSAMPLE_WIDTH, down_height), Image.Resampling.BOX)
    dw, dh = down.size
    pixels = list(down.tobytes())
    rows_with_ink: list[int] = []
    cols_with_ink: list[int] = []
    for row in range(dh):
        for col in range(dw):
            if pixels[row * dw + col] < CONTENT_CROP_MARGIN_VALUE:
                rows_with_ink.append(row)
                cols_with_ink.append(col)
                break
    if not rows_with_ink:
        # Blank page: keep the full frame (blankish detection handles it)
        return image
    for row in range(dh - 1, -1, -1):
        for col in range(dw):
            if pixels[row * dw + col] < CONTENT_CROP_MARGIN_VALUE:
                rows_with_ink.append(row)
                break
    top = min(rows_with_ink)
    bottom = max(rows_with_ink)
    cols_with_ink = []
    for col in range(dw):
        for row in range(dh):
            if pixels[row * dw + col] < CONTENT_CROP_MARGIN_VALUE:
                cols_with_ink.append(col)
                break
    left = min(cols_with_ink)
    right = max(cols_with_ink)
    # Map back to source coordinates
    scale_x = width / dw
    scale_y = height / dh
    crop_left = max(0, int(left * scale_x))
    crop_top = max(0, int(top * scale_y))
    crop_right = min(width, int((right + 1) * scale_x))
    crop_bottom = min(height, int((bottom + 1) * scale_y))
    if crop_right - crop_left < 4 or crop_bottom - crop_top < 4:
        return image  # degenerate crop, keep full frame
    pad_x = (crop_right - crop_left) * CONTENT_CROP_PAD_NUM // CONTENT_CROP_PAD_DEN
    pad_y = (crop_bottom - crop_top) * CONTENT_CROP_PAD_NUM // CONTENT_CROP_PAD_DEN
    crop_left = max(0, crop_left - pad_x)
    crop_top = max(0, crop_top - pad_y)
    crop_right = min(width, crop_right + pad_x)
    crop_bottom = min(height, crop_bottom + pad_y)
    if (crop_left, crop_top, crop_right, crop_bottom) == (0, 0, width, height):
        return image
    return gray.crop((crop_left, crop_top, crop_right, crop_bottom))


# ---------------------------------------------------------------------------
# Fingerprint assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FingerprintResult:
    """Per-image fingerprint payload (mirrors SampleFingerprint fields)."""

    ahash: str
    dhash: str
    phash: str
    aspect_bucket: str
    blankish: bool
    blank_stats: dict[str, int | bool]
    fingerprint_version: str


def fingerprint_image(image: Image.Image) -> FingerprintResult:
    """Compute all three hashes + bucket + blankish from a decoded image."""

    cropped = _content_crop(image)
    width, height = cropped.size
    stats = blankish_stats(cropped)
    return FingerprintResult(
        ahash=ahash_hex(cropped),
        dhash=dhash_hex(cropped),
        phash=phash_hex(cropped),
        aspect_bucket=aspect_bucket(width, height),
        blankish=bool(stats["blankish"]),
        blank_stats=stats,
        fingerprint_version=FINGERPRINT_VERSION,
    )


def fingerprint_path_cached(path: str | Path) -> FingerprintResult:
    """``fingerprint_path`` sharing the decoded-thumbnail cache (R3-H1)."""

    with safe_load_image_cached(path) as image:
        return fingerprint_image(image)


def fingerprint_path(path: str | Path) -> FingerprintResult:
    """Single-decode fingerprint of the image at ``path``."""

    with safe_load_image(path) as image:
        return fingerprint_image(image)


# ---------------------------------------------------------------------------
# Safe load helper (shared with artifacts.py)
# ---------------------------------------------------------------------------


class ImageDecodeError(ValueError):
    """Raised when an image cannot be decoded safely (single decode)."""


def safe_load_image(path: str | Path) -> Image.Image:
    """Verify-then-thumbnail single decode with hard safety rails.

    - Refuses symbolic links BEFORE any open (caller may map to PATH_SAFE).
    - ``Image.open`` runs under DecompressionBombWarning -> error, with the
      effective pixel ceiling taken from ``get_max_image_pixels()`` (env
      ``CLOUDA_MAX_IMAGE_PIXELS``, default 40M - R4-M1 fix).
    - ``verify()`` consumes the lazy header decode; then a fresh
      ``Image.open`` + ``thumbnail()`` + ``load()`` performs the single
      real pixel decode at bounded resolution (max side 512).
    - Returns an open image (context-managed by the caller).

    Module-level decoded-thumbnail cache: ``safe_load_image_cached`` lets
    the artifact and fingerprint stages share one decode per file per run
    (R3-H1 fix; the un-cached function keeps its exact prior behavior).
    """

    image_path = Path(path)
    if image_path.is_symlink():
        raise ImageDecodeError(f"Refusing to open symbolic link: {image_path}")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        effective_ceiling = get_max_image_pixels()
        previous_ceiling = Image.MAX_IMAGE_PIXELS
        # Only override the global when the caller has not explicitly set a
        # custom ceiling (tests monkeypatch Image.MAX_IMAGE_PIXELS directly).
        caller_customized = previous_ceiling != _PILLOW_DEFAULT_MAX_PIXELS
        if not caller_customized:
            Image.MAX_IMAGE_PIXELS = effective_ceiling
        try:
            with Image.open(image_path) as probe:
                probe.verify()
            with Image.open(image_path) as reopened:
                reopened.thumbnail((512, 512), Image.Resampling.BOX)
                reopened.load()
                return reopened.copy()
        finally:
            Image.MAX_IMAGE_PIXELS = previous_ceiling


#: Effective decompression-bomb ceiling (R4-M1): env-driven, default 40M px
#: matching .env.example's CLOUDA_MAX_IMAGE_PIXELS and the strictest limit
#: already used elsewhere in the repo.
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000


_PILLOW_DEFAULT_MAX_PIXELS = Image.MAX_IMAGE_PIXELS


def get_max_image_pixels() -> int:
    """Effective pixel ceiling from env CLOUDA_MAX_IMAGE_PIXELS."""

    import os

    raw = os.environ.get("CLOUDA_MAX_IMAGE_PIXELS", "")
    if not raw:
        return DEFAULT_MAX_IMAGE_PIXELS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_IMAGE_PIXELS
    return value if value > 0 else DEFAULT_MAX_IMAGE_PIXELS


#: Per-process decoded-thumbnail cache keyed by resolved path + mtime_ns.
#: Bounded to DECODE_CACHE_LIMIT entries; decode_count tracks real decodes
#: for observability (R3-H1).
_DECODE_CACHE: dict[str, tuple[int, Image.Image]] = {}
_DECODE_CACHE_LIMIT = 256
DECODE_COUNT = {"decodes": 0}


def safe_load_image_cached(path: str | Path) -> Image.Image:
    """Cached ``safe_load_image``: one decode per (path, mtime) per process.

    Returns a COPY of the cached thumbnail so callers cannot mutate the
    shared cache entry. Cache hits do not increment DECODE_COUNT.
    """

    resolved = str(Path(path).resolve())
    try:
        mtime = Path(path).stat().st_mtime_ns
    except OSError as exc:
        raise ImageDecodeError(f"Cannot stat image: {path}") from exc
    cached = _DECODE_CACHE.get(resolved)
    if cached is not None and cached[0] == mtime:
        return cached[1].copy()
    image = safe_load_image(path)
    DECODE_COUNT["decodes"] += 1
    if len(_DECODE_CACHE) >= _DECODE_CACHE_LIMIT:
        # Deterministic eviction: drop the first-inserted entry.
        oldest = next(iter(_DECODE_CACHE))
        del _DECODE_CACHE[oldest]
    _DECODE_CACHE[resolved] = (mtime, image)
    return image.copy()


def reset_decode_cache() -> None:
    """Clear the decode cache and counter (for tests / new gate runs)."""

    _DECODE_CACHE.clear()
    DECODE_COUNT["decodes"] = 0
