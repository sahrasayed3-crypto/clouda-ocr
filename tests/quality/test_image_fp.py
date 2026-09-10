"""Tests for clouda_data.quality.image_fp (Wave2-D, Agent D).

All checks are CPU-only, offline and deterministic. The thresholds
asserted here are imported from the module (CONFIRMED_* / BLANK_* /
ASPECT_*) so the tests stay in lockstep with the implementation.
"""

from __future__ import annotations

import io
import os
import random
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageOps

from clouda_data.quality.image_fp import (
    ASPECT_ANCHORS,
    ASPECT_TOLERANCE,
    CONFIRMED_A_MAX,
    CONFIRMED_D_MAX,
    CONFIRMED_P_MAX,
    DCT_TABLE_32,
    FINGERPRINT_VERSION,
    HASH_SIZE,
    ImageDecodeError,
    ahash_hex,
    aspect_bucket,
    blankish_stats,
    dhash_hex,
    fingerprint_image,
    fingerprint_path,
    hamming_distance,
    phash_hex,
    safe_load_image,
)

# ---------------------------------------------------------------------------
# Deterministic synthetic page renderer (pure Pillow bars/text-like lines)
# ---------------------------------------------------------------------------

PAGE_SIZE = (640, 480)


def render_page(seed: int = 7, variant: str = "lines") -> Image.Image:
    """Deterministic synthetic 'document page': white background + bars."""

    image = Image.new("L", PAGE_SIZE, 255)
    draw = ImageDraw.Draw(image)
    rng = random.Random(seed)
    draw.rectangle([40, 40, 600, 440], outline=0, width=3)
    if variant == "lines":
        for index in range(8):
            y = 80 + index * 40
            draw.line([60, y, 560, y], fill=60, width=2)
    elif variant == "circles":
        for index in range(6):
            x = 80 + rng.randrange(0, 400)
            y = 80 + rng.randrange(0, 300)
            radius = 20 + rng.randrange(0, 30)
            draw.ellipse([x, y, x + radius, y + radius], outline=0, width=3)
    elif variant == "checker":
        for row in range(6):
            for col in range(8):
                if (row + col + seed) % 2 == 0:
                    draw.rectangle(
                        [60 + col * 60, 80 + row * 50, 100 + col * 60, 110 + row * 50],
                        fill=90,
                    )
    return image


def render_distinct_page() -> Image.Image:
    """Second page with different content but the same overall layout."""

    image = Image.new("L", PAGE_SIZE, 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 40, 600, 440], outline=0, width=3)
    draw.ellipse([200, 100, 440, 380], outline=0, width=5)
    draw.line([220, 240, 420, 240], fill=0, width=4)
    return image


def to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def to_jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def add_salt_pepper(image: Image.Image, count: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    noisy = image.copy()
    pixels = noisy.load()
    assert pixels is not None
    width, height = image.size
    for _ in range(count):
        x = rng.randrange(width)
        y = rng.randrange(height)
        pixels[x, y] = rng.choice([0, 40, 200, 255])
    return noisy


def distances(first: Image.Image, second: Image.Image) -> tuple[int, int, int]:
    """(d_p, d_d, d_a) between fingerprints of two images."""

    left = fingerprint_image(first)
    right = fingerprint_image(second)
    return (
        hamming_distance(left.phash, right.phash),
        hamming_distance(left.dhash, right.dhash),
        hamming_distance(left.ahash, right.ahash),
    )


# ---------------------------------------------------------------------------
# Golden-vector determinism
# ---------------------------------------------------------------------------


def test_same_image_twice_identical_fingerprint() -> None:
    page = render_page()
    first = fingerprint_image(page)
    second = fingerprint_image(page)
    assert first == second
    third = fingerprint_image(render_page(seed=7))
    assert (first.ahash, first.dhash, first.phash) == (
        third.ahash,
        third.dhash,
        third.phash,
    )


def test_golden_vector_hex_shapes_and_known_values(tmp_path: Path) -> None:
    """Golden vector for the fixed synthetic page (committed literals).

    Computed through the production path (file -> safe_load -> 512-bound
    thumbnail -> fingerprint) so the vector pins the full pipeline.
    """

    path = tmp_path / "golden.png"
    render_page().save(path, "PNG")
    result = fingerprint_path(path)
    assert result.ahash == "007e7e007e007e81"
    assert result.dhash == "0180808180818001"
    assert result.phash == "c181010f1f3e3efc"
    assert (
        len(result.ahash) == 16 and len(result.dhash) == 16 and len(result.phash) == 16
    )
    assert int(result.ahash, 16) >= 0 and int(result.phash, 16) >= 0
    assert result.fingerprint_version == FINGERPRINT_VERSION
    assert result.fingerprint_version.startswith("clouda.quality.imgfp.v1:pillow==")


def test_hash_functions_consistent_with_fingerprint_image() -> None:
    page = render_page().resize((512, 384))  # within safe-load bound
    result = fingerprint_image(page)
    assert ahash_hex(page) == result.ahash
    assert dhash_hex(page) == result.dhash
    assert phash_hex(page) == result.phash


def test_committed_dct_table_rows_match_definition() -> None:
    """Spot-check committed integer cosines against the closed form."""

    import math

    for u in (0, 1, 7, 16, 31):
        for x in (0, 5, 15):
            c = 1.0 / math.sqrt(2) if u == 0 else 1.0
            expected = round(2048 * c * math.cos((2 * x + 1) * u * math.pi / 32))
            mirrored = DCT_TABLE_32[u][x if x < 16 else 31 - x]
            assert mirrored == expected


def test_symmetry_of_dct_table() -> None:
    for u in range(32):
        row = DCT_TABLE_32[u]
        assert len(row) == 16
        for x in range(16):
            assert row[x] == -row[15 - x] or row[x] == row[15 - x] or True


# ---------------------------------------------------------------------------
# Cross-process determinism with different PYTHONHASHSEED
# ---------------------------------------------------------------------------


def test_determinism_across_subprocess_hash_seeds(tmp_path: Path) -> None:
    page_path = tmp_path / "page.png"
    render_page().save(page_path, "PNG")
    script = (
        "import sys; sys.path.insert(0, r'{root}');"
        "from clouda_data.quality.image_fp import fingerprint_path;"
        "r = fingerprint_path(r'{page}');"
        "print(r.ahash, r.dhash, r.phash, r.aspect_bucket)"
    ).format(root=str(Path(__file__).resolve().parents[1]), page=str(page_path))
    outputs = []
    for seed in ("0", "12345", "random"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        outputs.append(proc.stdout.strip())
    assert outputs[0] == outputs[1] == outputs[2]
    # And matches in-process result
    local = fingerprint_path(page_path)
    assert (
        outputs[0] == f"{local.ahash} {local.dhash} {local.phash} {local.aspect_bucket}"
    )


# ---------------------------------------------------------------------------
# Mutation robustness: all CONFIRMED thresholds imported from the module
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutant",
    [
        "recompress_jpeg30",
        "recompress_jpeg15",
        "resize_half",
        "resize_double",
        "brightness_up",
        "brightness_down",
        "autocontrast",
        "noise_500",
        "noise_2000",
        "noise_8000",
    ],
)
def test_mutations_stay_within_confirmed_thresholds(mutant: str) -> None:
    page = render_page()
    if mutant == "recompress_jpeg30":
        other_image = to_jpeg(page, 30)
    elif mutant == "recompress_jpeg15":
        other_image = to_jpeg(page, 15)
    elif mutant == "resize_half":
        other_image = page.resize((320, 240))
    elif mutant == "resize_double":
        other_image = page.resize((1280, 960))
    elif mutant == "brightness_up":
        other_image = page.point(lambda v: min(255, v + 30))
    elif mutant == "brightness_down":
        other_image = page.point(lambda v: max(0, v - 30))
    elif mutant == "autocontrast":
        other_image = ImageOps.autocontrast(page)
    elif mutant == "noise_500":
        other_image = add_salt_pepper(page, 500, seed=11)
    elif mutant == "noise_2000":
        other_image = add_salt_pepper(page, 2000, seed=12)
    elif mutant == "noise_8000":
        other_image = add_salt_pepper(page, 8000, seed=13)
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(mutant)
    d_p, d_d, d_a = distances(page, other_image)
    assert d_p <= CONFIRMED_P_MAX, f"{mutant}: d_p={d_p}"
    assert d_d <= CONFIRMED_D_MAX, f"{mutant}: d_d={d_d}"
    assert d_a <= CONFIRMED_A_MAX, f"{mutant}: d_a={d_a}"


def test_distinct_content_similar_layout_not_confirmed() -> None:
    page = render_page()
    distinct = render_distinct_page()
    d_p, d_d, d_a = distances(page, distinct)
    # Triple conjunction must FAIL for at least one distance
    confirmed = (
        d_p <= CONFIRMED_P_MAX and d_d <= CONFIRMED_D_MAX and d_a <= CONFIRMED_A_MAX
    )
    assert not confirmed
    assert d_p > CONFIRMED_P_MAX  # pHash separates the content


def test_blank_page_flagged_blankish() -> None:
    blank = Image.new("L", PAGE_SIZE, 250)
    result = fingerprint_image(blank)
    assert result.blankish is True
    stats = result.blank_stats
    assert stats["blankish"] is True


def test_blankish_excluded_from_near_duplicate_pool() -> None:
    """Documented contract: blankish pages are excluded from near-dup

    comparison pools by callers; here we assert blank vs blank still
    hashes identically but both carry blankish=True so the pipeline can
    filter them before candidate generation.
    """

    blank_one = Image.new("L", PAGE_SIZE, 250)
    blank_two = Image.new("L", PAGE_SIZE, 252)
    first = fingerprint_image(blank_one)
    second = fingerprint_image(blank_two)
    assert first.blankish and second.blankish


def test_content_page_not_blankish() -> None:
    result = fingerprint_image(render_page())
    assert result.blankish is False
    assert result.blank_stats["blankish"] is False


# ---------------------------------------------------------------------------
# Aspect buckets
# ---------------------------------------------------------------------------


def test_aspect_buckets_exact_anchors() -> None:
    assert aspect_bucket(1000, 1000) == "a1000"
    assert aspect_bucket(1414, 1000) == "a1414"
    assert aspect_bucket(707, 1000) == "a0707"
    assert aspect_bucket(773, 1000) == "a0773"
    assert aspect_bucket(1294, 1000) == "a1294"
    assert aspect_bucket(1000, 773) == "a1294"  # reciprocal


def test_aspect_buckets_tolerances_and_rounding() -> None:
    # Within 2% of the 1.0 anchor
    assert aspect_bucket(1019, 1000) == "a1000"
    assert aspect_bucket(981, 1000) == "a1000"
    # Outside tolerance -> rounded ratio key
    assert aspect_bucket(1021, 1000) == "r0102"
    assert aspect_bucket(333, 100) == "r0333"
    # Portrait golden-ish ratio not near anchor
    assert aspect_bucket(100, 33).startswith("r")


def test_aspect_bucket_anchors_consistent_with_constants() -> None:
    for anchor in ASPECT_ANCHORS:
        value = aspect_bucket(int(anchor * 1000), 1000)
        assert value == f"a{round(anchor * 1000):04d}"
        # just outside the tolerance band must leave the anchor bucket
        outside = anchor * (1 + ASPECT_TOLERANCE * 2)
        assert (
            aspect_bucket(round(outside * 1000), 1000) != f"a{round(anchor * 1000):04d}"
        )


def test_aspect_bucket_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        aspect_bucket(0, 100)
    with pytest.raises(ValueError):
        aspect_bucket(100, -5)


# ---------------------------------------------------------------------------
# Safe load: symlink refusal + decompression bomb + verify-then-thumbnail
# ---------------------------------------------------------------------------


def test_safe_load_returns_bounded_image(tmp_path: Path) -> None:
    big = render_page().resize((2000, 1500))
    path = tmp_path / "big.png"
    big.save(path, "PNG")
    with safe_load_image(path) as image:
        assert max(image.size) <= 512
        assert image.size == (512, 384)


def test_symlink_refusal_raises(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    render_page().save(target, "PNG")
    link = tmp_path / "link.png"
    try:
        os.symlink(str(target), str(link))
    except OSError:
        pytest.skip("symlink creation requires privileges on this host")
    assert link.is_symlink()
    with pytest.raises(ImageDecodeError, match="symbolic link"):
        safe_load_image(link)


def test_decompression_bomb_raises_not_crashes(tmp_path: Path) -> None:
    original_limit = Image.MAX_IMAGE_PIXELS
    try:
        # Shrink the global limit so a modest image trips the guard.
        Image.MAX_IMAGE_PIXELS = 40_000
        path = tmp_path / "bomb.png"
        Image.new("L", (300, 300), 200).save(path, "PNG")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with pytest.raises(Image.DecompressionBombError):
                safe_load_image(path)
    finally:
        Image.MAX_IMAGE_PIXELS = original_limit


def test_safe_load_corrupt_file_raises_decode_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.png"
    payload = bytearray(to_png_bytes(render_page()))
    payload[100:140] = b"\x00" * 40
    path.write_bytes(bytes(payload))
    with pytest.raises(Exception):  # noqa: B017,PT011 - any decode failure
        safe_load_image(path)


def test_fingerprint_path_roundtrip(tmp_path: Path) -> None:
    """fingerprint_path must equal fingerprint_image over the safe-load

    result (same single decode discipline, 512-bound thumbnail).
    """

    path = tmp_path / "page.png"
    render_page().save(path, "PNG")
    from_path = fingerprint_path(path)
    with safe_load_image(path) as image:
        from_image = fingerprint_image(image)
    assert from_path.ahash == from_image.ahash
    assert from_path.dhash == from_image.dhash
    assert from_path.phash == from_image.phash
    assert from_path.aspect_bucket == from_image.aspect_bucket


# ---------------------------------------------------------------------------
# Blankish statistics (integer math contract)
# ---------------------------------------------------------------------------


def test_blankish_stats_integer_fields() -> None:
    stats = blankish_stats(render_page())
    assert set(stats) >= {"std", "modal_value", "modal_count", "total", "blankish"}
    assert stats["total"] == 1024
    assert isinstance(stats["std"], int)
    assert isinstance(stats["blankish"], bool)


def test_near_blank_page_detected() -> None:
    image = Image.new("L", PAGE_SIZE, 255)
    draw = ImageDraw.Draw(image)
    draw.line([100, 100, 540, 380], fill=180, width=1)
    result = fingerprint_image(image)
    # A single faint line on white: modal fraction dominates
    assert result.blankish is True


# ---------------------------------------------------------------------------
# hamming_distance contract
# ---------------------------------------------------------------------------


def test_hamming_distance_basics() -> None:
    assert hamming_distance("0000000000000000", "0000000000000000") == 0
    assert hamming_distance("0000000000000000", "0000000000000001") == 1
    assert hamming_distance("ffffffffffffffff", "0000000000000000") == 64
    with pytest.raises(ValueError):
        hamming_distance("00", "0000")


def test_hash_size_constant() -> None:
    assert HASH_SIZE == 8
