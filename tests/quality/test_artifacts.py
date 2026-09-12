"""Artifact integrity checks over clouda_data.quality.artifacts."""

from __future__ import annotations

import io
import zlib

from PIL import Image

from clouda_data.quality.config import HeuristicsPolicy
from clouda_data.quality.models import IssueCode, IssueSeverity
from clouda_data.pretraining.hashing import sha256_file

from tests.quality.conftest import render_arabic_page, save_png

from clouda_data.quality.artifacts import validate_artifact
from clouda_data.pretraining.schema import DatasetSample

SMALL_POLICY = HeuristicsPolicy(max_pixels=1_000_000)


def _sample(**overrides):
    kwargs = {
        "sample_id": "smp_test",
        "source_id": "src1",
        "image_path": "imgs/page.png",
        "text": "نص تجريبي طويل بما يكفي",
    }
    kwargs.update(overrides)
    return DatasetSample(**kwargs)


def _codes(issues):
    return {issue.code for issue in issues}


def _severity_of(issues, code):
    return {issue.severity for issue in issues if issue.code == code}


def _write_clean_page(
    root, path="imgs/page.png", size=(512, 512), text="مرحبا بالعالم"
):
    image = render_arabic_page(7, "plain", text, size=size)
    image_path = save_png(image, root / path)
    return image_path, sha256_file(image_path)


def test_zero_byte_image_reports_non_empty_file_error(tmp_path):
    image_path = tmp_path / "imgs" / "empty.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"")
    sample = _sample(image_path="imgs/empty.png", file_sha256=None)
    issues = validate_artifact(sample, tmp_path, SMALL_POLICY)
    assert IssueCode.NON_EMPTY_FILE in _codes(issues)
    assert IssueSeverity.ERROR in _severity_of(issues, IssueCode.NON_EMPTY_FILE)


def test_truncated_png_reports_image_decode_error(tmp_path):
    image_path, _ = _write_clean_page(tmp_path)
    data = image_path.read_bytes()
    image_path.write_bytes(data[: len(data) // 2])
    sample = _sample(file_sha256=sha256_file(image_path))
    issues = validate_artifact(sample, tmp_path, SMALL_POLICY)
    assert IssueCode.IMAGE_DECODE in _codes(issues)
    assert IssueSeverity.ERROR in _severity_of(issues, IssueCode.IMAGE_DECODE)


def test_hash_mismatch_after_file_modification(tmp_path):
    image_path, original_hash = _write_clean_page(tmp_path)
    with open(image_path, "ab") as handle:
        handle.write(b"\x00")
    sample = _sample(file_sha256=original_hash)
    issues = validate_artifact(sample, tmp_path, SMALL_POLICY)
    assert IssueCode.HASH_MISMATCH in _codes(issues)
    assert IssueSeverity.ERROR in _severity_of(issues, IssueCode.HASH_MISMATCH)


def test_missing_hash_is_info_only(tmp_path):
    image_path, _ = _write_clean_page(tmp_path)
    sample = _sample(file_sha256=None)
    issues = validate_artifact(sample, tmp_path, SMALL_POLICY)
    assert IssueCode.HASH_MISMATCH in _codes(issues)
    assert _severity_of(issues, IssueCode.HASH_MISMATCH) == {IssueSeverity.INFO}


def test_bad_hash_format_is_critical(tmp_path):
    image_path, _ = _write_clean_page(tmp_path)
    sample = _sample(file_sha256="deadbeef")
    issues = validate_artifact(sample, tmp_path, SMALL_POLICY)
    assert IssueCode.HASH_MISMATCH in _codes(issues)
    assert IssueSeverity.CRITICAL in _severity_of(issues, IssueCode.HASH_MISMATCH)


def test_palette_mode_reports_image_mode_warning(tmp_path):
    source = render_arabic_page(11, "plain", "مرحبا", size=(128, 96)).convert("P")
    image_path = tmp_path / "imgs" / "palette.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    source.save(image_path, format="PNG")
    sample = _sample(
        image_path="imgs/palette.png",
        file_sha256=sha256_file(image_path),
        width=128,
        height=96,
    )
    issues = validate_artifact(sample, tmp_path, SMALL_POLICY)
    assert IssueCode.IMAGE_MODE in _codes(issues)
    assert IssueSeverity.WARNING in _severity_of(issues, IssueCode.IMAGE_MODE)


def test_blank_synthetic_page_reports_blank_page(tmp_path):
    blank = Image.new("L", (128, 96), 255)
    image_path = tmp_path / "imgs" / "blank.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    blank.save(image_path, format="PNG")
    sample = _sample(
        image_path="imgs/blank.png",
        file_sha256=sha256_file(image_path),
        width=128,
        height=96,
    )
    issues = validate_artifact(sample, tmp_path, SMALL_POLICY)
    assert IssueCode.BLANK_PAGE in _codes(issues)
    assert IssueSeverity.WARNING in _severity_of(issues, IssueCode.BLANK_PAGE)


def test_near_blank_page_reports_near_blank(tmp_path):
    near_blank = Image.new("L", (128, 96), 255)
    # Sprinkle just enough ink to sit between blank_std and near_blank_std.
    for x in range(6):
        near_blank.putpixel((10 + x, 10 + x), 0)
    image_path = tmp_path / "imgs" / "near_blank.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    near_blank.save(image_path, format="PNG")
    sample = _sample(
        image_path="imgs/near_blank.png",
        file_sha256=sha256_file(image_path),
        width=128,
        height=96,
    )
    issues = validate_artifact(sample, tmp_path, SMALL_POLICY)
    assert IssueCode.NEAR_BLANK_PAGE in _codes(issues)
    assert IssueSeverity.WARNING in _severity_of(issues, IssueCode.NEAR_BLANK_PAGE)


def test_extreme_dimensions_from_metadata(tmp_path):
    # Simulate a decode failure (corrupt body) while the manifest metadata
    # still records extreme dimensions; EXTREME_DIMENSIONS must remain a
    # warning derived from sample.width/height.
    header = Image.new("L", (128, 96), 255)
    buffer = io.BytesIO()
    header.save(buffer, format="PNG")
    payload = bytearray(buffer.getvalue())
    # Corrupt every IDAT chunk so verify()/load() fail but the header stands.
    idat = zlib.crc32(b"IDAT")  # noqa: F841 - marker for readability
    for index in range(len(payload) - 8, len(payload) - 4):
        payload[index] ^= 0xFF
    image_path = tmp_path / "imgs" / "extreme.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(bytes(payload))
    sample = _sample(file_sha256=None, width=50000, height=50)
    issues = validate_artifact(sample, tmp_path, SMALL_POLICY)
    assert IssueCode.EXTREME_DIMENSIONS in _codes(issues)
    assert IssueSeverity.WARNING in _severity_of(issues, IssueCode.EXTREME_DIMENSIONS)
    assert IssueSeverity.ERROR not in _severity_of(issues, IssueCode.EXTREME_DIMENSIONS)


def test_extreme_dimensions_decoded_warning_not_error(tmp_path):
    # Small MAX_IMAGE_PIXELS monkeypatch so Pillow accepts opening a
    # 50000x50 image (50000*50 = 2.5MP, but side > extreme_max_side).
    original_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 100_000_000
    try:
        wide = Image.new("L", (50000, 50), 255)
        image_path = tmp_path / "imgs" / "wide.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        wide.save(image_path, format="PNG")
    finally:
        Image.MAX_IMAGE_PIXELS = original_limit
    sample = _sample(file_sha256=None, width=50000, height=50)
    issues = validate_artifact(sample, tmp_path, SMALL_POLICY)
    assert IssueCode.EXTREME_DIMENSIONS in _codes(issues)
    assert IssueSeverity.WARNING in _severity_of(issues, IssueCode.EXTREME_DIMENSIONS)
    assert IssueSeverity.ERROR not in _severity_of(issues, IssueCode.EXTREME_DIMENSIONS)


def test_extreme_aspect_ratio_warning(tmp_path):
    tall = Image.new("L", (100, 10000), 255)
    image_path = tmp_path / "imgs" / "tall.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    tall.save(image_path, format="PNG")
    sample = _sample(
        image_path="imgs/tall.png", file_sha256=None, width=100, height=10000
    )
    issues = validate_artifact(sample, tmp_path, SMALL_POLICY)
    assert IssueCode.EXTREME_ASPECT_RATIO in _codes(issues)
    assert IssueSeverity.WARNING in _severity_of(issues, IssueCode.EXTREME_ASPECT_RATIO)


def test_tiny_file_reports_suspiciously_small_image(tmp_path):
    tiny = Image.new("L", (16, 16), 255)
    image_path = tmp_path / "imgs" / "tiny.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    tiny.save(image_path, format="PNG", optimize=True)
    assert image_path.stat().st_size < 1024
    sample = _sample(image_path="imgs/tiny.png", file_sha256=None, width=16, height=16)
    issues = validate_artifact(sample, tmp_path, SMALL_POLICY)
    assert IssueCode.SUSPICIOUSLY_SMALL_IMAGE in _codes(issues)
    assert IssueSeverity.WARNING in _severity_of(
        issues, IssueCode.SUSPICIOUSLY_SMALL_IMAGE
    )


def test_very_short_gt_warning(tmp_path):
    image_path, _ = _write_clean_page(tmp_path)
    sample = _sample(file_sha256=None, text="x")
    issues = validate_artifact(sample, tmp_path, SMALL_POLICY)
    assert IssueCode.VERY_SHORT_GT in _codes(issues)
    assert IssueSeverity.WARNING in _severity_of(issues, IssueCode.VERY_SHORT_GT)


def test_clean_page_reports_no_issues(tmp_path):
    image_path, file_hash = _write_clean_page(tmp_path)
    assert image_path.stat().st_size >= 1024
    sample = _sample(file_sha256=file_hash, width=512, height=512)
    issues = validate_artifact(sample, tmp_path, SMALL_POLICY)
    assert issues == []


def test_path_escape_reports_path_safe(tmp_path):
    image_path, _ = _write_clean_page(tmp_path, path="imgs/page.png")
    # Place the actual file outside the root and reference it via traversal.
    outside = tmp_path.parent / "outside_escape.png"
    outside.write_bytes(image_path.read_bytes())
    try:
        sample = _sample(image_path="../../etc/passwd.png", file_sha256=None)
        issues = validate_artifact(sample, tmp_path, SMALL_POLICY)
        assert IssueCode.PATH_SAFE in _codes(issues)
        assert IssueSeverity.ERROR in _severity_of(issues, IssueCode.PATH_SAFE)
    finally:
        outside.unlink(missing_ok=True)
