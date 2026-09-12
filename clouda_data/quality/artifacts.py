"""Per-sample artifact integrity validation (Agent I).

``validate_artifact`` extends the pretraining validation semantics with
integrity checks that only make sense on disk: zero-byte files, hash
manifests, image modes, and warn-only heuristic flags (blank pages,
extreme dimensions, suspiciously small artifacts, very short ground
truth). The original files are never modified: every check is read-only,
the image is decoded exactly once through the shared single-decode
helper, and symbolic links are refused before any open.

Base findings are produced by :func:`clouda_data.pretraining.validation.
validate_sample`; its finding codes map onto the closed
:class:`~clouda_data.quality.models.IssueCode` vocabulary below.
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Any

from clouda_data.pretraining.hashing import SHA256_RE, sha256_file
from clouda_data.pretraining.schema import DatasetSample
from clouda_data.pretraining.validation import (
    Finding,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    validate_sample,
)
from clouda_data.quality.config import HeuristicsPolicy
from clouda_data.quality.image_fp import (
    ImageDecodeError,
    safe_load_image_cached as safe_load_image,
)
from clouda_data.quality.models import IssueCode, IssueSeverity, QualityIssue

__all__ = ["validate_artifact", "map_finding_severity", "map_finding_code"]

#: Decoded PIL modes that must be re-saved before archival (warning only).
SUSPECT_IMAGE_MODES = frozenset({"P", "1", "I", "I;16", "F", "CMYK", "YCbCr"})

#: Thumbnail width for blank/near-blank statistics.
_BLANK_THUMB_WIDTH = 256

#: Minimum width/height for the ``tiny_image`` heuristic baseline.
_TINY_DIMENSION = 8

#: Format of a valid SHA-256 hex digest (same grammar as SHA256_RE).
_SHA256_FULLMATCH = re.compile(r"^[0-9a-f]{64}$")

#: Mapping from pretraining finding codes to the quality-issue vocabulary.
_FINDING_TO_CODE: dict[str, str] = {
    "missing_image": IssueCode.MISSING_IMAGE,
    "path_escape": IssueCode.PATH_SAFE,
    "unsupported_format": IssueCode.IMAGE_DECODE,
    "unreadable_image": IssueCode.IMAGE_DECODE,
    "invalid_dimensions": IssueCode.IMAGE_DIMENSIONS,
    "oversized_image": IssueCode.IMAGE_DIMENSIONS,
    "tiny_image": IssueCode.IMAGE_DIMENSIONS,
    "missing_text": IssueCode.GT_MISSING,
    "empty_text": IssueCode.GT_EMPTY,
    "text_too_long": IssueCode.GT_EMPTY,
    "invalid_control_characters": IssueCode.GT_EMPTY,
    "invalid_encoding": IssueCode.GT_EMPTY,
    "malformed_metadata": IssueCode.GT_MISSING,
    "missing_provenance": IssueCode.GT_MISSING,
}

#: Pretraining severities that map onto INFO.
_INFO_SEVERITIES = frozenset({SEVERITY_INFO})


def map_finding_code(finding: Finding) -> str:
    """Map a pretraining finding code onto an IssueCode constant."""

    return _FINDING_TO_CODE.get(finding.code, IssueCode.IMAGE_DECODE)


def map_finding_severity(severity: str) -> IssueSeverity:
    """Map a pretraining severity string onto IssueSeverity."""

    if severity == SEVERITY_ERROR:
        return IssueSeverity.ERROR
    if severity == SEVERITY_WARNING:
        return IssueSeverity.WARNING
    return IssueSeverity.INFO


def _issue(
    code: str,
    severity: IssueSeverity,
    sample: DatasetSample,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> QualityIssue:
    return QualityIssue(
        code=code,
        severity=severity,
        sample_ids=(sample.sample_id,),
        canonical_key=f"{code}:{sample.source_id}:{sample.sample_id}",
        message=message,
        evidence=dict(evidence or {}),
    )


def _base_issues(sample: DatasetSample, root: Path) -> list[QualityIssue]:
    """Reuse pretraining validation, mapping findings onto issue codes."""

    status, findings = validate_sample(sample, root)
    del status  # severity rides on the findings themselves
    issues: list[QualityIssue] = []
    for finding in findings:
        issues.append(
            _issue(
                map_finding_code(finding),
                map_finding_severity(finding.severity),
                sample,
                finding.message,
            )
        )
    return issues


def _hash_issue(
    sample: DatasetSample, image_path: Path, root: Path
) -> list[QualityIssue]:
    """Hash-manifest check: None -> info, bad format -> critical, else compare."""

    if sample.file_sha256 is None:
        return [
            _issue(
                IssueCode.HASH_MISMATCH,
                IssueSeverity.INFO,
                sample,
                "no file hash recorded in manifest",
            )
        ]
    if not SHA256_RE.fullmatch(sample.file_sha256):
        return [
            _issue(
                IssueCode.HASH_MISMATCH,
                IssueSeverity.CRITICAL,
                sample,
                "manifest file_sha256 is not a SHA-256 hex digest; failing closed",
            )
        ]
    actual = sha256_file(image_path)
    if actual != sample.file_sha256:
        return [
            _issue(
                IssueCode.HASH_MISMATCH,
                IssueSeverity.ERROR,
                sample,
                "file hash differs from manifest file_sha256",
                {"expected": sample.file_sha256, "actual": actual},
            )
        ]
    return []


def _blank_issues(
    image: Any,
    sample: DatasetSample,
    thresholds: HeuristicsPolicy,
) -> list[QualityIssue]:
    """Blank / near-blank page heuristics over a 256-wide grayscale thumbnail."""

    thumbnail = image.convert("L").resize(
        (_BLANK_THUMB_WIDTH, max(1, image.height * _BLANK_THUMB_WIDTH // image.width)),
    )
    pixels = list(thumbnail.tobytes())
    std = statistics.pstdev(pixels) if len(pixels) > 1 else 0.0
    ink = sum(1 for value in pixels if value < 200) / len(pixels)
    issues: list[QualityIssue] = []
    if std < thresholds.blank_std:
        issues.append(
            _issue(
                IssueCode.BLANK_PAGE,
                IssueSeverity.WARNING,
                sample,
                f"page is blank (thumbnail stddev {std:.2f})",
                {"std": round(std, 3), "ink_fraction": round(ink, 6)},
            )
        )
    elif std < thresholds.near_blank_std or ink < 0.001:
        issues.append(
            _issue(
                IssueCode.NEAR_BLANK_PAGE,
                IssueSeverity.WARNING,
                sample,
                f"page is near blank (thumbnail stddev {std:.2f})",
                {"std": round(std, 3), "ink_fraction": round(ink, 6)},
            )
        )
    return issues


def _dimension_issues(
    width: int,
    height: int,
    sample: DatasetSample,
    thresholds: HeuristicsPolicy,
) -> list[QualityIssue]:
    """Extreme-dimension / aspect-ratio / metadata-mismatch warnings."""

    issues: list[QualityIssue] = []
    if (
        width > thresholds.extreme_max_side
        or height > thresholds.extreme_max_side
        or width * height > thresholds.extreme_max_pixels
    ):
        issues.append(
            _issue(
                IssueCode.EXTREME_DIMENSIONS,
                IssueSeverity.WARNING,
                sample,
                f"unusual image dimensions: {width}x{height}",
                {"width": width, "height": height},
            )
        )
    if max(width, height) / min(width, height) > thresholds.max_aspect_ratio:
        issues.append(
            _issue(
                IssueCode.EXTREME_ASPECT_RATIO,
                IssueSeverity.WARNING,
                sample,
                f"extreme aspect ratio: {width}x{height}",
                {"width": width, "height": height},
            )
        )
    if sample.width is not None and sample.width != width:
        issues.append(
            _issue(
                IssueCode.METADATA_DIMENSION_MISMATCH,
                IssueSeverity.WARNING,
                sample,
                f"manifest width {sample.width} differs from decoded {width}",
                {"manifest_width": sample.width, "decoded_width": width},
            )
        )
    if sample.height is not None and sample.height != height:
        issues.append(
            _issue(
                IssueCode.METADATA_DIMENSION_MISMATCH,
                IssueSeverity.WARNING,
                sample,
                f"manifest height {sample.height} differs from decoded {height}",
                {"manifest_height": sample.height, "decoded_height": height},
            )
        )
    return issues


def validate_artifact(
    sample: DatasetSample,
    root: Path | str,
    thresholds: HeuristicsPolicy,
) -> list[QualityIssue]:
    """Validate one artifact on disk. Read-only; never raises on bad data.

    Checks: symlink refusal, zero-byte refusal, streaming hash manifest,
    single decode with mode/dimension heuristics, and ground-truth
    length. Every problem becomes a :class:`QualityIssue` carrying only
    IDs, codes, and counts -- never sample content.
    """

    root_path = Path(root)
    issues: list[QualityIssue] = list(_base_issues(sample, root_path))

    if sample.image_path is None:
        return issues

    image_path = root_path / str(sample.image_path)

    # Symlink refusal BEFORE any open (mirrors pretraining path_escape but
    # keyed to the physical filesystem rather than path traversal).
    if image_path.is_symlink():
        issues.append(
            _issue(
                IssueCode.PATH_SAFE,
                IssueSeverity.ERROR,
                sample,
                "image path is a symbolic link; refusing to open",
            )
        )
        return issues

    # Zero-byte and oversized artifacts are stat-only (checked BEFORE decode).
    try:
        file_size = image_path.stat().st_size
    except OSError:
        file_size = None
    if file_size == 0:
        issues.append(
            _issue(
                IssueCode.NON_EMPTY_FILE,
                IssueSeverity.ERROR,
                sample,
                "image file is empty (zero bytes)",
            )
        )
        return issues
    if file_size is not None and file_size > thresholds.max_artifact_bytes:
        issues.append(
            _issue(
                IssueCode.VERY_LARGE_ARTIFACT,
                IssueSeverity.WARNING,
                sample,
                f"artifact exceeds {thresholds.max_artifact_bytes} bytes",
                {"file_size": file_size},
            )
        )
        return issues
    if file_size is not None and file_size < thresholds.small_image_min_bytes:
        issues.append(
            _issue(
                IssueCode.SUSPICIOUSLY_SMALL_IMAGE,
                IssueSeverity.WARNING,
                sample,
                f"image file suspiciously small: {file_size} bytes",
                {"file_size": file_size},
            )
        )
    # Hash manifest (after zero-byte and size bounds, before decode).
    issues.extend(_hash_issue(sample, image_path, root_path))

    # Single decode via the shared safe-load helper.
    image = None
    try:
        image = safe_load_image(image_path)
    except ImageDecodeError:
        image = None
    except OSError:
        image = None
    except Exception:  # noqa: BLE001 - decoder crashes must not abort the gate
        image = None

    decoded: Any = image
    if decoded is not None:
        try:
            width, height = decoded.size
            issues.extend(_dimension_issues(width, height, sample, thresholds))
            if decoded.mode in SUSPECT_IMAGE_MODES:
                issues.append(
                    _issue(
                        IssueCode.IMAGE_MODE,
                        IssueSeverity.WARNING,
                        sample,
                        f"image mode {decoded.mode!r} should be re-saved before archival",
                        {"mode": decoded.mode},
                    )
                )
            if file_size is not None and (
                max(width, height) < _TINY_DIMENSION and (width + height) / 2 < 32
            ):
                issues.append(
                    _issue(
                        IssueCode.SUSPICIOUSLY_SMALL_IMAGE,
                        IssueSeverity.WARNING,
                        sample,
                        f"tiny image with tiny average dimension: {width}x{height}",
                        {"width": width, "height": height},
                    )
                )
            issues.extend(_blank_issues(decoded, sample, thresholds))
        finally:
            decoded.close()

    if decoded is None:
        # EXTREME_DIMENSIONS must remain detectable from manifest metadata
        # when the decode is unavailable (fail-open to warning, not error).
        if sample.width is not None and sample.height is not None:
            meta_w = int(sample.width)
            meta_h = int(sample.height)
            if (
                meta_w > thresholds.extreme_max_side
                or meta_h > thresholds.extreme_max_side
                or meta_w * meta_h > thresholds.extreme_max_pixels
            ):
                issues.append(
                    _issue(
                        IssueCode.EXTREME_DIMENSIONS,
                        IssueSeverity.WARNING,
                        sample,
                        f"unusual image dimensions from metadata: {meta_w}x{meta_h}",
                        {"width": meta_w, "height": meta_h},
                    )
                )

    text = sample.raw_text if sample.raw_text is not None else sample.text
    if text is not None:
        stripped = text.strip()
        if 0 < len(stripped) < thresholds.min_gt_warn_chars:
            issues.append(
                _issue(
                    IssueCode.VERY_SHORT_GT,
                    IssueSeverity.WARNING,
                    sample,
                    f"ground truth suspiciously short: {len(stripped)!r}",
                    {"length": len(stripped)},
                )
            )

    return issues
