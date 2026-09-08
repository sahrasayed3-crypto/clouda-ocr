"""Sample validation independent of any OCR model.

Validation returns structured findings classified by severity. One bad
sample never aborts the dataset; errors simply mark the sample for
exclusion with a machine-readable reason.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from .discovery import IMAGE_EXTENSIONS
from .schema import DatasetSample, ValidationStatus

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"
SEVERITY_EXCLUSION = "exclusion"


@dataclass(frozen=True)
class Finding:
    code: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationThresholds:
    """Validation limits (kept as a plain class for cheap construction)."""

    min_width: int = 8
    min_height: int = 8
    max_pixels: int = 100_000_000
    max_text_chars: int = 100_000
    require_text: bool = True
    require_image: bool = True


def _path_inside(root: Path, relative: str | None) -> bool:
    if relative is None:
        return True
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_image(
    sample: DatasetSample,
    root: Path,
    findings: list[Finding],
    thresholds: ValidationThresholds,
) -> None:
    from PIL import Image  # local import keeps module import cheap

    image_path = root / str(sample.image_path)
    if not image_path.is_file():
        findings.append(
            Finding(
                "missing_image", SEVERITY_ERROR, f"image not found: {sample.image_path}"
            )
        )
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(image_path) as image:
                width, height = image.size
                if width <= 0 or height <= 0:
                    findings.append(
                        Finding(
                            "invalid_dimensions",
                            SEVERITY_ERROR,
                            f"invalid image dimensions: {width}x{height}",
                        )
                    )
                    return
                if width * height > thresholds.max_pixels:
                    findings.append(
                        Finding(
                            "oversized_image",
                            SEVERITY_ERROR,
                            f"image exceeds max pixels: {width * height}",
                        )
                    )
                    return
                image.verify()
            with Image.open(image_path) as reopened:
                reopened.load()
    except Exception:  # noqa: BLE001 - any decoder failure is a finding
        findings.append(
            Finding(
                "unreadable_image",
                SEVERITY_ERROR,
                f"image cannot be decoded: {sample.image_path}",
            )
        )
        return
    if width < thresholds.min_width or height < thresholds.min_height:
        findings.append(
            Finding(
                "tiny_image",
                SEVERITY_WARNING,
                f"image below minimum dimensions: {width}x{height}",
            )
        )


def validate_sample(
    sample: DatasetSample,
    dataset_root: Path,
    *,
    thresholds: ValidationThresholds | None = None,
) -> tuple[ValidationStatus, list[Finding]]:
    """Validate one sample. Never raises on bad data."""

    findings: list[Finding] = []
    root = Path(dataset_root)
    limits = thresholds or ValidationThresholds()

    if not sample.sample_id or not sample.source_id:
        findings.append(
            Finding("missing_provenance", SEVERITY_ERROR, "sample or source id missing")
        )

    if sample.image_path is not None:
        if not _path_inside(root, sample.image_path):
            findings.append(
                Finding(
                    "path_escape", SEVERITY_ERROR, "image path escapes dataset root"
                )
            )
        elif (
            PurePosixPath(str(sample.image_path)).suffix.lower() not in IMAGE_EXTENSIONS
        ):
            findings.append(
                Finding(
                    "unsupported_format",
                    SEVERITY_ERROR,
                    f"unsupported image format: {sample.image_path}",
                )
            )
        else:
            _validate_image(sample, root, findings, limits)
    elif limits.require_image:
        findings.append(Finding("missing_image", SEVERITY_ERROR, "sample has no image"))

    text = sample.raw_text if sample.raw_text is not None else sample.text
    if text is None:
        if limits.require_text:
            findings.append(
                Finding("missing_text", SEVERITY_ERROR, "sample has no text")
            )
    else:
        if not text.strip():
            findings.append(
                Finding("empty_text", SEVERITY_ERROR, "text is empty or whitespace")
            )
        elif len(text) > limits.max_text_chars:
            findings.append(
                Finding(
                    "text_too_long",
                    SEVERITY_ERROR,
                    f"text length {len(text)} exceeds limit",
                )
            )
        for ch in text:
            if ord(ch) < 32 and ch not in "\n\t\r":
                findings.append(
                    Finding(
                        "invalid_control_characters",
                        SEVERITY_WARNING,
                        "raw text contains control characters",
                    )
                )
                break

    if sample.provenance.get("encoding_ok") is False:
        findings.append(
            Finding("invalid_encoding", SEVERITY_WARNING, "text is not valid UTF-8")
        )
    if sample.provenance.get("malformed_metadata") is True:
        findings.append(
            Finding("malformed_metadata", SEVERITY_ERROR, "record could not be parsed")
        )

    status = ValidationStatus.OK
    for finding in findings:
        if finding.severity == SEVERITY_ERROR:
            status = ValidationStatus.ERROR
        elif finding.severity == SEVERITY_WARNING and status == ValidationStatus.OK:
            status = ValidationStatus.WARNING
    return status, findings


def apply_validation(
    samples: list[DatasetSample],
    dataset_root: Path,
    *,
    thresholds: ValidationThresholds | None = None,
) -> tuple[list[DatasetSample], dict[str, object]]:
    """Validate all samples, returning updated samples and a report."""

    updated: list[DatasetSample] = []
    counts = {"ok": 0, "warning": 0, "error": 0, "excluded": 0}
    severity_counts = {
        SEVERITY_INFO: 0,
        SEVERITY_WARNING: 0,
        SEVERITY_ERROR: 0,
        SEVERITY_EXCLUSION: 0,
    }
    for sample in samples:
        status, findings = validate_sample(sample, dataset_root, thresholds=thresholds)
        prior_validation_codes = {
            str(finding.get("code"))
            for finding in sample.validation_findings
            if isinstance(finding, dict)
        }
        exclusion_reason = (
            None
            if sample.exclusion_reason in prior_validation_codes
            else sample.exclusion_reason
        )
        quality_flags = [
            flag
            for flag in sample.quality_flags
            if flag not in {"validation_error", "validation_warning"}
        ]
        if status == ValidationStatus.ERROR:
            first_error = next(
                finding for finding in findings if finding.severity == SEVERITY_ERROR
            )
            exclusion_reason = exclusion_reason or first_error.code
            counts["error"] += 1
            quality_flags.append("validation_error")
        elif status == ValidationStatus.WARNING:
            counts["warning"] += 1
            quality_flags.append("validation_warning")
        else:
            counts["ok"] += 1
        for finding in findings:
            severity_counts[finding.severity] = (
                severity_counts.get(finding.severity, 0) + 1
            )
        updated.append(
            sample.evolve(
                validation_status=status,
                validation_findings=[finding.to_dict() for finding in findings],
                exclusion_reason=exclusion_reason,
                quality_flags=sorted(set(quality_flags)),
            )
        )
    report = {
        "counts": counts,
        "findings_by_severity": severity_counts,
        "total": len(updated),
        "schema_version": "clouda.pretraining.validation.v1",
    }
    return updated, report
