"""Sample validation independent of any OCR model.

Validation returns structured findings classified by severity. One bad
sample never aborts the dataset; errors simply mark the sample for
exclusion with a machine-readable reason.
"""

from __future__ import annotations

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


class ValidationThresholds:
    """Validation limits (kept as a plain class for cheap construction)."""

    def __init__(
        self,
        *,
        min_width: int = 8,
        min_height: int = 8,
        max_pixels: int = 100_000_000,
        max_text_chars: int = 100_000,
        require_text: bool = True,
        require_image: bool = True,
    ) -> None:
        self.min_width = min_width
        self.min_height = min_height
        self.max_pixels = max_pixels
        self.max_text_chars = max_text_chars
        self.require_text = require_text
        self.require_image = require_image


def _path_inside(root: Path, relative: str | None) -> bool:
    if relative is None:
        return True
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_image(sample: DatasetSample, root: Path, findings: list[Finding]) -> None:
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
        with Image.open(image_path) as image:
            width, height = image.size
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
    if width < 0 or height < 0:
        findings.append(
            Finding("invalid_dimensions", SEVERITY_ERROR, "negative dimensions")
        )
        return
    if width < _thresholds.min_width or height < _thresholds.min_height:
        findings.append(
            Finding(
                "tiny_image",
                SEVERITY_WARNING,
                f"image below minimum dimensions: {width}x{height}",
            )
        )
    if width * height > _thresholds.max_pixels:
        findings.append(
            Finding(
                "oversized_image",
                SEVERITY_WARNING,
                f"image exceeds max pixels: {width * height}",
            )
        )


_thresholds = ValidationThresholds()


def set_thresholds(thresholds: ValidationThresholds) -> None:
    """Install the active validation thresholds (module-level by design)."""

    global _thresholds
    _thresholds = thresholds


def validate_sample(
    sample: DatasetSample, dataset_root: Path
) -> tuple[ValidationStatus, list[Finding]]:
    """Validate one sample. Never raises on bad data."""

    findings: list[Finding] = []
    root = Path(dataset_root)

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
            _validate_image(sample, root, findings)
    elif _thresholds.require_image:
        findings.append(Finding("missing_image", SEVERITY_ERROR, "sample has no image"))

    text = sample.raw_text if sample.raw_text is not None else sample.text
    if text is None:
        if _thresholds.require_text:
            findings.append(
                Finding("missing_text", SEVERITY_ERROR, "sample has no text")
            )
    else:
        if not text.strip():
            findings.append(
                Finding("empty_text", SEVERITY_ERROR, "text is empty or whitespace")
            )
        elif len(text) > _thresholds.max_text_chars:
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
    samples: list[DatasetSample], dataset_root: Path
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
        status, findings = validate_sample(sample, dataset_root)
        exclusion_reason = sample.exclusion_reason
        quality_flags = list(sample.quality_flags)
        if status == ValidationStatus.ERROR:
            exclusion_reason = exclusion_reason or findings[0].code
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
                quality_flags=quality_flags,
            )
        )
    report = {
        "counts": counts,
        "findings_by_severity": severity_counts,
        "total": len(updated),
        "schema_version": "clouda.pretraining.validation.v1",
    }
    return updated, report
