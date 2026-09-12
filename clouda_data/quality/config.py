"""Versioned configuration for the dataset quality gate.

Mirrors the :mod:`clouda_data.pretraining.config` pattern: a frozen top-level
config dataclass with nested frozen policy dataclasses, canonical-JSON
``fingerprint()``, ``version + fingerprint`` ``identity()``, and strict
``from_mapping`` that rejects unknown fields and wrong schema versions.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from clouda_data.pretraining.normalize import NormalizationPolicy

QUALITY_GATE_CONFIG_VERSION = "clouda.quality.config.v1"

# Image decode guard: can be overridden via environment so very large scans
# can raise the Pillow ceiling without code changes.
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000

SEVERITY_ACTIONS = frozenset({"info", "warn", "fail"})

DEFAULT_CRITICAL_LEAKAGE_CODES: tuple[str, ...] = (
    "LEAK_MALFORMED_PROTECTION",
    "LEAK_EXACT_HASH",
    "LEAK_PAGE_IDENTITY",
    "LEAK_NEAR_IMAGE",
    "LEAK_DERIVED_PAGE",
    "LEAK_GROUP_STRADDLE",
)

DEFAULT_KEEP_PREFERENCE: tuple[str, ...] = (
    "protected",
    "canonical_valid",
    "clean_over_distorted",
    "stable_sample_id",
)


class QualityGateConfigError(ValueError):
    """Raised when a quality-gate configuration is invalid."""


def _resolve_max_pixels(raw: Any) -> int:
    if raw is None:
        env_value = os.environ.get("CLOUDA_MAX_IMAGE_PIXELS", "")
        if not env_value:
            return DEFAULT_MAX_IMAGE_PIXELS
        try:
            return int(env_value)
        except ValueError as exc:
            raise QualityGateConfigError(
                "CLOUDA_MAX_IMAGE_PIXELS must be an integer; got " f"{env_value!r}."
            ) from exc
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise QualityGateConfigError("max_pixels must be an integer.")
    return raw


@dataclass(frozen=True)
class ExactDuplicatePolicy:
    """Exact-duplicate detection knobs (wrapping pretraining dedupe)."""

    include_raw_text_hash: bool = True


@dataclass(frozen=True)
class ImageFingerprintPolicy:
    """Perceptual image fingerprinting policy."""

    algorithm: str | None = "dhash"
    hash_size: int = 8
    hamming_candidate: int = 12
    hamming_confirmed: int = 10
    max_bucket: int = 4096


@dataclass(frozen=True)
class TextNearDuplicatePolicy:
    """Near-text duplicate policy built on pretraining normalization."""

    normalization: NormalizationPolicy = field(default_factory=NormalizationPolicy)
    shingle_k: int = 4
    minhash_perms: int = 128
    lsh_bands: int = 16
    lsh_rows: int = 8
    jaccard_near: float = 0.85
    jaccard_review: float = 0.70
    min_text_chars: int = 40


@dataclass(frozen=True)
class HeuristicsPolicy:
    """Artifact heuristics thresholds (default WARN severity)."""

    blank_std: float = 4.0
    near_blank_std: float = 12.0
    extreme_max_side: int = 40_000
    extreme_max_pixels: int = 400_000_000
    max_aspect_ratio: float = 50.0
    small_image_min_bytes: int = 1024
    max_artifact_bytes: int = 2_000_000_000
    min_gt_warn_chars: int = 3
    max_pixels: int = field(default_factory=lambda: _resolve_max_pixels(None))


@dataclass(frozen=True)
class SeverityPolicy:
    """Per-code severity overrides over the default action."""

    default: str = "warn"
    overrides: dict[str, str] = field(default_factory=dict)
    strict_escalates_warn: bool = True


@dataclass(frozen=True)
class LeakagePolicy:
    """Leakage / holdout-protection policy (contracts-only, never hand-rolled)."""

    critical_codes: tuple[str, ...] = DEFAULT_CRITICAL_LEAKAGE_CODES
    eval_eval_warn: bool = True


@dataclass(frozen=True)
class KeepExcludePolicy:
    """Deterministic keep/exclude preference order for duplicates."""

    exclude_duplicate: bool = True
    exclude_holdout: bool = True
    exclude_error: bool = True
    exclude_conflicting_duplicate: bool = False
    keep_preference: tuple[str, ...] = DEFAULT_KEEP_PREFERENCE


@dataclass(frozen=True)
class ResourceLimits:
    """Scan resource bounds."""

    workers: int = 1
    max_samples: int = 0  # 0 = unlimited
    decode_budget: int = 0  # 0 = unlimited decodes


@dataclass(frozen=True)
class QualityPaths:
    """Filesystem locations for index and resume state."""

    index_dir: str = ""
    resume: str = ""


@dataclass(frozen=True)
class QualityGateConfig:
    """Everything the quality gate needs, with safe defaults."""

    exact_duplicate: ExactDuplicatePolicy = field(default_factory=ExactDuplicatePolicy)
    image_fingerprint: ImageFingerprintPolicy = field(
        default_factory=ImageFingerprintPolicy
    )
    text_near_duplicate: TextNearDuplicatePolicy = field(
        default_factory=TextNearDuplicatePolicy
    )
    heuristics: HeuristicsPolicy = field(default_factory=HeuristicsPolicy)
    severity: SeverityPolicy = field(default_factory=SeverityPolicy)
    leakage: LeakagePolicy = field(default_factory=LeakagePolicy)
    keep_exclude: KeepExcludePolicy = field(default_factory=KeepExcludePolicy)
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    paths: QualityPaths = field(default_factory=QualityPaths)
    config_version: str = QUALITY_GATE_CONFIG_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["image_fingerprint"]["algorithm"] = self.image_fingerprint.algorithm
        payload["text_near_duplicate"]["normalization"] = asdict(
            self.text_near_duplicate.normalization
        )
        payload["keep_exclude"]["keep_preference"] = list(
            self.keep_exclude.keep_preference
        )
        payload["leakage"]["critical_codes"] = list(self.leakage.critical_codes)
        return payload

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def identity(self) -> str:
        return f"{self.config_version}+{self.fingerprint()}"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> QualityGateConfig:
        data = dict(data)
        version = data.pop("config_version", QUALITY_GATE_CONFIG_VERSION)
        if version != QUALITY_GATE_CONFIG_VERSION:
            raise QualityGateConfigError(
                f"Unsupported quality-gate config version: {version!r}"
            )
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise QualityGateConfigError(
                f"Unknown quality-gate config fields: {sorted(unknown)}"
            )
        kwargs = dict(data)

        exact_data = kwargs.pop("exact_duplicate", {}) or {}
        kwargs["exact_duplicate"] = ExactDuplicatePolicy(**exact_data)

        image_data = dict(kwargs.pop("image_fingerprint", {}) or {})
        kwargs["image_fingerprint"] = ImageFingerprintPolicy(**image_data)

        text_data = dict(kwargs.pop("text_near_duplicate", {}) or {})
        normalization_data = text_data.pop("normalization", {}) or {}
        text_data["normalization"] = NormalizationPolicy(**normalization_data)
        kwargs["text_near_duplicate"] = TextNearDuplicatePolicy(**text_data)

        heuristics_data = dict(kwargs.pop("heuristics", {}) or {})
        heuristics_data["max_pixels"] = _resolve_max_pixels(
            heuristics_data.get("max_pixels")
        )
        kwargs["heuristics"] = HeuristicsPolicy(**heuristics_data)

        severity_data = dict(kwargs.pop("severity", {}) or {})
        severity_data["overrides"] = dict(severity_data.get("overrides", {}) or {})
        kwargs["severity"] = SeverityPolicy(**severity_data)

        leakage_data = dict(kwargs.pop("leakage", {}) or {})
        leakage_data["critical_codes"] = tuple(
            leakage_data.get("critical_codes", DEFAULT_CRITICAL_LEAKAGE_CODES)
        )
        kwargs["leakage"] = LeakagePolicy(**leakage_data)

        keep_data = dict(kwargs.pop("keep_exclude", {}) or {})
        keep_data["keep_preference"] = tuple(
            keep_data.get("keep_preference", DEFAULT_KEEP_PREFERENCE)
        )
        kwargs["keep_exclude"] = KeepExcludePolicy(**keep_data)

        limits_data = dict(kwargs.pop("resource_limits", {}) or {})
        kwargs["resource_limits"] = ResourceLimits(**limits_data)

        paths_data = dict(kwargs.pop("paths", {}) or {})
        kwargs["paths"] = QualityPaths(**paths_data)

        config = cls(**kwargs)
        _validate(config)
        return config


def _validate(config: QualityGateConfig) -> None:
    image = config.image_fingerprint
    if image.algorithm not in {None, "dhash", "ahash", "phash"}:
        raise QualityGateConfigError(
            "image_fingerprint.algorithm must be None|dhash|ahash|phash"
        )
    if image.hash_size not in {8}:
        raise QualityGateConfigError("image_fingerprint.hash_size must be 8")
    for name in ("hamming_candidate", "hamming_confirmed"):
        value = getattr(image, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise QualityGateConfigError(
                f"image_fingerprint.{name} must be a non-negative integer"
            )
    if image.hamming_confirmed > image.hamming_candidate:
        raise QualityGateConfigError(
            "hamming_confirmed must be <= hamming_candidate (triple-conjunction)"
        )
    if (
        isinstance(image.max_bucket, bool)
        or not isinstance(image.max_bucket, int)
        or image.max_bucket < 1
    ):
        raise QualityGateConfigError("image_fingerprint.max_bucket must be >= 1")

    text = config.text_near_duplicate
    if text.shingle_k < 1 or text.minhash_perms < 1:
        raise QualityGateConfigError(
            "text_near_duplicate.shingle_k and minhash_perms must be >= 1"
        )
    if text.lsh_bands < 1 or text.lsh_rows < 1:
        raise QualityGateConfigError(
            "text_near_duplicate.lsh_bands and lsh_rows must be >= 1"
        )
    if not 0.0 <= text.jaccard_review <= text.jaccard_near <= 1.0:
        raise QualityGateConfigError(
            "text_near_duplicate thresholds must satisfy 0 <= jaccard_review"
            " <= jaccard_near <= 1"
        )
    if text.min_text_chars < 0:
        raise QualityGateConfigError("text_near_duplicate.min_text_chars must be >= 0")

    heuristics = config.heuristics
    for name in ("blank_std", "near_blank_std", "max_aspect_ratio"):
        value = getattr(heuristics, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise QualityGateConfigError(f"heuristics.{name} must be numeric")
    if heuristics.blank_std < 0 or heuristics.near_blank_std < 0:
        raise QualityGateConfigError("heuristics std thresholds must be >= 0")
    if heuristics.blank_std > heuristics.near_blank_std:
        raise QualityGateConfigError("heuristics.blank_std must be <= near_blank_std")
    for name in (
        "extreme_max_side",
        "extreme_max_pixels",
        "small_image_min_bytes",
        "max_artifact_bytes",
        "max_pixels",
        "min_gt_warn_chars",
    ):
        value = getattr(heuristics, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise QualityGateConfigError(f"heuristics.{name} must be an integer >= 0")
    if heuristics.max_pixels < 1:
        raise QualityGateConfigError("heuristics.max_pixels must be >= 1")

    severity = config.severity
    if severity.default not in SEVERITY_ACTIONS:
        raise QualityGateConfigError(
            f"severity.default must be one of {sorted(SEVERITY_ACTIONS)}; got "
            f"{severity.default!r}"
        )
    for code, action in severity.overrides.items():
        if action not in SEVERITY_ACTIONS:
            raise QualityGateConfigError(
                f"severity.overrides[{code!r}] must be one of "
                f"{sorted(SEVERITY_ACTIONS)}; got {action!r}"
            )

    keep = config.keep_exclude
    if keep.exclude_conflicting_duplicate and not keep.exclude_duplicate:
        raise QualityGateConfigError(
            "exclude_conflicting_duplicate=true requires exclude_duplicate=true "
            "(conflicting duplicates merge into duplicate clusters only when "
            "leakage-safe merge semantics are enabled)."
        )

    limits = config.resource_limits
    for name in ("workers", "max_samples", "decode_budget"):
        value = getattr(limits, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise QualityGateConfigError(
                f"resource_limits.{name} must be a non-negative integer"
            )
    if limits.workers < 1:
        raise QualityGateConfigError("resource_limits.workers must be at least 1")


def load_quality_gate_config(path: str | None = None) -> QualityGateConfig:
    """Load a config from a JSON file, or defaults when ``path`` is None."""

    if path is None:
        return QualityGateConfig()
    import pathlib

    text = pathlib.Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise QualityGateConfigError("Quality config file must contain a mapping.")
    return QualityGateConfig.from_mapping(data)
