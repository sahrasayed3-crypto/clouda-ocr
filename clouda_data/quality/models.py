"""Frozen, versioned dataclasses for the dataset quality gate.

Every artifact produced by ``clouda_data.quality`` (issues, fingerprints,
duplicate groups, leakage findings, run results) is a plain, JSON-serializable
value object. All classes are frozen dataclasses that serialize to
canonical-JSON-compatible dictionaries (``sort_keys=True``, compact
separators, ``ensure_ascii=False``) with explicit schema-version checks,
following the ``clouda_data.pretraining.schema`` pattern. None of these
objects contain protected sample content: callers must pass IDs and codes
only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, TypeVar

QUALITY_ISSUE_SCHEMA_VERSION = "clouda.quality.issue.v1"
QUALITY_SAMPLE_FINGERPRINT_SCHEMA_VERSION = "clouda.quality.sample_fingerprint.v1"
QUALITY_EXACT_DUPLICATE_GROUP_SCHEMA_VERSION = "clouda.quality.exact_duplicate_group.v1"
QUALITY_NEAR_DUPLICATE_CANDIDATE_SCHEMA_VERSION = (
    "clouda.quality.near_duplicate_candidate.v1"
)
QUALITY_DUPLICATE_CLUSTER_SCHEMA_VERSION = "clouda.quality.duplicate_cluster.v1"
QUALITY_LEAKAGE_FINDING_SCHEMA_VERSION = "clouda.quality.leakage_finding.v1"
QUALITY_ARTIFACT_CHECK_SCHEMA_VERSION = "clouda.quality.artifact_check.v1"
QUALITY_HEALTH_SCHEMA_VERSION = "clouda.dataset.health.v1"
QUALITY_EXCLUSION_SCHEMA_VERSION = "clouda.quality.exclusion.v1"
QUALITY_RUN_SCHEMA_VERSION = "clouda.quality.run.v1"
QUALITY_GATE_RESULT_SCHEMA_VERSION = "clouda.quality.gate_result.v1"

NEAR_DUPLICATE_LEVELS = (
    "CANDIDATE",
    "LIKELY_DUPLICATE",
    "CONFIRMED_NEAR_DUPLICATE",
)

_A = TypeVar("_A")


def canonical_json(payload: Any) -> str:
    """Deterministic JSON encoding used for hashing and identity."""

    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def fingerprint_of(payload: dict[str, Any]) -> str:
    """SHA-256 hex digest of the canonical JSON of ``payload``."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class GateVerdict(str, Enum):
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL = "FAIL"


class IssueCode:
    """Closed vocabulary of quality-issue codes (module-level constants)."""

    MISSING_IMAGE = "MISSING_IMAGE"
    NON_EMPTY_FILE = "NON_EMPTY_FILE"
    HASH_MISMATCH = "HASH_MISMATCH"
    IMAGE_DECODE = "IMAGE_DECODE"
    IMAGE_DIMENSIONS = "IMAGE_DIMENSIONS"
    IMAGE_MODE = "IMAGE_MODE"
    GT_MISSING = "GT_MISSING"
    GT_EMPTY = "GT_EMPTY"
    PATH_SAFE = "PATH_SAFE"
    BLANK_PAGE = "BLANK_PAGE"
    NEAR_BLANK_PAGE = "NEAR_BLANK_PAGE"
    EXTREME_DIMENSIONS = "EXTREME_DIMENSIONS"
    EXTREME_ASPECT_RATIO = "EXTREME_ASPECT_RATIO"
    SUSPICIOUSLY_SMALL_IMAGE = "SUSPICIOUSLY_SMALL_IMAGE"
    VERY_LARGE_ARTIFACT = "VERY_LARGE_ARTIFACT"
    VERY_SHORT_GT = "VERY_SHORT_GT"
    METADATA_DIMENSION_MISMATCH = "METADATA_DIMENSION_MISMATCH"
    DUP_EXACT = "DUP_EXACT"
    DUP_NEAR_IMAGE = "DUP_NEAR_IMAGE"
    DUP_TEXT_NEAR = "DUP_TEXT_NEAR"
    LEAK_MALFORMED_PROTECTION = "LEAK_MALFORMED_PROTECTION"
    LEAK_EXACT_HASH = "LEAK_EXACT_HASH"
    LEAK_PAGE_IDENTITY = "LEAK_PAGE_IDENTITY"
    LEAK_NEAR_IMAGE = "LEAK_NEAR_IMAGE"
    LEAK_DERIVED_PAGE = "LEAK_DERIVED_PAGE"
    LEAK_GT_TEXT = "LEAK_GT_TEXT"
    LEAK_GROUP_STRADDLE = "LEAK_GROUP_STRADDLE"

    @classmethod
    def all_codes(cls) -> frozenset[str]:
        return frozenset(
            value
            for key, value in vars(cls).items()
            if key.isupper() and isinstance(value, str)
        )


def _versioned(cls: type[_A]) -> type[_A]:
    """Attach canonical-JSON ``to_dict``/``from_dict`` to an artifact dataclass.

    ``from_dict`` follows the ``DatasetSample`` pattern: unknown fields are
    rejected and the artifact's ``schema_version`` is checked before
    construction.
    """

    def to_dict(self: Any) -> dict[str, Any]:
        payload = asdict(self)
        # Enums serialize as their values for stable canonical JSON.
        for key, value in list(payload.items()):
            if isinstance(value, Enum):
                payload[key] = value.value
        return payload

    @classmethod  # type: ignore[misc]
    def from_dict(cls: type[_A], data: dict[str, Any]) -> _A:
        known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown fields for {cls.__name__}: {sorted(unknown)}")
        expected = cls.schema_version  # type: ignore[attr-defined]
        version = data.get("schema_version", expected)
        if version != expected:
            raise ValueError(
                f"Unsupported {cls.__name__} schema version: {version!r}; "
                f"expected {expected!r}."
            )
        kwargs = dict(data)
        for f in fields(cls):  # type: ignore[arg-type]
            value = kwargs.get(f.name)
            if isinstance(value, Enum):
                kwargs[f.name] = value.value
        return cls(**kwargs)  # type: ignore[return-value]

    cls.to_dict = to_dict  # type: ignore[attr-defined]
    cls.from_dict = from_dict  # type: ignore[attr-defined]
    return cls


@dataclass(frozen=True)
@_versioned
class QualityIssue:
    """One quality finding covering a set of samples (IDs + codes only)."""

    code: str
    severity: IssueSeverity
    sample_ids: tuple[str, ...]
    canonical_key: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    schema_version: str = QUALITY_ISSUE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.code not in IssueCode.all_codes():
            raise ValueError(f"Unknown issue code: {self.code!r}")
        object.__setattr__(self, "sample_ids", tuple(sorted(self.sample_ids)))


@dataclass(frozen=True)
@_versioned
class SampleFingerprint:
    """Perceptual fingerprint of one sample image."""

    sample_id: str
    source_id: str
    ahash: str
    dhash: str
    phash: str
    aspect_bucket: str
    blankish: bool
    blank_stats: dict[str, Any] = field(default_factory=dict)
    fingerprint_version: str = ""
    schema_version: str = QUALITY_SAMPLE_FINGERPRINT_SCHEMA_VERSION


@dataclass(frozen=True)
@_versioned
class ExactDuplicateGroup:
    """A group of samples sharing one exact duplicate key."""

    canonical_key: str
    member_ids: tuple[str, ...]
    schema_version: str = QUALITY_EXACT_DUPLICATE_GROUP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "member_ids", tuple(sorted(self.member_ids)))


@dataclass(frozen=True)
@_versioned
class NearDuplicateCandidate:
    """One near-duplicate pair candidate with per-hash distances."""

    sample_id_a: str
    sample_id_b: str
    level: str
    distances: dict[str, int] = field(default_factory=dict)
    schema_version: str = QUALITY_NEAR_DUPLICATE_CANDIDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.level not in NEAR_DUPLICATE_LEVELS:
            raise ValueError(f"Unsupported near-duplicate level: {self.level!r}")
        if self.sample_id_a == self.sample_id_b:
            raise ValueError(
                "Near-duplicate candidate cannot pair a sample with itself."
            )
        if self.sample_id_b < self.sample_id_a:
            object.__setattr__(self, "sample_id_a", self.sample_id_b)
            object.__setattr__(self, "sample_id_b", self.sample_id_a)


@dataclass(frozen=True)
@_versioned
class DuplicateCluster:
    """A deterministic cluster of duplicate/near-duplicate samples."""

    cluster_id: str
    member_ids: tuple[str, ...]
    level: str
    evidence: dict[str, Any] = field(default_factory=dict)
    schema_version: str = QUALITY_DUPLICATE_CLUSTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "member_ids", tuple(sorted(self.member_ids)))
        if not self.cluster_id.startswith("CLU_"):
            raise ValueError(
                "DuplicateCluster.cluster_id must start with 'CLU_'; got "
                f"{self.cluster_id!r}."
            )


@dataclass(frozen=True)
@_versioned
class LeakageFinding:
    """One cross-split leakage finding (IDs + codes, never content)."""

    finding_id: str
    kind: str
    severity: IssueSeverity
    partitions: tuple[str, ...]
    sample_ids: tuple[str, ...]
    canonical_key: str
    corroborating_signals: tuple[str, ...]
    raw_split_values: dict[str, str]
    detail: str
    schema_version: str = QUALITY_LEAKAGE_FINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "partitions", tuple(sorted(set(self.partitions))))
        object.__setattr__(self, "sample_ids", tuple(sorted(self.sample_ids)))
        object.__setattr__(
            self, "corroborating_signals", tuple(sorted(self.corroborating_signals))
        )
        if not self.finding_id.startswith("LKG_"):
            raise ValueError(
                "LeakageFinding.finding_id must start with 'LKG_'; got "
                f"{self.finding_id!r}."
            )


@dataclass(frozen=True)
@_versioned
class ArtifactCheckResult:
    """Outcome of artifact integrity checks for one sample."""

    sample_id: str
    checks: dict[str, Any] = field(default_factory=dict)
    schema_version: str = QUALITY_ARTIFACT_CHECK_SCHEMA_VERSION


@dataclass(frozen=True)
@_versioned
class DatasetHealthSummary:
    """Counts-only health summary across dataset dimensions."""

    dimensions: dict[str, dict[str, Any]] = field(default_factory=dict)
    cross_tabs: dict[str, dict[str, Any]] = field(default_factory=dict)
    bucket_definitions: dict[str, Any] = field(default_factory=dict)
    schema_version: str = QUALITY_HEALTH_SCHEMA_VERSION


@dataclass(frozen=True)
@_versioned
class ExclusionDecision:
    """Deterministic keep/exclude decision for one sample."""

    sample_id: str
    reason_code: str
    reason_source: str
    evidence: dict[str, Any] = field(default_factory=dict)
    schema_version: str = QUALITY_EXCLUSION_SCHEMA_VERSION


@dataclass(frozen=True)
@_versioned
class QualityRun:
    """Identity + lifecycle of one quality-gate run."""

    run_id: str
    manifest_sha256: str
    config_identity: str
    started_at: str
    finished_at: str
    verdict: GateVerdict
    severity_counts: dict[str, int] = field(default_factory=dict)
    issues: tuple[QualityIssue, ...] = ()
    schema_version: str = QUALITY_RUN_SCHEMA_VERSION


@dataclass(frozen=True)
@_versioned
class QualityGateResult:
    """Full output of one quality-gate run."""

    run_id: str
    verdict: GateVerdict
    issues: tuple[QualityIssue, ...] = ()
    clusters: tuple[DuplicateCluster, ...] = ()
    leakage_findings: tuple[LeakageFinding, ...] = ()
    health: DatasetHealthSummary = field(default_factory=DatasetHealthSummary)
    exclusions: tuple[ExclusionDecision, ...] = ()
    reason_codes: tuple[str, ...] = ()
    schema_version: str = QUALITY_GATE_RESULT_SCHEMA_VERSION
