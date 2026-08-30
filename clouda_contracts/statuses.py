from __future__ import annotations

from enum import StrEnum


class ObservationStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PENDING = "pending"
    MANUAL_REVIEW = "manual_review"


class DatasetUseStatus(StrEnum):
    APPROVED = "approved"
    APPROVED_WITH_CONDITIONS = "approved_with_conditions"
    RESEARCH_ONLY = "research_only"
    EVALUATION_ONLY = "evaluation_only"
    PENDING = "pending"
    BLOCKED = "blocked"
    EXPIRED = "expired"
