from __future__ import annotations

from enum import Enum


class PageStatus(str, Enum):
    QUEUED = "queued"
    RENDERING = "rendering"
    DISTORTING = "distorting"
    VALIDATING = "validating"
    DONE = "done"
    FAILED = "failed"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"
    CANCELLED = "cancelled"


ALLOWED_TRANSITIONS: dict[PageStatus, set[PageStatus]] = {
    PageStatus.QUEUED: {
        PageStatus.RENDERING,
        PageStatus.DISTORTING,
        PageStatus.CANCELLED,
    },
    PageStatus.RENDERING: {
        PageStatus.DISTORTING,
        PageStatus.FAILED,
        PageStatus.CANCELLED,
    },
    PageStatus.DISTORTING: {
        PageStatus.VALIDATING,
        PageStatus.FAILED,
        PageStatus.CANCELLED,
    },
    PageStatus.VALIDATING: {
        PageStatus.DONE,
        PageStatus.REJECTED,
        PageStatus.MANUAL_REVIEW,
        PageStatus.FAILED,
    },
    PageStatus.FAILED: {PageStatus.QUEUED, PageStatus.CANCELLED},
    PageStatus.REJECTED: {PageStatus.MANUAL_REVIEW, PageStatus.CANCELLED},
    PageStatus.MANUAL_REVIEW: {
        PageStatus.DONE,
        PageStatus.REJECTED,
        PageStatus.CANCELLED,
    },
    PageStatus.DONE: set(),
    PageStatus.CANCELLED: {PageStatus.QUEUED},
}


def assert_transition(current: PageStatus, target: PageStatus) -> None:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise ValueError(
            f"Invalid status transition: {current.value} -> {target.value}"
        )
