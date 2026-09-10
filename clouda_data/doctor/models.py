"""Typed domain models for the Clouda Environment Doctor.

The Doctor is a diagnostic-only subsystem: it inspects the active interpreter,
import resolution, optional dependency groups, renderers, the Training
Experiment Framework, GPU/CUDA, storage, and Git/worktree state, then reports
readiness. It never mutates the environment (no installs, no PATH edits, no
Git writes).

Model conventions:

- Every check carries an :class:`DoctorStatus` plus ``required``; exit codes
  derive only from *required* checks (Phase 19 / docs/doctor.md).
- ``details`` holds structured technical metadata (versions, paths, counts);
  ``remediation`` holds a short fix hint. Neither may contain secret values;
  environment-variable reporting is SET/NOT-SET only (see ``security.py``).
- Serialization is plain JSON-able dictionaries (``to_dict``) so
  ``clouda doctor --json`` is stable for the future Lab API/UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .security import redact_text, redact_value

DOCTOR_SCHEMA_VERSION = "clouda.ocr.doctor.v1"


class DoctorStatus(Enum):
    """Outcome of a single check or rollup of a section/report.

    ``SKIP`` marks checks that could not apply (subsystem absent from the
    branch base); ``INFO`` marks purely informational results. Neither ever
    fails the doctor.
    """

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"
    INFO = "INFO"


@dataclass
class DoctorCheck:
    """One named diagnostic with an outcome and optional fix hint."""

    id: str
    name: str
    subsystem: str
    status: DoctorStatus
    message: str
    required: bool = True
    details: dict[str, Any] = field(default_factory=dict)
    remediation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "subsystem": self.subsystem,
            "status": self.status.value,
            "message": redact_text(self.message),
            "required": self.required,
            "details": redact_value(dict(self.details)),
            "remediation": (
                redact_text(self.remediation) if self.remediation is not None else None
            ),
        }


@dataclass
class DoctorSection:
    """A named group of checks (Core, Data Factory, Rendering, ...)."""

    id: str
    name: str
    checks: list[DoctorCheck] = field(default_factory=list)

    def rollup_status(self) -> DoctorStatus:
        """Aggregate check outcomes: FAIL > WARN > SKIP > INFO > PASS.

        A section is ``PASS`` only when every check passed; only *required*
        FAILs make it ``FAIL`` (an optional FAIL rolls up as WARN — the doctor
        must not fail the run over optional capabilities), any WARN makes it
        ``WARN``, and sections composed solely of SKIP/INFO checks report
        ``SKIP``/``INFO``.
        """
        statuses = [check.status for check in self.checks]
        required_fail = any(
            check.status is DoctorStatus.FAIL and check.required
            for check in self.checks
        )
        optional_fail = any(
            check.status is DoctorStatus.FAIL and not check.required
            for check in self.checks
        )
        if required_fail:
            return DoctorStatus.FAIL
        if optional_fail or any(s is DoctorStatus.WARN for s in statuses):
            return DoctorStatus.WARN
        if statuses and all(s is DoctorStatus.SKIP for s in statuses):
            return DoctorStatus.SKIP
        if statuses and all(
            s in (DoctorStatus.SKIP, DoctorStatus.INFO) for s in statuses
        ):
            return DoctorStatus.INFO
        return DoctorStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.rollup_status().value,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass
class DoctorReport:
    """Complete doctor result: runtime, repo context, sections, rollup."""

    schema_version: str = DOCTOR_SCHEMA_VERSION
    timestamp: str = ""
    deep: bool = False
    runtime: dict[str, Any] = field(default_factory=dict)
    repository: dict[str, Any] = field(default_factory=dict)
    sections: list[DoctorSection] = field(default_factory=list)

    def overall_status(self) -> DoctorStatus:
        """Report rollup mirrors section rollup, ignoring SKIP/INFO."""
        statuses = [section.rollup_status() for section in self.sections]
        effective = [
            s for s in statuses if s not in (DoctorStatus.SKIP, DoctorStatus.INFO)
        ]
        if any(s is DoctorStatus.FAIL for s in effective):
            return DoctorStatus.FAIL
        if any(s is DoctorStatus.WARN for s in effective):
            return DoctorStatus.WARN
        if all(s is DoctorStatus.SKIP for s in statuses):
            return DoctorStatus.SKIP
        if statuses and all(
            s in (DoctorStatus.SKIP, DoctorStatus.INFO) for s in statuses
        ):
            return DoctorStatus.INFO
        return DoctorStatus.PASS

    def failed_required_checks(self) -> list[DoctorCheck]:
        return [
            check
            for section in self.sections
            for check in section.checks
            if check.required and check.status is DoctorStatus.FAIL
        ]

    def warn_checks(self) -> list[DoctorCheck]:
        return [
            check
            for section in self.sections
            for check in section.checks
            if check.status is DoctorStatus.WARN
        ]

    def readiness_label(self) -> str:
        """Human subsystem-readiness label (docs/doctor.md vocabulary)."""
        return {
            DoctorStatus.PASS: "READY",
            DoctorStatus.WARN: "PARTIALLY READY",
            DoctorStatus.FAIL: "NOT READY",
            DoctorStatus.SKIP: "NOT PRESENT",
            DoctorStatus.INFO: "INFO",
        }[self.overall_status()]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "deep": self.deep,
            "overall_status": self.overall_status().value,
            "readiness": self.readiness_label(),
            "runtime": dict(self.runtime),
            "repository": dict(self.repository),
            "sections": [section.to_dict() for section in self.sections],
        }

    def to_json(self, indent: int | None = 2) -> str:
        import json

        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=indent, sort_keys=False
        )
