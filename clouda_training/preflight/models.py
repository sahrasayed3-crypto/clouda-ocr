"""Typed domain model for the training preflight validator.

Fail-closed semantics
---------------------
- Any check whose status is ``FAIL`` and whose ``blocker`` flag is True makes
  the report :attr:`PreflightReport.final_status` ``NOT_READY``.
- A ``FAIL`` on a *non-blocker* check never blocks by itself: it is downgraded
  to a warning (report becomes ``READY_WITH_WARNINGS``) and surfaced in
  :attr:`PreflightReport.blockers`? No — it is surfaced in
  :attr:`PreflightReport.warnings` so the operator still sees it.
- ``UNAVAILABLE`` never blocks by itself; it always yields a warning entry.
  This keeps the report green-on-capability-missing (e.g. a capability is
  delegated to another canonical subsystem) while still being loud about it.
- ``SKIP`` is neutral: it neither blocks nor warns. A report composed only of
  ``SKIP``/``UNAVAILABLE``/``PASS`` checks with no warnings is ``READY``.

Aggregation rules (documented decision)
---------------------------------------
Given the union of all checks across all sections:

1. >= 1 blocker FAIL  -> ``NOT_READY``  (fail-closed dominates everything).
2. else >= 1 WARN, or >= 1 non-blocker FAIL, or >= 1 UNAVAILABLE
   -> ``READY_WITH_WARNINGS``.
3. else -> ``READY``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class PreflightStatus(str, Enum):
    """Outcome of a single preflight check."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"
    UNAVAILABLE = "UNAVAILABLE"


class PreflightSeverity(str, Enum):
    """How loud a check result should be reported."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class PreflightFinalStatus(str, Enum):
    """Aggregate verdict for the whole report."""

    READY = "READY"
    READY_WITH_WARNINGS = "READY_WITH_WARNINGS"
    NOT_READY = "NOT_READY"


@dataclass(frozen=True)
class PreflightCheck:
    """A single named check with a status and human-readable detail.

    ``blocker=True`` means a ``FAIL`` on this check makes the whole report
    ``NOT_READY``. ``WARN``/``UNAVAILABLE`` on a blocker still only warn —
    only ``FAIL`` blockers block.
    """

    name: str
    status: PreflightStatus
    detail: str = ""
    blocker: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass(frozen=True)
class PreflightSection:
    """A named group of checks (config, system, dataset, ...)."""

    name: str
    checks: tuple[PreflightCheck, ...] = ()

    def all_checks(self) -> tuple[PreflightCheck, ...]:
        return self.checks

    def worst_status(self) -> PreflightStatus | None:
        """Most severe status in this section, or None when it has no checks.

        Ordering: FAIL > WARN > UNAVAILABLE > SKIP > PASS.
        """
        order = (
            PreflightStatus.FAIL,
            PreflightStatus.WARN,
            PreflightStatus.UNAVAILABLE,
            PreflightStatus.SKIP,
            PreflightStatus.PASS,
        )
        present = {c.status for c in self.checks}
        for status in order:
            if status in present:
                return status
        return None

    def to_dict(self) -> dict[str, Any]:
        worst = self.worst_status()
        return {
            "name": self.name,
            "checks": [c.to_dict() for c in self.checks],
            "worst_status": worst.value if worst is not None else None,
        }


@dataclass(frozen=True)
class TrainingPlanSummary:
    """Pure-integer summary of the planned training run.

    Populated by ``clouda_training.preflight.plan.compute_training_plan``;
    the model only carries the values. ``None`` means "not computed".
    """

    effective_batch_size: int | None = None
    micro_batches_per_epoch: int | None = None
    optimizer_steps_per_epoch: int | None = None
    planned_optimizer_steps: int | None = None
    estimated_checkpoint_count: int | None = None
    world_size: int = 1
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["notes"] = list(self.notes)
        return data


@dataclass(frozen=True)
class TrainingBlocker:
    """Something that must be fixed before training may start."""

    check_name: str
    reason: str
    severity: PreflightSeverity = PreflightSeverity.ERROR

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        return data


@dataclass(frozen=True)
class TrainingWarning:
    """Something noteworthy that does not block training."""

    check_name: str
    message: str
    severity: PreflightSeverity = PreflightSeverity.WARNING

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["severity"] = self.severity.value
        return data


@dataclass(frozen=True)
class PreflightContext:
    """Identity-of-what-is-being-preflighted metadata for the report."""

    model_id: str | None = None
    adapter_type: str | None = None
    device: str | None = None
    precision: str | None = None
    dataset_id: str | None = None
    output_root: str | None = None
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreflightReport:
    """Aggregate preflight result across all sections."""

    sections: tuple[PreflightSection, ...] = ()
    context: PreflightContext | None = None
    training_plan: TrainingPlanSummary | None = None
    adapter_identity: str | None = None
    dataset_identity: str | None = None
    checkpoint_identity: str | None = None

    # -- aggregation ------------------------------------------------------

    def all_checks(self) -> tuple[PreflightCheck, ...]:
        return tuple(check for section in self.sections for check in section.checks)

    @property
    def blockers(self) -> tuple[TrainingBlocker, ...]:
        """Explicit blocker list: every ``FAIL`` check flagged as blocker."""
        return tuple(
            TrainingBlocker(check_name=c.name, reason=c.detail)
            for c in self.all_checks()
            if c.status is PreflightStatus.FAIL and c.blocker
        )

    @property
    def warnings(self) -> tuple[TrainingWarning, ...]:
        """Every non-blocking concern: WARN, non-blocker FAIL, UNAVAILABLE."""
        out: list[TrainingWarning] = []
        for c in self.all_checks():
            if c.status is PreflightStatus.WARN:
                out.append(TrainingWarning(check_name=c.name, message=c.detail))
            elif c.status is PreflightStatus.FAIL and not c.blocker:
                out.append(
                    TrainingWarning(
                        check_name=c.name,
                        message=f"non-blocking failure: {c.detail}",
                    )
                )
            elif c.status is PreflightStatus.UNAVAILABLE:
                out.append(
                    TrainingWarning(
                        check_name=c.name,
                        message=c.detail or "capability unavailable",
                        severity=PreflightSeverity.INFO,
                    )
                )
        return tuple(out)

    def final_status(self) -> PreflightFinalStatus:
        """Fail-closed aggregation (see module docstring for the rules)."""
        if self.blockers:
            return PreflightFinalStatus.NOT_READY
        if self.warnings:
            return PreflightFinalStatus.READY_WITH_WARNINGS
        return PreflightFinalStatus.READY

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Machine-readable snapshot of the full report."""
        final = self.final_status()
        return {
            "final_status": final.value,
            "ready": final is not PreflightFinalStatus.NOT_READY,
            "context": self.context.to_dict() if self.context else None,
            "sections": [s.to_dict() for s in self.sections],
            "blockers": [b.to_dict() for b in self.blockers],
            "warnings": [w.to_dict() for w in self.warnings],
            "training_plan": (
                self.training_plan.to_dict() if self.training_plan else None
            ),
            "adapter_identity": self.adapter_identity,
            "dataset_identity": self.dataset_identity,
            "checkpoint_identity": self.checkpoint_identity,
        }
