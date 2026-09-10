"""Clouda Environment Doctor.

Diagnostic-only readiness inspection of the active interpreter, import
resolution, dependency groups, Data Factory, renderers (WeasyPrint / native
RAQM), Training Experiment Framework, GPU/CUDA, storage, Git/worktree state,
and privacy-safe environment-variable reporting.

Entry points:
    from clouda_data.doctor import collect_report, render_human
    python -m clouda_data.pipeline.cli doctor [--json] [--deep] ...

The Doctor never mutates the environment: no installs, no PATH edits, no
Git writes, no dataset generation. Exit codes are defined in
``report.exit_code_for`` (0 = no required FAIL, 1 = required FAIL,
2 = internal error) and documented in docs/doctor.md.
"""

from .models import (
    DOCTOR_SCHEMA_VERSION,
    DoctorCheck,
    DoctorReport,
    DoctorSection,
    DoctorStatus,
)
from .report import collect_report, exit_code_for
from .human import render_human

__all__ = [
    "DOCTOR_SCHEMA_VERSION",
    "DoctorCheck",
    "DoctorReport",
    "DoctorSection",
    "DoctorStatus",
    "collect_report",
    "exit_code_for",
    "render_human",
]
