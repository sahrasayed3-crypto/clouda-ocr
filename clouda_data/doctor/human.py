"""Human-readable doctor output (Phase 22).

Concise, plain-text, status-prefixed lines. No ANSI-color dependence — the
status word itself carries the meaning so output survives redirection, CI
logs, and color-blind terminals.
"""

from __future__ import annotations

from .models import DoctorReport, DoctorStatus

_STATUS_MARKS: dict[DoctorStatus, str] = {
    DoctorStatus.PASS: "PASS",
    DoctorStatus.WARN: "WARN",
    DoctorStatus.FAIL: "FAIL",
    DoctorStatus.SKIP: "SKIP",
    DoctorStatus.INFO: "INFO",
}


def render_human(report: DoctorReport, *, verbose: bool = False) -> str:
    lines: list[str] = ["Clouda Environment Doctor", ""]
    for section in report.sections:
        lines.append(section.name)
        for check in section.checks:
            payload = check.to_dict()
            mark = _STATUS_MARKS[check.status]
            lines.append(f"  {mark} {payload['message']}")
            if verbose:
                if payload["details"]:
                    for key, value in payload["details"].items():
                        lines.append(f"        {key}: {value}")
                if payload["remediation"]:
                    lines.append(f"        fix: {payload['remediation']}")
            elif (
                check.status in (DoctorStatus.WARN, DoctorStatus.FAIL)
                and payload["remediation"]
            ):
                lines.append(f"        fix: {payload['remediation']}")
        lines.append("")
    lines.append(f"Overall: {report.readiness_label()}")
    if report.failed_required_checks():
        lines.append(
            f"Exit: 1 ({len(report.failed_required_checks())} required check(s) failed)"
        )
    else:
        lines.append("Exit: 0")
    return "\n".join(lines)
