"""Model, rollup, serialization, and exit-code contract tests."""

from __future__ import annotations

import json

from clouda_data.doctor.models import (
    DOCTOR_SCHEMA_VERSION,
    DoctorCheck,
    DoctorReport,
    DoctorSection,
    DoctorStatus,
)
from clouda_data.doctor.report import EXIT_CHECKS_FAILED, EXIT_OK, exit_code_for


def _check(
    status: DoctorStatus, *, required: bool = True, id_: str = "x"
) -> DoctorCheck:
    return DoctorCheck(
        id=id_,
        name="n",
        subsystem="s",
        status=status,
        message="m",
        required=required,
    )


def test_check_serialization_roundtrip():
    check = DoctorCheck(
        id="core.python-version",
        name="Python version",
        subsystem="core",
        status=DoctorStatus.PASS,
        message="ok",
        details={"python": "3.11.16"},
        remediation=None,
    )
    payload = check.to_dict()
    assert payload["status"] == "PASS"
    assert payload["details"] == {"python": "3.11.16"}
    text = json.dumps(payload)
    assert "PASS" in text


def test_section_rollup_fail_dominates():
    section = DoctorSection(
        id="s",
        name="S",
        checks=[_check(DoctorStatus.PASS), _check(DoctorStatus.FAIL)],
    )
    assert section.rollup_status() is DoctorStatus.FAIL


def test_section_rollup_warn_over_skip_and_pass():
    section = DoctorSection(
        id="s",
        name="S",
        checks=[
            _check(DoctorStatus.SKIP),
            _check(DoctorStatus.WARN),
            _check(DoctorStatus.PASS),
        ],
    )
    assert section.rollup_status() is DoctorStatus.WARN


def test_section_rollup_all_skip_is_skip():
    section = DoctorSection(
        id="s", name="S", checks=[_check(DoctorStatus.SKIP), _check(DoctorStatus.SKIP)]
    )
    assert section.rollup_status() is DoctorStatus.SKIP


def test_section_rollup_skip_and_info_is_info():
    section = DoctorSection(
        id="s", name="S", checks=[_check(DoctorStatus.SKIP), _check(DoctorStatus.INFO)]
    )
    assert section.rollup_status() is DoctorStatus.INFO


def test_section_rollup_empty_is_pass():
    assert DoctorSection(id="s", name="S").rollup_status() is DoctorStatus.PASS


def test_report_overall_ignores_skip_and_info():
    report = DoctorReport(
        timestamp="2026-01-01T00:00:00Z",
        sections=[
            DoctorSection(id="a", name="A", checks=[_check(DoctorStatus.SKIP)]),
            DoctorSection(id="b", name="B", checks=[_check(DoctorStatus.INFO)]),
            DoctorSection(id="c", name="C", checks=[_check(DoctorStatus.PASS)]),
        ],
    )
    assert report.overall_status() is DoctorStatus.PASS
    assert report.readiness_label() == "READY"


def test_report_readiness_labels():
    base = dict(timestamp="t")
    fail = DoctorReport(
        sections=[DoctorSection(id="a", name="A", checks=[_check(DoctorStatus.FAIL)])],
        **base,
    )
    warn = DoctorReport(
        sections=[DoctorSection(id="a", name="A", checks=[_check(DoctorStatus.WARN)])],
        **base,
    )
    assert fail.readiness_label() == "NOT READY"
    assert warn.readiness_label() == "PARTIALLY READY"


def test_report_failed_required_checks_ignores_optional_fail():
    report = DoctorReport(
        timestamp="t",
        sections=[
            DoctorSection(
                id="a",
                name="A",
                checks=[
                    _check(DoctorStatus.FAIL, required=False, id_="optional-fail"),
                    _check(DoctorStatus.PASS, id_="ok"),
                ],
            )
        ],
    )
    assert report.failed_required_checks() == []
    assert exit_code_for(report) == EXIT_OK
    assert report.overall_status() is DoctorStatus.WARN


def test_exit_code_required_fail():
    report = DoctorReport(
        timestamp="t",
        sections=[DoctorSection(id="a", name="A", checks=[_check(DoctorStatus.FAIL)])],
    )
    assert exit_code_for(report) == EXIT_CHECKS_FAILED


def test_report_to_json_is_stable_and_schema_versioned():
    report = DoctorReport(
        timestamp="2026-01-01T00:00:00Z",
        sections=[DoctorSection(id="a", name="A", checks=[_check(DoctorStatus.PASS)])],
    )
    first = report.to_json()
    second = report.to_json()
    assert first == second  # deterministic serialization
    payload = json.loads(first)
    assert payload["schema_version"] == DOCTOR_SCHEMA_VERSION
    assert payload["schema_version"] == "clouda.ocr.doctor.v1"
    assert payload["timestamp"] == "2026-01-01T00:00:00Z"
    assert payload["overall_status"] == "PASS"
    assert payload["readiness"] == "READY"
    assert isinstance(payload["sections"], list)
