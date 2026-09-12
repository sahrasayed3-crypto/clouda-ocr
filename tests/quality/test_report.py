"""Unit tests for clouda_data.quality.report and results_bridge."""

from __future__ import annotations

import json
from typing import Any

import pytest

from clouda_contracts.security import redact_mapping
from clouda_data.quality.models import (
    DatasetHealthSummary,
    ExclusionDecision,
    GateVerdict,
    IssueCode,
    IssueSeverity,
    LeakageFinding,
    QualityGateResult,
    QualityIssue,
)
from clouda_data.quality.report import (
    REQUIRED_REPORT_KEYS,
    build_report_payload,
    render_report,
    verdict_for,
)
from clouda_data.quality.results_bridge import (
    ResultsBridgeError,
    persist_quality_summary,
    quality_summary,
)

SECRET_SNIPPET = "SECRET-PROTECTED-PAYLOAD-XYZ"


def _issue(
    code: str,
    severity: IssueSeverity,
    sample_ids: tuple[str, ...],
    message: str = "issue message",
    evidence: dict[str, Any] | None = None,
) -> QualityIssue:
    return QualityIssue(
        code=code,
        severity=severity,
        sample_ids=sample_ids,
        canonical_key=code + ":" + ",".join(sorted(sample_ids)),
        message=message,
        evidence=evidence if evidence is not None else {},
    )


def _finding(sample_ids: tuple[str, ...], protected_value: str) -> LeakageFinding:
    return LeakageFinding(
        finding_id="LKG_" + "a" * 12,
        kind="exact_hash",
        severity=IssueSeverity.CRITICAL,
        partitions=("train", "holdout"),
        sample_ids=sample_ids,
        canonical_key="k1",
        corroborating_signals=("file_hash",),
        raw_split_values={"s1": "train", "s2": protected_value},
        detail="leak " + protected_value,
    )


def _result(
    issues: tuple[QualityIssue, ...] = (),
    findings: tuple[LeakageFinding, ...] = (),
    verdict: GateVerdict | None = None,
) -> QualityGateResult:
    if verdict is None:
        verdict = (
            GateVerdict.FAIL
            if any(
                i.severity in (IssueSeverity.ERROR, IssueSeverity.CRITICAL)
                for i in issues
            )
            else (
                GateVerdict.PASS_WITH_WARNINGS
                if any(i.severity is IssueSeverity.WARNING for i in issues)
                else GateVerdict.PASS
            )
        )
    return QualityGateResult(
        run_id="run-test-1",
        verdict=verdict,
        issues=issues,
        leakage_findings=findings,
        health=DatasetHealthSummary(dimensions={"split": {"train": 2}}),
        exclusions=(
            ExclusionDecision(
                sample_id="s2",
                reason_code=IssueCode.DUP_EXACT,
                reason_source="dedupe",
            ),
        ),
        reason_codes=(IssueCode.DUP_EXACT,),
    )


class TestVerdictFor:
    def test_critical_issue_fails(self) -> None:
        result = _result(
            issues=(_issue(IssueCode.HASH_MISMATCH, IssueSeverity.CRITICAL, ("s1",)),)
        )
        assert verdict_for(result) is GateVerdict.FAIL

    def test_error_issue_fails(self) -> None:
        result = _result(
            issues=(_issue(IssueCode.IMAGE_DECODE, IssueSeverity.ERROR, ("s1",)),)
        )
        assert verdict_for(result) is GateVerdict.FAIL

    def test_warning_only_passes_with_warnings(self) -> None:
        result = _result(
            issues=(
                _issue(IssueCode.BLANK_PAGE, IssueSeverity.WARNING, ("s1",)),
                _issue(IssueCode.VERY_SHORT_GT, IssueSeverity.INFO, ("s2",)),
            )
        )
        assert verdict_for(result) is GateVerdict.PASS_WITH_WARNINGS

    def test_clean_passes(self) -> None:
        assert verdict_for(_result()) is GateVerdict.PASS


class TestReportProtectionFilter:
    def test_protected_issue_content_stripped(self) -> None:
        result = _result(
            issues=(
                _issue(
                    IssueCode.HASH_MISMATCH,
                    IssueSeverity.CRITICAL,
                    ("s1",),
                    message="contains " + SECRET_SNIPPET,
                    evidence={"path": "holdout/x.png", "note": SECRET_SNIPPET},
                ),
            )
        )
        for fmt in ("json", "text"):
            out = render_report(result, fmt, protected_ids={"s1"})
            assert SECRET_SNIPPET not in out
            assert "holdout/x.png" not in out
        payload = build_report_payload(result, protected_ids={"s1"})
        issue = payload["issues"][0]
        assert issue["protected"] is True
        assert issue["sample_ids"] == ["s1"]
        assert "message" not in issue and "evidence" not in issue

    def test_unprotected_issue_keeps_redacted_content(self) -> None:
        result = _result(
            issues=(
                _issue(
                    IssueCode.HASH_MISMATCH,
                    IssueSeverity.CRITICAL,
                    ("s1",),
                    message="plain message",
                ),
            )
        )
        payload = build_report_payload(result, protected_ids={"other"})
        issue = payload["issues"][0]
        assert "protected" not in issue
        assert issue["message"] == "plain message"

    def test_protected_leakage_finding_stripped(self) -> None:
        result = _result(findings=(_finding(("s1", "s2"), SECRET_SNIPPET),))
        for fmt in ("json", "text"):
            out = render_report(result, fmt, protected_ids={"s1"})
            assert SECRET_SNIPPET not in out
        payload = build_report_payload(result, protected_ids={"s1"})
        finding = payload["leakage_findings"][0]
        assert finding["protected"] is True
        assert finding["sample_ids"] == ["s1", "s2"]
        assert "raw_split_values" not in finding and "detail" not in finding

    def test_unprotected_leakage_finding_keeps_redacted_splits(self) -> None:
        result = _result(findings=(_finding(("s1", "s2"), SECRET_SNIPPET),))
        payload = build_report_payload(result, protected_ids=set())
        finding = payload["leakage_findings"][0]
        assert finding["detail"] == "leak " + SECRET_SNIPPET
        assert finding["raw_split_values"] == {"s1": "train", "s2": SECRET_SNIPPET}


class TestRedaction:
    def test_redact_mapping_behavior_matches_contract(self) -> None:
        echo = {"api_key": "super-secret", "nested": {"authorization": "Bearer x"}}
        redacted = redact_mapping(echo)
        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["nested"]["authorization"] == "[REDACTED]"

    def test_redact_mapping_applied_to_issue_evidence(self) -> None:
        result = _result(
            issues=(
                _issue(
                    IssueCode.MISSING_IMAGE,
                    IssueSeverity.ERROR,
                    ("s1",),
                    evidence={"provenance": {"api_key": "super-secret", "ok": 1}},
                ),
            )
        )
        payload = build_report_payload(result)
        provenance = payload["issues"][0]["evidence"]["provenance"]
        assert provenance["api_key"] == "[REDACTED]"
        assert provenance["ok"] == 1


class TestJsonSchema:
    def test_required_keys_present(self) -> None:
        result = _result(
            issues=(_issue(IssueCode.DUP_EXACT, IssueSeverity.WARNING, ("s1", "s2")),)
        )
        payload = build_report_payload(
            result, artifact_paths={"report": "runs/x/report.json"}
        )
        for key in REQUIRED_REPORT_KEYS:
            assert key in payload, key
        assert payload["schema_version"] == "clouda.quality.run.v1"
        assert payload["verdict"] == "PASS_WITH_WARNINGS"
        assert payload["artifact_paths"] == {"report": "runs/x/report.json"}
        assert payload["severity_counts"]["warning"] == 1
        assert payload["reason_code_counts"] == {IssueCode.DUP_EXACT: 1}
        assert payload["clusters"]["count"] == 0
        assert payload["health"]["schema_version"] == "clouda.dataset.health.v1"
        assert payload["exclusions"][0]["sample_id"] == "s2"

    def test_render_json_roundtrip(self) -> None:
        doc = json.loads(render_report(_result(), "json"))
        assert doc["run_id"] == "run-test-1"
        assert doc["verdict"] == "PASS"


class TestTextFormat:
    def test_ends_with_gate_line(self) -> None:
        result = _result(
            issues=(_issue(IssueCode.BLANK_PAGE, IssueSeverity.WARNING, ("s1",)),)
        )
        lines = render_report(result, "text").strip().splitlines()
        assert lines[0] == "Dataset Health Report"
        assert lines[-1] == "QUALITY GATE: PASS_WITH_WARNINGS"

    def test_gate_line_pass_and_fail(self) -> None:
        assert render_report(_result(), "text").strip().endswith("QUALITY GATE: PASS")
        bad = _result(
            issues=(_issue(IssueCode.HASH_MISMATCH, IssueSeverity.CRITICAL, ("s9",)),)
        )
        assert render_report(bad, "text").strip().endswith("QUALITY GATE: FAIL")

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError):
            render_report(_result(), "yaml")


class FakeStore:
    def __init__(self) -> None:
        self.saved: dict[str, dict[str, Any]] = {}

    def save_summary(self, run_id: str, summary: dict[str, Any]) -> Any:
        self.saved[run_id] = summary
        return f"saved:{run_id}"


class TestResultsBridge:
    def test_persists_via_fake_store(self) -> None:
        store = FakeStore()
        summary = quality_summary(
            "run-42",
            {"severity_counts": {"critical": 1}},
            config_identity="cfg-abc",
            verdict="FAIL",
        )
        persist_quality_summary(store, "run-42", summary)
        saved = store.saved["run-42"]
        assert saved["metadata"]["kind"] == "dataset_quality_run"
        assert saved["metadata"]["config_identity"] == "cfg-abc"
        assert saved["metadata"]["verdict"] == "FAIL"
        assert saved["severity_counts"] == {"critical": 1}

    def test_store_without_save_summary_raises(self) -> None:
        class BadStore:
            pass

        with pytest.raises(ResultsBridgeError):
            persist_quality_summary(BadStore(), "run-1", {})

    def test_metadata_override_preserves_kind(self) -> None:
        summary = quality_summary(
            "run-7",
            {},
            config_identity="cfg",
            verdict="PASS",
            metadata={"verdict": "PASS_WITH_WARNINGS", "extra": 1},
        )
        assert summary["metadata"]["kind"] == "dataset_quality_run"
        assert summary["metadata"]["verdict"] == "PASS_WITH_WARNINGS"
        assert summary["metadata"]["extra"] == 1
