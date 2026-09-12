"""Unit tests for quality models + configuration (Wave2-A)."""

from __future__ import annotations

import pytest

from clouda_data.quality.config import (
    QUALITY_GATE_CONFIG_VERSION,
    HeuristicsPolicy,
    ImageFingerprintPolicy,
    QualityGateConfig,
    QualityGateConfigError,
    SeverityPolicy,
)
from clouda_data.quality.models import (
    GateVerdict,
    IssueCode,
    IssueSeverity,
    QualityIssue,
)


def _make_issue(severity: IssueSeverity = IssueSeverity.WARNING) -> QualityIssue:
    return QualityIssue(
        code=IssueCode.DUP_EXACT,
        severity=severity,
        sample_ids=("smp_b", "smp_a"),
        canonical_key="dup:sha256:abc",
        message="two samples share a file hash",
        evidence={"file_sha256": "abc"},
    )


class TestQualityIssue:
    def test_round_trip_to_dict_from_dict(self) -> None:
        issue = _make_issue()
        payload = issue.to_dict()
        restored = QualityIssue.from_dict(payload)
        assert restored == issue
        assert payload["severity"] == "warning"
        assert payload["schema_version"] == "clouda.quality.issue.v1"
        assert payload["sample_ids"] == ("smp_a", "smp_b")

    def test_unknown_field_rejected(self) -> None:
        payload = _make_issue().to_dict()
        payload["bogus"] = 1
        with pytest.raises(ValueError, match="Unknown fields"):
            QualityIssue.from_dict(payload)

    def test_wrong_schema_version_rejected(self) -> None:
        payload = _make_issue().to_dict()
        payload["schema_version"] = "clouda.quality.issue.v2"
        with pytest.raises(ValueError, match="Unsupported QualityIssue schema"):
            QualityIssue.from_dict(payload)


class TestQualityGateConfig:
    def test_round_trip_to_dict_from_mapping(self) -> None:
        config = QualityGateConfig()
        payload = config.to_dict()
        restored = QualityGateConfig.from_mapping(payload)
        assert restored == config
        assert restored.fingerprint() == config.fingerprint()

    def test_unknown_field_rejected(self) -> None:
        payload = QualityGateConfig().to_dict()
        payload["nonexistent_policy"] = {}
        with pytest.raises(QualityGateConfigError, match="Unknown quality-gate"):
            QualityGateConfig.from_mapping(payload)

    def test_wrong_config_version_rejected(self) -> None:
        payload = QualityGateConfig().to_dict()
        payload["config_version"] = "clouda.quality.config.v0"
        with pytest.raises(QualityGateConfigError, match="Unsupported quality-gate"):
            QualityGateConfig.from_mapping(payload)

    def test_fingerprint_stable_across_constructions(self) -> None:
        assert QualityGateConfig().fingerprint() == QualityGateConfig().fingerprint()
        other = QualityGateConfig.from_mapping(
            {"image_fingerprint": {"hash_size": 8, "hamming_candidate": 12}}
        )
        assert other.fingerprint() == QualityGateConfig().fingerprint()

    def test_invalid_severity_rejected(self) -> None:
        with pytest.raises(QualityGateConfigError, match="severity.default"):
            QualityGateConfig.from_mapping({"severity": {"default": "explode"}})
        with pytest.raises(QualityGateConfigError, match="severity.overrides"):
            QualityGateConfig.from_mapping(
                {"severity": {"overrides": {"BLANK_PAGE": "boom"}}}
            )

    def test_fingerprint_changes_with_any_threshold(self) -> None:
        base = QualityGateConfig().fingerprint()
        mutated = [
            QualityGateConfig(
                image_fingerprint=ImageFingerprintPolicy(hamming_candidate=11)
            ),
            QualityGateConfig(heuristics=HeuristicsPolicy(blank_std=5.0)),
            QualityGateConfig(heuristics=HeuristicsPolicy(max_pixels=80_000_000)),
            QualityGateConfig(severity=SeverityPolicy(default="fail")),
        ]
        for variant in mutated:
            assert variant.fingerprint() != base

    def test_identity_is_version_plus_fingerprint(self) -> None:
        config = QualityGateConfig()
        assert config.identity() == (
            f"{QUALITY_GATE_CONFIG_VERSION}+{config.fingerprint()}"
        )

    def test_max_pixels_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLOUDA_MAX_IMAGE_PIXELS", "55_000_000".replace("_", ""))
        config = QualityGateConfig()
        assert config.heuristics.max_pixels == 55000000

    def test_max_pixels_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CLOUDA_MAX_IMAGE_PIXELS", raising=False)
        assert QualityGateConfig().heuristics.max_pixels == 40_000_000

    def test_verdicts_and_severities(self) -> None:
        assert {v.value for v in GateVerdict} == {
            "PASS",
            "PASS_WITH_WARNINGS",
            "FAIL",
        }
        assert {s.value for s in IssueSeverity} == {
            "info",
            "warning",
            "error",
            "critical",
        }
