"""Human- and machine-readable quality-gate reports.

``render_report`` serializes a :class:`QualityGateResult` either as the full
``clouda.quality.run.v1`` JSON document or as a human-readable text layout
whose final line is the gate verdict line. Two invariants hold on every path:

- Protected samples never leak content: any issue or leakage finding whose
  sample rows are protected (caller-supplied ``protected_ids``) is reduced to
  IDs + codes only — no message, evidence, detail, or split values.
- Echoed provenance/metadata dictionaries are passed through
  :func:`clouda_contracts.security.redact_mapping` before serialization.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

from clouda_contracts.security import redact_mapping
from clouda_data.quality.models import (
    GateVerdict,
    IssueSeverity,
    LeakageFinding,
    QualityGateResult,
    QualityIssue,
)


def _artifact_dict(artifact: Any) -> dict[str, Any]:
    """Call the dynamically-attached ``to_dict`` on a models.py artifact."""

    to_dict = getattr(artifact, "to_dict")
    return to_dict()


REPORT_SCHEMA_VERSION = "clouda.quality.run.v1"

TEXT_REPORT_TITLE = "Dataset Health Report"
_GATE_LINE_PREFIX = "QUALITY GATE: "

_SEVERITY_ORDER: tuple[IssueSeverity, ...] = (
    IssueSeverity.INFO,
    IssueSeverity.WARNING,
    IssueSeverity.ERROR,
    IssueSeverity.CRITICAL,
)

_FAIL_SEVERITIES = frozenset({IssueSeverity.ERROR, IssueSeverity.CRITICAL})

_PROTECTED_FLAG = "protected"

REQUIRED_REPORT_KEYS: tuple[str, ...] = (
    "schema_version",
    "run_id",
    "verdict",
    "severity_counts",
    "reason_code_counts",
    "issues",
    "clusters",
    "leakage_findings",
    "health",
    "exclusions",
    "artifact_paths",
)


def verdict_for(result: QualityGateResult) -> GateVerdict:
    """FAIL on any error/critical issue, PASS_WITH_WARNINGS on warnings, else PASS."""

    severities = {issue.severity for issue in result.issues}
    if severities & _FAIL_SEVERITIES:
        return GateVerdict.FAIL
    if IssueSeverity.WARNING in severities:
        return GateVerdict.PASS_WITH_WARNINGS
    return GateVerdict.PASS


def _issue_payload(
    issue: QualityIssue, protected_ids: frozenset[str]
) -> dict[str, Any]:
    """IDs + codes only when a sample row is protected; else redacted evidence."""

    if protected_ids.intersection(issue.sample_ids):
        return {
            "code": issue.code,
            "severity": issue.severity.value,
            "sample_ids": list(issue.sample_ids),
            "canonical_key": issue.canonical_key,
            _PROTECTED_FLAG: True,
        }
    payload = _artifact_dict(issue)
    payload["evidence"] = redact_mapping(issue.evidence)
    return payload


def _leakage_payload(
    finding: LeakageFinding, protected_ids: frozenset[str]
) -> dict[str, Any]:
    """IDs + codes only when a sample row is protected; else redacted metadata."""

    if protected_ids.intersection(finding.sample_ids):
        return {
            "finding_id": finding.finding_id,
            "kind": finding.kind,
            "severity": finding.severity.value,
            "partitions": list(finding.partitions),
            "sample_ids": list(finding.sample_ids),
            "canonical_key": finding.canonical_key,
            "corroborating_signals": list(finding.corroborating_signals),
            _PROTECTED_FLAG: True,
        }
    payload = _artifact_dict(finding)
    payload["raw_split_values"] = redact_mapping(finding.raw_split_values)
    return payload


def _clusters_summary(
    result: QualityGateResult, protected_ids: frozenset[str]
) -> dict[str, Any]:
    """Cluster counts + per-cluster summaries (evidence dropped if protected)."""

    items: list[dict[str, Any]] = []
    level_counts: Counter[str] = Counter()
    for cluster in sorted(result.clusters, key=lambda c: c.cluster_id):
        level_counts[cluster.level] += 1
        contains_protected = bool(protected_ids.intersection(cluster.member_ids))
        items.append(
            {
                "cluster_id": cluster.cluster_id,
                "member_count": len(cluster.member_ids),
                "member_ids": list(cluster.member_ids),
                "level": cluster.level,
                "evidence": {} if contains_protected else cluster.evidence,
                _PROTECTED_FLAG: contains_protected,
            }
        )
    return {
        "count": len(result.clusters),
        "levels": dict(sorted(level_counts.items())),
        "items": items,
    }


def build_report_payload(
    result: QualityGateResult,
    *,
    protected_ids: frozenset[str] | set[str] = frozenset(),
    artifact_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Full ``clouda.quality.run.v1`` dictionary for one gate run."""

    protected = frozenset(protected_ids)
    severity_counts: Counter[str] = Counter(
        {severity.value: 0 for severity in _SEVERITY_ORDER}
    )
    for issue in result.issues:
        severity_counts[issue.severity.value] += 1
    reason_code_counts = Counter(issue.code for issue in result.issues)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_from": result.schema_version,
        "run_id": result.run_id,
        "verdict": result.verdict.value,
        "severity_counts": dict(sorted(severity_counts.items())),
        "reason_code_counts": dict(sorted(reason_code_counts.items())),
        "declared_reason_codes": sorted(result.reason_codes),
        "issue_count": len(result.issues),
        "issues": [
            _issue_payload(issue, protected)
            for issue in sorted(result.issues, key=lambda i: (i.severity.value, i.code))
        ],
        "clusters": _clusters_summary(result, protected),
        "leakage_findings": [
            _leakage_payload(finding, protected)
            for finding in sorted(result.leakage_findings, key=lambda f: f.finding_id)
        ],
        "health": _artifact_dict(result.health),
        "exclusions": [
            dict(redact_mapping(_artifact_dict(exclusion)))
            for exclusion in sorted(result.exclusions, key=lambda e: e.sample_id)
        ],
        "artifact_paths": redact_mapping(dict(artifact_paths or {})),
    }


def render_report(
    result: QualityGateResult,
    fmt: str,
    *,
    protected_ids: frozenset[str] | set[str] = frozenset(),
    artifact_paths: Mapping[str, str] | None = None,
) -> str:
    """Render ``result`` as ``'json'`` or ``'text'``; unknown formats raise."""

    protected = frozenset(protected_ids)
    verdict = result.verdict
    if fmt == "json":
        payload = build_report_payload(
            result,
            protected_ids=protected,
            artifact_paths=artifact_paths,
        )
        return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False)
    if fmt == "text":
        return _render_text(result, verdict, protected)
    raise ValueError(f"Unsupported report format: {fmt!r} (expected 'text' or 'json').")


def _render_text(
    result: QualityGateResult,
    verdict: GateVerdict,
    protected_ids: frozenset[str],
) -> str:
    severity_counts: Counter[str] = Counter(
        {severity.value: 0 for severity in _SEVERITY_ORDER}
    )
    for issue in result.issues:
        severity_counts[issue.severity.value] += 1
    protected_count = sum(
        1 for issue in result.issues if protected_ids.intersection(issue.sample_ids)
    )
    lines: list[str] = [
        TEXT_REPORT_TITLE,
        "=" * len(TEXT_REPORT_TITLE),
        f"Run: {result.run_id}",
        f"Verdict: {verdict.value}",
        (
            "Issues: "
            f"{len(result.issues)} total "
            f"(info={severity_counts[IssueSeverity.INFO.value]}, "
            f"warning={severity_counts[IssueSeverity.WARNING.value]}, "
            f"error={severity_counts[IssueSeverity.ERROR.value]}, "
            f"critical={severity_counts[IssueSeverity.CRITICAL.value]})"
        ),
        f"Protected-issue rows withheld: {protected_count}",
        f"Clusters: {len(result.clusters)}",
        f"Leakage findings: {len(result.leakage_findings)}",
        f"Exclusions: {len(result.exclusions)}",
        _GATE_LINE_PREFIX + verdict.value,
    ]
    return "\n".join(lines) + "\n"
