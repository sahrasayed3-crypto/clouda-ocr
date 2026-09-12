"""Quality gate orchestrator: ties all stages into one scan pass.

Pipeline position (see docs/data_quality/CLOUDA_DATA_QUALITY.md):

    canonical manifest
        -> manifest adapter (load, identity, artifact paths)
        -> artifact integrity checks
        -> exact duplicate classification
        -> image fingerprint + near-duplicate candidates
        -> text near-duplicate candidates
        -> cross-split leakage analysis
        -> dataset health summary
        -> keep/exclude policy
        -> PASS / PASS_WITH_WARNINGS / FAIL + reports + derived manifest

The original manifest is never modified. Protected rows are never excluded;
they are returned for quarantine. Stage timings are observability metadata
only and never enter the run identity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clouda_data.pretraining.schema import DatasetSample
from clouda_data.quality import (
    artifacts,
    exact_dup,
    leakage,
    manifest_adapter,
    near_index,
    policy as policy_module,
    text_dup,
)
from clouda_data.quality.config import QualityGateConfig
from clouda_data.quality.models import (
    GateVerdict,
    IssueSeverity,
    QualityGateResult,
    QualityIssue,
    QualityRun,
)
from clouda_data.quality.run_state import _utc_now_iso as run_state_utc_now

GATE_SCHEMA_VERSION = "clouda.quality.run.v1"

ALGORITHM_VERSIONS: dict[str, str] = {
    "imgfp": "clouda.quality.imgfp.v1",
    "textdup": text_dup.TEXT_POLICY_VERSION,
    "exactdup": exact_dup.EXACT_DUP_SCHEMA_VERSION,
}


@dataclass
class StageTimings:
    """Wall-clock seconds per stage (observability only, not identity)."""

    load: float = 0.0
    artifacts: float = 0.0
    exact: float = 0.0
    near: float = 0.0
    leakage: float = 0.0
    health: float = 0.0
    policy: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "load_s": round(self.load, 3),
            "artifacts_s": round(self.artifacts, 3),
            "exact_s": round(self.exact, 3),
            "near_s": round(self.near, 3),
            "leakage_s": round(self.leakage, 3),
            "health_s": round(self.health, 3),
            "policy_s": round(self.policy, 3),
            "total_s": round(self.total, 3),
        }


@dataclass
class ScanOutput:
    """Everything one quality-scan produces."""

    run: QualityRun
    result: QualityGateResult
    samples: list[DatasetSample]
    exclusions: list[Any] = field(default_factory=list)
    quarantine_ids: tuple[str, ...] = ()
    excluded_ids: frozenset[str] = frozenset()
    timings: StageTimings = field(default_factory=StageTimings)
    header: dict[str, Any] = field(default_factory=dict)
    manifest_sha256: str = ""


def run_quality_gate(
    manifest_path: str,
    config: QualityGateConfig | None = None,
    *,
    root: str | None = None,
    max_samples: int = 0,
    no_near_duplicates: bool = False,
    cross_split_only: bool = False,
) -> ScanOutput:
    """Execute the full quality gate over a canonical manifest.

    ``manifest_path`` is the canonical pretraining manifest (JSONL with
    ``clouda.pretraining.manifest.v1`` header). The original file is only
    read, never written. ``root`` defaults to the manifest's parent
    directory for artifact resolution.
    """

    timings = StageTimings()
    started = time.perf_counter()
    cfg = config or QualityGateConfig()

    # Stage 1: load -----------------------------------------------------
    t0 = time.perf_counter()
    header, samples = manifest_adapter.load_manifest(manifest_path)
    manifest_sha = manifest_adapter.manifest_sha256(manifest_path)
    if max_samples > 0:
        samples = samples[:max_samples]
    artifact_root = root or str(Path(manifest_path).resolve().parent)
    identity = manifest_adapter.run_identity(
        manifest_sha,
        cfg.identity(),
        ALGORITHM_VERSIONS,
    )
    timings.load = time.perf_counter() - t0

    started_at = run_state_utc_now()
    run = QualityRun(
        run_id=str(identity["run_id"]),
        manifest_sha256=manifest_sha,
        config_identity=cfg.identity(),
        started_at=started_at,
        finished_at="",
        verdict=GateVerdict.PASS,
        severity_counts={},
        issues=(),
    )

    issues: list[QualityIssue] = []

    # Stage 2: artifact integrity ---------------------------------------
    if not cross_split_only:
        t0 = time.perf_counter()
        for sample in samples:
            issues.extend(
                artifacts.validate_artifact(sample, artifact_root, cfg.heuristics)
            )
        timings.artifacts = time.perf_counter() - t0

    # Stage 3: exact duplicates -----------------------------------------
    t0 = time.perf_counter()
    classified, exact_report = exact_dup.classify_exact_duplicates(samples)
    timings.exact = time.perf_counter() - t0

    # Stage 4: near duplicates (image + text) ---------------------------
    confirmed_pairs: list[tuple[str, str]] = []
    if not cross_split_only and not no_near_duplicates:
        t0 = time.perf_counter()
        fingerprints = near_index.build_fingerprints(classified, artifact_root, cfg)
        candidates = near_index.candidate_pairs(
            fingerprints, cfg, page_meta=near_index.page_index_meta(classified)
        )
        confirmed_pairs = [
            (candidate.sample_id_a, candidate.sample_id_b)
            for candidate in candidates
            if candidate.level == near_index.LEVEL_CONFIRMED
        ]
        # Re-classify with confirmed near pairs unioned in.
        classified, exact_report = exact_dup.classify_exact_duplicates(
            samples, confirmed_near_pairs=confirmed_pairs
        )
        timings.near = time.perf_counter() - t0

    # Stage 5: leakage ----------------------------------------------------
    t0 = time.perf_counter()
    leak_report = leakage.detect_leakage(
        classified, confirmed_near_pairs=confirmed_pairs
    )
    timings.leakage = time.perf_counter() - t0

    # Stage 6: health -----------------------------------------------------
    t0 = time.perf_counter()
    from clouda_data.quality.health import compute_health_summary

    health = compute_health_summary(classified)
    timings.health = time.perf_counter() - t0

    # Stage 7: keep/exclude policy ---------------------------------------
    t0 = time.perf_counter()
    clusters = exact_dup.families_to_clusters(classified, exact_report)
    exclusions = policy_module.decide_exclusions(classified, clusters, issues, cfg)
    excluded_ids = frozenset(decision.sample_id for decision in exclusions)
    quarantine_ids = policy_module.quarantine_sample_ids(classified, exclusions)
    timings.policy = time.perf_counter() - t0

    # Verdict -------------------------------------------------------------
    all_issues = list(issues)
    for finding in leak_report.findings:
        all_issues.append(leakage.finding_to_issue(finding))
    has_critical_or_error = any(
        issue.severity in (IssueSeverity.CRITICAL, IssueSeverity.ERROR)
        for issue in all_issues
    )
    has_warnings = any(issue.severity == IssueSeverity.WARNING for issue in all_issues)
    if has_critical_or_error or not leak_report.passed:
        verdict = GateVerdict.FAIL
    elif has_warnings:
        verdict = GateVerdict.PASS_WITH_WARNINGS
    else:
        verdict = GateVerdict.PASS

    severity_counts: dict[str, int] = {}
    for issue in all_issues:
        key = issue.severity.value
        severity_counts[key] = severity_counts.get(key, 0) + 1

    run = QualityRun(
        run_id=str(identity["run_id"]),
        manifest_sha256=manifest_sha,
        config_identity=cfg.identity(),
        started_at=started_at,
        finished_at=run_state_utc_now(),
        verdict=verdict,
        severity_counts=severity_counts,
        issues=(),
    )

    result = QualityGateResult(
        run_id=run.run_id,
        verdict=verdict,
        issues=tuple(all_issues),
        clusters=tuple(clusters),
        leakage_findings=tuple(leak_report.findings),
        health=health,
        exclusions=tuple(exclusions),
        reason_codes=tuple(sorted({issue.code for issue in all_issues})),
        schema_version=GATE_SCHEMA_VERSION,
    )

    timings.total = time.perf_counter() - started
    return ScanOutput(
        run=run,
        result=result,
        samples=classified,
        exclusions=list(exclusions),
        quarantine_ids=tuple(sorted(quarantine_ids)),
        excluded_ids=excluded_ids,
        timings=timings,
        header=header,
        manifest_sha256=manifest_sha,
    )
