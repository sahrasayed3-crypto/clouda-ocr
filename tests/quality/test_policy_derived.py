"""Tests for quality policy + derived manifest (Wave2-K contract)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import pytest

from clouda_data.pretraining.manifest import read_manifest, write_manifest
from clouda_data.pretraining.schema import DuplicateState, ValidationStatus
from clouda_data.quality import manifest_adapter
from clouda_data.quality.derived import (
    DerivedManifestValidationError,
    revalidate_derived,
    write_clean_manifest,
    write_quarantine_manifest,
)
from clouda_data.quality.models import ExclusionDecision, GateVerdict, QualityRun
from clouda_data.quality.policy import (
    decide_exclusions,
    exclusion_report,
    quarantine_sample_ids,
)
from clouda_data.training_data.input_contract import validate_canonical_manifest
from conftest import make_manifest, make_row


@dataclass(frozen=True)
class FakeCluster:
    cluster_id: str
    member_ids: tuple[str, ...]
    level: str = "CONFIRMED_NEAR_DUPLICATE"
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FakeIssue:
    code: str
    severity: str = "error"
    sample_ids: tuple[str, ...] = ()
    canonical_key: str = ""
    message: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


def make_run(manifest_sha: str, run_id: str = "QRUN_" + "0" * 16) -> QualityRun:
    return QualityRun(
        run_id=run_id,
        manifest_sha256=manifest_sha,
        config_identity="clouda.quality.config.v1+test",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        verdict=GateVerdict.PASS,
    )


class TestClusterPolicy:
    def test_holdout_member_kept_and_training_dup_excluded(
        self, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        training = make_row("smp_train", target_split="train")
        holdout = make_row("smp_holdout", target_split="holdout")
        samples = [training, holdout]
        clusters = [FakeCluster("CLU_" + "a" * 12, ("smp_holdout", "smp_train"))]
        decisions = decide_exclusions(samples, clusters, [], make_run("a" * 64))

        excluded = {d.sample_id for d in decisions if d.reason_source == "dedupe"}
        quarantined = {d.sample_id for d in decisions if d.reason_source == "split"}
        assert "smp_train" in excluded
        assert "smp_holdout" in quarantined
        assert "smp_holdout" not in excluded
        assert quarantine_sample_ids(samples, decisions) == ("smp_holdout",)

    def test_deterministic_across_shuffled_input(self) -> None:
        samples = [
            make_row("smp_1", transformations=["recompress"]),
            make_row("smp_2"),
            make_row("smp_3", transformations=["resize"]),
        ]
        clusters = [FakeCluster("CLU_" + "b" * 12, ("smp_1", "smp_2", "smp_3"))]
        run = make_run("a" * 64)
        first = decide_exclusions(samples, clusters, [], run)
        for seed in range(5):
            order = list(samples)
            random.Random(seed).shuffle(order)
            shuffled = decide_exclusions(order, list(clusters), [], run)
            assert [(d.sample_id, d.reason_code) for d in shuffled] == [
                (d.sample_id, d.reason_code) for d in first
            ]
        # canonical-valid wins regardless of order
        assert "smp_2" not in {d.sample_id for d in first}

    def test_canonical_valid_preferred_over_clean(self) -> None:
        canonical = make_row(
            "smp_z",
            duplicate_state=DuplicateState.CANONICAL,
        )
        clean = make_row("smp_a")
        cluster = FakeCluster("CLU_" + "c" * 12, ("smp_a", "smp_z"))
        decisions = decide_exclusions(
            [canonical, clean], [cluster], [], make_run("a" * 64)
        )
        excluded = {d.sample_id for d in decisions if d.reason_source == "dedupe"}
        assert excluded == {"smp_a"}

    def test_validation_issue_exclusion(self) -> None:
        bad = make_row("smp_bad", validation_status=ValidationStatus.ERROR)
        issue = FakeIssue(code="IMAGE_DECODE", sample_ids=("smp_bad",))
        decisions = decide_exclusions([bad], [], [issue], make_run("a" * 64))
        assert len(decisions) == 1
        assert decisions[0].reason_code == "IMAGE_DECODE"
        assert decisions[0].reason_source == "validation"


class TestExclusionReport:
    def test_schema_and_counts(self) -> None:
        decisions = [
            ExclusionDecision("smp_1", "duplicate", "dedupe"),
            ExclusionDecision("smp_2", "duplicate", "dedupe"),
            ExclusionDecision("smp_3", "IMAGE_DECODE", "validation"),
        ]
        report = exclusion_report(decisions, total_samples=10, quarantine_ids=["smp_9"])
        assert report["schema_version"] == "clouda.pretraining.exclusion.v1"
        assert report["counts_by_reason"] == {
            "IMAGE_DECODE": 1,
            "duplicate": 2,
        }
        assert report["excluded_count"] == 3
        assert report["quarantine_count"] == 1


class TestDerivedManifest:
    def _write_and_check(self, tmp_path: Any, twice: bool) -> Any:
        rows = [
            make_row("smp_keep", text="احتفظ"),
            make_row("smp_drop", text="استبعد"),
            make_row("smp_holdout", target_split="holdout", text="خاص"),
        ]
        manifest_path = make_manifest(tmp_path / "src", rows)
        source_sha = manifest_path.read_bytes()
        samples = rows
        exclusions = [
            ExclusionDecision("smp_drop", "duplicate", "dedupe"),
        ]
        quarantine = ("smp_holdout",)
        run = make_run("a" * 64)

        kwargs: dict[str, Any] = {}
        result = write_clean_manifest(
            manifest_path,
            samples,
            exclusions,
            quarantine,
            run,
            tmp_path / "out" / "clean.jsonl",
        )
        first_bytes = (tmp_path / "out" / "clean.jsonl").read_bytes()
        if twice:
            write_clean_manifest(
                manifest_path,
                samples,
                exclusions,
                quarantine,
                run,
                tmp_path / "out" / "clean.jsonl",
                **kwargs,
            )
            assert (tmp_path / "out" / "clean.jsonl").read_bytes() == first_bytes
        assert manifest_path.read_bytes() == source_sha

        qpath = write_quarantine_manifest(
            samples,
            quarantine,
            run,
            result["source_manifest_sha256"],
            tmp_path / "out" / "quarantine.jsonl",
        )
        assert manifest_path.read_bytes() == source_sha

        clean_ids = {
            line.split('"sample_id": "')[1].split('"')[0]
            for line in (tmp_path / "out" / "clean.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[1:]
        }
        quarantine_ids = {
            line.split('"sample_id": "')[1].split('"')[0]
            for line in qpath.read_text(encoding="utf-8").splitlines()[1:]
        }
        assert clean_ids == {"smp_keep"}
        assert quarantine_ids == {"smp_holdout"}
        assert not (clean_ids & quarantine_ids)

        revalidate_derived(tmp_path / "out" / "clean.jsonl", {"smp_drop"})
        return manifest_path, result

    def test_clean_write_invariants(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        self._write_and_check(tmp_path, twice=False)

    def test_derived_manifest_byte_identical_across_runs(
        self, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        self._write_and_check(tmp_path, twice=True)

    def test_header_lineage(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        rows = [make_row("smp_a", text="نص")]
        manifest_path = make_manifest(tmp_path / "src", rows)
        source_sha = manifest_adapter.manifest_sha256(manifest_path)
        result = write_clean_manifest(
            manifest_path,
            rows,
            [],
            (),
            make_run(source_sha, run_id="QRUN_abcdef0123456789"),
            tmp_path / "out" / "clean.jsonl",
        )
        header_line = (
            (tmp_path / "out" / "clean.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()[0]
        )
        assert '"source_manifest_sha256": "' + source_sha in header_line
        assert '"quality_run_id": "QRUN_abcdef0123456789"' in header_line
        assert '"config_identity"' in header_line
        assert (
            '"derived_dataset_version": "derived-1.0.0+' + source_sha[:12]
            in header_line
        )
        assert '"exclusion_report_sha256"' in header_line
        assert '"verdict": "PASS"' in header_line
        assert result["clean_row_count"] == 1

    def test_clean_manifest_is_accepted_by_training_loader_contract(
        self, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        rows = [make_row("smp_train", target_split="train", text="نص تدريب")]
        source = tmp_path / "src" / "manifest.jsonl"
        write_manifest(
            source,
            [row.to_dict() for row in rows],
            metadata={"dataset_id": "ocr-source", "dataset_version": "v3"},
        )
        source_sha = manifest_adapter.manifest_sha256(source)
        output = tmp_path / "out" / "clean.jsonl"

        result = write_clean_manifest(
            source,
            rows,
            [],
            (),
            make_run(source_sha),
            output,
        )

        expected_version = f"derived-1.0.0+{source_sha[:12]}"
        identity = validate_canonical_manifest(
            output,
            dataset_id="ocr-source",
            dataset_version=expected_version,
        )
        header, written_rows = read_manifest(output)
        assert identity.dataset_id == "ocr-source"
        assert identity.dataset_version == expected_version
        assert header["source_dataset_version"] == "v3"
        assert header["source_manifest_sha256"] == source_sha
        assert result["dataset_id"] == "ocr-source"
        assert result["dataset_version"] == expected_version
        assert written_rows[0]["dataset_id"] == "ocr-source"
        assert written_rows[0]["dataset_version"] == expected_version

    def test_revalidate_catches_injected_protected_row(
        self, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        rows = [make_row("smp_a", text="نص")]
        manifest_path = make_manifest(tmp_path / "src", rows)
        out = tmp_path / "out" / "clean.jsonl"
        write_clean_manifest(manifest_path, rows, [], (), make_run("c" * 64), out)
        # Inject a protected row.
        protected = make_row("smp_evil", target_split="holdout", text="سر")
        lines = out.read_text(encoding="utf-8").splitlines()
        header = lines[0]
        injected = header.replace('"_row_count": 1', '"_row_count": 2') + "\n"
        body = (
            "\n".join(lines[1:])
            + "\n"
            + "\n".join(
                [
                    __import__("json").dumps(
                        protected.to_dict(), ensure_ascii=False, sort_keys=True
                    )
                ]
            )
        )
        out.write_text(injected + body + "\n", encoding="utf-8")
        with pytest.raises(DerivedManifestValidationError) as excinfo:
            revalidate_derived(out, set())
        assert "smp_evil" in str(excinfo.value)

    def test_revalidate_catches_excluded_reappearance(
        self, tmp_path
    ) -> None:  # type: ignore[no-untyped-def]
        rows = [make_row("smp_a", text="نص")]
        manifest_path = make_manifest(tmp_path / "src", rows)
        out = tmp_path / "out" / "clean.jsonl"
        write_clean_manifest(manifest_path, rows, [], (), make_run("d" * 64), out)
        with pytest.raises(DerivedManifestValidationError) as excinfo:
            revalidate_derived(out, {"smp_a"})
        assert "smp_a" in str(excinfo.value)

    def test_quarantine_disjoint_enforced(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        rows = [make_row("smp_a", text="نص")]
        manifest_path = make_manifest(tmp_path / "src", rows)
        with pytest.raises(ValueError):
            write_clean_manifest(
                manifest_path,
                rows,
                [ExclusionDecision("smp_a", "duplicate", "dedupe")],
                ("smp_a",),
                make_run("e" * 64),
                tmp_path / "out" / "clean.jsonl",
            )
