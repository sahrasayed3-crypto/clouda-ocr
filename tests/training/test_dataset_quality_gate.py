"""Quality-verdict enforcement in the canonical training dataset validator.

If a manifest header records a quality-gate verdict, a FAIL verdict must
block training (fail-closed) and any recorded verdict must flow into run
metadata for provenance. Absence of the field stays valid so manifests
produced before the quality gate existed keep working.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clouda_training.experiments.config import (
    DatasetSection,
    ExperimentConfig,
    ExperimentSection,
    ModelSection,
    RuntimeSection,
)
from clouda_training.experiments.dataset import validate_training_dataset


def _write_manifest(tmp_path: Path, header: dict) -> Path:
    rows = [
        header,
        {
            "sample_id": "sample-1",
            "target_split": "train",
            "source_id": "synthetic-test",
            "source_license": "Apache-2.0",
        },
    ]
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _config(manifest: Path) -> DatasetSection:
    return DatasetSection(
        dataset_id="synthetic-test",
        dataset_version="v1",
        manifest_path=manifest,
        split="train",
    )


_BASE_HEADER = {
    "_schema_version": "clouda.pretraining.manifest.v1",
    "_row_count": 1,
    "dataset_id": "synthetic-test",
    "dataset_version": "v1",
    "dataset_role": "training",
}


def test_failed_quality_verdict_blocks_training(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path, dict(_BASE_HEADER, quality_gate_verdict="FAIL")
    )
    with pytest.raises(PermissionError, match="quality"):
        validate_training_dataset(_config(manifest))


def test_lowercase_fail_verdict_blocks_training(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path, dict(_BASE_HEADER, quality_gate_verdict="fail")
    )
    with pytest.raises(PermissionError, match="quality"):
        validate_training_dataset(_config(manifest))


def test_pass_verdict_is_recorded(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path, dict(_BASE_HEADER, quality_gate_verdict="PASS_WITH_WARNINGS")
    )
    identity = validate_training_dataset(_config(manifest))
    assert identity.quality_verdict == "PASS_WITH_WARNINGS"


def test_missing_verdict_is_none(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, dict(_BASE_HEADER))
    identity = validate_training_dataset(_config(manifest))
    assert identity.quality_verdict is None


def test_unknown_verdict_value_fails_closed(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path, dict(_BASE_HEADER, quality_gate_verdict="SORT_OF_OK")
    )
    with pytest.raises(ValueError, match="quality_gate_verdict"):
        validate_training_dataset(_config(manifest))


def test_run_metadata_records_quality_verdict(tmp_path: Path) -> None:
    """run_experiment records the header verdict (absent -> null)."""
    manifest = _write_manifest(
        tmp_path, dict(_BASE_HEADER, quality_gate_verdict="PASS")
    )
    config = ExperimentConfig(
        experiment=ExperimentSection(name="quality_verdict_probe"),
        model=ModelSection(
            model_id="synthetic/linear",
            revision="probe-v1",
            model_family="synthetic",
            adapter_type="mock",
        ),
        dataset=DatasetSection(
            dataset_id="synthetic-test",
            dataset_version="v1",
            manifest_path=manifest,
            split="train",
        ),
        runtime=RuntimeSection(
            output_root=tmp_path / "runs", dry_run=True, device="cpu"
        ),
    )
    from clouda_training.experiments import RunStatus, run_experiment

    handle = run_experiment(config)
    assert handle.status is RunStatus.COMPLETED
    metadata = json.loads((handle.path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset_quality_verdict"] == "PASS"
