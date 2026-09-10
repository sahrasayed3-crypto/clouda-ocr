"""Read-only readiness checks for the current canonical backend stack."""

from __future__ import annotations

import importlib
import importlib.util
import io
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from .models import DoctorCheck, DoctorSection, DoctorStatus

RESULTS_MODULES = (
    "clouda_data.results",
    "clouda_data.results.service",
    "clouda_data.results.store",
    "clouda_data.results.models",
)
LAB_MODULES = (
    "clouda_lab.results_service",
    "clouda_lab.dataset_selection",
    "clouda_lab.training_orchestrator",
)
TRAINING_DATA_MODULES = (
    "clouda_data.training_data",
    "clouda_data.training_data.input_contract",
    "clouda_data.training_data.sharding",
    "clouda_data.training_data.loader",
    "clouda_data.training_data.artifacts",
)


def _failed_imports(module_names: tuple[str, ...]) -> list[str]:
    failed: list[str] = []
    for module_name in module_names:
        try:
            # Optional native libraries may print diagnostics while importing.
            # Keep the Doctor's stdout reserved for its human/JSON contract.
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - diagnostic boundary
            failed.append(f"{module_name}: {type(exc).__name__}")
    return failed


def _writable_probe(root: Path | None) -> tuple[bool, str | None]:
    if root is None:
        return False, "repository root unavailable"
    probe_root = root
    while not probe_root.exists() and probe_root != probe_root.parent:
        probe_root = probe_root.parent
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".clouda-doctor-backend-", dir=probe_root, delete=False
        ) as handle:
            handle.write(b"probe")
            probe = Path(handle.name)
        probe.unlink(missing_ok=True)
        return True, None
    except Exception as exc:  # noqa: BLE001 - report, never crash
        return False, f"{type(exc).__name__}"


def check_results_store(repo_root: Path | None) -> DoctorSection:
    failed = _failed_imports(RESULTS_MODULES)
    package_ok = not failed
    writable, error = _writable_probe(repo_root)
    return DoctorSection(
        id="results-store",
        name="Results Store",
        checks=[
            DoctorCheck(
                id="results.package",
                name="Results Store package",
                subsystem="results",
                status=DoctorStatus.PASS if package_ok else DoctorStatus.FAIL,
                message=(
                    "RESULTS STORE: READY — package, service, store, and contracts import"
                    if package_ok
                    else "RESULTS STORE: NOT READY — imports failed"
                ),
                details={"failed_imports": failed},
                remediation=(
                    None if package_ok else "Reinstall the current project package."
                ),
            ),
            DoctorCheck(
                id="results.storage",
                name="Results Store state path",
                subsystem="results",
                status=DoctorStatus.PASS if writable else DoctorStatus.WARN,
                message=(
                    "Configured repository/state parent is writable"
                    if writable
                    else "Results state parent is not writable"
                ),
                required=False,
                details={"root": str(repo_root) if repo_root else None, "error": error},
            ),
        ],
    )


def check_lab_backend() -> DoctorSection:
    failed = _failed_imports(LAB_MODULES)
    ready = not failed
    return DoctorSection(
        id="lab-backend",
        name="Clouda Lab Backend",
        checks=[
            DoctorCheck(
                id="lab.services",
                name="Lab services",
                subsystem="lab",
                status=DoctorStatus.PASS if ready else DoctorStatus.FAIL,
                message=(
                    "CLOUDA LAB BACKEND: READY — analysis, selection, and orchestrator services import"
                    if ready
                    else "CLOUDA LAB BACKEND: NOT READY — service imports failed"
                ),
                details={"failed_imports": failed},
                remediation=None if ready else "Reinstall the current project package.",
            )
        ],
    )


def check_training_data(repo_root: Path | None) -> DoctorSection:
    failed = _failed_imports(TRAINING_DATA_MODULES)
    ready = not failed
    writable, error = _writable_probe(repo_root)
    try:
        from clouda_data.training_data.torch_adapter import torch_available

        has_torch = torch_available()
    except Exception:  # noqa: BLE001
        has_torch = False
    return DoctorSection(
        id="training-data",
        name="Training Data Engineering",
        checks=[
            DoctorCheck(
                id="training-data.engine",
                name="Training Data engine",
                subsystem="training-data",
                status=DoctorStatus.PASS if ready else DoctorStatus.FAIL,
                message=(
                    "TRAINING DATA ENGINEERING: READY — manifest, sharding, loader, and artifacts import"
                    if ready
                    else "TRAINING DATA ENGINEERING: NOT READY — imports failed"
                ),
                details={"failed_imports": failed},
                remediation=None if ready else "Reinstall the current project package.",
            ),
            DoctorCheck(
                id="training-data.storage",
                name="Training Data runtime path",
                subsystem="training-data",
                status=DoctorStatus.PASS if writable else DoctorStatus.WARN,
                message=(
                    "Training-data runtime parent is writable"
                    if writable
                    else "Training-data runtime parent is not writable"
                ),
                required=False,
                details={"root": str(repo_root) if repo_root else None, "error": error},
            ),
            DoctorCheck(
                id="training-data.torch",
                name="Optional PyTorch adapter",
                subsystem="training-data",
                status=DoctorStatus.PASS if has_torch else DoctorStatus.INFO,
                message=(
                    "Optional PyTorch adapter available"
                    if has_torch
                    else "Optional PyTorch adapter unavailable; core loader remains ready"
                ),
                required=False,
                details={
                    "torch_present": importlib.util.find_spec("torch") is not None
                },
            ),
        ],
    )


def run_backend_deep_smoke(
    temp_root: Path,
) -> tuple[DoctorStatus, str, dict[str, Any]]:
    """Exercise Results/Lab/Training Data with tiny synthetic local records."""
    try:
        from clouda_data.pretraining.manifest import write_manifest
        from clouda_data.results.service import ResultsService
        from clouda_data.training_data.loader import StreamingTrainingDataLoader
        from clouda_data.training_data.models import (
            BatchConfig,
            ShardConfig,
            ShuffleConfig,
            ShuffleMode,
            TrainingDataConfig,
            ValidationMode,
        )
        from clouda_data.training_data.sharding import build_shards
        from clouda_lab.results_service import StoredResultsAnalysisService
        from clouda_lab.training_orchestrator import TrainingOrchestrator

        temp_root.mkdir(parents=True, exist_ok=True)
        results = ResultsService(temp_root / "results")
        dataset = results.register_dataset(
            dataset_id="doctor-synthetic", version="v1", splits=("train",)
        )
        StoredResultsAnalysisService(results)
        TrainingOrchestrator(temp_root / "runs")

        rows = [
            {
                "sample_id": f"doctor-{index}",
                "source_id": "doctor-synthetic",
                "target_split": "train",
                "text": text,
                "provenance": {"origin": "environment-doctor"},
            }
            for index, text in enumerate(("مرحبا", "اختبار"), start=1)
        ]
        manifest = write_manifest(
            temp_root / "manifest.jsonl",
            rows,
            metadata={
                "dataset_id": "doctor-synthetic",
                "dataset_version": "v1",
            },
        )
        shard_root = temp_root / "training-data"
        build_shards(
            manifest,
            shard_root,
            ShardConfig(samples_per_shard=1),
            dataset_id="doctor-synthetic",
            dataset_version="v1",
        )
        loader = StreamingTrainingDataLoader(
            shard_index_path=shard_root / "shard_index.json",
            loader_config=TrainingDataConfig(
                dataset_id="doctor-synthetic",
                dataset_version="v1",
                shuffle=ShuffleConfig(mode=ShuffleMode.NONE),
                batch=BatchConfig(batch_size=1),
                validation_mode=ValidationMode.NONE,
            ),
            manifest_path=manifest,
            dataset_root=temp_root,
        )
        sample_ids = list(loader.iter_sample_ids(epoch=0))
        return (
            DoctorStatus.PASS,
            "Results/Lab/Training Data deep smoke completed",
            {
                "results_dataset_registered": dataset["dataset_id"]
                == "doctor-synthetic",
                "lab_services_constructed": True,
                "loader_samples": len(sample_ids),
            },
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic isolation
        return (
            DoctorStatus.FAIL,
            f"Backend deep smoke failed: {type(exc).__name__}",
            {"error_type": type(exc).__name__},
        )
