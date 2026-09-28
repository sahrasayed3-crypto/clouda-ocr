"""Training Framework (Phase 10), GPU/CUDA (Phases 10-11), PDF/image (Phase 9) checks.

Training readiness is split deliberately:

- ``check_training_framework`` — engineering readiness of
  ``clouda_training.experiments`` (imports, config layer, MockTrainer,
  registry, checkpoint metadata, output-root writability). No training runs.
- ``check_gpu_training`` — hardware readiness (torch + CUDA), reported
  separately so an orchestration-only machine is not conflated with a
  training machine.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .environment import installed_version
from .models import DoctorCheck, DoctorSection, DoctorStatus

TRAINING_IMPORT_MODULES: tuple[str, ...] = (
    "clouda_training",
    "clouda_training.cli",
    "clouda_training.experiments",
    "clouda_training.experiments.config",
    "clouda_training.experiments.runs",
    "clouda_training.experiments.trainer",
    "clouda_training.experiments.registry",
    "clouda_training.experiments.checkpoints",
    "clouda_training.checkpoints.metadata",
    "clouda_training.config.models",
    "clouda_training.planner",
    "clouda_training.exporter",
)

_MOCK_EXPERIMENT_CONFIG = """\
schema_version: 1
experiment:
  name: clouda_doctor_mock
  description: Deterministic offline Doctor validation.
  tags: [doctor, offline, mock]
model:
  model_id: mock/clouda-ocr
  revision: fixture-v1
  model_family: multimodal-ocr
  adapter_type: mock
dataset:
  dataset_id: clouda-doctor-fixture
  dataset_version: fixture-v1
  manifest_path: mock-manifest.jsonl
  split: train
  sample_limit: 2
training:
  seed: 20260909
  epochs: 1
  max_steps: 6
  batch_size: 2
  gradient_accumulation_steps: 1
  learning_rate: 0.0001
checkpoint:
  save_strategy: steps
  save_steps: 2
  save_total_limit: 2
evaluation:
  enabled: true
  eval_split: validation
  eval_steps: 2
  metrics: [cer, wer]
runtime:
  device: cpu
  num_workers: 0
  output_root: runs
  dry_run: true
  offline: true
  deterministic: true
tracking:
  enabled: true
  backend: jsonl
  log_steps: 1
"""

_MOCK_TRAINING_MANIFEST = """\
{"_row_count":2,"_schema_version":"clouda.pretraining.manifest.v1","dataset_role":"training","preprocessing_version":"clouda.pretraining.normalize.v1"}
{"sample_id":"doctor-ar-1","source_id":"clouda-doctor","source_license":"Apache-2.0","source_path":"doctor/ar-1.png","target_split":"train"}
{"sample_id":"doctor-ar-2","source_id":"clouda-doctor","source_license":"Apache-2.0","source_path":"doctor/ar-2.png","target_split":"train"}
"""


def _write_mock_experiment_config(directory: Path) -> Path:
    """Create Doctor's deterministic training fixture without a repo checkout."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "clouda-doctor-mock-experiment.yaml"
    path.write_text(_MOCK_EXPERIMENT_CONFIG, encoding="utf-8")
    (directory / "mock-manifest.jsonl").write_text(
        _MOCK_TRAINING_MANIFEST, encoding="utf-8"
    )
    return path


def check_training_framework(runs_root: Path | None = None) -> DoctorSection:
    """Engineering readiness of the Training Experiment Framework."""
    checks: list[DoctorCheck] = []
    if installed_version("torch") is None:
        return DoctorSection(
            id="training-framework",
            name="Training Framework",
            checks=[
                DoctorCheck(
                    id="training.imports",
                    name="Training framework imports",
                    subsystem="training",
                    status=DoctorStatus.SKIP,
                    message=(
                        "Training capability unavailable: optional torch dependency "
                        "is not installed."
                    ),
                    required=False,
                    remediation='pip install "clouda-ocr[training,training-torch]"',
                )
            ],
        )
    failed: list[str] = []
    for module_name in TRAINING_IMPORT_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{module_name}: {type(exc).__name__}")
    if failed:
        checks.append(
            DoctorCheck(
                id="training.imports",
                name="Training framework imports",
                subsystem="training",
                status=DoctorStatus.FAIL,
                message="Training framework imports failed: " + "; ".join(failed),
                details={"failed": failed},
                remediation='pip install -e ".[training]"',
            )
        )
    else:
        # Config layer + MockTrainer + lifecycle wiring (no execution).
        try:
            from clouda_training.experiments import MockTrainer  # type: ignore[attr-defined]
            from clouda_training.experiments.config import load_experiment_config

            with tempfile.TemporaryDirectory(prefix="clouda-doctor-training-") as tmp:
                config = load_experiment_config(
                    _write_mock_experiment_config(Path(tmp))
                )
            has_mock = MockTrainer is not None
            checks.append(
                DoctorCheck(
                    id="training.imports",
                    name="Training framework imports",
                    subsystem="training",
                    status=DoctorStatus.PASS if has_mock else DoctorStatus.FAIL,
                    message=(
                        "Framework imports, config layer, and MockTrainer available "
                        f"(example config hash {config.hash[:12]})"
                        if has_mock
                        else "MockTrainer missing from clouda_training.experiments"
                    ),
                    details={
                        "modules": len(TRAINING_IMPORT_MODULES),
                        "example_config_hash": config.hash,
                        "config_origin": "generated",
                    },
                    remediation=(
                        None
                        if has_mock
                        else "Reinstall clouda_training (pip install -e .)."
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001
            checks.append(
                DoctorCheck(
                    id="training.imports",
                    name="Training framework imports",
                    subsystem="training",
                    status=DoctorStatus.FAIL,
                    message=f"Config/MockTrainer verification failed: {type(exc).__name__}: {exc}",
                    details={"error": str(exc)},
                    remediation='pip install "clouda-ocr[training,training-torch]"',
                )
            )

    # Output-root writability (never mutates anything outside a temp probe).
    probe_root = runs_root or Path("runs")
    writable, detail = _probe_writable(probe_root)
    checks.append(
        DoctorCheck(
            id="training.output-root",
            name="Training runs output root",
            subsystem="training",
            status=DoctorStatus.PASS if writable else DoctorStatus.FAIL,
            message=(
                f"Runs root writable: {probe_root}"
                if writable
                else f"Runs root NOT writable: {probe_root}"
            ),
            required=False,
            details=detail,
            remediation=(
                None
                if writable
                else "Choose a writable --runs-root (or fix permissions)."
            ),
        )
    )
    return DoctorSection(
        id="training-framework", name="Training Framework", checks=checks
    )


def _probe_writable(root: Path) -> tuple[bool, dict[str, Any]]:
    """Create/delete one temp file under root (created if missing)."""
    detail: dict[str, Any] = {"root": str(root)}
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=".clouda-doctor-", dir=root, delete=False
        ) as handle:
            handle.write(b"doctor")
            probe = Path(handle.name)
        probe.unlink(missing_ok=True)
        detail["writable"] = True
        return True, detail
    except Exception as exc:  # noqa: BLE001
        detail["writable"] = False
        detail["error"] = f"{type(exc).__name__}: {exc}"
        return False, detail


def run_training_dry_run(output_root: Path) -> tuple[DoctorStatus, str, dict[str, Any]]:
    """Deep mode: one tiny offline MockTrainer lifecycle in a temp directory.

    Uses ``clouda_training.cli``'s own dry-run path (runtime.dry_run=true,
    runtime.offline=true forced) so the doctor exercises the real command
    surface. Returns (status, message, details); caller cleans up output_root.
    """
    try:
        from clouda_training.cli import main as training_main

        output_root.mkdir(parents=True, exist_ok=True)
        config_path = _write_mock_experiment_config(output_root)
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            rc = training_main(
                [
                    "dry-run",
                    str(config_path),
                    "--override",
                    f"runtime.output_root={output_root.as_posix()}",
                    "--json",
                ]
            )
        if rc == 0:
            return (
                DoctorStatus.PASS,
                "Offline MockTrainer dry-run completed",
                {"exit_code": 0},
            )
        return DoctorStatus.FAIL, f"Training dry-run exited {rc}", {"exit_code": rc}
    except Exception as exc:  # noqa: BLE001
        return (
            DoctorStatus.FAIL,
            f"Training dry-run failed: {type(exc).__name__}: {exc}",
            {"error": str(exc)},
        )


# ---------------------------------------------------------------------------
# Phase 11 — GPU / CUDA
# ---------------------------------------------------------------------------


def check_gpu() -> DoctorSection:
    """GPU/CUDA diagnostics; absence is a clean result, never a crash."""
    checks: list[DoctorCheck] = []
    torch_version = installed_version("torch")

    if torch_version is None:
        checks.append(
            DoctorCheck(
                id="gpu.torch",
                name="PyTorch",
                subsystem="gpu",
                status=DoctorStatus.INFO,
                message="PyTorch not installed — GPU training readiness unknown/not applicable",
                required=False,
                details={"torch": None},
                remediation='pip install -e ".[training]" plus a torch build when GPU training is needed',
            )
        )
        checks.append(
            DoctorCheck(
                id="gpu.cuda",
                name="CUDA GPU",
                subsystem="gpu",
                status=DoctorStatus.INFO,
                message="GPU TRAINING: NOT READY — torch not installed",
                required=False,
                details={"cuda_available": False},
            )
        )
        return DoctorSection(id="gpu", name="GPU / CUDA", checks=checks)

    import torch  # type: ignore[import-not-found]

    cuda_available = bool(torch.cuda.is_available())
    cuda_build = torch.version.cuda
    detail: dict[str, Any] = {
        "torch": torch_version,
        "cuda_build": cuda_build,
        "cuda_available": cuda_available,
    }
    if not cuda_available:
        detail["reason"] = (
            "torch.cuda.is_available() is False (CPU-only build or no driver)"
        )
        checks.append(
            DoctorCheck(
                id="gpu.cuda",
                name="CUDA GPU",
                subsystem="gpu",
                status=DoctorStatus.INFO,
                message="GPU TRAINING: NOT READY — no CUDA-capable GPU detected (torch available)",
                required=False,
                details=detail,
            )
        )
    else:
        count = torch.cuda.device_count()
        devices = []
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
        bf16 = torch.cuda.is_bf16_supported()
        detail["device_count"] = count
        detail["devices"] = devices
        detail["bf16_supported"] = bf16
        checks.append(
            DoctorCheck(
                id="gpu.cuda",
                name="CUDA GPU",
                subsystem="gpu",
                status=DoctorStatus.PASS,
                message=f"CUDA available: {count} device(s); BF16 {'supported' if bf16 else 'not supported'}",
                required=False,
                details=detail,
            )
        )

    nvidia_smi = shutil.which("nvidia-smi")
    checks.append(
        DoctorCheck(
            id="gpu.nvidia-smi",
            name="nvidia-smi tooling",
            subsystem="gpu",
            status=DoctorStatus.INFO,
            message=(
                f"nvidia-smi found at {nvidia_smi}"
                if nvidia_smi
                else "nvidia-smi not found on PATH"
            ),
            required=False,
            details={"path": nvidia_smi},
        )
    )
    return DoctorSection(id="gpu", name="GPU / CUDA", checks=checks)
