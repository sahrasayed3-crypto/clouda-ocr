"""Factory, WeasyPrint, RAQM, training, and GPU check tests (Phases 6-11).

RAQM tests are the security-critical ones: they pin the contract that the
vendored backend's ``_HAS_RAQM``-style "enum exists" signal must NEVER be
reported as ready when the native library is missing.
"""

from __future__ import annotations

from types import ModuleType


import clouda_data.doctor.factory as doctor_factory
import clouda_data.doctor.training as doctor_training
from clouda_data.doctor.factory import (
    check_factory,
    check_raqm,
    check_weasyprint,
    native_raqm_available,
    raqm_fallback_warning,
)
from clouda_data.doctor.models import DoctorStatus
from clouda_data.doctor.training import check_gpu, check_training_framework

# ---------------------------------------------------------------------------
# Phase 6 — Data Factory
# ---------------------------------------------------------------------------


def test_factory_ready_state(clean_worktree_root, monkeypatch):
    # Force the module-import probes so the test doesn't depend on host venv.
    monkeypatch.setattr(doctor_factory, "_module_importable", lambda name: True)
    section = check_factory(
        clean_worktree_root,
        cv2_available=True,
        numpy_available=True,
        yaml_available=True,
    )
    by_id = {c.id: c for c in section.checks}
    assert by_id["factory.package"].status is DoctorStatus.PASS
    assert by_id["factory.fonts"].status is DoctorStatus.PASS
    assert by_id["factory.profiles"].status is DoctorStatus.PASS
    assert by_id["factory.fonts"].details["count"] >= 1


def test_factory_missing_cv2_fails(clean_worktree_root, monkeypatch):
    monkeypatch.setattr(doctor_factory, "_module_importable", lambda name: True)
    section = check_factory(
        clean_worktree_root,
        cv2_available=False,
        numpy_available=True,
        yaml_available=True,
    )
    by_id = {c.id: c for c in section.checks}
    assert by_id["factory.package"].status is DoctorStatus.FAIL
    assert "cv2" in by_id["factory.package"].message
    assert "factory" in (by_id["factory.package"].remediation or "")


def test_factory_missing_numpy_fails(clean_worktree_root):
    section = check_factory(
        clean_worktree_root,
        cv2_available=True,
        numpy_available=False,
        yaml_available=True,
    )
    by_id = {c.id: c for c in section.checks}
    assert by_id["factory.package"].status is DoctorStatus.FAIL


# ---------------------------------------------------------------------------
# Phase 7 — WeasyPrint
# ---------------------------------------------------------------------------


def test_weasyprint_absent_reports_not_ready(monkeypatch):
    monkeypatch.setattr(doctor_factory, "installed_version", lambda name: None)
    section = check_weasyprint()
    check = section.checks[0]
    assert check.status is DoctorStatus.FAIL
    assert check.required is False
    assert check.details["version"] is None
    assert "factory-render" in (check.remediation or "")


def test_weasyprint_import_error_captured(monkeypatch):
    monkeypatch.setattr(doctor_factory, "installed_version", lambda name: None)

    def boom(name):
        raise ModuleNotFoundError("No module named 'weasyprint'")

    monkeypatch.setattr(doctor_factory.importlib, "import_module", boom)
    section = check_weasyprint()
    check = section.checks[0]
    assert check.status is DoctorStatus.FAIL
    assert "ModuleNotFoundError" in check.message


def test_weasyprint_available_with_successful_smoke(monkeypatch):
    monkeypatch.setattr(doctor_factory, "installed_version", lambda name: "66.1")
    # Force the import probe: the module "exists" in this scenario.
    monkeypatch.setattr(doctor_factory, "_module_importable", lambda name: True)
    monkeypatch.setattr(
        doctor_factory, "_weasyprint_smoke", lambda: (DoctorStatus.PASS, "smoke ok")
    )
    section = check_weasyprint()
    check = section.checks[0]
    assert check.status is DoctorStatus.PASS
    assert check.details["version"] == "66.1"


def test_weasyprint_import_ok_but_render_fails(monkeypatch):
    monkeypatch.setattr(doctor_factory, "installed_version", lambda name: "66.1")
    monkeypatch.setattr(doctor_factory, "_module_importable", lambda name: True)
    monkeypatch.setattr(
        doctor_factory,
        "_weasyprint_smoke",
        lambda: (DoctorStatus.FAIL, "render failed (native libraries?)"),
    )
    section = check_weasyprint()
    check = section.checks[0]
    assert check.status is DoctorStatus.FAIL
    assert check.details["version"] == "66.1"


# ---------------------------------------------------------------------------
# Phase 8 — RAQM (native detection contract)
# ---------------------------------------------------------------------------


def test_raqm_not_ready_when_native_flag_false(monkeypatch):
    monkeypatch.setattr(doctor_factory, "native_raqm_available", lambda: False)
    section = check_raqm()
    check = section.checks[0]
    assert check.status is DoctorStatus.FAIL
    assert check.details["native_raqm_flag"] is False
    assert check.details["module_flag_is_unreliable"] is True
    assert "false positive" in check.message.lower()
    assert check.required is False


def test_raqm_false_positive_protection(monkeypatch):
    """The enum existing must NOT make the doctor report PASS.

    Regression guard: if someone later 'simplifies' native_raqm_available to
    check ImageFont.Layout.RAQM (the vendored backend's mistake), this test
    fails — that signal reports True on hosts without libraqm.
    """
    monkeypatch.setattr(doctor_factory, "native_raqm_available", lambda: False)
    section = check_raqm()
    statuses = [c.status for c in section.checks]
    assert DoctorStatus.PASS not in statuses


def test_raqm_fallback_warning_detected(monkeypatch):
    monkeypatch.setattr(doctor_factory, "native_raqm_available", lambda: True)
    monkeypatch.setattr(
        doctor_factory,
        "raqm_fallback_warning",
        lambda: "Raqm layout was requested, but Raqm is not available. "
        "Falling back to basic layout.",
    )
    section = check_raqm()
    check = section.checks[0]
    assert check.status is DoctorStatus.FAIL
    assert check.details["fallback_warning"]


def test_raqm_ready_when_native_and_no_fallback(monkeypatch):
    monkeypatch.setattr(doctor_factory, "native_raqm_available", lambda: True)
    monkeypatch.setattr(doctor_factory, "raqm_fallback_warning", lambda: None)
    section = check_raqm()
    check = section.checks[0]
    assert check.status is DoctorStatus.PASS


def test_native_raqm_available_reads_real_flag(monkeypatch):
    """native_raqm_available must read PIL._imagingft.HAVE_RAQM, not the enum."""
    fake_ft = ModuleType("PIL._imagingft")
    fake_ft.HAVE_RAQM = False

    class FakePil(ModuleType):
        pass

    import sys as _sys

    monkeypatch.setitem(_sys.modules, "PIL._imagingft", fake_ft)
    assert native_raqm_available() is False

    fake_ft.HAVE_RAQM = True
    assert native_raqm_available() is True


def test_raqm_fallback_warning_function_smoke(clean_worktree_root):
    """The real function returns a string or None without raising."""
    result = raqm_fallback_warning()
    assert result is None or isinstance(result, str)


# ---------------------------------------------------------------------------
# Phase 10 — Training framework
# ---------------------------------------------------------------------------


def test_training_framework_ready(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    section = check_training_framework(runs_root=tmp_path / "runs")
    by_id = {c.id: c for c in section.checks}
    assert by_id["training.imports"].status is DoctorStatus.PASS
    assert by_id["training.output-root"].status is DoctorStatus.PASS
    assert by_id["training.imports"].details["example_config_hash"]


def test_training_framework_writable_root(tmp_path):
    section = check_training_framework(runs_root=tmp_path)
    by_id = {c.id: c for c in section.checks}
    assert by_id["training.output-root"].status is DoctorStatus.PASS


def test_training_dry_run_deep(tmp_path):
    """Deep mode dry-run is offline, fast, and cleans up via caller."""
    from clouda_data.doctor.training import run_training_dry_run

    output_root = tmp_path / "deep-runs"
    status, message, detail = run_training_dry_run(output_root)
    assert status is DoctorStatus.PASS
    assert detail["exit_code"] == 0
    # cleanup contract: caller removes the tree
    import shutil

    shutil.rmtree(output_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Phase 11 — GPU
# ---------------------------------------------------------------------------


def test_gpu_no_torch_is_clean_info(monkeypatch):
    monkeypatch.setattr(doctor_training, "installed_version", lambda name: None)
    section = check_gpu()
    by_id = {c.id: c for c in section.checks}
    assert by_id["gpu.torch"].status is DoctorStatus.INFO
    assert by_id["gpu.cuda"].status is DoctorStatus.INFO
    assert "NOT READY" in by_id["gpu.cuda"].message
    # No GPU must never be a doctor failure:
    assert all(c.status is not DoctorStatus.FAIL for c in section.checks)


class _FakeTorch(ModuleType):
    """Minimal torch stand-in for CUDA-present / CPU-only scenarios."""

    def __init__(self, *, cuda_available: bool, devices=None, bf16=False):
        super().__init__("torch")
        self.version = ModuleType("torch.version")
        self.version.cuda = "12.4" if cuda_available else None
        self.cuda = self._cuda(cuda_available, devices or [], bf16)

    class _cuda:
        def __init__(self, available, devices, bf16):
            self._available = available
            self._devices = devices
            self._bf16 = bf16

        def is_available(self):
            return self._available

        def device_count(self):
            return len(self._devices)

        def get_device_properties(self, index):
            name, mem, major, minor = self._devices[index]
            return SimpleNamespace(
                name=name, total_memory=mem, major=major, minor=minor
            )

        def is_bf16_supported(self):
            return self._bf16


from types import SimpleNamespace  # noqa: E402


def test_gpu_cpu_only_torch(monkeypatch):
    fake = _FakeTorch(cuda_available=False)
    import sys as _sys

    monkeypatch.setattr(
        doctor_training,
        "installed_version",
        lambda name: "2.4.0" if name == "torch" else None,
    )
    monkeypatch.setitem(_sys.modules, "torch", fake)
    monkeypatch.setattr(
        doctor_training, "shutil", SimpleNamespace(which=lambda p: None)
    )
    section = check_gpu()
    by_id = {c.id: c for c in section.checks}
    assert by_id["gpu.cuda"].status is DoctorStatus.INFO
    assert "NOT READY" in by_id["gpu.cuda"].message
    assert by_id["gpu.cuda"].details["cuda_available"] is False
    assert by_id["gpu.cuda"].details["torch"] == "2.4.0"


def test_gpu_mocked_cuda_gpu(monkeypatch):
    fake = _FakeTorch(
        cuda_available=True,
        devices=[("NVIDIA A100", 80 * 1024**3, 8, 0)],
        bf16=True,
    )
    import sys as _sys

    monkeypatch.setattr(
        doctor_training,
        "installed_version",
        lambda name: "2.4.0" if name == "torch" else None,
    )
    monkeypatch.setitem(_sys.modules, "torch", fake)
    monkeypatch.setattr(
        doctor_training, "shutil", SimpleNamespace(which=lambda p: None)
    )
    section = check_gpu()
    by_id = {c.id: c for c in section.checks}
    cuda_check = by_id["gpu.cuda"]
    assert cuda_check.status is DoctorStatus.PASS
    assert cuda_check.details["device_count"] == 1
    assert cuda_check.details["devices"][0]["name"] == "NVIDIA A100"
    assert cuda_check.details["bf16_supported"] is True
    assert cuda_check.details["cuda_build"] == "12.4"


def test_gpu_bf16_unsupported_reported(monkeypatch):
    fake = _FakeTorch(
        cuda_available=True,
        devices=[("Old GPU", 4 * 1024**3, 6, 1)],
        bf16=False,
    )
    import sys as _sys

    monkeypatch.setattr(
        doctor_training,
        "installed_version",
        lambda name: "2.4.0" if name == "torch" else None,
    )
    monkeypatch.setitem(_sys.modules, "torch", fake)
    monkeypatch.setattr(
        doctor_training, "shutil", SimpleNamespace(which=lambda p: None)
    )
    section = check_gpu()
    by_id = {c.id: c for c in section.checks}
    assert by_id["gpu.cuda"].details["bf16_supported"] is False
    assert "not supported" in by_id["gpu.cuda"].message
