"""Readiness checks for systems added after the Doctor branch forked."""

from __future__ import annotations

from clouda_data.doctor.backend import (
    check_lab_backend,
    check_results_store,
    check_training_data,
    run_backend_deep_smoke,
)
from clouda_data.doctor.models import DoctorStatus
from clouda_data.doctor.report import collect_report


def test_current_backend_sections_are_ready(tmp_path):
    results = check_results_store(tmp_path)
    lab = check_lab_backend()
    training_data = check_training_data(tmp_path)

    assert results.id == "results-store"
    assert results.checks[0].status is DoctorStatus.PASS
    assert lab.id == "lab-backend"
    assert lab.checks[0].status is DoctorStatus.PASS
    assert training_data.id == "training-data"
    assert training_data.checks[0].status is DoctorStatus.PASS
    assert "TRAINING DATA ENGINEERING: READY" in training_data.checks[0].message


def test_training_data_reports_missing_torch_as_optional(tmp_path, monkeypatch):
    import clouda_data.training_data.torch_adapter as torch_adapter

    monkeypatch.setattr(torch_adapter, "torch_available", lambda: False)
    section = check_training_data(tmp_path)
    torch_check = next(
        check for check in section.checks if check.id == "training-data.torch"
    )
    assert torch_check.status is DoctorStatus.INFO
    assert torch_check.required is False
    assert "optional" in torch_check.message.lower()


def test_backend_deep_smoke_is_offline_and_cleans_by_caller(tmp_path):
    status, message, details = run_backend_deep_smoke(tmp_path / "deep")
    assert status is DoctorStatus.PASS
    assert "Results/Lab/Training Data" in message
    assert details["results_dataset_registered"] is True
    assert details["loader_samples"] == 2


def test_collect_report_uses_current_backend_sections():
    report = collect_report(
        include_factory=False,
        include_render=False,
        include_training=False,
        include_gpu=False,
        include_git=False,
        include_storage=False,
        include_deps=False,
    )
    section_ids = {section.id for section in report.sections}
    assert {"results-store", "lab-backend", "training-data"} <= section_ids
    assert "optional-subsystems" not in section_ids


def test_results_store_missing_is_reported_not_raised(tmp_path, monkeypatch):
    import clouda_data.doctor.backend as backend

    real_import = backend.importlib.import_module

    def missing_results(name):
        if name.startswith("clouda_data.results"):
            raise ImportError("synthetic missing Results Store")
        return real_import(name)

    monkeypatch.setattr(backend.importlib, "import_module", missing_results)
    section = check_results_store(tmp_path)
    assert section.checks[0].status is DoctorStatus.FAIL
    assert "ImportError" in section.checks[0].details["failed_imports"][0]


def test_deep_backend_failure_is_a_report_check(monkeypatch):
    import clouda_data.doctor.report as report_module

    monkeypatch.setattr(
        report_module,
        "run_backend_deep_smoke",
        lambda _root: (DoctorStatus.FAIL, "synthetic backend failure", {}),
    )
    report = report_module.collect_report(
        deep=True,
        include_factory=False,
        include_render=False,
        include_training=False,
        include_gpu=False,
        include_git=False,
        include_storage=False,
        include_deps=False,
    )
    backend_check = next(
        check
        for section in report.sections
        for check in section.checks
        if check.id == "deep.backend-smoke"
    )
    assert backend_check.status is DoctorStatus.FAIL
    assert backend_check.required is False
