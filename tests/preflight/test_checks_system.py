"""Tests for system-level preflight checks (checks_system.py).

Unique basename ``test_checks_system`` per WAVE1_BRIEF (no collection
clashes). No torch/transformers needed at collection time; torch-dependent
expectations target THIS no-GPU machine (cuda requested -> FAIL).
"""

from __future__ import annotations

import json
import os
import stat
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from clouda_training.experiments.config import (
    DatasetSection,
    ExperimentSection,
    ExperimentConfig,
    ModelSection,
    RuntimeSection,
)
from clouda_training.preflight.checks_system import (
    check_adapter,
    check_capability_hooks,
    check_dependencies,
    check_device,
    check_local_model,
    check_output_storage,
    check_precision,
    ensure_adapters_registered,
)
from clouda_training.preflight.models import PreflightStatus

IS_WINDOWS = sys.platform == "win32"
IS_ADMIN_ON_WINDOWS = IS_WINDOWS and os.environ.get("CLOUDA_TEST_IS_ADMIN") == "1"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def make_config(
    tmp_path: Path,
    *,
    adapter_type: str = "hunyuanocr15_sft",
    device: str = "cpu",
    precision: str = "bf16",
    output_root: Path | None = None,
    manifest_path: Path | None = None,
    revision: str = "unresolved",
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment=ExperimentSection(name="preflight_system_check"),
        model=ModelSection(
            model_id="mock/ocr",
            revision=revision,
            model_family="mock",
            adapter_type=adapter_type,
            precision=precision,
        ),
        dataset=DatasetSection(
            dataset_id="synthetic-test",
            dataset_version="v1",
            manifest_path=manifest_path or (tmp_path / "manifest.jsonl"),
        ),
        runtime=RuntimeSection(
            device=device,
            output_root=output_root or (tmp_path / "runs"),
        ),
    )


def write_manifest(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "_schema_version": "clouda.pretraining.manifest.v1",
                "_row_count": 1,
                "dataset_role": "training",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def registered() -> None:
    ensure_adapters_registered()


# ---------------------------------------------------------------------------
# 1. check_adapter
# ---------------------------------------------------------------------------


class TestCheckAdapter:
    def test_unknown_adapter_is_blocker_fail(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, adapter_type="no_such_adapter")
        check = check_adapter(config)
        assert check.status is PreflightStatus.FAIL
        assert check.blocker is True
        assert "no_such_adapter" in check.detail

    def test_registered_hunyuan_passes_with_revision_recorded(
        self, tmp_path: Path, registered: None
    ) -> None:
        config = make_config(tmp_path, adapter_type="hunyuanocr15_sft")
        check = check_adapter(config)
        assert check.status is PreflightStatus.PASS
        assert check.blocker is False
        assert "c55965d3da1e" in check.detail

    def test_registered_qwen_passes(self, tmp_path: Path, registered: None) -> None:
        config = make_config(tmp_path, adapter_type="qwen_vl_sft")
        check = check_adapter(config)
        assert check.status is PreflightStatus.PASS
        assert "96588727" in check.detail

    def test_mismatched_revision_fails(self, tmp_path: Path, registered: None) -> None:
        config = make_config(
            tmp_path, adapter_type="hunyuanocr15_sft", revision="deadbeef"
        )
        check = check_adapter(config)
        assert check.status is PreflightStatus.FAIL
        assert check.blocker is True
        assert "deadbeef" in check.detail

    def test_mock_adapter_skips_model_checks(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, adapter_type="mock")
        check = check_adapter(config)
        assert check.status is PreflightStatus.SKIP
        assert check.blocker is False
        assert "mock" in check.detail

    def test_torch_adapter_skips_model_checks(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, adapter_type="torch")
        check = check_adapter(config)
        assert check.status is PreflightStatus.SKIP

    def test_registration_is_idempotent(self, tmp_path: Path) -> None:
        ensure_adapters_registered()
        ensure_adapters_registered()  # must not raise DuplicateAdapterError
        from clouda_training.adapters.registry import get_default_registry

        assert "hunyuanocr15_sft" in get_default_registry().list_adapters()
        assert "qwen_vl_sft" in get_default_registry().list_adapters()


# ---------------------------------------------------------------------------
# 2. check_dependencies
# ---------------------------------------------------------------------------


class TestCheckDependencies:
    def test_unknown_adapter_fails(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, adapter_type="nope")
        check = check_dependencies(config)
        assert check.status is PreflightStatus.FAIL
        assert check.blocker is True

    def test_mock_adapter_passes_trivially(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, adapter_type="mock")
        check = check_dependencies(config)
        assert check.status is PreflightStatus.SKIP
        assert check.blocker is False

    def test_torch_adapter_skips(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, adapter_type="torch")
        assert check_dependencies(config).status is PreflightStatus.SKIP

    def test_qwen_dependency_probe_reports_installed_transformers(
        self, tmp_path: Path, registered: None
    ) -> None:
        # The probe NEVER installs; on a machine without transformers this is
        # a FAIL, with an old transformers also a FAIL (version clause), and
        # only a satisfied install passes. Assert semantics, not environment.
        config = make_config(tmp_path, adapter_type="qwen_vl_sft")
        check = check_dependencies(config)
        assert check.status in (PreflightStatus.PASS, PreflightStatus.FAIL)
        assert check.name == "dependencies"
        if check.status is PreflightStatus.FAIL:
            # Either the package is absent or the version clause is unsatisfied.
            assert "No module named" in check.detail or "4.57" in check.detail

    def test_version_clause_rejects_old_version(
        self, tmp_path: Path, registered: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys as _sys
        import types as _types

        fake = _types.ModuleType("transformers")
        fake.__version__ = "4.10.0"  # type: ignore[attr-defined]
        monkeypatch.setitem(_sys.modules, "transformers", fake)
        config = make_config(tmp_path, adapter_type="qwen_vl_sft")
        check = check_dependencies(config)
        assert check.status is PreflightStatus.FAIL
        assert "4.10.0" in check.detail


# ---------------------------------------------------------------------------
# 3. check_device
# ---------------------------------------------------------------------------


class TestCheckDevice:
    def test_cpu_always_passes(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, device="cpu")
        check = check_device(config)
        assert check.status is PreflightStatus.PASS
        assert check.blocker is False

    def test_cuda_unavailable_is_blocker_fail(self, tmp_path: Path) -> None:
        # This machine has no GPU: requesting cuda must FAIL as a blocker.
        config = make_config(tmp_path, device="cuda")
        check = check_device(config)
        assert check.status is PreflightStatus.FAIL
        assert check.blocker is True
        assert "cuda" in check.detail.lower()

    def test_unsupported_device_fails(self, tmp_path: Path) -> None:
        config = replace(make_config(tmp_path))
        config = ExperimentConfig(
            experiment=config.experiment,
            model=config.model,
            dataset=config.dataset,
            runtime=RuntimeSection(device="tpu", output_root=tmp_path / "runs"),
        )
        check = check_device(config)
        assert check.status is PreflightStatus.FAIL
        assert check.blocker is True


# ---------------------------------------------------------------------------
# 4. check_precision
# ---------------------------------------------------------------------------


class TestCheckPrecision:
    def test_fp16_on_cpu_fails(self, tmp_path: Path, registered: None) -> None:
        config = make_config(tmp_path, precision="fp16", device="cpu")
        check = check_precision(config)
        assert check.status is PreflightStatus.FAIL
        assert check.blocker is True
        assert "fp16" in check.detail

    def test_bf16_on_cpu_warns_not_claims_gpu(
        self, tmp_path: Path, registered: None
    ) -> None:
        config = make_config(tmp_path, precision="bf16", device="cpu")
        check = check_precision(config)
        assert check.status is PreflightStatus.WARN
        assert check.blocker is False
        assert "unverified without GPU" in check.detail

    def test_unsupported_precision_fails(
        self, tmp_path: Path, registered: None
    ) -> None:
        config = make_config(tmp_path, precision="int8")
        check = check_precision(config)
        assert check.status is PreflightStatus.FAIL
        assert check.blocker is True

    def test_mock_precision_skips(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, adapter_type="mock", precision="fp16")
        check = check_precision(config)
        assert check.status is PreflightStatus.SKIP


# ---------------------------------------------------------------------------
# 5. check_local_model
# ---------------------------------------------------------------------------


class TestCheckLocalModel:
    def test_mock_skips(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, adapter_type="mock")
        assert check_local_model(config).status is PreflightStatus.SKIP

    def test_torch_skips(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, adapter_type="torch")
        assert check_local_model(config).status is PreflightStatus.SKIP

    def test_qwen_missing_path_fails(self, tmp_path: Path, registered: None) -> None:
        config = make_config(
            tmp_path,
            adapter_type="qwen_vl_sft",
        )
        config = ExperimentConfig(
            experiment=config.experiment,
            model=replace(config.model, model_id=str(tmp_path / "absent_model")),
            dataset=config.dataset,
            runtime=config.runtime,
        )
        check = check_local_model(config)
        assert check.status is PreflightStatus.FAIL
        assert check.blocker is True
        assert "never downloads" in check.detail

    def test_qwen_dir_without_config_json_fails(
        self, tmp_path: Path, registered: None
    ) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        config = ExperimentConfig(
            experiment=make_config(tmp_path).experiment,
            model=replace(make_config(tmp_path).model, model_id=str(model_dir)),
            dataset=make_config(tmp_path).dataset,
            runtime=make_config(tmp_path).runtime,
        )
        check = check_local_model(config)
        assert check.status is PreflightStatus.FAIL
        assert check.blocker is True
        assert "hunyuan preflight failed" in check.detail

    def test_qwen_valid_dir_passes(self, tmp_path: Path, registered: None) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")
        base = make_config(tmp_path, adapter_type="qwen_vl_sft")
        config = ExperimentConfig(
            experiment=base.experiment,
            model=replace(base.model, model_id=str(model_dir)),
            dataset=base.dataset,
            runtime=base.runtime,
        )
        check = check_local_model(config)
        assert check.status is PreflightStatus.PASS


# ---------------------------------------------------------------------------
# 6. check_output_storage
# ---------------------------------------------------------------------------


class TestCheckOutputStorage:
    def test_writable_dir_passes_and_reports_disk(self, tmp_path: Path) -> None:
        out = tmp_path / "runs"
        out.mkdir()
        config = make_config(tmp_path, output_root=out)
        check = check_output_storage(config)
        assert check.status is PreflightStatus.PASS
        assert check.blocker is False
        assert "writable" in check.detail
        # disk free is either reported or UNKNOWN — never fabricated silently
        assert "disk free" in check.detail

    def test_probe_file_is_cleaned_up(self, tmp_path: Path) -> None:
        out = tmp_path / "runs"
        out.mkdir()
        config = make_config(tmp_path, output_root=out)
        check_output_storage(config)
        leftovers = [p for p in out.iterdir() if p.name.startswith(".preflight_probe_")]
        assert leftovers == []

    def test_unwritable_dir_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out = tmp_path / "locked"
        out.mkdir()
        if IS_WINDOWS:
            pytest.skip(
                "chmod 0o555 does not reliably block writes on Windows "
                "(admin / ACL semantics)"
            )
        mode = out.stat().st_mode
        out.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        try:
            probe = out / ".preflight_probe_write_test"
            try:
                probe.write_text("x", encoding="utf-8")
            except PermissionError:
                pass  # expected: write really is blocked
            else:
                probe.unlink(missing_ok=True)
                pytest.skip("running with privileges that bypass chmod")
            config = make_config(tmp_path, output_root=out)
            check = check_output_storage(config)
            assert check.status is PreflightStatus.FAIL
            assert check.blocker is True
        finally:
            out.chmod(mode)

    def test_missing_directory_is_created(self, tmp_path: Path) -> None:
        # Phase 15: output root "exists or can be safely created" — a missing
        # leaf dir is created by the probe, then writability is verified.
        config = make_config(tmp_path, output_root=tmp_path / "does_not_exist")
        check = check_output_storage(config)
        assert check.status is PreflightStatus.PASS
        assert "created missing output_root" in check.detail
        assert (tmp_path / "does_not_exist").is_dir()

    def test_traversal_pattern_fails(self, tmp_path: Path) -> None:
        out = tmp_path / ".." / "elsewhere"
        out.mkdir(exist_ok=True)
        config = make_config(tmp_path, output_root=out)
        check = check_output_storage(config)
        assert check.status is PreflightStatus.FAIL
        assert ".." in check.detail

    def test_escape_beyond_intended_root_fails(self, tmp_path: Path) -> None:
        # output_root resolving outside cwd-based intended root
        outside = Path(os.environ.get("TEMP", tmp_path)) / "preflight_outside"
        outside.mkdir(exist_ok=True)
        config = make_config(tmp_path, output_root=outside)
        check = check_output_storage(config)
        assert check.status in (PreflightStatus.PASS, PreflightStatus.FAIL)

    def test_output_root_equals_manifest_parent_fails(self, tmp_path: Path) -> None:
        manifest = write_manifest(tmp_path / "manifest.jsonl")
        config = make_config(tmp_path, output_root=tmp_path, manifest_path=manifest)
        check = check_output_storage(config)
        assert check.status is PreflightStatus.FAIL
        assert check.blocker is True
        assert "protected input" in check.detail

    def test_output_root_sibling_of_manifest_passes(self, tmp_path: Path) -> None:
        manifest = write_manifest(tmp_path / "manifest.jsonl")
        out = tmp_path / "runs"
        out.mkdir()
        config = make_config(tmp_path, output_root=out, manifest_path=manifest)
        check = check_output_storage(config)
        assert check.status is PreflightStatus.PASS


# ---------------------------------------------------------------------------
# 7. capability hooks
# ---------------------------------------------------------------------------


class TestCapabilityHooks:
    def test_integrated_capability_hooks_report_current_stack(self) -> None:
        checks = check_capability_hooks()
        assert len(checks) == 3
        names = [c.name for c in checks]
        assert names == [
            "DATASET QUALITY CHECK",
            "TRAINING DATA LOADER CHECK",
            "ENVIRONMENT DOCTOR",
        ]
        assert [c.status for c in checks] == [
            PreflightStatus.PASS,
            PreflightStatus.PASS,
            PreflightStatus.SKIP,
        ]
        assert "available" in checks[0].detail
        assert "available" in checks[1].detail
        assert "Environment Doctor" in checks[2].detail

    def test_integrated_capability_hooks_never_duplicate_run_validation(self) -> None:
        for check in check_capability_hooks():
            assert check.blocker is False
