"""Fail-closed model training-use approval guard.

A real training run must never start against a model that has not been
explicitly approved for the intended training use in a reviewed approval
catalog. These tests pin the catalog schema, the matching rules (explicit
approval only, explicit rejection wins, wildcards), and the enforcement
points in the experiment runtime.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from clouda_training.adapters.approval import (
    ModelTrainingNotApproved,
    get_training_approval,
    load_approval_catalog,
    require_training_approval,
)


@pytest.fixture()
def torch_config(tmp_path: Path):
    """Minimal CPU experiment config (mirrors tests/runtime/conftest.py)."""
    rows = [
        {
            "_schema_version": "clouda.pretraining.manifest.v1",
            "_row_count": 1,
            "dataset_role": "training",
        },
        {
            "sample_id": "sample-1",
            "target_split": "train",
            "source_id": "synthetic-test",
            "source_path": "page.png",
            "source_license": "Apache-2.0",
        },
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    from clouda_training.experiments import load_experiment_config

    payload: dict[str, object] = {
        "schema_version": 1,
        "experiment": {"name": "approval_probe", "tags": ["test"]},
        "model": {
            "model_id": "synthetic/linear",
            "revision": "probe-v1",
            "model_family": "synthetic",
            "adapter_type": "torch",
        },
        "dataset": {
            "dataset_id": "synthetic-test",
            "dataset_version": "v1",
            "manifest_path": str(manifest),
            "split": "train",
        },
        "training": {
            "seed": 17,
            "max_steps": 12,
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "learning_rate": 0.05,
            "scheduler": "linear",
            "max_grad_norm": 1.0,
        },
        "checkpoint": {
            "save_strategy": "steps",
            "save_steps": 4,
            "save_total_limit": 3,
        },
        "evaluation": {"enabled": False},
        "runtime": {
            "device": "cpu",
            "output_root": str(tmp_path / "runs"),
            "dry_run": False,
            "offline": True,
            "deterministic": True,
        },
        "tracking": {"enabled": True, "backend": "jsonl", "log_steps": 1},
    }
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return load_experiment_config(config_path)


def _write_catalog(tmp_path: Path, approvals: list[dict[str, Any]]) -> Path:
    path = tmp_path / "approvals.json"
    path.write_text(
        json.dumps({"schema_version": 1, "approvals": approvals}, indent=2),
        encoding="utf-8",
    )
    return path


_APPROVED = {
    "adapter_type": "hunyuanocr15_sft",
    "model_id": "tencent/HunyuanOCR-1.5",
    "revision": "c55965d3da1e",
    "approved": True,
    "license_id": "license-upstream-allowlist",
    "approved_by": "reviewer-a",
    "approved_at": "2026-09-22T00:00:00Z",
    "notes": "test fixture",
}


def test_approved_model_returns_record(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path, [_APPROVED])
    approval = get_training_approval(
        "hunyuanocr15_sft",
        "tencent/HunyuanOCR-1.5",
        "c55965d3da1e",
        catalog_path=catalog,
    )
    assert approval is not None
    assert approval.license_id == "license-upstream-allowlist"
    assert approval.adapter_type == "hunyuanocr15_sft"


def test_wildcard_revision_matches(tmp_path: Path) -> None:
    record = dict(_APPROVED, revision="*")
    catalog = _write_catalog(tmp_path, [record])
    approval = get_training_approval(
        "hunyuanocr15_sft",
        "tencent/HunyuanOCR-1.5",
        "any-revision",
        catalog_path=catalog,
    )
    assert approval is not None


def test_unknown_model_is_not_approved(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path, [_APPROVED])
    assert (
        get_training_approval(
            "qwen_vl_sft", "Qwen/Qwen3-VL-4B", "main", catalog_path=catalog
        )
        is None
    )


def test_explicit_rejection_wins_over_approval(tmp_path: Path) -> None:
    approved = dict(_APPROVED, revision="*")
    rejected = dict(_APPROVED, revision="c55965d3da1e", approved=False, notes="revoked")
    catalog = _write_catalog(tmp_path, [approved, rejected])
    with pytest.raises(ModelTrainingNotApproved, match="revoked|not approved"):
        require_training_approval(
            "hunyuanocr15_sft",
            "tencent/HunyuanOCR-1.5",
            "c55965d3da1e",
            catalog_path=catalog,
        )


def test_missing_catalog_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ModelTrainingNotApproved, match="catalog"):
        require_training_approval(
            "hunyuanocr15_sft", "m", "r", catalog_path=tmp_path / "missing.json"
        )


def test_malformed_catalog_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "approvals.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelTrainingNotApproved, match="catalog"):
        require_training_approval("hunyuanocr15_sft", "m", "r", catalog_path=path)


def test_unknown_schema_version_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "approvals.json"
    path.write_text(
        json.dumps({"schema_version": 999, "approvals": []}), encoding="utf-8"
    )
    with pytest.raises(ModelTrainingNotApproved, match="schema"):
        require_training_approval("hunyuanocr15_sft", "m", "r", catalog_path=path)


def test_approved_record_without_license_fails_closed(tmp_path: Path) -> None:
    record = dict(_APPROVED)
    record.pop("license_id")
    catalog = _write_catalog(tmp_path, [record])
    with pytest.raises(ModelTrainingNotApproved, match="license"):
        require_training_approval(
            "hunyuanocr15_sft",
            "tencent/HunyuanOCR-1.5",
            "c55965d3da1e",
            catalog_path=catalog,
        )


def test_load_approval_catalog_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ModelTrainingNotApproved):
        load_approval_catalog(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# Runtime enforcement
# ---------------------------------------------------------------------------


def _register_runtime_test_adapter() -> str:
    from clouda_training.adapters.capabilities import ModelCapabilities
    from clouda_training.adapters.descriptor import ModelAdapterDescriptor
    from clouda_training.adapters.registry import get_default_registry
    from clouda_training.runtime.adapter import SyntheticLinearAdapter

    adapter_type = "approval_test_adapter"
    registry = get_default_registry()
    if registry.is_registered(adapter_type):
        return adapter_type

    def factory(*, config):
        return SyntheticLinearAdapter()

    registry.register(
        ModelAdapterDescriptor(
            adapter_type=adapter_type,
            adapter_version="1.0.0",
            model_family="synthetic",
            task_family="test",
            capabilities=ModelCapabilities(
                supports_full_finetune=True,
                supports_cpu_smoke=True,
                supports_resume=True,
            ),
            supported_precision=("float32",),
            supported_devices=("cpu",),
            checkpoint_compatibility_id="approval-test-v1",
            upstream_revision="synthetic-v1",
        ),
        factory,
    )
    return adapter_type


@pytest.fixture()
def _runtime_test_adapter():
    """Register a real-adapter stand-in and clean the registry afterwards."""
    adapter_type = _register_runtime_test_adapter()
    yield adapter_type
    from clouda_training.adapters.registry import get_default_registry

    get_default_registry().unregister(adapter_type)


def _real_config(torch_config, adapter_type: str):
    return dataclasses.replace(
        torch_config,
        model=dataclasses.replace(
            torch_config.model,
            model_id="synthetic/approval",
            adapter_type=adapter_type,
        ),
    )


def test_real_run_blocked_without_approval(
    torch_config, tmp_path: Path, _runtime_test_adapter
) -> None:
    adapter_type = _runtime_test_adapter
    config = _real_config(torch_config, adapter_type)
    from clouda_training.experiments import run_experiment

    missing = tmp_path / "no-approvals.json"
    with pytest.raises(ModelTrainingNotApproved):
        run_experiment(config, approval_catalog=missing)


def test_real_run_with_approval_records_provenance(
    torch_config, tmp_path: Path, _runtime_test_adapter
) -> None:
    adapter_type = _runtime_test_adapter
    config = _real_config(torch_config, adapter_type)
    catalog = _write_catalog(
        tmp_path,
        [
            dict(
                _APPROVED,
                adapter_type=adapter_type,
                model_id="synthetic/approval",
                revision="*",
            )
        ],
    )
    from clouda_training.experiments import RunStatus, run_experiment

    handle = run_experiment(
        config,
        approval_catalog=catalog,
        fail_at_step=None,
    )
    assert handle.status is RunStatus.COMPLETED
    metadata = json.loads((handle.path / "metadata.json").read_text(encoding="utf-8"))
    recorded = metadata["model_training_approval"]
    assert recorded["adapter_type"] == adapter_type
    assert recorded["license_id"] == "license-upstream-allowlist"
    assert recorded["source_catalog"] == str(catalog)


def test_mock_and_torch_adapters_need_no_approval(torch_config, tmp_path: Path) -> None:
    from clouda_training.experiments import RunStatus, run_experiment

    missing = tmp_path / "no-approvals.json"
    handle = run_experiment(torch_config, approval_catalog=missing)
    assert handle.status is RunStatus.COMPLETED


# ---------------------------------------------------------------------------
# Preflight surfacing
# ---------------------------------------------------------------------------


def test_preflight_reports_unapproved_real_adapter_as_blocker(
    torch_config, tmp_path: Path, _runtime_test_adapter
) -> None:
    adapter_type = _runtime_test_adapter
    config = _real_config(torch_config, adapter_type)
    from clouda_training.preflight.checks_system import check_model_approval

    check = check_model_approval(config, catalog_path=tmp_path / "missing.json")
    assert check.status.value == "FAIL"
    assert check.blocker is True


def test_preflight_approved_real_adapter_passes(
    torch_config, tmp_path: Path, _runtime_test_adapter
) -> None:
    adapter_type = _runtime_test_adapter
    config = _real_config(torch_config, adapter_type)
    catalog = _write_catalog(
        tmp_path,
        [
            dict(
                _APPROVED,
                adapter_type=adapter_type,
                model_id="synthetic/approval",
                revision="*",
            )
        ],
    )
    from clouda_training.preflight.checks_system import check_model_approval

    check = check_model_approval(config, catalog_path=catalog)
    assert check.status.value == "PASS"
    assert check.blocker is False


def test_preflight_skips_approval_for_mock(torch_config, tmp_path: Path) -> None:
    from clouda_training.preflight.checks_system import check_model_approval

    check = check_model_approval(torch_config, catalog_path=tmp_path / "missing.json")
    assert check.status.value == "SKIP"


# ---------------------------------------------------------------------------
# Catalog packaging (source tree + wheel) and env-var override
# ---------------------------------------------------------------------------


@pytest.fixture()
def _no_approvals_env(monkeypatch):
    """Isolate tests from any operator-set approval catalog."""
    monkeypatch.delenv("CLOUDA_MODEL_APPROVALS", raising=False)


def test_source_tree_finds_default_empty_catalog(_no_approvals_env) -> None:
    from clouda_training.adapters.approval import default_catalog_path
    from clouda_training.adapters.approval import load_approval_catalog

    path = default_catalog_path()
    assert path.is_file(), f"default catalog missing: {path}"
    payload = load_approval_catalog(path)
    assert payload["schema_version"] == 1
    assert payload["approvals"] == [], "shipped default must approve no model"


def test_packaged_resource_catalog_is_valid_and_empty(_no_approvals_env) -> None:
    """The wheel-shipped copy exists inside the package and blocks training."""
    from clouda_training.adapters import approval

    packaged = approval.PACKAGE_ROOT / "resources" / approval.PACKAGED_CATALOG_NAME
    assert packaged.is_file(), (
        "packaged default catalog missing — pip installs would have no "
        "approval catalog at all"
    )
    payload = approval.load_approval_catalog(packaged)
    assert payload["approvals"] == []


def test_repo_catalog_and_packaged_resource_are_identical(_no_approvals_env) -> None:
    """Source-tree runs prefer configs/, installed wheels use the packaged copy.

    The two defaults must parse to the same JSON so both execution modes
    enforce the same approval policy.
    """
    from clouda_training.adapters import approval

    repo_copy = approval.PACKAGE_ROOT.parent / approval.DEFAULT_CATALOG_RELPATH
    packaged_copy = approval.PACKAGE_ROOT / "resources" / approval.PACKAGED_CATALOG_NAME
    assert repo_copy.is_file(), f"repo-canonical catalog missing: {repo_copy}"
    assert packaged_copy.is_file(), f"packaged catalog missing: {packaged_copy}"
    with repo_copy.open(encoding="utf-8") as handle:
        repo_payload = json.load(handle)
    with packaged_copy.open(encoding="utf-8") as handle:
        packaged_payload = json.load(handle)
    assert repo_payload == packaged_payload
    assert (
        repo_payload["approvals"] == []
    ), "neither default catalog may approve a model"


def test_default_empty_catalog_blocks_real_model(_no_approvals_env) -> None:
    from clouda_training.adapters.approval import ModelTrainingNotApproved
    from clouda_training.adapters.approval import require_training_approval

    with pytest.raises(ModelTrainingNotApproved):
        require_training_approval(
            "hunyuanocr15_sft", "tencent/HunyuanOCR-1.5", "c55965d3da1e"
        )


def test_env_var_override_selects_deployment_catalog(
    tmp_path: Path, monkeypatch
) -> None:
    from clouda_training.adapters.approval import (
        default_catalog_path,
        get_training_approval,
    )

    catalog = _write_catalog(
        tmp_path,
        [
            dict(
                _APPROVED,
                adapter_type="qwen_vl_sft",
                model_id="Qwen/Qwen3-VL-4B",
                revision="*",
            )
        ],
    )
    monkeypatch.setenv("CLOUDA_MODEL_APPROVALS", str(catalog))
    assert default_catalog_path() == catalog
    approval = get_training_approval("qwen_vl_sft", "Qwen/Qwen3-VL-4B", "main")
    assert approval is not None
    assert approval.license_id == "license-upstream-allowlist"


def test_malformed_env_override_fails_closed(tmp_path: Path, monkeypatch) -> None:
    from clouda_training.adapters.approval import ModelTrainingNotApproved
    from clouda_training.adapters.approval import load_approval_catalog

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("CLOUDA_MODEL_APPROVALS", str(broken))
    with pytest.raises(ModelTrainingNotApproved, match="unreadable"):
        load_approval_catalog()


def test_missing_env_override_fails_closed_without_fallback(
    tmp_path: Path, monkeypatch
) -> None:
    """A broken override must never silently fall back to the packaged default."""
    from clouda_training.adapters.approval import ModelTrainingNotApproved
    from clouda_training.adapters.approval import require_training_approval

    monkeypatch.setenv("CLOUDA_MODEL_APPROVALS", str(tmp_path / "gone.json"))
    with pytest.raises(ModelTrainingNotApproved, match="not found"):
        require_training_approval(
            "hunyuanocr15_sft", "tencent/HunyuanOCR-1.5", "c55965d3da1e"
        )
