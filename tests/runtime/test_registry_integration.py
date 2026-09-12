"""Canonical adapter registry -> experiment runtime integration."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from clouda_training.adapters.capabilities import ModelCapabilities  # noqa: E402
from clouda_training.adapters.descriptor import ModelAdapterDescriptor  # noqa: E402
from clouda_training.adapters.registry import get_default_registry  # noqa: E402
from clouda_training.experiments import (  # noqa: E402
    RunStatus,
    list_checkpoints,
    list_runs,
    resume_run,
    run_experiment,
)
from clouda_training.runtime.adapter import SyntheticLinearAdapter  # noqa: E402


class RegistrySyntheticAdapter(SyntheticLinearAdapter):
    def identity(self) -> dict[str, Any]:
        return {
            "adapter_id": "registry_runtime_test",
            "adapter_version": "1.0.0",
            "upstream_revision": "synthetic-v1",
        }


def test_registered_adapter_runs_and_resumes_through_experiment_framework(
    torch_config,
) -> None:
    registry = get_default_registry()
    adapter_type = "registry_runtime_test"
    seen_configs: list[Any] = []

    def factory(*, config):
        seen_configs.append(config)
        return RegistrySyntheticAdapter()

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
            checkpoint_compatibility_id="registry-runtime-test-v1",
            upstream_revision="synthetic-v1",
        ),
        factory,
    )
    config = dataclasses.replace(
        torch_config,
        model=dataclasses.replace(
            torch_config.model,
            model_id="synthetic/registry",
            adapter_type=adapter_type,
        ),
    )
    try:
        with pytest.raises(KeyboardInterrupt):
            run_experiment(config, interrupt_at_step=9)
        interrupted = list_runs(
            config.runtime.output_root, status=RunStatus.INTERRUPTED
        )[0]
        resumed = resume_run(interrupted.run_id, config.runtime.output_root)

        assert resumed.status is RunStatus.COMPLETED
        assert len(seen_configs) == 2
        assert all(item.model.adapter_type == adapter_type for item in seen_configs)
        checkpoint = list_checkpoints(resumed.path)[-1]
        metadata = json.loads(
            (checkpoint.path / "metadata.json").read_text(encoding="utf-8")
        )
        assert metadata["adapter_identity"] == {
            "adapter_id": "registry_runtime_test",
            "adapter_version": "1.0.0",
            "upstream_revision": "synthetic-v1",
        }
    finally:
        registry.unregister(adapter_type)
