from __future__ import annotations

from pathlib import Path

import pytest

from clouda_contracts.storage import StorageRoots
from clouda_models.local_models import resolve_local_checkpoint
from clouda_models.metadata import ModelMetadata
from clouda_models.registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "models" / "registry.v1.json"


def test_placeholder_registry_is_valid_and_disabled() -> None:
    registry = ModelRegistry.load(REGISTRY)
    registry.validate_no_enabled_placeholder()
    record = registry.get("future-vlm-adapter")
    assert record.deployment_status == "disabled"
    assert registry.deployable() == ()


def test_unpinned_model_cannot_be_enabled() -> None:
    registry = ModelRegistry(
        (
            ModelMetadata(
                model_id="unsafe",
                base_model_name="provider/model",
                model_revision="main",
                license="unknown",
                model_type="vlm",
                parameter_count=None,
                architecture="unknown",
                active_parameter_count=None,
                supported_image_inputs=("png",),
                tokenizer_revision="main",
                processor_revision="main",
                training_method="adapter",
                dataset_manifest_ids=(),
                checkpoint_uri=None,
                deployment_status="evaluation",
                commercial_use_status="pending",
                attribution_requirements="",
            ),
        )
    )
    with pytest.raises(ValueError, match="unpinned"):
        registry.validate_no_enabled_placeholder()


def test_local_checkpoint_stays_under_model_root(tmp_path: Path) -> None:
    roots = StorageRoots.from_env({"CLOUDA_STATE_HOME": str(tmp_path)})
    record = ModelRegistry.load(REGISTRY).get("future-vlm-adapter")
    executable = ModelMetadata.from_dict(
        {
            **record.to_dict(),
            "checkpoint_uri": "model://unsafe/model.exe",
        }
    )
    with pytest.raises(ValueError, match="Executable"):
        resolve_local_checkpoint(executable, roots=roots)
