"""Cross-adapter resume rejection + optional-import smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from clouda_training.hunyuan.adapter import HunyuanOCR15SFTAdapter  # noqa: E402
from clouda_training.qwen.adapter import QwenVLSFTAdapter  # noqa: E402
from clouda_training.runtime.torch_backend import TorchTrainerBackend  # noqa: E402
from tests.multimodel.mock_qwen import (  # noqa: E402
    MockQwen3VLForConditionalGeneration,
    MockQwenProcessor,
)


def _build_backend(tmp_path: Path, adapter, model, config) -> TorchTrainerBackend:
    from clouda_training.experiments.metrics import MetricLogger

    metrics_logger = MetricLogger(tmp_path / "metrics.jsonl", "probe")

    backend = TorchTrainerBackend.__new__(TorchTrainerBackend)
    backend.torch = torch
    backend.metrics = metrics_logger
    backend.checkpoints = None
    backend.config = config
    backend.adapter = adapter
    backend.model = model
    backend.device = torch.device("cpu")
    backend.fail_at_step = None
    backend.interrupt_at_step = None
    backend._build_optimizer_and_scheduler()
    return backend


def _minimal_config(tmp_path: Path):
    from clouda_training.experiments.config import (
        CheckpointSection,
        RuntimeSection,
        TrackingSection,
        TrainingSection,
    )

    config = type("C", (), {})()
    config.training = TrainingSection(
        seed=29, max_steps=2, batch_size=2, learning_rate=0.01
    )
    config.checkpoint = CheckpointSection(save_strategy="steps", save_steps=1)
    config.tracking = TrackingSection(enabled=True, log_steps=1)
    config.runtime = RuntimeSection(device="cpu", output_root=tmp_path)
    return config


def test_cross_adapter_resume_rejected(tmp_path: Path) -> None:
    """A checkpoint saved by the qwen adapter must NOT resume under the
    hunyuan adapter (checkpoint adapter-identity gate, fail-closed)."""

    config = _minimal_config(tmp_path)
    qwen_model = MockQwen3VLForConditionalGeneration()
    qwen_adapter = QwenVLSFTAdapter(local_model_path="unused-mock")
    qwen_adapter.model = qwen_model
    qwen_adapter.processor = MockQwenProcessor()

    backend = _build_backend(tmp_path, qwen_adapter, qwen_model, config)
    # the shim needs a checkpoint manager that remembers saves
    from types import SimpleNamespace

    saved: list[Path] = []

    class Mgr:
        def save(self, **kw):
            directory = tmp_path / "ckpts" / f"step-{kw['step']:08d}"
            directory.mkdir(parents=True, exist_ok=True)
            saved.append(directory)
            return SimpleNamespace(path=directory)

        def latest(self):
            return SimpleNamespace(path=saved[-1]) if saved else None

    backend.checkpoints = Mgr()
    result = backend.train(start_step=0)
    assert result.final_step == 2
    assert saved, "expected a checkpoint to be saved"

    # Now try to resume the SAME checkpoint with the HUNYUAN adapter
    hunyuan_adapter = HunyuanOCR15SFTAdapter(local_model_path="unused-mock")
    hunyuan_adapter.model = MockQwen3VLForConditionalGeneration()  # same shape
    hunyuan_adapter.processor = MockQwenProcessor()
    hunyuan_backend = _build_backend(
        tmp_path, hunyuan_adapter, hunyuan_adapter.model, config
    )
    hunyuan_backend.checkpoints = Mgr()
    with pytest.raises(RuntimeError, match="adapter identity mismatch"):
        hunyuan_backend.train(start_step=2)


def test_same_adapter_resume_identity_stable(tmp_path: Path) -> None:
    """Two identically-configured qwen adapters produce identical identity —
    resume under the same adapter must NOT be rejected on identity."""
    a = QwenVLSFTAdapter(
        local_model_path="x",
        tune_vision=True,
        tune_projector=True,
        tune_llm=True,
    )
    b = QwenVLSFTAdapter(local_model_path="completely/different/path")
    assert a.identity() == b.identity()


def test_optional_imports_smoke() -> None:
    """All adapter framework entry points import cleanly (torch may be absent
    in this env — these imports must not require it)."""
    import clouda_training.adapters  # noqa: F401
    import clouda_training.hunyuan  # noqa: F401
    import clouda_training.qwen  # noqa: F401

    from clouda_training.adapters.data_adapter import (  # noqa: F401
        DataAdapterRegistry,
        ModelTrainingDataAdapter,
        get_default_data_adapter_registry,
    )
    from clouda_training.adapters.registry import (  # noqa: F401
        ModelAdapterRegistry,
        get_default_registry,
    )
    from clouda_training.hunyuan.data_adapter import HunyuanTrainingDataAdapter
    from clouda_training.hunyuan.descriptor import HUNYUAN_DESCRIPTOR
    from clouda_training.qwen.data_adapter import QwenTrainingDataAdapter
    from clouda_training.qwen.descriptor import QWEN_DESCRIPTOR

    # descriptors build without any heavy import
    assert HUNYUAN_DESCRIPTOR.adapter_type
    assert QWEN_DESCRIPTOR.adapter_type
    assert QwenTrainingDataAdapter().describe_contract()
    assert HunyuanTrainingDataAdapter().describe_contract()


def test_checkpoint_metadata_carries_adapter_identity(tmp_path: Path) -> None:
    """The identity persisted in checkpoints matches descriptor facts."""
    adapter = QwenVLSFTAdapter(local_model_path="x")
    identity = adapter.identity()
    # roundtrip through JSON (checkpoint metadata is json-serialized)
    restored = json.loads(json.dumps(identity))
    assert restored == identity
    assert restored["adapter_id"] == "qwen_vl_sft"
