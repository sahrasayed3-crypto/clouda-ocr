"""QwenVLSFTAdapter unit tests (torch-dependent, synthetic fixtures)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from clouda_training.qwen.adapter import (  # noqa: E402
    QwenAdapterError,
    QwenVLSFTAdapter,
    model_class_available,
    transformers_version_ok,
)
from clouda_training.qwen.descriptor import QWEN_DESCRIPTOR  # noqa: E402
from tests.multimodel.mock_qwen import (  # noqa: E402
    MockQwen3VLForConditionalGeneration,
    MockQwenProcessor,
)


def test_identity_no_machine_paths(tmp_path) -> None:
    adapter = QwenVLSFTAdapter(local_model_path=str(tmp_path / "m"))
    meta = adapter.identity()
    assert meta["adapter_id"] == "qwen_vl_sft"
    assert meta["upstream_revision"] == ("96588727e44c78b25ba03ea03b8e12f7e64fd0da")
    assert meta["trainable"] == {"vision": True, "projector": True, "llm": True}
    assert str(tmp_path) not in str(meta)


def test_identity_mirrors_descriptor_facts(tmp_path) -> None:
    meta = QwenVLSFTAdapter(local_model_path=str(tmp_path / "m")).identity()
    assert meta["upstream_repository"] == QWEN_DESCRIPTOR.upstream_repository
    assert meta["adapter_version"] == QWEN_DESCRIPTOR.adapter_version


def test_local_only_path_enforced(tmp_path) -> None:
    adapter = QwenVLSFTAdapter(local_model_path=str(tmp_path / "missing"))
    with pytest.raises(QwenAdapterError, match="never downloads"):
        adapter.build_model()


def test_missing_config_json_rejected(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    adapter = QwenVLSFTAdapter(local_model_path=str(empty))
    with pytest.raises(QwenAdapterError, match="config.json"):
        adapter.build_model()


def test_processor_not_loaded_make_batch_fails(tmp_path) -> None:
    adapter = QwenVLSFTAdapter(local_model_path=str(tmp_path / "m"))
    with pytest.raises(QwenAdapterError, match="processor not loaded"):
        adapter.make_batch(step=0, batch_size=1, seed=1)


def test_trainable_component_selection_mock(tmp_path) -> None:
    adapter = QwenVLSFTAdapter(local_model_path=str(tmp_path / "x"))
    model = MockQwen3VLForConditionalGeneration()
    adapter.model = model
    adapter._apply_trainable_selection()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "expected trainable params with all tune flags on"

    # freeze vision -> visual params frozen (incl. merger)
    adapter2 = QwenVLSFTAdapter(
        local_model_path="x", tune_vision=False, tune_projector=False
    )
    model2 = MockQwen3VLForConditionalGeneration()
    adapter2.model = model2
    adapter2._apply_trainable_selection()
    assert all(
        not p.requires_grad
        for n, p in model2.named_parameters()
        if n.startswith("visual")
    )
    # llm stays trainable
    assert any(
        p.requires_grad
        for n, p in model2.named_parameters()
        if n.startswith("language_model") or n.startswith("lm_head")
    )

    # freeze projector only -> visual.merger frozen, rest of visual trainable
    adapter3 = QwenVLSFTAdapter(local_model_path="x", tune_projector=False)
    model3 = MockQwen3VLForConditionalGeneration()
    adapter3.model = model3
    adapter3._apply_trainable_selection()
    assert all(
        not p.requires_grad
        for n, p in model3.named_parameters()
        if n.startswith("visual.merger")
    )
    assert any(
        p.requires_grad
        for n, p in model3.named_parameters()
        if n.startswith("visual.proj")
    )


def test_loss_extraction_real_output(tmp_path) -> None:
    adapter = QwenVLSFTAdapter(local_model_path=str(tmp_path / "x"))
    model = MockQwen3VLForConditionalGeneration()
    batch = {
        "input_ids": torch.randint(1, 512, (2, 16)),
        "labels": torch.randint(1, 512, (2, 16)),
    }
    loss = adapter.forward_loss(model, batch)
    assert loss is not None
    assert loss.requires_grad


def test_loss_absence_fails_explicitly(tmp_path) -> None:
    adapter = QwenVLSFTAdapter(local_model_path=str(tmp_path / "x"))

    class NoLoss:
        pass

    class MockModel:
        def forward(self, **kw):
            out = NoLoss()
            out.logits = torch.zeros(1)
            return out

        def __call__(self, **kw):
            return self.forward(**kw)

    with pytest.raises(QwenAdapterError, match="no loss"):
        adapter.forward_loss(MockModel(), {"input_ids": torch.zeros(1, 1).long()})


def test_model_state_roundtrip(tmp_path) -> None:
    adapter = QwenVLSFTAdapter(local_model_path=str(tmp_path / "x"))
    model = MockQwen3VLForConditionalGeneration()
    state = adapter.model_state(model)
    assert state
    model2 = MockQwen3VLForConditionalGeneration()
    adapter.load_model_state(model2, state)
    for (k1, v1), (k2, v2) in zip(
        model.state_dict().items(), model2.state_dict().items()
    ):
        assert k1 == k2
        assert torch.equal(v1, v2)


def test_make_batch_processor_bound(tmp_path) -> None:
    processor = MockQwenProcessor()
    adapter = QwenVLSFTAdapter(local_model_path="unused", processor=processor)
    batch = adapter.make_batch(step=3, batch_size=2, seed=17)
    assert set(batch) >= {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape == (2, processor.seq_len)


def test_native_transformers_class_available() -> None:
    """The verified upstream symbol imports natively (no trust_remote_code)."""
    if not transformers_version_ok():
        pytest.skip("requires optional transformers>=4.57 with native Qwen3-VL")
    assert transformers_version_ok() is True
    assert model_class_available() is True
