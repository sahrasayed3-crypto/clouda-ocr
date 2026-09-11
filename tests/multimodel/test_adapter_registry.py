"""Registry, descriptor, and capability tests (both adapters)."""

from __future__ import annotations

import pytest

from clouda_training.adapters.capabilities import ModelCapabilities
from clouda_training.adapters.data_adapter import get_default_data_adapter_registry
from clouda_training.adapters.descriptor import (
    UnsupportedCapabilityError,
)
from clouda_training.adapters.registry import get_default_registry
from clouda_training.hunyuan.descriptor import (
    CHECKPOINT_COMPATIBILITY_ID as HUNYUAN_CKPT_ID,
)
from clouda_training.hunyuan.descriptor import HUNYUAN_DESCRIPTOR
from clouda_training.qwen.descriptor import (
    CHECKPOINT_COMPATIBILITY_ID as QWEN_CKPT_ID,
)
from clouda_training.qwen.descriptor import QWEN_DESCRIPTOR


class TestRegistry:
    def test_both_adapters_registered(self) -> None:
        registry = get_default_registry()
        assert registry.is_registered("hunyuanocr15_sft")
        assert registry.is_registered("qwen_vl_sft")
        assert registry.list_adapters() == ("hunyuanocr15_sft", "qwen_vl_sft")

    def test_both_data_adapters_registered(self) -> None:
        registry = get_default_data_adapter_registry()
        assert registry.is_registered("hunyuanocr15_sft_data")
        assert registry.is_registered("qwen_vl_sft_data")

    def test_registration_is_idempotent(self) -> None:
        from clouda_training.hunyuan.registration import register_hunyuan_adapters
        from clouda_training.qwen.registration import register_qwen_adapters

        # repeated registration must never raise DuplicateAdapterError
        register_hunyuan_adapters()
        register_hunyuan_adapters()
        register_qwen_adapters()
        register_qwen_adapters()
        assert get_default_registry().list_adapters() == (
            "hunyuanocr15_sft",
            "qwen_vl_sft",
        )

    def test_duplicate_registration_on_fresh_registry_raises(self) -> None:
        from clouda_training.adapters.data_adapter import DataAdapterRegistry
        from clouda_training.adapters.registry import ModelAdapterRegistry
        from clouda_training.hunyuan.data_adapter import (
            HUNYUAN_DATA_ADAPTER_TYPE,
            HunyuanTrainingDataAdapter,
        )
        from clouda_training.hunyuan.descriptor import HUNYUAN_DESCRIPTOR
        from clouda_training.hunyuan.registration import register_hunyuan_adapters
        from clouda_training.qwen.data_adapter import QwenTrainingDataAdapter
        from clouda_training.qwen.registration import register_qwen_adapters

        fresh = ModelAdapterRegistry(name="fresh")
        register_hunyuan_adapters(model_registry=fresh)
        with pytest.raises(Exception, match="already registered"):
            fresh.register(HUNYUAN_DESCRIPTOR, lambda **kw: None)

        fresh_data = DataAdapterRegistry(name="fresh-data")
        fresh_data.register("qwen_vl_sft_data", QwenTrainingDataAdapter)
        with pytest.raises(Exception, match="already registered"):
            fresh_data.register("qwen_vl_sft_data", QwenTrainingDataAdapter)
        fresh_data.register(HUNYUAN_DATA_ADAPTER_TYPE, HunyuanTrainingDataAdapter)
        with pytest.raises(Exception, match="already registered"):
            fresh_data.register(HUNYUAN_DATA_ADAPTER_TYPE, HunyuanTrainingDataAdapter)
        # and the guarded path stays idempotent on the fresh registries
        register_hunyuan_adapters(model_registry=fresh)
        register_qwen_adapters(
            model_registry=fresh,
            data_registry=DataAdapterRegistry(name="fresh-data-2"),
        )

    def test_lazy_factory_builds_without_torch(self) -> None:
        registry = get_default_registry()
        adapter = registry.create("qwen_vl_sft", local_model_path="unused")
        assert adapter.identity()["adapter_id"] == "qwen_vl_sft"
        adapter = registry.create("hunyuanocr15_sft", local_model_path="unused")
        assert adapter.identity()["adapter_id"] == "hunyuanocr15_sft"

    def test_unknown_adapter_lists_registered(self) -> None:
        from clouda_training.adapters.registry import UnknownAdapterError

        with pytest.raises(UnknownAdapterError, match="qwen_vl_sft"):
            get_default_registry().get("qwen_vl_sft_typo")


class TestDescriptors:
    def test_hunyuan_descriptor_facts(self) -> None:
        d = HUNYUAN_DESCRIPTOR
        assert d.adapter_type == "hunyuanocr15_sft"
        assert d.upstream_revision == "c55965d3da1e"
        assert d.upstream_repository == "Tencent-Hunyuan/HunyuanOCR"
        assert d.model_family == "hunyuanocr15"
        caps = d.capabilities
        assert caps.supports_local_only_loading is True
        assert caps.supports_packed_sequences is True
        assert caps.supports_resume is True
        assert caps.supports_cpu_smoke is True  # via mock
        assert caps.real_weights_validated is False
        assert caps.gpu_validated is False
        # honest unverified claims stay False
        assert caps.supports_lora is False
        assert caps.supports_fp16 is False

    def test_hunyuan_checkpoint_compatibility_id_deterministic(self) -> None:
        assert HUNYUAN_CKPT_ID == "hunyuanocr15-sft@c55965d3da1e"
        # deterministic across imports
        from clouda_training.hunyuan.descriptor import HUNYUAN_DESCRIPTOR as again

        assert again.checkpoint_compatibility_id == HUNYUAN_CKPT_ID

    def test_qwen_descriptor_facts(self) -> None:
        d = QWEN_DESCRIPTOR
        assert d.adapter_type == "qwen_vl_sft"
        assert d.upstream_revision == "96588727e44c78b25ba03ea03b8e12f7e64fd0da"
        assert d.upstream_repository == "QwenLM/Qwen3-VL"
        assert d.model_family == "qwen3_vl"
        caps = d.capabilities
        assert caps.supports_multimodal_batches is True
        assert caps.supports_packed_sequences is True
        assert caps.supports_local_only_loading is True
        assert caps.supports_resume is True
        assert caps.supports_cpu_smoke is True
        assert caps.real_weights_validated is False
        assert caps.gpu_validated is False
        # native transformers >= 4.57 — no trust_remote_code anywhere
        assert all(
            "trust_remote_code" not in dep for dep in d.required_optional_dependencies
        )
        assert any(
            dep.startswith("transformers>=") for dep in d.required_optional_dependencies
        )

    def test_qwen_checkpoint_compatibility_id_deterministic(self) -> None:
        assert QWEN_CKPT_ID == ("qwen3vl-sft@96588727e44c78b25ba03ea03b8e12f7e64fd0da")

    def test_identity_dict_excludes_paths_and_capabilities(self) -> None:
        for descriptor in (HUNYUAN_DESCRIPTOR, QWEN_DESCRIPTOR):
            identity = descriptor.identity_dict()
            flat = str(identity)
            assert "local_model_path" not in flat
            assert "capabilities" not in identity
            assert identity["adapter_type"] == descriptor.adapter_type
            assert identity["upstream_revision"] == descriptor.upstream_revision

    def test_descriptor_is_frozen(self) -> None:
        for descriptor in (HUNYUAN_DESCRIPTOR, QWEN_DESCRIPTOR):
            with pytest.raises(Exception):
                descriptor.adapter_type = "mutated"  # type: ignore[misc]

    def test_capabilities_are_frozen(self) -> None:
        for caps in (HUNYUAN_DESCRIPTOR.capabilities, QWEN_DESCRIPTOR.capabilities):
            assert isinstance(caps, ModelCapabilities)
            with pytest.raises(Exception):
                caps.real_weights_validated = True  # type: ignore[misc]

    def test_require_capabilities_fails_closed(self) -> None:
        with pytest.raises(UnsupportedCapabilityError, match="gpu_validated"):
            HUNYUAN_DESCRIPTOR.require_capabilities("gpu_validated")
        with pytest.raises(UnsupportedCapabilityError, match="unknown capability"):
            HUNYUAN_DESCRIPTOR.require_capabilities("nonexistent_flag")
        # supported flags pass
        QWEN_DESCRIPTOR.require_capabilities(
            "supports_multimodal_batches", "supports_packed_sequences"
        )

    def test_descriptor_type_enforced_by_registry(self) -> None:
        from clouda_training.adapters.registry import ModelAdapterRegistry

        fresh = ModelAdapterRegistry(name="type-check")
        with pytest.raises(TypeError, match="ModelAdapterDescriptor"):
            fresh.register(
                {"adapter_type": "bogus"},  # type: ignore[arg-type]
                lambda **kw: None,
            )

    def test_summary_flags_are_honest(self) -> None:
        summary = QWEN_DESCRIPTOR.capabilities.summary()
        assert summary["real_weights_validated"] is False
        assert summary["gpu_validated"] is False
        assert summary["supports_local_only_loading"] is True
        hunyuan_summary = HUNYUAN_DESCRIPTOR.capabilities.summary()
        assert hunyuan_summary["real_weights_validated"] is False
        assert hunyuan_summary["gpu_validated"] is False
