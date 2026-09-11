"""Regression tests for review findings (header protection, validator
robustness, adapter-identity checkpoint integration)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from clouda_training.hunyuan.exporter import export_raw_jsonl  # noqa: E402
from clouda_training.hunyuan.models import (  # noqa: E402
    ARABIC_DOCUMENT_OCR_PROMPT,
    HunyuanExportConfig,
)
from clouda_training.hunyuan.validators import (  # noqa: E402
    PackedSchemaError,
    RawSchemaError,
    validate_packed_line,
    validate_raw_jsonl,
    validate_raw_line,
)


def _write_manifest(path: Path, rows: list[dict]) -> str:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_rows(dataset_id: str = "s-ar") -> list[dict]:
    return [
        {
            "_schema_version": "clouda.pretraining.manifest.v1",
            "_row_count": 1,
            "dataset_id": dataset_id,
            "dataset_version": "v1",
        },
        {
            "sample_id": "ar-1",
            "target_split": "train",
            "source_id": dataset_id,
            "image_path": "p/1.png",
            "ground_truth": "نص عربي",
            "source_license": "Apache-2.0",
        },
    ]


def _config(manifest: Path, manifest_hash: str, tmp_path: Path) -> HunyuanExportConfig:
    return HunyuanExportConfig(
        dataset_id="s-ar",
        dataset_version="v1",
        manifest_path=str(manifest),
        manifest_hash=manifest_hash,
        split="train",
        image_root=str(tmp_path / "root"),
        prompt_profile=ARABIC_DOCUMENT_OCR_PROMPT,
    )


# ------------------------------------------------------- header protection (HIGH)


def test_protected_manifest_header_refuses_export(tmp_path: Path) -> None:
    rows = _clean_rows()
    rows[0]["protected"] = True  # whole-dataset protection marker in HEADER
    manifest = tmp_path / "protected_manifest.jsonl"
    h = _write_manifest(manifest, rows)
    with pytest.raises(PermissionError, match="protected"):
        export_raw_jsonl(_config(manifest, h, tmp_path), tmp_path / "out.jsonl")
    assert not (tmp_path / "out.jsonl").exists()


def test_evaluation_only_manifest_header_refuses_export(tmp_path: Path) -> None:
    rows = _clean_rows()
    rows[0]["purpose"] = "evaluation_only"
    rows[0]["protected"] = True
    manifest = tmp_path / "eval_manifest.jsonl"
    h = _write_manifest(manifest, rows)
    with pytest.raises(PermissionError):
        export_raw_jsonl(_config(manifest, h, tmp_path), tmp_path / "out.jsonl")


def test_clean_header_still_exports(tmp_path: Path) -> None:
    manifest = tmp_path / "clean.jsonl"
    h = _write_manifest(manifest, _clean_rows())
    report = export_raw_jsonl(_config(manifest, h, tmp_path), tmp_path / "out.jsonl")
    assert report.exported_count == 1


# ------------------------------------------------- validator robustness (MEDIUM)


def test_non_dict_conversation_turn_raises_raw_schema_error() -> None:
    sample = {
        "image_path": ["/abs/img.png"],
        "conversations": ["not-a-dict", {"from": "gpt", "value": "x"}],
    }
    with pytest.raises(RawSchemaError, match="JSON objects"):
        validate_raw_line(sample)


def test_non_dict_turn_in_packed_wraps_as_packed_error() -> None:
    bad_raw = {
        "image_path": ["/abs/img.png"],
        "conversations": [42],
    }
    packed = {
        "packed_samples": [bad_raw],
        "cu_seqlens": [0, 10],
        "total_tokens": 10,
    }
    with pytest.raises(PackedSchemaError, match="malformed embedded raw sample"):
        validate_packed_line(packed)


def test_non_dict_turn_in_file_reports_clean_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {
                "image_path": ["/a.png"],
                "conversations": ["x", {"from": "gpt", "value": "y"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RawSchemaError, match="line 1"):
        validate_raw_jsonl(bad)


# --------------------------- adapter identity persisted + resume rejection (MED)


def test_checkpoint_persists_adapter_identity_and_rejects_change(
    tmp_path: Path,
) -> None:
    from clouda_training.hunyuan.adapter import HunyuanOCR15SFTAdapter
    from tests.hunyuan.mock_hunyuan import MockHunyuanVLForConditionalGeneration
    from clouda_training.runtime.torch_backend import TorchTrainerBackend

    adapter = HunyuanOCR15SFTAdapter(local_model_path="x", processor=None)
    model = MockHunyuanVLForConditionalGeneration()
    adapter.model = model

    backend = TorchTrainerBackend.__new__(TorchTrainerBackend)
    backend.torch = __import__("torch")
    backend.adapter = adapter
    backend.model = model

    class _Mgr:
        def __init__(self, root: Path):
            self.root = root
            self.latest_info = None

        def save(self, *, step, epoch, metrics):
            d = self.root / f"step-{step:08d}"
            d.mkdir(parents=True, exist_ok=True)
            self.latest_info = type("Info", (), {"path": d})()
            return self.latest_info

        def latest(self):
            return self.latest_info

    mgr = _Mgr(tmp_path)
    backend.checkpoints = mgr
    # minimal config shim for optimizer construction
    from clouda_training.experiments.config import TrainingSection

    backend.config = type("C", (), {})()
    backend.config.training = TrainingSection(
        seed=17, max_steps=4, batch_size=2, learning_rate=0.01
    )
    backend._build_optimizer_and_scheduler()
    backend._save_checkpoint(step=3, metrics={"loss": 0.5})

    metadata = json.loads(
        (mgr.latest_info.path / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["adapter_identity"]["adapter_id"] == "hunyuanocr15_sft"
    assert metadata["adapter_identity"]["upstream_revision"] == "c55965d3da1e"
    assert str(tmp_path) not in json.dumps(metadata["adapter_identity"])

    # Same identity -> restore path proceeds past the identity gate.
    backend._restore_from_latest_checkpoint(
        start_step=3
    )  # must not raise identity error

    # Changed trainable config -> resume must be rejected.
    adapter2 = HunyuanOCR15SFTAdapter(
        local_model_path="x", processor=None, tune_llm=False
    )
    adapter2.model = MockHunyuanVLForConditionalGeneration()
    backend.adapter = adapter2
    with pytest.raises(RuntimeError, match="adapter identity mismatch"):
        backend._restore_from_latest_checkpoint(start_step=3)
