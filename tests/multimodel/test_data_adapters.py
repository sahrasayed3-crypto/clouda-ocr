"""Data adapter tests: hunyuanocr15_sft_data + qwen_vl_sft_data (no torch)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clouda_training.adapters.data_adapter import (
    ModelTrainingDataAdapter,
    get_default_data_adapter_registry,
)
from clouda_training.hunyuan.data_adapter import (
    HUNYUAN_DATA_ADAPTER_TYPE,
    HunyuanDataAdapterError,
    HunyuanTrainingDataAdapter,
)
from clouda_training.hunyuan.exporter import build_raw_sample
from clouda_training.hunyuan.models import HunyuanExportConfig
from clouda_training.qwen.data_adapter import (
    QwenDataAdapterError,
    QwenExportConfig,
    QwenTrainingDataAdapter,
    build_qwen_record,
)


def _clean_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not r.get("protected")]


class TestQwenDataAdapter:
    def test_isinstance_protocol(self) -> None:
        adapter = QwenTrainingDataAdapter()
        assert isinstance(adapter, ModelTrainingDataAdapter)

    def test_export_produces_official_format(
        self, canonical_rows, qwen_export_config: QwenExportConfig
    ) -> None:
        adapter = QwenTrainingDataAdapter()
        records = adapter.export(_clean_rows(canonical_rows), qwen_export_config)
        assert len(records) == 2
        record = records[0]
        assert set(record) == {"image", "conversations"}
        assert record["image"].endswith("pages\\page_001.png") or record[
            "image"
        ].endswith("pages/page_001.png")
        human, gpt = record["conversations"]
        assert human["from"] == "human"
        assert gpt["from"] == "gpt"
        # <image> tag only in the human turn, exactly once, before the prompt
        assert human["value"].startswith("<image>\n")
        assert human["value"].count("<image>") == 1
        assert "<image>" not in gpt["value"]
        assert gpt["value"] == canonical_rows[0]["ground_truth"]

    def test_records_are_json_serializable(
        self, canonical_rows, qwen_export_config: QwenExportConfig
    ) -> None:
        records = QwenTrainingDataAdapter().export(
            _clean_rows(canonical_rows), qwen_export_config
        )
        line = json.dumps(records[0], ensure_ascii=False)
        assert json.loads(line) == records[0]

    def test_lineage_report(
        self, canonical_rows, qwen_export_config: QwenExportConfig
    ) -> None:
        import hashlib

        adapter = QwenTrainingDataAdapter()
        rows = _clean_rows(canonical_rows)
        adapter.export(rows, qwen_export_config)
        report = adapter.last_lineage_report
        assert report is not None
        payload = report.to_dict()
        assert payload["sample_ids"] == ["ar-001", "ar-002"]
        assert (
            payload["manifest_sha256"]
            == hashlib.sha256(
                json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
        )
        assert (
            payload["prompt_sha256"]
            == hashlib.sha256(
                qwen_export_config.prompt_text.encode("utf-8")
            ).hexdigest()
        )
        assert payload["upstream_revision"] == (
            "96588727e44c78b25ba03ea03b8e12f7e64fd0da"
        )
        # prompt identity is in the report, NOT in the wire format
        adapter = QwenTrainingDataAdapter()
        records = adapter.export(rows, qwen_export_config)
        assert "prompt_sha256" not in json.dumps(records, ensure_ascii=False)
        assert report.to_dict()["sample_ids"] == ["ar-001", "ar-002"]

    def test_protected_rows_skipped_and_counted(
        self, canonical_rows, qwen_export_config: QwenExportConfig
    ) -> None:
        adapter = QwenTrainingDataAdapter()
        records = adapter.export(canonical_rows, qwen_export_config)
        assert len(records) == 2  # protected row skipped
        assert adapter.last_lineage_report.skipped_counts.get("protected") == 1

    def test_validate_ok(
        self, canonical_rows, qwen_export_config: QwenExportConfig
    ) -> None:
        adapter = QwenTrainingDataAdapter()
        records = adapter.export(_clean_rows(canonical_rows), qwen_export_config)
        summary = adapter.validate(records)
        assert summary["valid"] is True
        assert summary["errors"] == []

    @pytest.mark.parametrize(
        "broken",
        [
            {"image": "", "conversations": []},
            {"conversations": [{"from": "human", "value": "hi"}]},
            {
                "image": "a.png",
                "conversations": [
                    {"from": "gpt", "value": "no human first"},
                    {"from": "gpt", "value": "x"},
                ],
            },
            {
                "image": "a.png",
                "conversations": [
                    {"from": "human", "value": "missing image tag"},
                    {"from": "gpt", "value": "x"},
                ],
            },
            {
                "image": "a.png",
                "conversations": [
                    {"from": "human", "value": "<image>\n<p>\n<image>"},
                    {"from": "gpt", "value": "x"},
                ],
            },
            {
                "image": "a.png",
                "conversations": [
                    {"from": "human", "value": "<image>\nq"},
                    {"from": "gpt", "value": "   "},
                ],
            },
            {
                "image": "a.png",
                "conversations": [
                    {"from": "system", "value": "<image>\nq"},
                    {"from": "gpt", "value": "x"},
                ],
            },
        ],
    )
    def test_validate_failures(self, broken) -> None:
        summary = QwenTrainingDataAdapter().validate([broken])
        assert summary["valid"] is False
        assert summary["errors"], f"expected errors for {broken!r}"

    def test_describe_contract(self) -> None:
        contract = QwenTrainingDataAdapter().describe_contract()
        assert contract["data_contract_version"] == "1"
        assert contract["upstream_revision"] == (
            "96588727e44c78b25ba03ea03b8e12f7e64fd0da"
        )
        assert contract["image_tag"]["value"] == "<image>"
        assert contract["image_tag"]["level"] == "text"
        assert contract["loss_masking"].startswith("labels = -100")
        # machine-assertable: image stays a file path field
        assert "file path" in contract["fields"]["image"]["description"]

    def test_wrong_config_type_rejected(self, canonical_rows) -> None:
        with pytest.raises(QwenDataAdapterError, match="QwenExportConfig"):
            QwenTrainingDataAdapter().export(canonical_rows, object())

    def test_malformed_row_fails_closed(
        self, qwen_export_config: QwenExportConfig
    ) -> None:
        with pytest.raises(QwenDataAdapterError, match="sample_id"):
            build_qwen_record({"image": "x.png"}, qwen_export_config)
        with pytest.raises(QwenDataAdapterError, match="ground truth"):
            build_qwen_record({"sample_id": "s", "image": "x.png"}, qwen_export_config)
        with pytest.raises(QwenDataAdapterError, match="image"):
            build_qwen_record(
                {"sample_id": "s", "ground_truth": "t"}, qwen_export_config
            )


class TestHunyuanDataAdapter:
    def test_isinstance_protocol(self) -> None:
        assert isinstance(HunyuanTrainingDataAdapter(), ModelTrainingDataAdapter)

    def test_registry_roundtrip(self, hunyuan_export_config, canonical_rows) -> None:
        factory = get_default_data_adapter_registry().get(HUNYUAN_DATA_ADAPTER_TYPE)
        adapter = factory()
        records = adapter.export(_clean_rows(canonical_rows), hunyuan_export_config)
        assert len(records) == 2
        assert records[0]["conversations"][0]["from"] == "human"
        assert "<image>" in records[0]["conversations"][0]["value"]

    def test_export_wraps_bridge_exporter(
        self, canonical_rows, hunyuan_export_config: HunyuanExportConfig
    ) -> None:
        adapter = HunyuanTrainingDataAdapter()
        rows = _clean_rows(canonical_rows)
        records = adapter.export(rows, hunyuan_export_config)
        # parity with the frozen bridge exporter, row by row
        for row, record in zip(rows, records):
            sample = build_raw_sample(row, hunyuan_export_config)
            assert record == sample.to_upstream_dict()

    def test_lineage_report(
        self, canonical_rows, hunyuan_export_config: HunyuanExportConfig
    ) -> None:
        adapter = HunyuanTrainingDataAdapter()
        adapter.export(_clean_rows(canonical_rows), hunyuan_export_config)
        report = adapter.last_lineage_report
        assert report is not None
        assert report["sample_ids"] == ["ar-001", "ar-002"]
        assert report["manifest_sha256"] == hunyuan_export_config.manifest_hash
        assert report["prompt_identity"] == ("clouda_arabic_document_ocr@v1")
        assert report["upstream_revision"] == "c55965d3da1e"

    def test_protected_row_fails_closed(
        self, canonical_rows, hunyuan_export_config: HunyuanExportConfig
    ) -> None:
        from clouda_training.hunyuan.exporter import HunyuanExportError

        adapter = HunyuanTrainingDataAdapter()
        # the wrapped frozen bridge exporter raises its own error type;
        # per-row skip semantics belong to export_raw_jsonl, not this wrapper
        with pytest.raises(HunyuanExportError, match="protected"):
            adapter.export(canonical_rows, hunyuan_export_config)

    def test_duplicate_sample_ids_rejected(
        self, hunyuan_export_config: HunyuanExportConfig
    ) -> None:
        row = {
            "sample_id": "dup",
            "target_split": "train",
            "image_path": "p/1.png",
            "ground_truth": "t",
        }
        with pytest.raises(HunyuanDataAdapterError, match="duplicate"):
            HunyuanTrainingDataAdapter().export([row, dict(row)], hunyuan_export_config)

    def test_validate_uses_frozen_bridge_validator(
        self, canonical_rows, hunyuan_export_config: HunyuanExportConfig
    ) -> None:
        adapter = HunyuanTrainingDataAdapter()
        records = adapter.export(_clean_rows(canonical_rows), hunyuan_export_config)
        assert adapter.validate(records)["valid"] is True
        broken = [dict(records[0], image_path=[])]
        summary = adapter.validate(broken)
        assert summary["valid"] is False
        assert any("image_path" in e for e in summary["errors"])

    def test_validate_duplicate_image_paths(self, tmp_path: Path) -> None:
        from clouda_training.hunyuan.models import ARABIC_DOCUMENT_OCR_PROMPT

        config = HunyuanExportConfig(
            dataset_id="d",
            dataset_version="v1",
            manifest_path=str(tmp_path / "m.jsonl"),
            manifest_hash="0" * 64,
            split="train",
            image_root=str(tmp_path),
            prompt_profile=ARABIC_DOCUMENT_OCR_PROMPT,
        )
        row = {
            "sample_id": "r1",
            "target_split": "train",
            "image_path": "p/1.png",
            "ground_truth": "t",
        }
        records = HunyuanTrainingDataAdapter().export([row], config)
        duplicated = [dict(records[0]), dict(records[0])]
        summary = HunyuanTrainingDataAdapter().validate(duplicated)
        assert summary["valid"] is False
        assert any("duplicate image path" in e for e in summary["errors"])

    def test_describe_contract(self) -> None:
        contract = HunyuanTrainingDataAdapter().describe_contract()
        assert contract["data_contract_version"] == "1"
        assert contract["upstream_revision"] == "c55965d3da1e"
        assert contract["fields"]["image_path"]["type"] == "list[string]"

    def test_wrong_config_type_rejected(self, canonical_rows) -> None:
        with pytest.raises(HunyuanDataAdapterError, match="HunyuanExportConfig"):
            HunyuanTrainingDataAdapter().export(canonical_rows, object())


class TestCrossFamilyContracts:
    def test_contracts_are_json_serializable_and_distinct(self) -> None:
        qwen = QwenTrainingDataAdapter().describe_contract()
        hunyuan = HunyuanTrainingDataAdapter().describe_contract()
        for contract in (qwen, hunyuan):
            json.dumps(contract, ensure_ascii=False)  # must not raise
        assert qwen["upstream_repository"] != hunyuan["upstream_repository"]
        assert qwen["record_format"] != hunyuan["record_format"]
        # wire formats differ: qwen image is a string path, hunyuan a list
        assert qwen["fields"]["image"]["type"] == "string"
        assert hunyuan["fields"]["image_path"]["type"] == "list[string]"
