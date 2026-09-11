"""Regression tests for independent-review findings (image-tag leakage)."""

from __future__ import annotations

from clouda_training.qwen.data_adapter import (
    QwenExportConfig,
    QwenTrainingDataAdapter,
)
from clouda_training.qwen.models import QWEN_IMAGE_PLACEHOLDER

_ADAPTER = QwenTrainingDataAdapter()
_CONFIG = QwenExportConfig(image_root="/abs/root")


def _validate(records):
    return _ADAPTER.validate(records)


def test_image_tag_in_gpt_turn_rejected() -> None:
    record = {
        "image": "/abs/root/p/1.png",
        "conversations": [
            {"from": "human", "value": f"{QWEN_IMAGE_PLACEHOLDER}\nprompt"},
            {"from": "gpt", "value": f"answer with {QWEN_IMAGE_PLACEHOLDER} inside"},
        ],
    }
    result = _validate([record])
    assert not result["valid"]
    assert any("gpt turn" in e for e in result["errors"])


def test_image_tag_in_later_human_turn_rejected() -> None:
    record = {
        "image": "/abs/root/p/1.png",
        "conversations": [
            {"from": "human", "value": f"{QWEN_IMAGE_PLACEHOLDER}\nprompt"},
            {"from": "gpt", "value": "answer"},
            {"from": "human", "value": f"follow up {QWEN_IMAGE_PLACEHOLDER}"},
            {"from": "gpt", "value": "answer2"},
        ],
    }
    result = _validate([record])
    assert not result["valid"]
    assert any("first human turn" in e for e in result["errors"])


def test_clean_multi_turn_record_still_valid() -> None:
    record = {
        "image": "/abs/root/p/1.png",
        "conversations": [
            {"from": "human", "value": f"{QWEN_IMAGE_PLACEHOLDER}\nprompt"},
            {"from": "gpt", "value": "answer <vision tokens never appear> ok"},
            {"from": "human", "value": "follow up without tag"},
            {"from": "gpt", "value": "answer2"},
        ],
    }
    result = _validate([record])
    assert result["valid"], result["errors"]
