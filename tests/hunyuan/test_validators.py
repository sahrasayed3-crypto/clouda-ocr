"""Raw + packed schema validator tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clouda_training.hunyuan.validators import (
    PackedSchemaError,
    RawSchemaError,
    validate_packed_jsonl,
    validate_raw_jsonl,
)


def _raw_sample(image="img.png", prompt="<image>\nاستخرج", gt="نص عربي"):
    return {
        "image_path": [image],
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": gt},
        ],
    }


def _write_jsonl(path: Path, rows: list) -> Path:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    return path


# ---------------- raw ----------------


def test_valid_raw_jsonl(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path / "raw.jsonl", [_raw_sample(), _raw_sample("b.png")])
    assert validate_raw_jsonl(p) == {"valid": True, "samples": 2}


def test_malformed_image_path_rejected(tmp_path: Path) -> None:
    bad = _raw_sample()
    bad["image_path"] = []
    p = _write_jsonl(tmp_path / "r.jsonl", [bad])
    with pytest.raises(RawSchemaError, match="image_path"):
        validate_raw_jsonl(p)


def test_wrong_roles_rejected(tmp_path: Path) -> None:
    bad = _raw_sample()
    bad["conversations"][0]["from"] = "system"
    p = _write_jsonl(tmp_path / "r.jsonl", [bad])
    with pytest.raises(RawSchemaError, match="role"):
        validate_raw_jsonl(p)


def test_missing_image_placeholder_rejected(tmp_path: Path) -> None:
    bad = _raw_sample(prompt="no placeholder here")
    p = _write_jsonl(tmp_path / "r.jsonl", [bad])
    with pytest.raises(RawSchemaError, match="<image>"):
        validate_raw_jsonl(p)


def test_empty_gt_rejected(tmp_path: Path) -> None:
    bad = _raw_sample(gt="   ")
    p = _write_jsonl(tmp_path / "r.jsonl", [bad])
    with pytest.raises(RawSchemaError, match="ground truth"):
        validate_raw_jsonl(p)


def test_invalid_utf8_reports_line(tmp_path: Path) -> None:
    p = tmp_path / "r.jsonl"
    p.write_bytes(b'{"image_path": ["\xff\xfe"], "conversations": []}\n')
    with pytest.raises(RawSchemaError):
        validate_raw_jsonl(p)


def test_image_existence_check_mode(tmp_path: Path) -> None:
    existing = tmp_path / "exists.png"
    existing.write_bytes(b"png")
    p = _write_jsonl(tmp_path / "r.jsonl", [_raw_sample(str(existing))])
    assert validate_raw_jsonl(p, check_image_exists=True)["valid"]
    missing = _write_jsonl(
        tmp_path / "m.jsonl", [_raw_sample(str(tmp_path / "no.png"))]
    )
    with pytest.raises(RawSchemaError, match="does not exist"):
        validate_raw_jsonl(missing, check_image_exists=True)


# ---------------- packed ----------------


def _packed(total_tokens=1000, cu=None, samples=None):
    return {
        "packed_samples": samples or [_raw_sample()],
        "cu_seqlens": cu or [0, total_tokens],
        "total_tokens": total_tokens,
    }


def test_valid_packed(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path / "p.jsonl", [_packed(), _packed(500, [0, 250, 500])])
    summary = validate_packed_jsonl(p)
    assert summary == {"valid": True, "packs": 2, "samples": 2}


def test_total_tokens_exceeds_pack_length(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path / "p.jsonl", [_packed(total_tokens=30000)])
    with pytest.raises(PackedSchemaError, match="exceeds"):
        validate_packed_jsonl(p, pack_length=20480)


def test_cu_seqlens_not_starting_at_zero(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path / "p.jsonl", [_packed(cu=[10, 1000])])
    with pytest.raises(PackedSchemaError, match="start at 0"):
        validate_packed_jsonl(p)


def test_non_monotonic_cu_seqlens(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path / "p.jsonl", [_packed(cu=[0, 500, 400, 1000])])
    with pytest.raises(PackedSchemaError, match="monotonic"):
        validate_packed_jsonl(p)


def test_final_boundary_mismatch(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path / "p.jsonl", [_packed(cu=[0, 900])])
    with pytest.raises(PackedSchemaError, match="does not match"):
        validate_packed_jsonl(p)


def test_malformed_embedded_raw_sample(tmp_path: Path) -> None:
    bad_inner = {"image_path": [], "conversations": []}
    p = _write_jsonl(
        tmp_path / "p.jsonl",
        [_packed(samples=[bad_inner])],
    )
    with pytest.raises(PackedSchemaError, match="image_path"):
        validate_packed_jsonl(p)
