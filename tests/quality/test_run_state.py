"""Tests for clouda_data.quality.run_state (resume state, Agent L)."""

from __future__ import annotations

import json

import pytest

from clouda_data.quality.run_state import (
    CorruptStateError,
    QualityRunState,
    StaleResumeError,
    checkpoint,
    load_state,
    save_state,
    start_or_resume,
)


def make_state(**overrides: object) -> QualityRunState:
    base: dict[str, object] = {
        "manifest_sha256": "a" * 64,
        "row_count": 100,
        "config_identity": "clouda.quality.config.v1+deadbeef",
        "algorithm_versions": {"exact_hash": "v1", "image_fp": "v2"},
        "stage": "dedup",
        "stage_cursor": {"band_id": 3},
        "processed_count": 42,
        "run_id": "run-0001",
    }
    base.update(overrides)
    return QualityRunState(**base)  # type: ignore[arg-type]


def test_fresh_start_writes_state(tmp_path) -> None:
    expected = make_state()
    state, resumed = start_or_resume(tmp_path, expected)
    assert resumed is False
    assert state is expected
    assert (tmp_path / "run_state.json").exists()
    loaded = load_state(tmp_path)
    assert loaded == expected


def test_immediate_resume_matches(tmp_path) -> None:
    expected = make_state()
    start_or_resume(tmp_path, expected)
    loaded, resumed = start_or_resume(tmp_path, expected)
    assert resumed is True
    assert loaded == expected


def test_resume_preserves_checkpointed_progress(tmp_path) -> None:
    expected = make_state()
    start_or_resume(tmp_path, expected)
    checkpoint(tmp_path, "near_dedup", {"band_id": 7}, 90)
    loaded, resumed = start_or_resume(tmp_path, expected)
    assert resumed is True
    assert loaded.stage == "near_dedup"
    assert loaded.stage_cursor == {"band_id": 7}
    assert loaded.processed_count == 90


def test_changed_manifest_sha256_raises(tmp_path) -> None:
    start_or_resume(tmp_path, make_state())
    expected = make_state(manifest_sha256="b" * 64)
    with pytest.raises(StaleResumeError) as excinfo:
        start_or_resume(tmp_path, expected)
    message = str(excinfo.value)
    assert "b" * 64 in message
    assert "a" * 64 in message


def test_changed_config_identity_raises(tmp_path) -> None:
    start_or_resume(tmp_path, make_state())
    expected = make_state(config_identity="clouda.quality.config.v1+cafebabe")
    with pytest.raises(StaleResumeError) as excinfo:
        start_or_resume(tmp_path, expected)
    assert "cafebabe" in str(excinfo.value)


def test_changed_algorithm_versions_raises(tmp_path) -> None:
    start_or_resume(tmp_path, make_state())
    expected = make_state(algorithm_versions={"exact_hash": "v1", "image_fp": "v3"})
    with pytest.raises(StaleResumeError) as excinfo:
        start_or_resume(tmp_path, expected)
    assert "algorithm_versions" in str(excinfo.value)


def test_mismatch_message_lists_each_field(tmp_path) -> None:
    start_or_resume(tmp_path, make_state())
    expected = make_state(
        manifest_sha256="c" * 64,
        config_identity="other-identity",
    )
    with pytest.raises(StaleResumeError) as excinfo:
        start_or_resume(tmp_path, expected)
    message = str(excinfo.value)
    assert "manifest_sha256" in message
    assert "config_identity" in message
    assert "expected=" in message
    assert "found=" in message


def test_checkpoint_updates_progress_preserves_identity(tmp_path) -> None:
    original = make_state()
    save_state(original, tmp_path)
    updated = checkpoint(tmp_path, "clustering", {"sample_pk": 9}, 55)
    assert updated.manifest_sha256 == original.manifest_sha256
    assert updated.config_identity == original.config_identity
    assert updated.algorithm_versions == original.algorithm_versions
    assert updated.run_id == original.run_id
    assert updated.stage == "clustering"
    assert updated.stage_cursor == {"sample_pk": 9}
    assert updated.processed_count == 55
    loaded = load_state(tmp_path)
    assert loaded == updated


def test_checkpoint_without_state_raises(tmp_path) -> None:
    with pytest.raises(CorruptStateError):
        checkpoint(tmp_path, "dedup", {}, 0)


def test_load_state_missing_returns_none(tmp_path) -> None:
    assert load_state(tmp_path) is None


def test_corrupt_json_raises(tmp_path) -> None:
    (tmp_path / "run_state.json").write_text("{not json!!", encoding="utf-8")
    with pytest.raises(CorruptStateError):
        load_state(tmp_path)


def test_corrupt_unknown_field_raises(tmp_path) -> None:
    payload = make_state().to_dict()
    payload["bogus"] = 1
    (tmp_path / "run_state.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CorruptStateError):
        load_state(tmp_path)


def test_atomic_write_leaves_no_tmp_files(tmp_path) -> None:
    save_state(make_state(), tmp_path)
    checkpoint(tmp_path, "dedup", {}, 1)
    leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run_state.json"]


def test_updated_at_is_utc_isoformat() -> None:
    state = make_state()
    assert state.updated_at.endswith("+00:00")
