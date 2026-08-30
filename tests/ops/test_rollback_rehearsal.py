from pathlib import Path

from tools.ops.rollback_rehearsal import run_rehearsal


def test_rollback_rehearsal_uses_synthetic_isolated_root(tmp_path: Path):
    report = run_rehearsal(tmp_path / "rollback")

    assert report["status"] == "PASS"
    assert report["checks"]["writers_frozen"] is True
    assert report["checks"]["workers_stopped"] is True
    assert report["checks"]["queue_state_preserved"] is True
    assert report["checks"]["release_metadata_recorded"] is True
    assert report["checks"]["version_marker_rolled_back"] is True
    assert report["checks"]["backup_valid"] is True
    assert report["checks"]["isolated_restore"] is True
    assert report["restore"]["integrity_check"] == "ok"
    assert report["restore"]["conversion_rows"] == 1
    assert report["limits"]["production_touched"] is False
    assert report["limits"]["secrets_required"] is False
