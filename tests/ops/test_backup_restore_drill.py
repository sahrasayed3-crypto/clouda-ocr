from pathlib import Path

from tools.ops.backup_restore_drill import run_drill


def test_backup_restore_drill_uses_synthetic_isolated_root(tmp_path: Path):
    report = run_drill(tmp_path / "drill")

    assert report["status"] == "PASS"
    assert report["validation"]["valid"] is True
    assert report["validation"]["contains_database"] is True
    assert report["restore"]["integrity_check"] == "ok"
    assert report["restore"]["conversion_rows"] == 1
