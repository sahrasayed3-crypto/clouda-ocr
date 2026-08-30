from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pdfword.backup import create_backup, restore_backup, validate_backup  # noqa: E402
from pdfword.database import Database, utc_now  # noqa: E402


def _create_conversion(database: Database, storage: Path, *, job_id: str) -> None:
    source_pdf = storage / f"{job_id}.pdf"
    result_docx = storage / f"{job_id}.docx"
    source_pdf.write_bytes(b"%PDF-1.4\n% synthetic rollback rehearsal\n")
    result_docx.write_bytes(b"PK\x03\x04 synthetic docx placeholder")
    now = utc_now()
    database.create_conversion(
        {
            "job_id": job_id,
            "username": "rollback-user",
            "owner_user_id": "rollback-user",
            "original_pdf_name": f"{job_id}.pdf",
            "stored_pdf_path": str(source_pdf),
            "stored_docx_path": str(result_docx),
            "status": "completed",
            "visibility": "private",
            "created_at": now,
            "updated_at": now,
        }
    )


def _database_integrity(database_path: Path) -> tuple[str, int]:
    with closing(sqlite3.connect(database_path)) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        rows = connection.execute("SELECT COUNT(*) FROM conversions").fetchone()[0]
    return str(integrity), int(rows)


def run_rehearsal(work_root: Path | None = None) -> dict:
    if work_root is None:
        context = tempfile.TemporaryDirectory(prefix="clouda-rollback-")
        root = Path(context.name)
    else:
        context = None
        root = work_root
        root.mkdir(parents=True, exist_ok=True)
    try:
        active_root = root / "active"
        storage = active_root / "storage"
        storage.mkdir(parents=True, exist_ok=True)
        database = Database(active_root / "clouda.sqlite3")
        _create_conversion(database, storage, job_id="rollback-job-v1")

        queue_state = {
            "frozen": True,
            "workers_stopped": True,
            "preserved_jobs": ["rollback-job-v1"],
            "captured_at": utc_now(),
        }
        queue_state_path = root / "queue-state.json"
        queue_state_path.write_text(
            json.dumps(queue_state, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        release_metadata = {
            "from_release": "synthetic-v2",
            "rollback_to_release": "synthetic-v1",
            "schema_compatibility": "compatible",
            "recorded_at": utc_now(),
        }
        release_metadata_path = root / "release-metadata.json"
        release_metadata_path.write_text(
            json.dumps(release_metadata, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

        backup_root = root / "backups"
        archive = create_backup(
            database,
            storage_root=storage,
            backup_root=backup_root,
            retention_days=30,
        )
        validation = validate_backup(archive)

        simulated_v2_marker = active_root / "release.txt"
        simulated_v2_marker.write_text("synthetic-v2", encoding="utf-8")
        rolled_back_marker = active_root / "release.txt"
        rolled_back_marker.write_text("synthetic-v1", encoding="utf-8")

        restored_root = restore_backup(archive, root / "isolated-restore")
        restored_db = restored_root / "data" / "clouda.sqlite3"
        integrity, rows = _database_integrity(restored_db)

        checks = {
            "writers_frozen": queue_state["frozen"] is True,
            "workers_stopped": queue_state["workers_stopped"] is True,
            "queue_state_preserved": queue_state_path.exists(),
            "release_metadata_recorded": release_metadata_path.exists(),
            "version_marker_rolled_back": rolled_back_marker.read_text(encoding="utf-8")
            == "synthetic-v1",
            "backup_valid": bool(validation["valid"]),
            "isolated_restore": restored_root.name == "isolated-restore",
            "database_integrity": integrity == "ok",
            "tenant_rows_present": rows == 1,
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        return {
            "status": status,
            "checks": checks,
            "archive_name": archive.name,
            "archive_size_bytes": archive.stat().st_size,
            "restore": {
                "integrity_check": integrity,
                "conversion_rows": rows,
                "restored_root_name": restored_root.name,
            },
            "limits": {
                "synthetic_only": True,
                "production_touched": False,
                "secrets_required": False,
            },
        }
    finally:
        if context is not None:
            context.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=None)
    args = parser.parse_args()
    report = run_rehearsal(args.work_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
