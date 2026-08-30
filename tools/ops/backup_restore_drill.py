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


def run_drill(work_root: Path | None = None) -> dict:
    if work_root is None:
        context = tempfile.TemporaryDirectory(prefix="clouda-drill-")
        root = Path(context.name)
    else:
        context = None
        root = work_root
        root.mkdir(parents=True, exist_ok=True)
    try:
        storage = root / "storage"
        storage.mkdir(parents=True, exist_ok=True)
        (storage / "sample.txt").write_text(
            "synthetic backup payload", encoding="utf-8"
        )
        database = Database(root / "source.sqlite3")
        database.create_conversion(
            {
                "job_id": "synthetic-job",
                "username": "synthetic-user",
                "original_pdf_name": "synthetic.pdf",
                "stored_pdf_path": str(storage / "synthetic.pdf"),
                "stored_docx_path": str(storage / "synthetic.docx"),
                "status": "completed",
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
        )
        backup_root = root / "backups"
        archive = create_backup(
            database,
            storage_root=storage,
            backup_root=backup_root,
            retention_days=30,
        )
        validation = validate_backup(archive)
        restored_root = restore_backup(archive, root / "restored")
        restored_db = restored_root / "data" / "clouda.sqlite3"
        with closing(sqlite3.connect(restored_db)) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            rows = connection.execute("SELECT COUNT(*) FROM conversions").fetchone()[0]
        return {
            "status": "PASS" if validation["valid"] and integrity == "ok" else "FAIL",
            "archive_name": archive.name,
            "archive_size_bytes": archive.stat().st_size,
            "validation": {
                "valid": bool(validation["valid"]),
                "contains_database": bool(validation["contains_database"]),
                "members": int(validation["members"]),
            },
            "restore": {
                "integrity_check": integrity,
                "conversion_rows": rows,
                "restored_root_name": restored_root.name,
            },
        }
    finally:
        if context is not None:
            context.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=None)
    args = parser.parse_args()
    report = run_drill(args.work_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
