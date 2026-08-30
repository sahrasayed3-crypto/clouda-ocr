from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone

from .database import Database


def _contained_file(path_value: str, storage_root: Path) -> Path | None:
    if not path_value:
        return None
    root = storage_root.resolve()
    path = Path(path_value)
    resolved = (
        (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()
    )
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def cleanup_expired_guest_jobs(
    database: Database, *, storage_root: Path, now_offset_seconds: int = 0
) -> dict:
    threshold = (
        datetime.now(timezone.utc) + timedelta(seconds=now_offset_seconds)
    ).isoformat()
    with database.connect() as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT job_id, stored_pdf_path, stored_docx_path
                FROM guest_jobs
                WHERE deleted_at='' AND claimed_at=''
                  AND expires_at <= ?
                """,
                (threshold,),
            ).fetchall()
        ]
    files_removed = 0
    for row in rows:
        for key in ("stored_pdf_path", "stored_docx_path"):
            path = _contained_file(str(row.get(key) or ""), storage_root)
            if path and path.is_file():
                path.unlink()
                files_removed += 1
    jobs_expired = database.expire_guest_jobs(now_offset_seconds=now_offset_seconds)
    return {"jobs_expired": jobs_expired, "files_removed": files_removed}
