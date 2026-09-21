import os
import shutil
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .database import Database


def cleanup_temporary_directories(
    storage_root: str | Path = "conversions",
    retention_hours: int = 24,
    database: Database | None = None,
) -> dict:
    root = Path(storage_root).resolve()
    if not root.exists():
        return {"deleted": 0, "bytes_freed": 0, "errors": []}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, retention_hours))

    # Cross-process liveness from shared DB — fail-safe on error
    try:
        db = database or Database()
        active_scopes = db.active_conversion_scope_ids()
    except Exception as exc:
        return {
            "deleted": 0,
            "bytes_freed": 0,
            "errors": [f"DB liveness lookup failed: {exc}"],
        }

    deleted = 0
    bytes_freed = 0
    errors: list[str] = []
    for temporary in root.glob("*/*/temporary"):
        try:
            resolved = temporary.resolve()
            if root not in resolved.parents:
                continue
            if temporary.parent.name in active_scopes:
                continue
            modified = datetime.fromtimestamp(
                temporary.stat().st_mtime, tz=timezone.utc
            )
            if modified > cutoff:
                continue
            size = sum(
                path.stat().st_size for path in temporary.rglob("*") if path.is_file()
            )
            shutil.rmtree(resolved)
            deleted += 1
            bytes_freed += size
        except (OSError, ValueError) as exc:
            errors.append(f"{temporary}: {exc}")
    return {"deleted": deleted, "bytes_freed": bytes_freed, "errors": errors}


def _storage_root_from_env() -> Path:
    """Resolve the cleanup root the same way the app does.

    Production deployments configure an absolute ``STORAGE_ROOT``
    (deploy/production.env.example); without this, the timer unit's
    cwd-relative default silently cleaned the wrong tree.
    """

    configured = os.getenv("STORAGE_ROOT", "").strip()
    if configured:
        return Path(configured)
    return Path("conversions")


if __name__ == "__main__":
    print(
        json.dumps(
            cleanup_temporary_directories(_storage_root_from_env()),
            ensure_ascii=False,
        )
    )
