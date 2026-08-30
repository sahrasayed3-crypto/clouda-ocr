from contextlib import closing
import shutil
from pathlib import Path

from .database import Database
from .job_queue import get_distributed_queue
from .settings import runtime_settings


def system_health(database: Database, storage_root: str | Path = "conversions") -> dict:
    checks: dict[str, dict] = {}
    try:
        with closing(database.connect()) as connection:
            connection.execute("SELECT 1").fetchone()
        checks["database"] = {"ok": True, "detail": str(database.path)}
    except Exception as exc:
        checks["database"] = {"ok": False, "detail": str(exc)}

    root = Path(storage_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        checks["storage"] = {"ok": True, "detail": str(root.resolve())}
    except Exception as exc:
        checks["storage"] = {"ok": False, "detail": str(exc)}

    try:
        usage = shutil.disk_usage(root)
        free_gb = usage.free / (1024**3)
        checks["disk"] = {"ok": free_gb >= 1.0, "detail": f"{free_gb:.2f} GB free"}
    except Exception as exc:
        checks["disk"] = {"ok": False, "detail": str(exc)}

    try:
        queue = get_distributed_queue()
        checks["redis"] = {
            "ok": queue.ping(),
            "detail": f"{queue.pending_count()} pending in {runtime_settings().rq_queue_name}",
        }
    except Exception as exc:
        checks["redis"] = {"ok": False, "detail": str(exc)}
    with closing(database.connect()) as connection:
        worker = connection.execute("""
            SELECT worker_name, last_heartbeat FROM conversions
            WHERE last_heartbeat IS NOT NULL ORDER BY last_heartbeat DESC LIMIT 1
            """).fetchone()
    checks["worker"] = {
        "ok": worker is not None,
        "detail": (
            "none"
            if worker is None
            else f"{worker['worker_name']}: {worker['last_heartbeat']}"
        ),
    }
    with closing(database.connect()) as connection:
        error = connection.execute("""
            SELECT error_message, updated_at FROM conversions
            WHERE error_message IS NOT NULL ORDER BY updated_at DESC LIMIT 1
            """).fetchone()
    checks["last_error"] = {
        "ok": error is None,
        "detail": (
            "none"
            if error is None
            else f"{error['updated_at']}: {error['error_message']}"
        ),
    }
    last_backup = database.last_backup()
    checks["last_backup"] = {
        "ok": bool(last_backup and last_backup.get("status") == "completed"),
        "detail": (
            "none"
            if not last_backup
            else f"{last_backup['created_at']}: {last_backup['status']}"
        ),
    }
    critical = [checks["database"]["ok"], checks["storage"]["ok"], checks["disk"]["ok"]]
    if not all(critical):
        status = "unhealthy"
    elif not all(item["ok"] for item in checks.values()):
        status = "degraded"
    else:
        status = "healthy"
    return {"status": status, "checks": checks}
