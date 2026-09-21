"""Periodic server maintenance.

Two production gaps closed here:

- A conversion whose dispatch was deferred (Redis unavailable at upload
  time) stayed ``pending`` forever: nothing re-attempted the enqueue and
  ``Database.list_pending_conversions`` had no caller.
- Expired guest jobs were never expired/purged by any production path:
  ``cleanup_expired_guest_jobs`` existed but nothing invoked it, so
  expired guest PDFs/DOCX files accumulated without bound.

``run_maintenance_cycle`` executes one pass of both; the worker API runs
it on a background thread (see ``worker_api._maintenance_loop``).
"""

from __future__ import annotations

from pathlib import Path

from .database import Database
from .guest_trial import cleanup_expired_guest_jobs


def re_dispatch_deferred_conversions(database: Database) -> int:
    """Re-enqueue pending conversions that were never queued.

    A row still pending with an empty ``rq_job_id`` means its dispatch was
    deferred. Rows that already carry an ``rq_job_id`` are left alone: the
    queue owns them and re-enqueueing would double-run.
    """

    from .settings import runtime_settings

    if runtime_settings().local_processing_enabled:
        return 0
    from .worker_api import _dispatch_conversion_job

    dispatched = 0
    for row in database.list_pending_conversions():
        if row.get("rq_job_id"):
            continue
        status = _dispatch_conversion_job(database, row["job_id"], actor="maintenance")
        if status == "queued":
            dispatched += 1
    return dispatched


def run_maintenance_cycle(database: Database, *, storage_root: Path | str) -> dict:
    """One maintenance pass: deferred dispatch + expired-guest cleanup."""

    return {
        "re_dispatched": re_dispatch_deferred_conversions(database),
        "guest_retention": cleanup_expired_guest_jobs(
            database, storage_root=Path(storage_root)
        ),
    }
