"""Server maintenance-cycle tests: deferred dispatch + guest retention."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import fakeredis
import pytest
from rq import Queue

from pdfword import worker_api
from pdfword.database import Database, utc_now
from pdfword.job_queue import DistributedJobQueue
from pdfword.maintenance import (
    re_dispatch_deferred_conversions,
    run_maintenance_cycle,
)


@pytest.fixture()
def database(tmp_path: Path) -> Database:
    return Database(tmp_path / "maintenance.sqlite3")


def _fake_queue(monkeypatch: pytest.MonkeyPatch) -> Queue:
    queue = Queue("clouda:pdf_conversion", connection=fakeredis.FakeRedis())
    backend = DistributedJobQueue()
    monkeypatch.setattr(backend, "_queue", lambda: queue)
    monkeypatch.setattr(worker_api, "get_distributed_queue", lambda: backend)
    return queue


def _create_pending(database: Database, tmp_path: Path) -> dict:
    job_root = tmp_path / "storage" / "alice" / "job-m"
    job_root.mkdir(parents=True)
    pdf = job_root / "input.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    database.create_conversion(
        {
            "job_id": "job-m",
            "username": "alice",
            "original_pdf_name": "input.pdf",
            "stored_pdf_path": str(pdf),
            "output_docx_name": "output.docx",
            "stored_docx_path": str(job_root / "output.docx"),
            "page_from": 1,
            "page_to": 1,
            "page_numbers": "[1]",
            "status": "pending",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
    )
    return database.get_conversion("job-m")


def test_deferred_pending_job_is_redispatched(database, tmp_path, monkeypatch):
    _fake_queue(monkeypatch)
    row = _create_pending(database, tmp_path)
    assert row["rq_job_id"] in ("", None)

    dispatched = re_dispatch_deferred_conversions(database)

    assert dispatched == 1
    assert database.get_conversion("job-m")["rq_job_id"]


def test_already_queued_pending_job_is_not_redispatched(
    database, tmp_path, monkeypatch
):
    queue = _fake_queue(monkeypatch)
    _create_pending(database, tmp_path)
    database.update_conversion(
        database.get_conversion("job-m")["id"], {"rq_job_id": "rq-existing"}
    )

    dispatched = re_dispatch_deferred_conversions(database)

    assert dispatched == 0
    assert len(queue.jobs) == 0


def test_maintenance_cycle_expires_guest_jobs_and_files(database, tmp_path):
    storage_root = tmp_path / "storage"
    stored_pdf = storage_root / "guest" / "upload.pdf"
    stored_pdf.parent.mkdir(parents=True)
    stored_pdf.write_bytes(b"%PDF-1.4 stale")
    session = database.create_guest_session(lifetime_seconds=3600)
    expires = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    database.create_guest_job(
        {
            "guest_scope_id": session["guest_scope_id"],
            "original_pdf_name": "upload.pdf",
            "stored_pdf_path": str(stored_pdf),
            "stored_docx_path": "",
            "page_count": 1,
            "size_bytes": 13,
            "expires_at": expires,
        }
    )

    report = run_maintenance_cycle(database, storage_root=storage_root)

    assert report["guest_retention"]["jobs_expired"] >= 1
    assert report["guest_retention"]["files_removed"] >= 1
    assert not stored_pdf.exists()


def test_maintenance_cycle_reports_dispatch_deferred_when_queue_unavailable(
    database, tmp_path, monkeypatch
):
    def broken_queue():
        raise RuntimeError("redis down")

    monkeypatch.setattr(worker_api, "get_distributed_queue", broken_queue)
    _create_pending(database, tmp_path)

    report = run_maintenance_cycle(database, storage_root=tmp_path / "storage")

    assert report["re_dispatched"] == 0
    assert database.get_conversion("job-m")["status"] == "pending"
