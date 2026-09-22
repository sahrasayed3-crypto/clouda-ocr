"""Regression tests for job lifecycle wiring defects found in the deep
engineering session (docs/engineering/CODE_AUDIT_CURRENT.md):

- Retry must clear the stale rq_job_id and re-dispatch (previously the row
  stayed pending forever because maintenance skips rows with an rq_job_id).
- Cancel must attempt to cancel the RQ job (previously the worker kept
  running / the queued job later ran to completion).
- start must not resurrect a cancelled job (previously an RQ retry flipped
  cancelled -> pending -> processing against the user's cancel).
- The single-use guest result token must survive a concurrent download race
  (consume_guest_result_token now takes the write lock before its SELECT).
"""

from __future__ import annotations

import fakeredis
import pytest
from fastapi.testclient import TestClient
from rq import Queue

from pdfword.database import Database, utc_now
from pdfword.job_queue import DistributedJobQueue
from pdfword.maintenance import re_dispatch_deferred_conversions
from pdfword.worker_api import app

from tests.test_multi_user_auth import fake_token, login

API_KEY = "test-worker-secret"  # pragma: allowlist secret


@pytest.fixture
def client_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APP_ROLE", "server")
    monkeypatch.setenv("WORKER_API_KEY", API_KEY)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "clouda.sqlite3"))
    monkeypatch.setenv("CLOUDA_AUTH_VERIFIER", "fake")
    monkeypatch.setenv("CLOUDA_FIREBASE_PROJECT_ID", "local-test-project")
    monkeypatch.setenv("CLOUDA_SESSION_COOKIE_SECURE", "false")
    database = Database(tmp_path / "clouda.sqlite3")
    client = TestClient(app)
    profile, _csrf = login(client, fake_token("alice-id", "alice@example.com"))
    user_id = profile["user"]["user_id"]
    return client, database, user_id


def create_conversion(
    database: Database, job_id: str = "job-a", owner_user_id: str = ""
) -> dict:
    database.create_conversion(
        {
            "job_id": job_id,
            "username": "alice",
            "owner_user_id": owner_user_id,
            "original_pdf_name": "input.pdf",
            "stored_pdf_path": "storage/alice/input.pdf",
            "output_docx_name": "output.docx",
            "stored_docx_path": "storage/alice/output.docx",
            "page_from": 1,
            "page_to": 1,
            "page_numbers": "[1]",
            "status": "pending",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
    )
    row = database.get_conversion(job_id)
    assert row is not None
    return row


class RecordingQueue:
    """Stands in for the RQ adapter; records enqueue/cancel calls."""

    def __init__(self, *, fail_enqueue: bool = False):
        self.enqueued: list[str] = []
        self.cancelled: list[str] = []
        self.fail_enqueue = fail_enqueue

    def enqueue(self, job_id: str):
        if self.fail_enqueue:
            raise ConnectionError("redis unavailable")
        self.enqueued.append(job_id)

        class Job:
            id = f"rq-{job_id}"

        return Job()

    def cancel(self, job_id: str) -> bool:
        self.cancelled.append(job_id)
        return True


def test_retry_clears_stale_rq_job_id_and_redispatches(
    client_env, monkeypatch: pytest.MonkeyPatch
):
    _client, database, user_id = client_env
    row = create_conversion(database, owner_user_id=user_id)
    database.update_conversion(row["id"], {"rq_job_id": "stale-rq-job"})
    database.transition_conversion("job-a", "failed", error_message="boom")
    queue = RecordingQueue()
    monkeypatch.setattr("pdfword.worker_api.get_distributed_queue", lambda: queue)

    # retry is a user endpoint requiring CSRF/session; exercise the handler
    # directly with a context tuple shaped like _require_csrf's return value
    # (database, user, session_row, session_id).
    from pdfword.worker_api import retry_user_document  # noqa: PLC0415

    result = retry_user_document(
        "job-a", context=(database, {"user_id": user_id}, None, None)
    )

    assert result["status"] == "pending"
    refreshed = database.get_conversion("job-a")
    assert refreshed["status"] == "pending"
    assert refreshed["rq_job_id"] == "rq-job-a"
    assert queue.enqueued == ["job-a"]


def test_retry_survives_redis_outage_and_maintenance_redispatches(
    client_env, monkeypatch: pytest.MonkeyPatch
):
    _, database, user_id = client_env
    row = create_conversion(database, owner_user_id=user_id)
    database.update_conversion(row["id"], {"rq_job_id": "stale-rq-job"})
    database.transition_conversion("job-a", "failed", error_message="boom")

    from pdfword.worker_api import retry_user_document  # noqa: PLC0415

    retry_user_document(
        "job-a",
        context=(
            database,
            {"user_id": user_id},
            None,
            None,
        ),
    )
    row = database.get_conversion("job-a")
    assert row is not None
    assert row["status"] == "pending"
    # The stale RQ reference must be gone so maintenance can re-dispatch.
    assert not row["rq_job_id"]

    monkeypatch.setenv("LOCAL_PROCESSING_ENABLED", "false")
    queue = RecordingQueue()
    monkeypatch.setattr("pdfword.worker_api.get_distributed_queue", lambda: queue)
    dispatched = re_dispatch_deferred_conversions(database)
    assert dispatched == 1
    assert queue.enqueued == ["job-a"]


def test_cancel_attempts_to_stop_the_rq_job(
    client_env, monkeypatch: pytest.MonkeyPatch
):
    _, database, user_id = client_env
    create_conversion(database, owner_user_id=user_id)
    queue = RecordingQueue()
    monkeypatch.setattr("pdfword.worker_api.get_distributed_queue", lambda: queue)

    from pdfword.worker_api import cancel_user_document  # noqa: PLC0415

    result = cancel_user_document(
        "job-a", context=(database, {"user_id": user_id}, None, None)
    )

    assert result["status"] == "cancelled"
    assert queue.cancelled == ["job-a"]
    assert database.get_conversion("job-a")["status"] == "cancelled"


def test_cancel_survives_redis_outage(client_env, monkeypatch: pytest.MonkeyPatch):
    _, database, user_id = client_env
    create_conversion(database, owner_user_id=user_id)

    class BrokenQueue:
        def cancel(self, _job_id: str):
            raise ConnectionError("redis unavailable")

    monkeypatch.setattr(
        "pdfword.worker_api.get_distributed_queue", lambda: BrokenQueue()
    )

    from pdfword.worker_api import cancel_user_document  # noqa: PLC0415

    result = cancel_user_document(
        "job-a", context=(database, {"user_id": user_id}, None, None)
    )
    assert result["status"] == "cancelled"


def test_worker_start_cannot_resurrect_a_cancelled_job(client_env):
    client, database, user_id = client_env
    create_conversion(database, owner_user_id=user_id)
    database.transition_conversion("job-a", "cancelled")
    headers = {"X-Worker-API-Key": API_KEY}

    response = client.post(
        "/internal/jobs/job-a/start",
        headers=headers,
        json={"worker_name": "worker-1"},
    )

    assert response.status_code == 409
    assert database.get_conversion("job-a")["status"] == "cancelled"


def test_enqueue_dispatch_lock_prevents_duplicate_queue_entries(
    monkeypatch: pytest.MonkeyPatch,
):
    redis = fakeredis.FakeRedis()
    queue = Queue("clouda:pdf_conversion", connection=redis)
    backend = DistributedJobQueue()
    monkeypatch.setattr(backend, "_queue", lambda: queue)

    first = backend.enqueue("job-a")
    assert queue.count == 1

    # A live queued job is returned as-is; no second entry is pushed.
    again = backend.enqueue("job-a")
    assert again.get_status(refresh=True) == "queued"
    assert queue.count == 1
    assert queue.get_job_ids() == [first.id]


def test_enqueue_dispatch_lock_rejects_concurrent_dispatch(
    monkeypatch: pytest.MonkeyPatch,
):
    redis = fakeredis.FakeRedis()
    queue = Queue("clouda:pdf_conversion", connection=redis)
    backend = DistributedJobQueue()
    monkeypatch.setattr(backend, "_queue", lambda: queue)

    lock_key = "clouda:pdf_conversion:dispatch:job-a"
    assert redis.set(lock_key, "1", nx=True, ex=30)

    with pytest.raises(RuntimeError, match="dispatch already in progress"):
        backend.enqueue("job-a")
    assert queue.count == 0


def test_single_use_guest_result_token_is_concurrent_safe(tmp_path):
    database = Database(tmp_path / "db.sqlite3")
    with database.transaction() as connection:
        connection.execute("""
            INSERT INTO guest_sessions(
                guest_scope_id, session_id_hash, created_at, expires_at, last_seen_at
            ) VALUES ('scope-1', 'hash-1', '2026-01-01T00:00:00+00:00',
                      '2099-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')
            """)
    from concurrent.futures import ThreadPoolExecutor

    created = database.create_guest_job(
        {
            "job_id": "guest-1",
            "guest_scope_id": "scope-1",
            "original_pdf_name": "input.pdf",
            "stored_pdf_path": "storage/guest/input.pdf",
            "page_count": 1,
            "size_bytes": 100,
            "expires_at": "2099-01-01T00:00:00+00:00",
        }
    )
    token = created["result_token"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _: database.consume_guest_result_token(
                    "guest-1", "scope-1", token
                ),
                range(2),
            )
        )

    consumed = [r for r in results if r is not None]
    assert len(consumed) == 1
