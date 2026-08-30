import ast
import io
import json
import logging
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fakeredis
import pytest
import requests
from fastapi.testclient import TestClient
from rq import Queue

from pdfword.database import Database, utc_now
from pdfword.job_queue import DistributedJobQueue
from pdfword.worker_client import WorkerApiClient
from pdfword.worker_api import app, _dispatch_conversion_job
from pdfword import worker_tasks

API_KEY = "test-worker-secret"  # pragma: allowlist secret


def valid_docx_bytes() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',
        )
        archive.writestr(
            "word/document.xml",
            '<document xmlns="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>',
        )
    return output.getvalue()


def create_job(database: Database, storage: Path, job_id: str = "job-a") -> dict:
    job_root = storage / "alice" / job_id
    job_root.mkdir(parents=True)
    pdf = job_root / "input.pdf"
    pdf.write_bytes(b"%PDF-1.4 test")
    docx = job_root / "output.docx"
    database.create_conversion(
        {
            "job_id": job_id,
            "username": "alice",
            "original_pdf_name": "input.pdf",
            "stored_pdf_path": str(pdf),
            "output_docx_name": "output.docx",
            "stored_docx_path": str(docx),
            "page_from": 1,
            "page_to": 1,
            "page_numbers": "[1]",
            "status": "pending",
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
    )
    result = database.get_conversion(job_id)
    assert result is not None
    return result


@pytest.fixture
def api_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    storage = tmp_path / "storage"
    database_path = tmp_path / "clouda.sqlite3"
    monkeypatch.setenv("APP_ROLE", "server")
    monkeypatch.setenv("WORKER_API_KEY", API_KEY)
    monkeypatch.setenv("STORAGE_ROOT", str(storage))
    monkeypatch.setenv("DATABASE_PATH", str(database_path))
    database = Database(database_path)
    return TestClient(app), database, storage


def test_rq_payload_contains_only_job_id(monkeypatch: pytest.MonkeyPatch):
    redis = fakeredis.FakeRedis()
    queue = Queue("clouda:pdf_conversion", connection=redis)
    backend = DistributedJobQueue()
    monkeypatch.setattr(backend, "_queue", lambda: queue)

    job = backend.enqueue("job-only-id")

    assert job.args == ("job-only-id",)
    assert "clouda" not in json.dumps(job.kwargs).lower()


def test_distributed_queue_uses_redis_namespace(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLOUDA_REDIS_NAMESPACE", "clouda-prod")
    monkeypatch.setenv("RQ_QUEUE_NAME", "conversion")

    backend = DistributedJobQueue()

    assert backend.queue_name == "clouda-prod:conversion"


def test_distributed_dispatch_updates_rq_job_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("LOCAL_PROCESSING_ENABLED", "false")
    database = Database(tmp_path / "db.sqlite3")
    storage = tmp_path / "storage"
    create_job(database, storage)

    class FakeQueue:
        def enqueue(self, job_id: str):
            assert job_id == "job-a"

            class Job:
                id = "rq-job-a"

            return Job()

    monkeypatch.setattr("pdfword.worker_api.get_distributed_queue", lambda: FakeQueue())

    status = _dispatch_conversion_job(database, "job-a", actor="test-user")

    assert status == "queued"
    row = database.get_conversion("job-a")
    assert row is not None
    assert row["rq_job_id"] == "rq-job-a"


def test_distributed_dispatch_deferred_keeps_pending_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("LOCAL_PROCESSING_ENABLED", "false")
    database = Database(tmp_path / "db.sqlite3")
    storage = tmp_path / "storage"
    create_job(database, storage)

    class FailingQueue:
        def enqueue(self, _job_id: str):
            raise ConnectionError("redis unavailable")

    monkeypatch.setattr(
        "pdfword.worker_api.get_distributed_queue", lambda: FailingQueue()
    )

    status = _dispatch_conversion_job(database, "job-a", actor="test-user")

    row = database.get_conversion("job-a")
    assert row is not None
    assert status == "dispatch_deferred"
    assert row["status"] == "pending"
    assert row["rq_job_id"] in {"", None}


def test_local_processing_dispatch_fails_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("LOCAL_PROCESSING_ENABLED", "true")
    database = Database(tmp_path / "db.sqlite3")
    storage = tmp_path / "storage"
    create_job(database, storage)

    with pytest.raises(RuntimeError, match="not a supported dispatch mode"):
        _dispatch_conversion_job(database, "job-a", actor="test-user")


def test_distributed_dispatch_reaches_completed_status(
    api_environment, monkeypatch: pytest.MonkeyPatch
):
    client, database, storage = api_environment
    monkeypatch.setenv("LOCAL_PROCESSING_ENABLED", "false")
    create_job(database, storage)

    class FakeQueue:
        def enqueue(self, job_id: str):
            assert job_id == "job-a"

            class Job:
                id = "rq-job-a"

            return Job()

    monkeypatch.setattr("pdfword.worker_api.get_distributed_queue", lambda: FakeQueue())

    assert _dispatch_conversion_job(database, "job-a", actor="test-user") == "queued"

    class FakeWorkerApiClient:
        headers = {"X-Worker-API-Key": API_KEY}

        def start(self, job_id: str, worker_name: str):
            response = client.post(
                f"/internal/jobs/{job_id}/start",
                headers=self.headers,
                json={"worker_name": worker_name},
            )
            assert response.status_code == 200
            return response.json()

        def download_input(self, job_id: str, target: Path):
            response = client.get(
                f"/internal/jobs/{job_id}/input", headers=self.headers
            )
            assert response.status_code == 200
            target.write_bytes(response.content)

        def upload_result(self, job_id: str, worker_name: str, docx, metadata: dict):
            response = client.post(
                f"/internal/jobs/{job_id}/result",
                headers=self.headers,
                data={
                    "worker_name": worker_name,
                    "claim_token": database.get_conversion(job_id)["claim_token"],
                    "metadata": json.dumps(metadata),
                },
                files={
                    "result": (
                        "result.docx",
                        docx,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
            )
            assert response.status_code == 200
            return response.json()

        def fail(self, *_args):
            pytest.fail("The documented distributed flow must not report a failure")

    def fake_conversion(request):
        request.docx_path.write_bytes(valid_docx_bytes())
        return {"status": "completed", "text_quality_score": 99.0}

    monkeypatch.setenv("APP_ROLE", "worker")
    monkeypatch.setenv("TEMP_ROOT", str(storage / "temporary"))
    monkeypatch.setattr(worker_tasks, "WorkerApiClient", FakeWorkerApiClient)
    import pdfword.conversion_service as conversion_service

    monkeypatch.setattr(
        conversion_service, "execute_worker_conversion", fake_conversion
    )
    assert worker_tasks.run_remote_job("job-a")["status"] == "completed"
    assert database.get_conversion("job-a")["status"] == "completed"


def test_redis_failure_does_not_remove_pending_database_job(tmp_path: Path):
    database = Database(tmp_path / "db.sqlite3")
    storage = tmp_path / "storage"
    create_job(database, storage)

    conversion = database.get_conversion("job-a")
    assert conversion is not None
    assert conversion["status"] == "pending"
    assert (storage / "alice" / "job-a" / "input.pdf").is_file()


def test_api_requires_worker_key(api_environment):
    client, database, storage = api_environment
    create_job(database, storage)

    assert client.get("/internal/jobs/job-a").status_code == 401
    assert (
        client.get(
            "/internal/jobs/job-a", headers={"X-Worker-API-Key": "wrong"}
        ).status_code
        == 401
    )
    assert (
        client.get(
            "/internal/jobs/job-a", headers={"X-Worker-API-Key": API_KEY}
        ).status_code
        == 200
    )


def test_public_health_reports_local_processing_without_worker_key(
    api_environment, monkeypatch: pytest.MonkeyPatch
):
    client, _database, _storage = api_environment
    monkeypatch.setenv("LOCAL_PROCESSING_ENABLED", "true")

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "role": "server",
        "local_processing_enabled": True,
    }
    assert API_KEY not in response.text


def test_worker_cloud_status_is_reported_without_exposing_key(
    api_environment, monkeypatch: pytest.MonkeyPatch
):
    client, _database, _storage = api_environment
    redis = fakeredis.FakeRedis()
    monkeypatch.setattr("redis.Redis.from_url", lambda *_args, **_kwargs: redis)
    headers = {"X-Worker-API-Key": API_KEY}

    reported = client.post(
        "/internal/workers/status",
        headers=headers,
        json={
            "worker_name": "windows-worker-1",
            "cloud_available": True,
            "cloud_provider": "openrouter",
        },
    )
    assert reported.status_code == 200

    health = client.get("/internal/health", headers=headers)
    assert health.status_code == 200
    payload = health.json()
    assert payload["cloud_available"] is True
    assert payload["workers"][0]["cloud_provider"] == "openrouter"
    assert API_KEY not in health.text


def test_worker_status_rejects_unsafe_worker_identifiers(api_environment):
    client, _database, _storage = api_environment
    headers = {"X-Worker-API-Key": API_KEY}

    newline_name = client.post(
        "/internal/workers/status",
        headers=headers,
        json={
            "worker_name": "worker-1\nspoofed",
            "cloud_available": True,
            "cloud_provider": "openrouter",
        },
    )
    assert newline_name.status_code == 422

    unsafe_provider = client.post(
        "/internal/workers/status",
        headers=headers,
        json={
            "worker_name": "worker-1",
            "cloud_available": True,
            "cloud_provider": "provider with spaces",
        },
    )
    assert unsafe_provider.status_code == 422

    unknown_provider = client.post(
        "/internal/workers/status",
        headers=headers,
        json={
            "worker_name": "worker-1",
            "cloud_available": True,
            "cloud_provider": "unknown-provider",
        },
    )
    assert unknown_provider.status_code == 422


def test_worker_status_get_reports_registered_workers(
    api_environment, monkeypatch: pytest.MonkeyPatch
):
    client, _database, _storage = api_environment
    redis = fakeredis.FakeRedis()
    monkeypatch.setenv("CLOUDA_REDIS_NAMESPACE", "clouda-test")
    monkeypatch.setattr("redis.Redis.from_url", lambda *_args, **_kwargs: redis)
    headers = {"X-Worker-API-Key": API_KEY}

    client.post(
        "/internal/workers/status",
        headers=headers,
        json={
            "worker_name": "windows-worker-1",
            "cloud_available": False,
            "cloud_provider": "",
        },
    )
    response = client.get("/internal/workers/status", headers=headers)

    assert response.status_code == 200
    assert redis.get("clouda-test:worker-status:windows-worker-1")
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["redis_available"] is True
    assert payload["workers"][0]["worker_name"] == "windows-worker-1"


@pytest.mark.parametrize("error_type", [ConnectionError, TimeoutError])
def test_worker_status_degrades_when_redis_is_unavailable(
    api_environment, monkeypatch: pytest.MonkeyPatch, error_type
):
    client, _database, _storage = api_environment

    class BrokenRedis:
        def ping(self):
            raise error_type("redis unavailable")

        def set(self, *_args, **_kwargs):
            raise error_type("redis unavailable")

    monkeypatch.setattr("redis.Redis.from_url", lambda *_args, **_kwargs: BrokenRedis())
    headers = {"X-Worker-API-Key": API_KEY}

    get_response = client.get("/internal/workers/status", headers=headers)
    post_response = client.post(
        "/internal/workers/status",
        headers=headers,
        json={
            "worker_name": "windows-worker-1",
            "cloud_available": True,
            "cloud_provider": "openrouter",
        },
    )

    for response in (get_response, post_response):
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "degraded"
        assert payload["redis_available"] is False
        assert payload["workers"] == []
        assert "local conversion remains available" in payload["message"]


def test_worker_api_full_status_flow(api_environment):
    client, database, storage = api_environment
    create_job(database, storage)
    headers = {"X-Worker-API-Key": API_KEY}

    started = client.post(
        "/internal/jobs/job-a/start", headers=headers, json={"worker_name": "worker-1"}
    )
    assert started.status_code == 200
    claim_token = started.json()["claim_token"]
    assert database.get_conversion("job-a")["status"] == "processing"
    assert database.get_conversion("job-a")["attempt_count"] == 1

    heartbeat = client.post(
        "/internal/jobs/job-a/heartbeat",
        headers=headers,
        json={"worker_name": "worker-1", "claim_token": claim_token},
    )
    assert heartbeat.status_code == 200

    metadata = {
        "status": "manual_review",
        "file_type": "scan",
        "text_quality_score": 70,
        "layout_quality_score": 80,
        "final_quality_score": 75,
        "winning_engine": "pending:future_ocr_engine",
        "processing_time": 2.5,
        "cloud_attempts": [
            {
                "provider": "openrouter",
                "model": "openai/gpt-5-mini",
                "latency_ms": 1250,
                "prompt_tokens": 120,
                "completion_tokens": 40,
                "cost": 0.002,
                "cost_is_estimated": 0,
                "score": 70,
                "failure_reason": "quality below threshold",
            }
        ],
        "cost_limit_reached": True,
    }
    uploaded = client.post(
        "/internal/jobs/job-a/result",
        headers=headers,
        data={
            "worker_name": "worker-1",
            "claim_token": claim_token,
            "metadata": json.dumps(metadata),
        },
        files={
            "result": (
                "result.docx",
                io.BytesIO(valid_docx_bytes()),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert uploaded.status_code == 200
    row = database.get_conversion("job-a")
    assert row["status"] == "manual_review"
    assert "cost limit" in row["error_message"].lower()
    attempts = database.list_attempts(row["id"])
    assert attempts[-1]["model_name"] == "openai/gpt-5-mini"
    assert attempts[-1]["prompt_tokens"] == 120
    assert attempts[-1]["completion_tokens"] == 40
    assert attempts[-1]["cost"] == pytest.approx(0.002)
    assert attempts[-1]["quality_score"] == 70
    assert attempts[-1]["failure_reason"] == "quality below threshold"
    assert Path(row["stored_docx_path"]).read_bytes().startswith(b"PK")


def test_completed_worker_result_without_accepted_quality_becomes_manual_review(
    api_environment,
):
    client, database, storage = api_environment
    create_job(database, storage)
    headers = {"X-Worker-API-Key": API_KEY}
    started = client.post(
        "/internal/jobs/job-a/start", headers=headers, json={"worker_name": "worker-1"}
    )
    claim_token = started.json()["claim_token"]

    uploaded = client.post(
        "/internal/jobs/job-a/result",
        headers=headers,
        data={
            "worker_name": "worker-1",
            "claim_token": claim_token,
            "metadata": json.dumps({"status": "completed", "text_quality_score": None}),
        },
        files={"result": ("result.docx", io.BytesIO(valid_docx_bytes()))},
    )

    assert uploaded.status_code == 200
    row = database.get_conversion("job-a")
    assert row["status"] == "manual_review"
    assert "quality" in row["error_message"].lower()


def test_worker_claim_token_fences_stale_dispatcher(api_environment):
    client, database, storage = api_environment
    create_job(database, storage)
    headers = {"X-Worker-API-Key": API_KEY}
    first = client.post(
        "/internal/jobs/job-a/start", headers=headers, json={"worker_name": "worker-1"}
    )
    first_token = first.json()["claim_token"]
    database.transition_conversion("job-a", "pending")
    second = client.post(
        "/internal/jobs/job-a/start", headers=headers, json={"worker_name": "worker-1"}
    )
    second_token = second.json()["claim_token"]
    assert second_token != first_token

    stale = client.post(
        "/internal/jobs/job-a/result",
        headers=headers,
        data={
            "worker_name": "worker-1",
            "claim_token": first_token,
            "metadata": json.dumps({"status": "completed", "text_quality_score": 99.0}),
        },
        files={"result": ("result.docx", io.BytesIO(valid_docx_bytes()))},
    )
    assert stale.status_code == 409

    fresh = client.post(
        "/internal/jobs/job-a/result",
        headers=headers,
        data={
            "worker_name": "worker-1",
            "claim_token": second_token,
            "metadata": json.dumps({"status": "completed", "text_quality_score": 99.0}),
        },
        files={"result": ("result.docx", io.BytesIO(valid_docx_bytes()))},
    )
    assert fresh.status_code == 200
    assert database.get_conversion("job-a")["status"] == "completed"


def test_stale_processing_job_can_be_reclaimed_by_new_worker(
    api_environment, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("CLOUDA_PROCESSING_STALE_SECONDS", "1")
    client, database, storage = api_environment
    create_job(database, storage)
    headers = {"X-Worker-API-Key": API_KEY}

    first = client.post(
        "/internal/jobs/job-a/start", headers=headers, json={"worker_name": "worker-1"}
    )
    assert first.status_code == 200
    first_token = first.json()["claim_token"]
    row = database.get_conversion("job-a")
    database.update_conversion(
        row["id"],
        {
            "updated_at": "2000-01-01T00:00:00+00:00",
            "last_heartbeat": "2000-01-01T00:00:00+00:00",
        },
    )

    second = client.post(
        "/internal/jobs/job-a/start", headers=headers, json={"worker_name": "worker-2"}
    )

    assert second.status_code == 200, second.text
    second_token = second.json()["claim_token"]
    assert second_token != first_token
    reclaimed = database.get_conversion("job-a")
    assert reclaimed["status"] == "processing"
    assert reclaimed["worker_name"] == "worker-2"
    assert reclaimed["attempt_count"] == 2

    stale_result = client.post(
        "/internal/jobs/job-a/result",
        headers=headers,
        data={
            "worker_name": "worker-1",
            "claim_token": first_token,
            "metadata": json.dumps({"status": "completed", "text_quality_score": 99.0}),
        },
        files={"result": ("result.docx", io.BytesIO(valid_docx_bytes()))},
    )
    assert stale_result.status_code == 409

    fresh_result = client.post(
        "/internal/jobs/job-a/result",
        headers=headers,
        data={
            "worker_name": "worker-2",
            "claim_token": second_token,
            "metadata": json.dumps({"status": "completed", "text_quality_score": 99.0}),
        },
        files={"result": ("result.docx", io.BytesIO(valid_docx_bytes()))},
    )
    assert fresh_result.status_code == 200
    assert database.get_conversion("job-a")["status"] == "completed"


def test_duplicate_worker_result_upload_cannot_overwrite_winner(api_environment):
    client, database, storage = api_environment
    create_job(database, storage)
    headers = {"X-Worker-API-Key": API_KEY}
    started = client.post(
        "/internal/jobs/job-a/start", headers=headers, json={"worker_name": "worker-1"}
    )
    claim_token = started.json()["claim_token"]

    def upload(name: str, marker: str):
        return client.post(
            "/internal/jobs/job-a/result",
            headers=headers,
            data={
                "worker_name": "worker-1",
                "claim_token": claim_token,
                "metadata": json.dumps(
                    {"status": "completed", "text_quality_score": 99.0}
                ),
            },
            files={"result": (name, io.BytesIO(valid_docx_bytes() + marker.encode()))},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(
            executor.map(
                lambda args: upload(*args),
                [("first.docx", "WINNER_A"), ("second.docx", "WINNER_B")],
            )
        )

    statuses = sorted(response.status_code for response in responses)
    assert statuses == [200, 409]
    final_bytes = Path(
        database.get_conversion("job-a")["stored_docx_path"]
    ).read_bytes()
    assert sum(marker in final_bytes for marker in (b"WINNER_A", b"WINNER_B")) == 1


def test_worker_result_promotion_failure_is_recoverable_and_cleans_temp(
    api_environment, monkeypatch: pytest.MonkeyPatch
):
    client, database, storage = api_environment
    client = TestClient(app, raise_server_exceptions=False)
    create_job(database, storage)
    headers = {"X-Worker-API-Key": API_KEY}
    started = client.post(
        "/internal/jobs/job-a/start", headers=headers, json={"worker_name": "worker-1"}
    )
    claim_token = started.json()["claim_token"]
    target = Path(database.get_conversion("job-a")["stored_docx_path"])
    calls = 0

    def fail_once(_temporary_path, _target):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated promotion failure")
        return original_replace(_temporary_path, _target)

    original_replace = os.replace
    monkeypatch.setattr("pdfword.worker_api.os.replace", fail_once)

    failed = client.post(
        "/internal/jobs/job-a/result",
        headers=headers,
        data={
            "worker_name": "worker-1",
            "claim_token": claim_token,
            "metadata": json.dumps({"status": "completed", "text_quality_score": 99}),
        },
        files={"result": ("result.docx", io.BytesIO(valid_docx_bytes() + b"FIRST"))},
    )

    assert failed.status_code == 503
    failed_row = database.get_conversion("job-a")
    assert failed_row["status"] == "processing"
    assert failed_row["completed_at"] == ""
    assert failed_row["claim_token"] == claim_token
    assert failed_row["file_type"] is None
    assert failed_row["text_quality_score"] is None
    assert failed_row["winning_engine"] is None
    assert failed_row["processing_time"] == 0
    assert not target.exists()
    assert list(target.parent.glob("*.docx.part")) == []

    recovered = client.post(
        "/internal/jobs/job-a/result",
        headers=headers,
        data={
            "worker_name": "worker-1",
            "claim_token": claim_token,
            "metadata": json.dumps({"status": "completed", "text_quality_score": 99}),
        },
        files={
            "result": ("result.docx", io.BytesIO(valid_docx_bytes() + b"RECOVERED"))
        },
    )

    assert recovered.status_code == 200, recovered.text
    recovered_row = database.get_conversion("job-a")
    assert recovered_row["status"] == "completed"
    assert recovered_row["completed_at"]
    assert target.read_bytes().endswith(b"RECOVERED")
    assert list(target.parent.glob("*.docx.part")) == []

    replay = client.post(
        "/internal/jobs/job-a/result",
        headers=headers,
        data={
            "worker_name": "worker-1",
            "claim_token": claim_token,
            "metadata": json.dumps({"status": "completed", "text_quality_score": 99}),
        },
        files={"result": ("result.docx", io.BytesIO(valid_docx_bytes() + b"REPLAY"))},
    )
    assert replay.status_code == 409
    assert target.read_bytes().endswith(b"RECOVERED")


def test_promoted_result_recovers_if_database_completion_fails(
    api_environment, monkeypatch: pytest.MonkeyPatch
):
    client, database, storage = api_environment
    client = TestClient(app, raise_server_exceptions=False)
    create_job(database, storage)
    headers = {"X-Worker-API-Key": API_KEY}
    started = client.post(
        "/internal/jobs/job-a/start", headers=headers, json={"worker_name": "worker-1"}
    )
    claim_token = started.json()["claim_token"]
    target = Path(database.get_conversion("job-a")["stored_docx_path"])
    original_complete = Database.complete_conversion_finalization
    calls = 0

    def fail_once(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated sqlite outage")
        return original_complete(self, *args, **kwargs)

    monkeypatch.setattr(Database, "complete_conversion_finalization", fail_once)

    failed = client.post(
        "/internal/jobs/job-a/result",
        headers=headers,
        data={
            "worker_name": "worker-1",
            "claim_token": claim_token,
            "metadata": json.dumps({"status": "completed", "text_quality_score": 99}),
        },
        files={"result": ("result.docx", io.BytesIO(valid_docx_bytes() + b"PROMOTED"))},
    )

    assert failed.status_code == 503
    assert target.read_bytes().endswith(b"PROMOTED")
    assert database.get_conversion("job-a")["status"] == "finalizing"

    recovered = client.post(
        "/internal/jobs/job-a/start",
        headers=headers,
        json={"worker_name": "worker-1", "claim_token": claim_token},
    )

    assert recovered.status_code == 409
    row = database.get_conversion("job-a")
    assert row["status"] == "completed"
    assert row["completed_at"]
    assert target.read_bytes().endswith(b"PROMOTED")


def test_same_worker_duplicate_cannot_rollback_fresh_finalizing_without_file(
    api_environment,
):
    client, database, storage = api_environment
    create_job(database, storage)
    headers = {"X-Worker-API-Key": API_KEY}
    started = client.post(
        "/internal/jobs/job-a/start", headers=headers, json={"worker_name": "worker-1"}
    )
    claim_token = started.json()["claim_token"]
    target = Path(database.get_conversion("job-a")["stored_docx_path"])
    database.prepare_conversion_finalization(
        "job-a",
        "completed",
        worker_name="worker-1",
        claim_token=claim_token,
        extra={"stored_docx_path": str(target), "text_quality_score": 99},
    )
    assert not target.exists()

    duplicate = client.post(
        "/internal/jobs/job-a/result",
        headers=headers,
        data={
            "worker_name": "worker-1",
            "claim_token": claim_token,
            "metadata": json.dumps({"status": "completed", "text_quality_score": 99}),
        },
        files={"result": ("result.docx", io.BytesIO(valid_docx_bytes() + b"RETRY"))},
    )

    assert duplicate.status_code == 409
    row = database.get_conversion("job-a")
    assert row["status"] == "finalizing"
    assert row["worker_name"] == "worker-1"
    assert row["claim_token"] == claim_token
    assert not target.exists()


def test_stale_finalizing_without_file_can_be_requeued(api_environment, monkeypatch):
    monkeypatch.setenv("CLOUDA_FINALIZING_STALE_SECONDS", "1")
    client, database, storage = api_environment
    create_job(database, storage)
    headers = {"X-Worker-API-Key": API_KEY}
    started = client.post(
        "/internal/jobs/job-a/start", headers=headers, json={"worker_name": "worker-1"}
    )
    claim_token = started.json()["claim_token"]
    target = Path(database.get_conversion("job-a")["stored_docx_path"])
    database.prepare_conversion_finalization(
        "job-a",
        "completed",
        worker_name="worker-1",
        claim_token=claim_token,
        extra={"stored_docx_path": str(target), "text_quality_score": 99},
    )
    database.update_conversion(
        database.get_conversion("job-a")["id"],
        {"updated_at": "2000-01-01T00:00:00+00:00"},
    )

    restarted = client.post(
        "/internal/jobs/job-a/start", headers=headers, json={"worker_name": "worker-2"}
    )

    assert restarted.status_code == 200, restarted.text
    row = database.get_conversion("job-a")
    assert row["status"] == "processing"
    assert row["worker_name"] == "worker-2"
    assert row["claim_token"] != claim_token
    assert not target.exists()


def test_worker_start_cannot_steal_job_during_finalization(api_environment):
    client, database, storage = api_environment
    create_job(database, storage)
    headers = {"X-Worker-API-Key": API_KEY}
    started = client.post(
        "/internal/jobs/job-a/start", headers=headers, json={"worker_name": "worker-1"}
    )
    claim_token = started.json()["claim_token"]
    database.prepare_conversion_finalization(
        "job-a",
        "completed",
        worker_name="worker-1",
        claim_token=claim_token,
        extra={
            "stored_docx_path": database.get_conversion("job-a")["stored_docx_path"]
        },
    )

    stolen = client.post(
        "/internal/jobs/job-a/start", headers=headers, json={"worker_name": "worker-2"}
    )

    assert stolen.status_code == 409
    row = database.get_conversion("job-a")
    assert row["status"] == "finalizing"
    assert row["worker_name"] == "worker-1"
    assert row["claim_token"] == claim_token


def test_worker_result_path_rejects_reparse_or_symlink_escape(
    api_environment, tmp_path
):
    client, database, storage = api_environment
    row = create_job(database, storage)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = storage / "alice" / "job-a" / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"cannot create symlink in this environment: {exc}")
    database.update_conversion(
        row["id"], {"stored_docx_path": str(link / "escape.docx")}
    )
    client.post(
        "/internal/jobs/job-a/start",
        headers={"X-Worker-API-Key": API_KEY},
        json={"worker_name": "worker-1"},
    )
    claim_token = database.get_conversion("job-a")["claim_token"]

    response = client.post(
        "/internal/jobs/job-a/result",
        headers={"X-Worker-API-Key": API_KEY},
        data={
            "worker_name": "worker-1",
            "claim_token": claim_token,
            "metadata": json.dumps({"status": "completed", "text_quality_score": 99}),
        },
        files={"result": ("result.docx", io.BytesIO(valid_docx_bytes()))},
    )

    assert response.status_code == 403
    assert not (outside / "escape.docx").exists()


def test_worker_api_rejects_invalid_job_and_result_inputs(
    api_environment,
    monkeypatch: pytest.MonkeyPatch,
):
    client, database, storage = api_environment
    create_job(database, storage)
    headers = {"X-Worker-API-Key": API_KEY}

    assert client.get("/internal/jobs/bad$id", headers=headers).status_code == 400
    assert client.get("/internal/jobs/missing", headers=headers).status_code == 404
    assert client.get("/internal/jobs/job-a/input", headers=headers).status_code == 200

    assert (
        client.post(
            "/internal/jobs/job-a/start",
            headers=headers,
            json={"worker_name": "worker-1"},
        ).status_code
        == 200
    )
    claim_token = database.get_conversion("job-a")["claim_token"]
    assert (
        client.post(
            "/internal/jobs/job-a/start",
            headers=headers,
            json={"worker_name": "worker-1", "claim_token": claim_token},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/internal/jobs/job-a/start",
            headers=headers,
            json={"worker_name": "worker-2"},
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/internal/jobs/job-a/heartbeat",
            headers=headers,
            json={"worker_name": "worker-2"},
        ).status_code
        == 409
    )

    common_with_claim = {
        "worker_name": "worker-1",
        "claim_token": claim_token,
        "metadata": '{"status":"completed"}',
    }
    assert (
        client.post(
            "/internal/jobs/job-a/result",
            headers=headers,
            data=common_with_claim,
            files={"result": ("bad.txt", b"text", "text/plain")},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/internal/jobs/job-a/result",
            headers=headers,
            data={
                "worker_name": "worker-1",
                "claim_token": claim_token,
                "metadata": "{",
            },
            files={"result": ("result.docx", valid_docx_bytes())},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/internal/jobs/job-a/result",
            headers=headers,
            data={
                "worker_name": "worker-1",
                "claim_token": claim_token,
                "metadata": '{"status":"wrong"}',
            },
            files={"result": ("result.docx", valid_docx_bytes())},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/internal/jobs/job-a/result",
            headers=headers,
            data=common_with_claim,
            files={"result": ("result.docx", b"PK\x03\x04not-a-zip")},
        ).status_code
        == 400
    )
    monkeypatch.setenv("CLOUDA_MAX_RESULT_BYTES", "1024")
    assert (
        client.post(
            "/internal/jobs/job-a/result",
            headers=headers,
            data=common_with_claim,
            files={"result": ("result.docx", b"PK" + b"x" * 2048)},
        ).status_code
        == 413
    )


def test_worker_failure_endpoint_enforces_ownership(api_environment):
    client, database, storage = api_environment
    create_job(database, storage)
    headers = {"X-Worker-API-Key": API_KEY}
    payload = {"worker_name": "worker-1", "error": "authorization header present"}
    assert (
        client.post(
            "/internal/jobs/job-a/failure", headers=headers, json=payload
        ).status_code
        == 409
    )
    client.post(
        "/internal/jobs/job-a/start",
        headers=headers,
        json={"worker_name": "worker-1"},
    )
    claim_token = database.get_conversion("job-a")["claim_token"]
    payload["claim_token"] = claim_token
    response = client.post(
        "/internal/jobs/job-a/failure",
        headers=headers,
        json=payload,
    )
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert database.get_conversion("job-a")["error_message"] == "[redacted]"
    assert (
        client.post(
            "/internal/jobs/job-a/failure", headers=headers, json=payload
        ).json()["status"]
        == "failed"
    )


def test_api_rejects_path_outside_storage(api_environment, tmp_path: Path):
    client, database, storage = api_environment
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF")
    row = create_job(database, storage)
    database.update_conversion(row["id"], {"status": "pending"})
    with database.transaction() as connection:
        connection.execute(
            "UPDATE conversions SET stored_pdf_path = ? WHERE job_id = ?",
            (str(outside), "job-a"),
        )

    response = client.get(
        "/internal/jobs/job-a/input", headers={"X-Worker-API-Key": API_KEY}
    )
    assert response.status_code == 403


def test_worker_input_download_rejects_hidden_and_final_jobs(api_environment):
    client, database, storage = api_environment
    row = create_job(database, storage)
    headers = {"X-Worker-API-Key": API_KEY}

    database.hide_conversion("job-a", row["username"])
    assert client.get("/internal/jobs/job-a/input", headers=headers).status_code == 404

    row = create_job(database, storage, job_id="job-b")
    database.transition_conversion("job-b", "processing", worker_name="worker-1")
    database.transition_conversion("job-b", "completed")
    assert client.get("/internal/jobs/job-b/input", headers=headers).status_code in {
        404,
        409,
    }


def test_database_rejects_invalid_final_transition(tmp_path: Path):
    database = Database(tmp_path / "db.sqlite3")
    create_job(database, tmp_path / "storage")
    database.transition_conversion("job-a", "processing", worker_name="worker")
    database.transition_conversion("job-a", "completed")

    with pytest.raises(ValueError, match="Invalid status transition"):
        database.transition_conversion("job-a", "processing", worker_name="worker")


def test_server_entry_has_no_heavy_worker_imports():
    tree = ast.parse(Path("app.py").read_text(encoding="utf-8-sig"))
    top_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_imports.append(node.module or "")
    assert "pdfword.ocr_pipeline" not in top_imports
    assert "pdfword.conversion_service" not in top_imports


def test_server_modules_do_not_load_ocr():
    before = set(sys.modules)
    __import__("pdfword.worker_api")
    __import__("pdfword.health")
    loaded = set(sys.modules) - before
    assert "pdfword.ocr_pipeline" not in loaded
    assert "pdfword.ocr_pipeline" not in loaded


def test_worker_downloads_converts_uploads_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    events = []

    class FakeClient:
        def start(self, job_id, worker_name):
            events.append(("start", job_id))
            return {
                "page_numbers": [1],
                "fast_model": "fast",
                "accurate_model": "accurate",
                "settings": {},
            }

        def download_input(self, job_id, target):
            events.append(("download", job_id))
            target.write_bytes(b"%PDF")

        def upload_result(self, job_id, worker_name, docx, metadata):
            events.append(("upload", docx.read(), metadata["status"]))
            return {"job_id": job_id, "status": metadata["status"]}

        def heartbeat(self, job_id, worker_name):
            events.append(("heartbeat", job_id))

        def fail(self, job_id, worker_name, error):
            events.append(("fail", error))

    def fake_conversion(request):
        assert request.pdf_path.read_bytes() == b"%PDF"
        request.docx_path.write_bytes(b"PK\x03\x04worker-result")
        return {"status": "completed", "processing_time": 1}

    monkeypatch.setenv("APP_ROLE", "worker")
    monkeypatch.setenv("WORKER_API_KEY", API_KEY)
    monkeypatch.setenv("TEMP_ROOT", str(tmp_path / "temporary"))
    monkeypatch.setattr(worker_tasks, "WorkerApiClient", FakeClient)
    import pdfword.conversion_service as conversion_service

    monkeypatch.setattr(
        conversion_service, "execute_worker_conversion", fake_conversion
    )
    result = worker_tasks.run_remote_job("job-a")

    assert result["status"] == "completed"
    assert [event[0] for event in events] == ["start", "download", "upload"]
    assert list((tmp_path / "temporary").iterdir()) == []


def test_worker_reports_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    failures = []

    class FakeClient:
        def start(self, job_id, worker_name):
            return {
                "page_numbers": [1],
                "fast_model": "fast",
                "accurate_model": "accurate",
                "settings": {},
            }

        def download_input(self, job_id, target):
            target.write_bytes(b"%PDF")

        def heartbeat(self, job_id, worker_name):
            return None

        def fail(self, job_id, worker_name, error):
            failures.append((job_id, error))

    monkeypatch.setenv("APP_ROLE", "worker")
    monkeypatch.setenv("WORKER_API_KEY", API_KEY)
    monkeypatch.setenv("TEMP_ROOT", str(tmp_path / "temporary"))
    monkeypatch.setattr(worker_tasks, "WorkerApiClient", FakeClient)
    import pdfword.conversion_service as conversion_service

    monkeypatch.setattr(
        conversion_service,
        "execute_worker_conversion",
        lambda _request: (_ for _ in ()).throw(
            RuntimeError("authorization header present")
        ),
    )
    with caplog.at_level(logging.ERROR, logger="pdfword.worker_tasks"):
        with pytest.raises(RuntimeError, match="authorization header present"):
            worker_tasks.run_remote_job("job-a")
    assert failures == [("job-a", "[redacted]")]
    assert "RuntimeError" in caplog.text
    assert "authorization header present" not in caplog.text
    assert list((tmp_path / "temporary").iterdir()) == []


def test_worker_status_404_is_treated_as_legacy_server(
    monkeypatch: pytest.MonkeyPatch,
):
    response = requests.Response()
    response.status_code = 404
    response.url = "http://server/internal/workers/status"

    def fake_request(*_args, **_kwargs):
        return response

    monkeypatch.setenv("SERVER_BASE_URL", "http://server")
    monkeypatch.setenv("WORKER_API_KEY", API_KEY)
    monkeypatch.setattr(requests, "request", fake_request)

    result = WorkerApiClient().report_status("windows-worker-1", True, "openrouter")

    assert result == {"status": "unsupported", "legacy_server": True}


def test_worker_client_covers_json_stream_and_command_methods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    class Response:
        headers = {"content-type": "application/json"}

        def raise_for_status(self):
            return None

        def json(self):
            return {"status": "ok"}

        def iter_content(self, _size):
            return [b"part-a", b"", b"part-b"]

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        response = Response()
        if url.endswith("/input"):
            response.headers = {"content-type": "application/pdf"}
        return response

    monkeypatch.setenv("SERVER_BASE_URL", "http://server")
    monkeypatch.setenv("WORKER_API_KEY", API_KEY)
    monkeypatch.setattr(requests, "request", fake_request)
    client = WorkerApiClient()

    assert client.health()["status"] == "ok"
    assert client.correction_snapshot()["status"] == "ok"
    assert client.report_status("worker", False)["status"] == "ok"
    assert client.get_job("job-a")["status"] == "ok"
    target = tmp_path / "input.pdf"
    client.download_input("job-a", target)
    assert target.read_bytes() == b"part-apart-b"
    assert client.start("job-a", "worker")["status"] == "ok"
    client.heartbeat("job-a", "worker")
    assert (
        client.upload_result(
            "job-a",
            "worker",
            io.BytesIO(valid_docx_bytes()),
            {"status": "completed"},
        )["status"]
        == "ok"
    )
    assert client.fail("job-a", "worker", "error" * 1000)["status"] == "ok"
    assert all(call[2]["headers"]["X-Worker-API-Key"] == API_KEY for call in calls)
