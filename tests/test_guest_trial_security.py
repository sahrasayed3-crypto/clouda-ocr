from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from pdfword.auth import FakeAuthEmulator
from pdfword.database import Database
from pdfword.guest_trial import cleanup_expired_guest_jobs
from pdfword.worker_api import app

from tests.test_multi_user_auth import login

API_KEY = "test-worker-secret"  # pragma: allowlist secret


def signed_fake_token(uid: str, email: str) -> str:
    return FakeAuthEmulator("local-test-project").token(
        provider="password",
        uid=uid,
        email=email,
        state="verified",
        display_name="Alice",
    )


def pdf_bytes(pages: int = 1) -> bytes:
    output = io.BytesIO()
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    writer.write(output)
    return output.getvalue()


def docx_bytes(marker: str = "guest-result") -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',
        )
        archive.writestr(
            "word/document.xml",
            f'<document xmlns="http://schemas.openxmlformats.org/wordprocessingml/2006/main">{marker}</document>',
        )
    return output.getvalue()


@pytest.fixture
def guest_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLOUDA_DATABASE_PATH", str(tmp_path / "guest.sqlite3"))
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("CLOUDA_AUTH_VERIFIER", "fake")
    monkeypatch.setenv("CLOUDA_FIREBASE_PROJECT_ID", "local-test-project")
    monkeypatch.setenv("CLOUDA_SESSION_COOKIE_SECURE", "false")
    monkeypatch.setenv("CLOUDA_GUEST_MAX_BYTES", str(10 * 1024 * 1024))
    monkeypatch.setenv("CLOUDA_GUEST_MAX_PAGES", "5")
    monkeypatch.setenv("WORKER_API_KEY", API_KEY)

    class FakeQueue:
        def enqueue(self, job_id: str):
            return type("QueuedJob", (), {"id": job_id})()

    monkeypatch.setattr("pdfword.worker_api.get_distributed_queue", lambda: FakeQueue())
    return TestClient(app), Database(tmp_path / "guest.sqlite3"), tmp_path / "storage"


def start_guest_session(client: TestClient) -> dict[str, str]:
    response = client.post("/guest/session")
    assert response.status_code == 200, response.text
    return {"X-CSRF-Token": response.json()["csrf_token"]}


def test_guest_trial_requires_guest_csrf_header(guest_client):
    client, _database, _storage = guest_client
    start_guest_session(client)

    response = client.post(
        "/guest/trial",
        files={"file": ("small.pdf", pdf_bytes(1), "application/pdf")},
    )

    assert response.status_code == 403


def test_guest_trial_rejects_wrong_or_cross_guest_csrf_header(guest_client):
    client, _database, _storage = guest_client
    start_guest_session(client)

    wrong = client.post(
        "/guest/trial",
        headers={"X-CSRF-Token": "wrong-token"},
        files={"file": ("small.pdf", pdf_bytes(1), "application/pdf")},
    )
    assert wrong.status_code == 403

    other_guest = TestClient(app)
    other_guest_csrf = start_guest_session(other_guest)
    cross = client.post(
        "/guest/trial",
        headers=other_guest_csrf,
        files={"file": ("small.pdf", pdf_bytes(1), "application/pdf")},
    )
    assert cross.status_code == 403


def test_guest_trial_limits_file_type_pages_active_job_and_paid_dispatch(guest_client):
    client, database, storage = guest_client
    guest_csrf = start_guest_session(client)

    txt = client.post(
        "/guest/trial",
        headers=guest_csrf,
        files={"file": ("not.pdf", b"text", "text/plain")},
    )
    assert txt.status_code == 400

    too_many_pages = client.post(
        "/guest/trial",
        headers=guest_csrf,
        files={"file": ("large.pdf", pdf_bytes(6), "application/pdf")},
    )
    assert too_many_pages.status_code == 413

    accepted = client.post(
        "/guest/trial",
        headers=guest_csrf,
        files={"file": ("small.pdf", pdf_bytes(5), "application/pdf")},
    )
    assert accepted.status_code == 201, accepted.text
    payload = accepted.json()
    assert payload["provider_policy"] == "local_free_only"
    assert payload["paid_provider_allowed"] is False
    assert payload["page_count"] == 5
    assert "guest_scope_id" not in payload
    assert "@" not in database.get_guest_job(payload["job_id"])["stored_pdf_path"]

    second = client.post(
        "/guest/trial",
        headers=guest_csrf,
        files={"file": ("second.pdf", pdf_bytes(1), "application/pdf")},
    )
    assert second.status_code == 429

    other_client = TestClient(app)
    other_client.post("/guest/session")
    assert other_client.get(f"/guest/jobs/{payload['job_id']}").status_code == 404


@pytest.mark.parametrize("filename", ["../../pwned.pdf", r"..\..\pwned.pdf"])
def test_guest_trial_pathlike_filename_stays_within_tenant_uploads(
    guest_client, filename: str
):
    client, database, storage = guest_client
    guest_csrf = start_guest_session(client)

    response = client.post(
        "/guest/trial",
        headers=guest_csrf,
        files={"file": (filename, pdf_bytes(1), "application/pdf")},
    )

    assert response.status_code == 201, response.text
    job = database.get_guest_job(response.json()["job_id"])
    uploads = storage / "guests" / job["guest_scope_id"] / "uploads"
    stored_pdf = Path(job["stored_pdf_path"]).resolve()
    assert job["original_pdf_name"] == filename
    assert stored_pdf.is_relative_to(uploads.resolve())
    assert not [
        path
        for path in storage.rglob("pwned.pdf")
        if not path.resolve().is_relative_to(uploads.resolve())
    ]


def test_guest_cleanup_and_claim_to_account(guest_client):
    client, database, storage = guest_client
    guest_csrf = start_guest_session(client)
    accepted = client.post(
        "/guest/trial",
        headers=guest_csrf,
        files={"file": ("small.pdf", pdf_bytes(1), "application/pdf")},
    )
    job_id = accepted.json()["job_id"]

    claim_token = client.post("/guest/claim-token", headers=guest_csrf).json()[
        "claim_token"
    ]
    auth_payload, csrf = login(
        client, signed_fake_token("alice-id", "alice@example.com")
    )
    claim = client.post(
        "/guest/claim",
        headers=csrf,
        json={"job_id": job_id, "claim_token": claim_token},
    )
    assert claim.status_code == 200, claim.text
    assert claim.json()["owner_user_id"] == auth_payload["user"]["user_id"]
    assert (
        client.post(
            "/guest/claim",
            headers=csrf,
            json={"job_id": job_id, "claim_token": claim_token},
        ).status_code
        == 409
    )

    expired = database.expire_guest_jobs(now_offset_seconds=10_000)
    assert expired >= 0
    assert any(
        event["event_type"] == "guest_job_claimed"
        for event in database.list_auth_audit_events()
    )


def test_expired_guest_jobs_and_files_are_inaccessible_and_removed(guest_client):
    client, database, storage = guest_client
    guest_csrf = start_guest_session(client)
    accepted = client.post(
        "/guest/trial",
        headers=guest_csrf,
        files={"file": ("small.pdf", pdf_bytes(1), "application/pdf")},
    )
    assert accepted.status_code == 201, accepted.text
    job_id = accepted.json()["job_id"]
    stored_pdf = Path(database.get_guest_job(job_id)["stored_pdf_path"])
    assert stored_pdf.is_file()

    result = cleanup_expired_guest_jobs(
        database, storage_root=storage, now_offset_seconds=10_000
    )

    assert result["jobs_expired"] == 1
    assert result["files_removed"] >= 1
    assert not stored_pdf.exists()
    assert client.get(f"/guest/jobs/{job_id}").status_code == 404
    assert database.get_guest_job(job_id) is None


def test_guest_claim_token_replay_and_cross_guest_claim_are_rejected(guest_client):
    client, _database, _storage = guest_client
    guest_csrf = start_guest_session(client)
    accepted = client.post(
        "/guest/trial",
        headers=guest_csrf,
        files={"file": ("small.pdf", pdf_bytes(1), "application/pdf")},
    )
    job_id = accepted.json()["job_id"]
    claim_token = client.post("/guest/claim-token", headers=guest_csrf).json()[
        "claim_token"
    ]

    other_guest = TestClient(app)
    other_guest.post("/guest/session")
    auth_payload, csrf = login(
        other_guest, signed_fake_token("other-user", "other@example.com")
    )
    assert auth_payload["user"]["email"] == "other@example.com"
    stolen = other_guest.post(
        "/guest/claim",
        headers=csrf,
        json={"job_id": job_id, "claim_token": claim_token},
    )
    assert stolen.status_code in {403, 409}

    auth_payload, csrf = login(
        client, signed_fake_token("alice-id", "alice@example.com")
    )
    assert auth_payload["user"]["email"] == "alice@example.com"
    assert (
        client.post(
            "/guest/claim",
            headers=csrf,
            json={"job_id": job_id, "claim_token": claim_token},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/guest/claim",
            headers=csrf,
            json={"job_id": job_id, "claim_token": claim_token},
        ).status_code
        == 409
    )


def test_guest_daily_budget_survives_fresh_guest_session(guest_client, monkeypatch):
    monkeypatch.setenv("CLOUDA_GUEST_DAILY_BUDGET", "1")
    client, _database, _storage = guest_client
    guest_csrf = start_guest_session(client)
    first = client.post(
        "/guest/trial",
        headers=guest_csrf,
        files={"file": ("first.pdf", pdf_bytes(1), "application/pdf")},
    )
    assert first.status_code == 201, first.text

    fresh_guest = TestClient(app)
    fresh_guest_csrf = start_guest_session(fresh_guest)
    second = fresh_guest.post(
        "/guest/trial",
        headers=fresh_guest_csrf,
        files={"file": ("second.pdf", pdf_bytes(1), "application/pdf")},
    )

    assert second.status_code == 429
    assert second.headers["Retry-After"].isdigit()


def test_invalid_guest_upload_does_not_consume_daily_budget(guest_client, monkeypatch):
    monkeypatch.setenv("CLOUDA_GUEST_DAILY_BUDGET", "1")
    client, _database, _storage = guest_client
    guest_csrf = start_guest_session(client)
    invalid = client.post(
        "/guest/trial",
        headers=guest_csrf,
        files={"file": ("bad.pdf", b"not a pdf", "application/pdf")},
    )
    assert invalid.status_code == 400

    accepted = client.post(
        "/guest/trial",
        headers=guest_csrf,
        files={"file": ("first.pdf", pdf_bytes(1), "application/pdf")},
    )

    assert accepted.status_code == 201, accepted.text


def test_guest_full_backend_ocr_flow_through_one_time_download(guest_client):
    client, database, _storage = guest_client
    guest_csrf = start_guest_session(client)
    accepted = client.post(
        "/guest/trial",
        headers=guest_csrf,
        files={"file": ("small.pdf", pdf_bytes(1), "application/pdf")},
    )
    assert accepted.status_code == 201, accepted.text
    job_id = accepted.json()["job_id"]
    headers = {"X-Worker-API-Key": API_KEY}

    started = client.post(
        f"/internal/jobs/{job_id}/start",
        headers=headers,
        json={"worker_name": "guest-worker"},
    )
    assert started.status_code == 200, started.text
    claim_token = started.json()["claim_token"]
    uploaded = client.post(
        f"/internal/jobs/{job_id}/result",
        headers=headers,
        data={
            "worker_name": "guest-worker",
            "claim_token": claim_token,
            "metadata": json.dumps({"status": "completed", "text_quality_score": 99}),
        },
        files={"result": ("result.docx", io.BytesIO(docx_bytes()))},
    )
    assert uploaded.status_code == 200, uploaded.text
    status = client.get(f"/guest/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"

    result_token = accepted.json()["result_token"]
    downloaded = client.get(f"/guest/jobs/{job_id}/download?token={result_token}")
    assert downloaded.status_code == 200, downloaded.text
    assert downloaded.content.startswith(b"PK")
    replay = client.get(f"/guest/jobs/{job_id}/download?token={result_token}")
    assert replay.status_code in {403, 404, 409}

    other_guest = TestClient(app)
    other_guest.post("/guest/session")
    guessed = other_guest.get(f"/guest/jobs/{job_id}/download?token={result_token}")
    assert guessed.status_code == 404


def test_guest_result_download_survives_worker_promotion_recovery(
    guest_client, monkeypatch: pytest.MonkeyPatch
):
    client, database, _storage = guest_client
    guest_csrf = start_guest_session(client)
    accepted = client.post(
        "/guest/trial",
        headers=guest_csrf,
        files={"file": ("small.pdf", pdf_bytes(1), "application/pdf")},
    )
    job_id = accepted.json()["job_id"]
    result_token = accepted.json()["result_token"]
    headers = {"X-Worker-API-Key": API_KEY}
    started = client.post(
        f"/internal/jobs/{job_id}/start",
        headers=headers,
        json={"worker_name": "guest-worker"},
    )
    claim_token = started.json()["claim_token"]
    calls = 0

    def fail_once(_temporary_path, _target):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated promotion failure")
        return original_replace(_temporary_path, _target)

    original_replace = os.replace
    monkeypatch.setattr("pdfword.worker_api.os.replace", fail_once)

    first = client.post(
        f"/internal/jobs/{job_id}/result",
        headers=headers,
        data={
            "worker_name": "guest-worker",
            "claim_token": claim_token,
            "metadata": json.dumps({"status": "completed", "text_quality_score": 99}),
        },
        files={"result": ("result.docx", io.BytesIO(docx_bytes("first")))},
    )
    assert first.status_code == 503
    assert database.get_conversion(job_id)["status"] == "processing"
    assert database.get_guest_job(job_id)["status"] == "pending"

    second = client.post(
        f"/internal/jobs/{job_id}/result",
        headers=headers,
        data={
            "worker_name": "guest-worker",
            "claim_token": claim_token,
            "metadata": json.dumps({"status": "completed", "text_quality_score": 99}),
        },
        files={"result": ("result.docx", io.BytesIO(docx_bytes("recovered")))},
    )
    assert second.status_code == 200, second.text

    download = client.get(f"/guest/jobs/{job_id}/download?token={result_token}")
    assert download.status_code == 200, download.text
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        assert b"recovered" in archive.read("word/document.xml")


def test_guest_download_does_not_burn_token_before_finalizing_recovery(
    guest_client, monkeypatch: pytest.MonkeyPatch
):
    client, database, _storage = guest_client
    guest_csrf = start_guest_session(client)
    accepted = client.post(
        "/guest/trial",
        headers=guest_csrf,
        files={"file": ("small.pdf", pdf_bytes(1), "application/pdf")},
    )
    job_id = accepted.json()["job_id"]
    result_token = accepted.json()["result_token"]
    headers = {"X-Worker-API-Key": API_KEY}
    started = client.post(
        f"/internal/jobs/{job_id}/start",
        headers=headers,
        json={"worker_name": "guest-worker"},
    )
    claim_token = started.json()["claim_token"]
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
        f"/internal/jobs/{job_id}/result",
        headers=headers,
        data={
            "worker_name": "guest-worker",
            "claim_token": claim_token,
            "metadata": json.dumps({"status": "completed", "text_quality_score": 99}),
        },
        files={"result": ("result.docx", io.BytesIO(docx_bytes("promoted")))},
    )
    assert failed.status_code == 503
    assert database.get_conversion(job_id)["status"] == "finalizing"

    download = client.get(f"/guest/jobs/{job_id}/download?token={result_token}")
    assert download.status_code == 200, download.text
    with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
        assert b"promoted" in archive.read("word/document.xml")

    replay = client.get(f"/guest/jobs/{job_id}/download?token={result_token}")
    assert replay.status_code in {403, 404, 409}
