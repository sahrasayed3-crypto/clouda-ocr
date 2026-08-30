from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from pdfword.database import Database
from pdfword.worker_api import app

from tests.test_multi_user_auth import fake_token, login


def pdf_bytes(pages: int = 1) -> bytes:
    output = io.BytesIO()
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    writer.write(output)
    return output.getvalue()


@pytest.fixture
def tenant_clients(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLOUDA_DATABASE_PATH", str(tmp_path / "tenant.sqlite3"))
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("CLOUDA_AUTH_VERIFIER", "fake")
    monkeypatch.setenv("CLOUDA_FIREBASE_PROJECT_ID", "local-test-project")
    monkeypatch.setenv("CLOUDA_SESSION_COOKIE_SECURE", "false")
    database = Database(tmp_path / "tenant.sqlite3")
    alice = TestClient(app)
    bob = TestClient(app)
    admin = TestClient(app)
    alice_profile, alice_csrf = login(
        alice, fake_token("alice-id", "alice@example.com")
    )
    bob_profile, bob_csrf = login(bob, fake_token("bob-id", "bob@example.com"))
    admin_profile, admin_csrf = login(
        admin, fake_token("admin-id", "admin@example.com")
    )
    database.set_auth_user_role(
        admin_profile["user"]["user_id"],
        "admin",
        actor_user_id="system",
    )
    login(admin, fake_token("admin-id", "admin@example.com"))
    return (
        database,
        (alice, alice_profile, alice_csrf),
        (bob, bob_profile, bob_csrf),
        (admin, admin_profile, admin_csrf),
    )


def test_owner_scoped_documents_reject_idor_and_payload_owner_spoofing(tenant_clients):
    database, alice_ctx, bob_ctx, admin_ctx = tenant_clients
    alice, alice_profile, alice_csrf = alice_ctx
    bob, bob_profile, bob_csrf = bob_ctx
    admin, _admin_profile, admin_csrf = admin_ctx

    created = alice.post(
        "/user/documents",
        headers=alice_csrf,
        json={
            "original_pdf_name": "book.pdf",
            "page_count": 2,
            "owner_user_id": bob_profile["user"]["user_id"],
        },
    )
    assert created.status_code == 201, created.text
    created_payload = created.json()
    job_id = created_payload["job_id"]

    def assert_safe_document_payload(payload: dict) -> None:
        assert payload["page_count"] == 2
        assert payload["can_retry"] is False
        assert payload["can_cancel"] is True
        assert "stored_pdf_path" not in payload
        assert "stored_docx_path" not in payload
        assert "error_message" not in payload

    assert_safe_document_payload(created_payload)
    row = database.get_conversion(job_id)
    assert row["owner_user_id"] == alice_profile["user"]["user_id"]
    assert "alice@example.com" not in row["stored_pdf_path"]

    listed = alice.get("/user/documents").json()["documents"]
    assert [item["job_id"] for item in listed] == [job_id]
    assert_safe_document_payload(listed[0])
    detail = alice.get(f"/user/documents/{job_id}")
    assert detail.status_code == 200
    assert_safe_document_payload(detail.json())
    assert bob.get("/user/documents").json()["documents"] == []
    assert bob.get(f"/user/documents/{job_id}").status_code == 404
    assert bob.delete(f"/user/documents/{job_id}", headers=bob_csrf).status_code == 404
    assert (
        bob.post(f"/user/documents/{job_id}/retry", headers=bob_csrf).status_code == 404
    )
    assert bob.get(f"/user/documents/{job_id}/download").status_code == 404

    admin_read = admin.get(f"/admin/documents/{job_id}", headers=admin_csrf)
    assert admin_read.status_code == 200
    assert admin_read.json()["owner_user_id"] == alice_profile["user"]["user_id"]
    assert any(
        event["event_type"] == "admin_document_access"
        for event in database.list_auth_audit_events()
    )


def test_authenticated_pdf_upload_assigns_server_side_owner(tenant_clients):
    database, alice_ctx, bob_ctx, _admin_ctx = tenant_clients
    alice, alice_profile, alice_csrf = alice_ctx
    bob, _bob_profile, _bob_csrf = bob_ctx

    uploaded = alice.post(
        "/user/documents/upload",
        headers=alice_csrf,
        files={"file": ("owned.pdf", pdf_bytes(2), "application/pdf")},
    )

    assert uploaded.status_code == 201, uploaded.text
    job_id = uploaded.json()["job_id"]
    row = database.get_conversion(job_id)
    assert row["owner_user_id"] == alice_profile["user"]["user_id"]
    assert Path(row["stored_pdf_path"]).is_file()
    assert "alice@example.com" not in row["stored_pdf_path"]
    assert bob.get(f"/user/documents/{job_id}").status_code == 404


def test_guests_and_stale_sessions_cannot_access_user_documents(tenant_clients):
    database, alice_ctx, _bob_ctx, _admin_ctx = tenant_clients
    alice, _alice_profile, alice_csrf = alice_ctx
    created = alice.post(
        "/user/documents",
        headers=alice_csrf,
        json={"original_pdf_name": "book.pdf", "page_count": 1},
    )
    job_id = created.json()["job_id"]

    guest = TestClient(app)
    assert guest.post("/guest/session").status_code == 200
    assert guest.get(f"/user/documents/{job_id}").status_code == 401

    alice.post("/auth/logout-all", headers=alice_csrf)
    assert alice.get(f"/user/documents/{job_id}").status_code == 401


def test_cancel_download_path_manipulation_and_admin_audit_are_isolated(
    tenant_clients, tmp_path: Path
):
    database, alice_ctx, bob_ctx, admin_ctx = tenant_clients
    alice, alice_profile, alice_csrf = alice_ctx
    bob, _bob_profile, bob_csrf = bob_ctx
    admin, admin_profile, _admin_csrf = admin_ctx

    created = alice.post(
        "/user/documents",
        headers=alice_csrf,
        json={"original_pdf_name": "alice.pdf", "page_count": 1},
    )
    assert created.status_code == 201, created.text
    job_id = created.json()["job_id"]

    assert (
        bob.post(f"/user/documents/{job_id}/cancel", headers=bob_csrf).status_code
        == 404
    )
    cancelled = alice.post(f"/user/documents/{job_id}/cancel", headers=alice_csrf)
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"

    row = database.get_owner_conversion(job_id, alice_profile["user"]["user_id"])
    assert row is not None
    outside = tmp_path / "outside.docx"
    outside.write_bytes(b"PK\x03\x04")
    database.update_conversion(row["id"], {"stored_docx_path": str(outside)})
    assert alice.get(f"/user/documents/{job_id}/download").status_code == 403

    admin_read = admin.get(f"/admin/documents/{job_id}")
    assert admin_read.status_code == 200
    assert admin_read.json()["owner_user_id"] == alice_profile["user"]["user_id"]
    assert any(
        event["event_type"] == "admin_document_access"
        and event["actor_user_id"] == admin_profile["user"]["user_id"]
        for event in database.list_auth_audit_events()
    )


def test_delete_uses_owner_scope_when_legacy_username_differs(tenant_clients):
    database, alice_ctx, _bob_ctx, _admin_ctx = tenant_clients
    alice, alice_profile, alice_csrf = alice_ctx
    created = alice.post(
        "/user/documents",
        headers=alice_csrf,
        json={"original_pdf_name": "alice.pdf", "page_count": 1},
    )
    job_id = created.json()["job_id"]
    row = database.get_owner_conversion(job_id, alice_profile["user"]["user_id"])
    assert row is not None
    with database.transaction() as connection:
        connection.execute(
            "UPDATE conversions SET username=? WHERE job_id=?",
            ("legacy-local-user", job_id),
        )

    deleted = alice.delete(f"/user/documents/{job_id}", headers=alice_csrf)

    assert deleted.status_code == 200, deleted.text
    assert (
        database.get_owner_conversion(job_id, alice_profile["user"]["user_id"]) is None
    )
