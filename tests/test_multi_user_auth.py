from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pdfword.auth import (
    DisabledIdentityError,
    ExpiredIdentityError,
    FirebaseIdentityVerifier,
    FakeIdentityVerifier,
    InvalidIdentityError,
    RevokedIdentityError,
    WrongProjectIdentityError,
    VerifiedIdentity,
    identity_verifier_from_env,
)
from pdfword import worker_api
from pdfword.database import Database
from pdfword.worker_api import app

TEST_PASSWORD = "correct horse battery"  # pragma: allowlist secret
RESET_PASSWORD = "new passphrase"  # pragma: allowlist secret
REPLAY_PASSWORD = "replay passphrase"  # pragma: allowlist secret


def fake_token(
    uid: str,
    email: str = "alice@example.com",
    provider: str = "password",
    *,
    verified: bool = True,
    project: str = "local-test-project",
    display_name: str = "Alice",
    state_override: str | None = None,
) -> str:
    import hmac
    import hashlib

    state = (
        state_override if state_override else ("verified" if verified else "unverified")
    )
    payload = f"{project}:{provider}:{uid}:{email}:{state}:{display_name}"
    signature = hmac.new(
        b"dev_secret_key", payload.encode(), hashlib.sha256
    ).hexdigest()
    return f"fake:{payload}:{signature}"


@pytest.fixture
def auth_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLOUDA_DATABASE_PATH", str(tmp_path / "auth.sqlite3"))
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("CLOUDA_AUTH_VERIFIER", "fake")
    monkeypatch.setenv("CLOUDA_FIREBASE_PROJECT_ID", "local-test-project")
    monkeypatch.setenv("CLOUDA_SESSION_COOKIE_SECURE", "false")
    monkeypatch.setenv("CLOUDA_REQUIRE_VERIFIED_EMAIL", "true")
    monkeypatch.setenv("CLOUDA_ENV", "development")
    return TestClient(app), Database(tmp_path / "auth.sqlite3")


def login(client: TestClient, token: str) -> tuple[dict, dict[str, str]]:
    response = client.post(
        "/auth/session", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200, response.text
    csrf = response.json()["csrf_token"]
    return response.json(), {"X-CSRF-Token": csrf}


def test_fake_email_password_registration_login_reset_and_google_login(auth_client):
    client, database = auth_client

    registered = client.post(
        "/auth/dev/register",
        json={
            "email": "Alice@Example.com",
            "password": TEST_PASSWORD,
        },
    )
    assert registered.status_code == 201, registered.text

    unverified_login = client.post(
        "/auth/dev/login",
        json={
            "email": "alice@example.com",
            "password": TEST_PASSWORD,
        },
    )
    assert unverified_login.status_code == 403

    verify = client.post("/auth/dev/verify-email", json={"email": "alice@example.com"})
    assert verify.status_code == 200

    logged_in = client.post(
        "/auth/dev/login",
        json={
            "email": "alice@example.com",
            "password": TEST_PASSWORD,
        },
    )
    assert logged_in.status_code == 200, logged_in.text
    assert client.get("/auth/me").json()["email"] == "alice@example.com"

    start_reset = client.post(
        "/auth/dev/password-reset/start", json={"email": "alice@example.com"}
    )
    assert start_reset.status_code == 200
    reset_token = start_reset.json()["reset_token"]
    complete_reset = client.post(
        "/auth/dev/password-reset/complete",
        json={
            "reset_token": reset_token,
            "new_password": RESET_PASSWORD,
        },
    )
    assert complete_reset.status_code == 200
    assert (
        client.post(
            "/auth/dev/password-reset/complete",
            json={
                "reset_token": reset_token,
                "new_password": REPLAY_PASSWORD,
            },
        ).status_code
        == 403
    )

    logout = client.post(
        "/auth/logout", headers={"X-CSRF-Token": logged_in.json()["csrf_token"]}
    )
    assert logout.status_code == 200
    assert (
        client.post(
            "/auth/dev/login",
            json={
                "email": "alice@example.com",
                "password": TEST_PASSWORD,
            },
        ).status_code
        == 401
    )

    relogin = client.post(
        "/auth/dev/login",
        json={
            "email": "alice@example.com",
            "password": RESET_PASSWORD,
        },
    )
    assert relogin.status_code == 200

    google = client.post(
        "/auth/dev/google-login",
        json={"email": "alice@example.com", "google_user_id": "google-alice"},
    )
    assert google.status_code == 200
    events = [event["event_type"] for event in database.list_auth_audit_events()]
    assert "password_reset_completed" in events
    assert "login_success" in events


def test_fake_verifier_accepts_password_and_google_and_rejects_bad_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    verifier = FakeIdentityVerifier(project_id="local-test-project")

    password = verifier.verify_bearer_token(fake_token("email-uid"))
    google = verifier.verify_bearer_token(
        fake_token("google-uid", "alice@example.com", "google.com")
    )

    assert password.provider == "password"
    assert google.provider == "google.com"
    with pytest.raises(InvalidIdentityError):
        verifier.verify_bearer_token("not-a-fake-token")
    with pytest.raises(ExpiredIdentityError):
        verifier.verify_bearer_token(
            fake_token(
                "u", "e@example.com", state_override="expired", display_name="Name"
            )
        )
    with pytest.raises(RevokedIdentityError):
        verifier.verify_bearer_token(
            fake_token(
                "u", "e@example.com", state_override="revoked", display_name="Name"
            )
        )
    with pytest.raises(WrongProjectIdentityError):
        verifier.verify_bearer_token(
            fake_token("u", "e@example.com", project="wrong", display_name="Name")
        )
    with pytest.raises(InvalidIdentityError):
        verifier.verify_bearer_token(fake_token("u", "", display_name="Name"))
    with pytest.raises(InvalidIdentityError):
        verifier.verify_bearer_token(
            fake_token("u", "e@example.com", verified=False, display_name="Name")
        )
    with pytest.raises(DisabledIdentityError):
        verifier.verify_bearer_token(
            fake_token(
                "u", "e@example.com", state_override="disabled", display_name="Name"
            )
        )


def test_identity_configuration_fails_closed_in_production(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("CLOUDA_ENV", "production")
    monkeypatch.delenv("CLOUDA_AUTH_VERIFIER", raising=False)

    with pytest.raises(RuntimeError, match="Firebase"):
        identity_verifier_from_env()


def test_token_exchange_creates_opaque_cookie_session_without_leaking_token(
    auth_client, caplog: pytest.LogCaptureFixture
):
    client, database = auth_client
    token = fake_token("alice-provider-id")
    caplog.set_level(logging.INFO)

    payload, csrf_headers = login(client, token)
    profile = client.get("/auth/me")

    assert payload["user"]["email"] == "alice@example.com"
    assert payload["user"]["role"] == "user"
    assert "clouda_session=" in client.cookies.jar.__str__()
    assert profile.status_code == 200
    assert profile.json()["email"] == "alice@example.com"
    assert token not in profile.text
    assert token not in "\n".join(record.getMessage() for record in caplog.records)
    assert database.list_auth_audit_events()[0]["event_type"] == "login_success"
    assert client.post("/auth/logout").status_code == 403
    assert client.post("/auth/logout", headers=csrf_headers).status_code == 200
    assert client.get("/auth/me").status_code == 401


def test_firebase_rejects_unverified_and_missing_email_verification(monkeypatch):
    class FirebaseAuth:
        claims = {
            "aud": "local-test-project",
            "uid": "firebase-user",
            "email": "alice@example.com",
            "firebase": {"sign_in_provider": "google.com"},
        }

        @classmethod
        def verify_id_token(cls, *_args, **_kwargs):
            return dict(cls.claims)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "firebase_admin":

            class FirebaseAdmin:
                auth = FirebaseAuth

            return FirebaseAdmin
        if name == "firebase_admin.auth":
            return FirebaseAuth
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    verifier = FirebaseIdentityVerifier(
        "local-test-project", require_verified_email=True
    )

    FirebaseAuth.claims["email_verified"] = False
    with pytest.raises(InvalidIdentityError, match="verified"):
        verifier.verify_bearer_token("token")

    FirebaseAuth.claims.pop("email_verified")
    with pytest.raises(InvalidIdentityError, match="verified"):
        verifier.verify_bearer_token("token")

    FirebaseAuth.claims["email_verified"] = True
    identity = verifier.verify_bearer_token("token")
    assert identity.email == "alice@example.com"
    assert identity.email_verified is True


def test_session_exchange_rejects_constructed_unverified_identity(
    auth_client, monkeypatch
):
    client, _database = auth_client

    class ConstructedUnverifiedVerifier:
        def verify_bearer_token(self, _token):
            return VerifiedIdentity(
                provider_user_id="firebase-user",
                email="unverified@example.com",
                email_verified=False,
                provider="firebase",
            )

    monkeypatch.setattr(
        worker_api,
        "identity_verifier_from_env",
        lambda: ConstructedUnverifiedVerifier(),
    )

    response = client.post(
        "/auth/session", headers={"Authorization": "Bearer constructed"}
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid identity"


def test_session_idle_expiry_rejects_old_last_seen_and_refreshes_active(auth_client):
    client, database = auth_client
    login(client, fake_token("idle-user", "idle@example.com"))
    session_id = client.cookies.get("clouda_session")
    assert session_id
    session_hash = database.hash_secret(session_id)
    with database.transaction() as connection:
        connection.execute(
            """
            UPDATE auth_sessions
            SET last_seen_at='2000-01-01T00:00:00+00:00'
            WHERE session_id_hash=?
            """,
            (session_hash,),
        )

    assert client.get("/auth/me").status_code == 401

    fresh_client = TestClient(app)
    login(fresh_client, fake_token("active-user", "active@example.com"))
    fresh_session = fresh_client.cookies.get("clouda_session")
    assert fresh_session
    fresh_hash = database.hash_secret(fresh_session)
    before = database.get_session_by_hash_for_test(fresh_hash)["last_seen_at"]
    assert fresh_client.get("/auth/me").status_code == 200
    after = database.get_session_by_hash_for_test(fresh_hash)["last_seen_at"]
    assert after >= before


@pytest.mark.parametrize(
    "env_value",
    ["production", "staging", "", "prod", "Production "],
)
def test_dev_auth_endpoints_are_unavailable_outside_explicit_dev_or_test(
    auth_client, monkeypatch, env_value
):
    client, _database = auth_client
    monkeypatch.setenv("CLOUDA_ENV", env_value)
    monkeypatch.setenv("CLOUDA_AUTH_VERIFIER", "fake")

    response = client.post(
        "/auth/dev/register",
        json={"email": "blocked@example.com", "password": TEST_PASSWORD},
    )

    assert response.status_code == 404


def test_session_exchange_rotates_existing_cookie_and_blocks_fixation(auth_client):
    client, _database = auth_client
    first_payload, _first_csrf = login(client, fake_token("alice-provider-id"))
    old_session = client.cookies.get("clouda_session")
    assert old_session

    second_payload, _second_csrf = login(client, fake_token("alice-provider-id"))
    new_session = client.cookies.get("clouda_session")

    assert second_payload["csrf_token"] != first_payload["csrf_token"]
    assert new_session and new_session != old_session

    stale_client = TestClient(app)
    stale_client.cookies.set(
        "clouda_session", old_session, domain="testserver.local", path="/"
    )
    assert stale_client.get("/auth/me").status_code == 401

    fixed_client = TestClient(app)
    fixed_client.cookies.set(
        "clouda_session", "attacker-fixed-session", domain="testserver.local", path="/"
    )
    fixed_payload, _fixed_csrf = login(
        fixed_client, fake_token("fixed-user", "fixed@example.com")
    )
    assert fixed_client.cookies.get("clouda_session") != "attacker-fixed-session"
    assert fixed_payload["user"]["email"] == "fixed@example.com"


def test_disabled_and_deleted_users_cannot_use_existing_sessions(auth_client):
    client, database = auth_client
    payload, _csrf = login(client, fake_token("alice-provider-id"))
    user_id = payload["user"]["user_id"]

    database.set_auth_user_status(user_id, "disabled", actor_user_id="system")
    assert client.get("/auth/me").status_code == 401

    database.set_auth_user_status(user_id, "deleted", actor_user_id="system")
    assert client.get("/auth/me").status_code == 401


def test_user_deletion_request_revokes_session_and_records_audit(auth_client):
    client, database = auth_client
    payload, csrf = login(client, fake_token("alice-provider-id"))

    deletion = client.post("/auth/delete-account", headers=csrf)

    assert deletion.status_code == 200
    assert deletion.json()["status"] == "deletion_pending"
    assert client.get("/auth/me").status_code == 401
    user = database.get_auth_user(payload["user"]["user_id"])
    assert user is not None
    assert user["status"] == "deletion_pending"
    assert any(
        event["event_type"] == "account_deletion_requested"
        for event in database.list_auth_audit_events()
    )


def test_admin_bootstrap_and_role_changes_revoke_sessions(auth_client, monkeypatch):
    client, database = auth_client
    monkeypatch.setenv("CLOUDA_FIRST_ADMIN_BOOTSTRAP_TOKEN", "bootstrap-secret")
    admin, admin_csrf = login(client, fake_token("admin-provider", "admin@example.com"))

    bootstrap = client.post(
        "/admin/bootstrap",
        headers=admin_csrf | {"X-First-Admin-Token": "bootstrap-secret"},
    )
    assert bootstrap.status_code == 200
    assert bootstrap.json()["role"] == "admin"
    assert client.get("/admin/users").status_code == 401

    _admin, admin_csrf = login(
        client, fake_token("admin-provider", "admin@example.com")
    )
    other_client = TestClient(app)
    user, _user_csrf = login(
        other_client, fake_token("user-provider", "user@example.com")
    )
    promote = client.post(
        f"/admin/users/{user['user']['user_id']}/role",
        headers=admin_csrf,
        json={"role": "admin"},
    )
    assert promote.status_code == 200
    assert other_client.get("/auth/me").status_code == 401
    assert any(
        event["event_type"] == "role_changed"
        for event in database.list_auth_audit_events()
    )


def test_user_upload_quota_survives_session_rotation(auth_client, monkeypatch):
    client, _database = auth_client
    monkeypatch.setenv("CLOUDA_USER_UPLOADS_PER_MINUTE", "1")
    _payload, csrf = login(client, fake_token("quota-user", "quota@example.com"))

    first = client.post(
        "/user/documents",
        headers=csrf,
        json={"original_pdf_name": "first.pdf", "page_count": 1},
    )
    assert first.status_code == 201, first.text

    _payload, csrf = login(client, fake_token("quota-user", "quota@example.com"))
    second = client.post(
        "/user/documents",
        headers=csrf | {"X-Forwarded-For": "203.0.113.123"},
        json={"original_pdf_name": "second.pdf", "page_count": 1},
    )
    assert second.status_code == 429
    assert second.headers["Retry-After"].isdigit()


def test_separate_users_have_isolated_upload_quotas(auth_client, monkeypatch):
    client, _database = auth_client
    monkeypatch.setenv("CLOUDA_USER_UPLOADS_PER_MINUTE", "1")
    _alice, alice_csrf = login(client, fake_token("alice-quota", "alice@example.com"))
    assert (
        client.post(
            "/user/documents",
            headers=alice_csrf,
            json={"original_pdf_name": "alice.pdf", "page_count": 1},
        ).status_code
        == 201
    )
    bob = TestClient(app)
    _bob, bob_csrf = login(bob, fake_token("bob-quota", "bob@example.com"))
    assert (
        bob.post(
            "/user/documents",
            headers=bob_csrf,
            json={"original_pdf_name": "bob.pdf", "page_count": 1},
        ).status_code
        == 201
    )


def test_admin_action_quota_covers_role_changes(auth_client, monkeypatch):
    client, _database = auth_client
    monkeypatch.setenv("CLOUDA_FIRST_ADMIN_BOOTSTRAP_TOKEN", "bootstrap-secret")
    monkeypatch.setenv("CLOUDA_ADMIN_ACTIONS_PER_MINUTE", "1")
    _admin, admin_csrf = login(client, fake_token("admin-quota", "admin@example.com"))
    assert (
        client.post(
            "/admin/bootstrap",
            headers=admin_csrf | {"X-First-Admin-Token": "bootstrap-secret"},
        ).status_code
        == 200
    )
    _admin, admin_csrf = login(client, fake_token("admin-quota", "admin@example.com"))
    user_client = TestClient(app)
    user, _user_csrf = login(
        user_client, fake_token("target-quota", "target@example.com")
    )

    response = client.post(
        f"/admin/users/{user['user']['user_id']}/role",
        headers=admin_csrf,
        json={"role": "admin"},
    )

    assert response.status_code == 429


def test_state_changing_session_routes_require_csrf_or_non_cookie_auth():
    browser_exempt = {
        "/auth/session",
        "/auth/streamlit/bridge",
        "/guest/session",
    }
    dev_routes = {
        "/auth/dev/register",
        "/auth/dev/verify-email",
        "/auth/dev/login",
        "/auth/dev/google-login",
        "/auth/dev/password-reset/start",
        "/auth/dev/password-reset/complete",
    }
    non_cookie_auth = {
        route.path for route in app.routes if route.path.startswith("/internal/")
    }
    missing = []
    for route in app.routes:
        methods = getattr(route, "methods", set()) or set()
        if methods.isdisjoint({"POST", "PUT", "PATCH", "DELETE"}):
            continue
        path = route.path
        if path in browser_exempt or path in dev_routes or path in non_cookie_auth:
            continue
        dependency_names = {
            dependency.call.__name__
            for dependency in getattr(route, "dependant", ()).dependencies
            if getattr(dependency, "call", None) is not None
        }
        if dependency_names.isdisjoint({"_require_csrf", "_require_guest_csrf"}):
            missing.append(path)

    assert missing == []
