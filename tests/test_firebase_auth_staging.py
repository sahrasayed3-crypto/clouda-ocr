from __future__ import annotations

import builtins
from pathlib import Path
import tomllib

import pytest
from fastapi.testclient import TestClient

from pdfword import worker_api
from pdfword.auth import (
    DisabledIdentityError,
    FakeIdentityVerifier,
    FirebaseIdentityVerifier,
    WrongProjectIdentityError,
    auth_mode_from_env,
    firebase_client_config_from_env,
    identity_verifier_from_env,
)
from pdfword.database import Database
from pdfword.worker_api import app


@pytest.fixture
def firebase_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLOUDA_DATABASE_PATH", str(tmp_path / "firebase.sqlite3"))
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("CLOUDA_SESSION_COOKIE_SECURE", "false")
    monkeypatch.setenv("CLOUDA_REQUIRE_VERIFIED_EMAIL", "true")
    worker_api._fake_auth_emulators.clear()
    return TestClient(app), Database(tmp_path / "firebase.sqlite3")


def test_auth_mode_maps_local_emulator_and_staging(monkeypatch: pytest.MonkeyPatch):
    class FirebaseAuth:
        @staticmethod
        def verify_id_token(*_args, **_kwargs):
            return {}

    class FirebaseAdmin:
        _apps: list[object] = []
        auth = FirebaseAuth

        @staticmethod
        def initialize_app(*_args, **_kwargs):
            FirebaseAdmin._apps.append(object())

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "firebase_admin":
            return FirebaseAdmin
        if name == "firebase_admin.auth":
            return FirebaseAuth
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setenv("AUTH_MODE", "local")
    monkeypatch.setenv("CLOUDA_ENV", "development")
    monkeypatch.setenv("CLOUDA_FIREBASE_PROJECT_ID", "local-test-project")
    assert auth_mode_from_env() == "local"
    assert isinstance(identity_verifier_from_env(), FakeIdentityVerifier)

    monkeypatch.setenv("AUTH_MODE", "firebase_emulator")
    monkeypatch.setenv("FIREBASE_AUTH_EMULATOR_HOST", "127.0.0.1:9099")
    assert auth_mode_from_env() == "firebase_emulator"
    assert isinstance(identity_verifier_from_env(), FirebaseIdentityVerifier)

    monkeypatch.setenv("AUTH_MODE", "firebase_staging")
    monkeypatch.setenv("CLOUDA_ENV", "staging")
    monkeypatch.setenv("CLOUDA_FIREBASE_PROJECT_ID", "clouda-staging")
    monkeypatch.setenv("CLOUDA_STAGING_FIREBASE_PROJECT_ALLOWLIST", "clouda-staging")
    assert auth_mode_from_env() == "firebase_staging"


@pytest.mark.parametrize("mode", ["", "staging", "firebase_stage", "production"])
def test_unknown_auth_modes_fail_closed(monkeypatch: pytest.MonkeyPatch, mode: str):
    monkeypatch.setenv("AUTH_MODE", mode)
    with pytest.raises(RuntimeError, match="AUTH_MODE"):
        auth_mode_from_env()


def test_firebase_client_config_returns_only_public_web_values(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AUTH_MODE", "firebase_staging")
    monkeypatch.setenv("CLOUDA_ENV", "staging")
    monkeypatch.setenv("CLOUDA_FIREBASE_PROJECT_ID", "clouda-server")
    monkeypatch.setenv("CLOUDA_STAGING_FIREBASE_PROJECT_ALLOWLIST", "clouda-staging")
    monkeypatch.setenv("FIREBASE_CLIENT_API_KEY", "public-api-key")
    monkeypatch.setenv("FIREBASE_CLIENT_AUTH_DOMAIN", "clouda-staging.firebaseapp.com")
    monkeypatch.setenv("FIREBASE_CLIENT_PROJECT_ID", "clouda-staging")
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT_JSON_PATH", "F:/secrets/key.json")

    config = firebase_client_config_from_env()

    assert config == {
        "apiKey": "public-api-key",  # pragma: allowlist secret
        "authDomain": "clouda-staging.firebaseapp.com",
        "projectId": "clouda-staging",
    }
    assert "service" not in repr(config).casefold()
    assert "private" not in repr(config).casefold()


def test_firebase_staging_client_project_must_match_allowlist(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AUTH_MODE", "firebase_staging")
    monkeypatch.setenv("CLOUDA_ENV", "staging")
    monkeypatch.setenv("CLOUDA_FIREBASE_PROJECT_ID", "clouda-staging")
    monkeypatch.setenv("CLOUDA_STAGING_FIREBASE_PROJECT_ALLOWLIST", "clouda-staging")
    monkeypatch.setenv("FIREBASE_CLIENT_PROJECT_ID", "clouda-other-staging")
    monkeypatch.setenv("FIREBASE_CLIENT_API_KEY", "public-api-key")
    monkeypatch.setenv(
        "FIREBASE_CLIENT_AUTH_DOMAIN", "clouda-other-staging.firebaseapp.com"
    )

    with pytest.raises(RuntimeError, match="allowlist"):
        firebase_client_config_from_env()


def test_firebase_admin_rejects_wrong_issuer_and_disabled_user(
    monkeypatch: pytest.MonkeyPatch,
):
    class FirebaseUser:
        disabled = False

    class FirebaseAuth:
        claims = {
            "aud": "clouda-staging",
            "iss": "https://securetoken.google.com/wrong-project",
            "uid": "firebase-user",
            "sub": "firebase-user",
            "email": "alice@example.com",
            "email_verified": True,
            "firebase": {"sign_in_provider": "google.com"},
        }

        @classmethod
        def verify_id_token(cls, *_args, **_kwargs):
            return dict(cls.claims)

        @staticmethod
        def get_user(_uid):
            return FirebaseUser()

    class FirebaseCredentials:
        class Certificate:
            def __init__(self, path):
                self.path = path

    class FirebaseAdmin:
        _apps: list[object] = []
        auth = FirebaseAuth
        credentials = FirebaseCredentials

        @staticmethod
        def initialize_app(*_args, **_kwargs):
            FirebaseAdmin._apps.append(object())

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "firebase_admin":
            return FirebaseAdmin
        if name == "firebase_admin.auth":
            return FirebaseAuth
        if name == "firebase_admin.credentials":
            return FirebaseCredentials
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    verifier = FirebaseIdentityVerifier("clouda-staging")

    with pytest.raises(WrongProjectIdentityError):
        verifier.verify_bearer_token("token")

    FirebaseAuth.claims["iss"] = "https://securetoken.google.com/clouda-staging"
    FirebaseUser.disabled = True
    with pytest.raises(DisabledIdentityError):
        verifier.verify_bearer_token("token")


def test_public_firebase_config_endpoint_returns_only_web_values(
    firebase_client, monkeypatch: pytest.MonkeyPatch
):
    client, _database = firebase_client
    monkeypatch.setenv("AUTH_MODE", "firebase_staging")
    monkeypatch.setenv("CLOUDA_ENV", "staging")
    monkeypatch.setenv("CLOUDA_FIREBASE_PROJECT_ID", "clouda-staging")
    monkeypatch.setenv("CLOUDA_STAGING_FIREBASE_PROJECT_ALLOWLIST", "clouda-staging")
    monkeypatch.setenv("FIREBASE_CLIENT_API_KEY", "public-api-key")
    monkeypatch.setenv("FIREBASE_CLIENT_AUTH_DOMAIN", "clouda-staging.firebaseapp.com")

    response = client.get("/auth/firebase/config")

    assert response.status_code == 200
    assert response.json()["mode"] == "firebase_staging"
    assert response.json()["config"]["projectId"] == "clouda-staging"
    assert "service" not in response.text.casefold()
    assert "private" not in response.text.casefold()


def test_public_firebase_config_endpoint_hidden_in_local_mode(
    firebase_client, monkeypatch: pytest.MonkeyPatch
):
    client, _database = firebase_client
    monkeypatch.setenv("AUTH_MODE", "local")
    monkeypatch.setenv("CLOUDA_ENV", "development")

    assert client.get("/auth/firebase/config").status_code == 404


def test_streamlit_bridge_token_is_single_use_and_returns_session(
    firebase_client, monkeypatch: pytest.MonkeyPatch
):
    client, _database = firebase_client
    monkeypatch.setenv("AUTH_MODE", "local")
    monkeypatch.setenv("CLOUDA_ENV", "development")
    monkeypatch.setenv("CLOUDA_AUTH_VERIFIER", "fake")
    client.post(
        "/auth/dev/register",
        json={
            "email": "bridge@example.com",
            "password": "password-123",  # pragma: allowlist secret
        },
    )
    client.post("/auth/dev/verify-email", json={"email": "bridge@example.com"})

    login = client.post(
        "/auth/dev/login",
        headers={"X-Clouda-Streamlit-Bridge": "1"},
        json={
            "email": "bridge@example.com",
            "password": "password-123",  # pragma: allowlist secret
        },
    )

    assert login.status_code == 200
    bridge_token = login.json()["bridge_token"]
    bridge = client.post("/auth/streamlit/bridge", json={"bridge_token": bridge_token})
    assert bridge.status_code == 200
    assert bridge.json()["session_id"]
    assert bridge.json()["csrf_token"] == login.json()["csrf_token"]
    assert bridge.json()["user"]["email"] == "bridge@example.com"
    assert (
        client.post(
            "/auth/streamlit/bridge", json={"bridge_token": bridge_token}
        ).status_code
        == 401
    )


def test_streamlit_contains_firebase_web_sdk_helper():
    app_source = Path("app.py").read_text(encoding="utf-8")
    component_source = Path("pdfword/streamlit_firebase_auth/index.html").read_text(
        encoding="utf-8"
    )
    source = app_source + "\n" + component_source

    assert "/auth/firebase/config" in app_source
    assert "declare_component" in app_source
    assert "firebase-auth.js" in component_source
    assert "/auth/session" in component_source
    assert "X-Clouda-Streamlit-Bridge" in component_source
    assert "clouda_auth_bridge" in source
    assert "_consume_firebase_auth_bridge" in app_source
    assert "Streamlit.setComponentValue" in component_source
    assert "window.parent.location.href" not in source
    assert 'searchParams.set("clouda_auth_bridge"' not in source
    assert "sendPasswordResetEmail" in component_source
    assert "sendEmailVerification" in component_source


def test_firebase_streamlit_component_is_packaged():
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    package_data = config["tool"]["setuptools"]["package-data"]

    assert "streamlit_firebase_auth/index.html" in package_data["pdfword"]


def test_streamlit_server_logout_uses_bridged_session_state():
    source = Path("app.py").read_text(encoding="utf-8")
    assert "/auth/logout" in source
    assert "firebase_logout_requested" in source
    assert 'st.session_state.get("firebase_logout_requested")' in source
    assert '_api_request("POST", "/auth/logout", headers=_api_headers(True))' in source


def test_firebase_staging_requires_allowlisted_runtime_project(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AUTH_MODE", "firebase_staging")
    monkeypatch.setenv("CLOUDA_ENV", "staging")
    monkeypatch.setenv("CLOUDA_FIREBASE_PROJECT_ID", "clouda-staging")
    monkeypatch.delenv("CLOUDA_STAGING_FIREBASE_PROJECT_ALLOWLIST", raising=False)

    with pytest.raises(RuntimeError, match="allowlist"):
        identity_verifier_from_env()

    monkeypatch.setenv("CLOUDA_STAGING_FIREBASE_PROJECT_ALLOWLIST", "other-staging")
    with pytest.raises(RuntimeError, match="allowlist"):
        identity_verifier_from_env()


def test_firebase_staging_rejects_production_named_project(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AUTH_MODE", "firebase_staging")
    monkeypatch.setenv("CLOUDA_ENV", "staging")
    monkeypatch.setenv("CLOUDA_FIREBASE_PROJECT_ID", "clouda-production")
    monkeypatch.setenv("CLOUDA_STAGING_FIREBASE_PROJECT_ALLOWLIST", "clouda-production")

    with pytest.raises(RuntimeError, match="production"):
        identity_verifier_from_env()


def test_firebase_production_requires_allowlisted_runtime_project(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AUTH_MODE", "firebase_staging")
    monkeypatch.setenv("CLOUDA_ENV", "production")
    monkeypatch.setenv("CLOUDA_FIREBASE_PROJECT_ID", "clouda-prod-main")
    monkeypatch.delenv("CLOUDA_PRODUCTION_FIREBASE_PROJECT_ALLOWLIST", raising=False)

    with pytest.raises(RuntimeError, match="production project ID"):
        identity_verifier_from_env()

    monkeypatch.setenv(
        "CLOUDA_PRODUCTION_FIREBASE_PROJECT_ALLOWLIST", "clouda-other-prod"
    )
    with pytest.raises(RuntimeError, match="production project ID"):
        identity_verifier_from_env()


def test_firebase_staging_rejects_emulator_host(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUTH_MODE", "firebase_staging")
    monkeypatch.setenv("CLOUDA_ENV", "staging")
    monkeypatch.setenv("CLOUDA_FIREBASE_PROJECT_ID", "clouda-staging")
    monkeypatch.setenv("CLOUDA_STAGING_FIREBASE_PROJECT_ALLOWLIST", "clouda-staging")
    monkeypatch.setenv("FIREBASE_AUTH_EMULATOR_HOST", "127.0.0.1:9099")

    with pytest.raises(RuntimeError, match="emulator"):
        identity_verifier_from_env()


def test_firebase_emulator_requires_emulator_host(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUTH_MODE", "firebase_emulator")
    monkeypatch.setenv("CLOUDA_ENV", "development")
    monkeypatch.setenv("CLOUDA_FIREBASE_PROJECT_ID", "local-test-project")
    monkeypatch.delenv("FIREBASE_AUTH_EMULATOR_HOST", raising=False)

    with pytest.raises(RuntimeError, match="FIREBASE_AUTH_EMULATOR_HOST"):
        identity_verifier_from_env()


def test_firebase_emulator_config_exposes_emulator_host(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AUTH_MODE", "firebase_emulator")
    monkeypatch.setenv("CLOUDA_ENV", "development")
    monkeypatch.setenv("CLOUDA_FIREBASE_PROJECT_ID", "local-test-project")
    monkeypatch.setenv("FIREBASE_CLIENT_API_KEY", "emulator-public-key")
    monkeypatch.setenv("FIREBASE_CLIENT_AUTH_DOMAIN", "localhost")
    monkeypatch.setenv("FIREBASE_AUTH_EMULATOR_HOST", "127.0.0.1:9099")

    config = firebase_client_config_from_env()

    assert config["authEmulatorHost"] == "http://127.0.0.1:9099"


def test_live_smoke_refuses_without_safety_variable(monkeypatch: pytest.MonkeyPatch):
    from tools.firebase_staging_smoke import safety_check

    monkeypatch.delenv("CLOUDA_ALLOW_LIVE_FIREBASE_TESTS", raising=False)

    assert safety_check() == "blocked"


def test_live_smoke_exchanges_custom_token_for_id_token_and_session(
    monkeypatch: pytest.MonkeyPatch,
):
    from tools import firebase_staging_smoke

    calls: list[tuple[str, str, dict]] = []

    class FirebaseAuth:
        @staticmethod
        def create_user(**_kwargs):
            return type("User", (), {"uid": "firebase-user"})()

        @staticmethod
        def create_custom_token(uid):
            assert uid == "firebase-user"
            return b"custom-token"

        @staticmethod
        def delete_user(uid):
            assert uid == "firebase-user"

    class FirebaseAdmin:
        _apps: list[object] = []
        auth = FirebaseAuth

        @staticmethod
        def initialize_app(**_kwargs):
            FirebaseAdmin._apps.append(object())

    def fake_import(name, *args, **kwargs):
        if name == "firebase_admin":
            return FirebaseAdmin
        if name == "firebase_admin.auth":
            return FirebaseAuth
        return real_import(name, *args, **kwargs)

    class FakeResponse:
        def __init__(self, status_code=200, payload=None):
            self.status_code = status_code
            self._payload = payload or {}
            self.text = str(self._payload)

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.text)

        def json(self):
            return self._payload

    class FakeSession:
        next_id = 0

        def __init__(self):
            type(self).next_id += 1
            self.session_id = type(self).next_id
            self.headers = {}
            self.logged_out = False
            self.logout_all = False

        def post(self, url, **kwargs):
            calls.append(("POST", url, kwargs))
            if url.endswith("/auth/session"):
                assert kwargs["headers"]["Authorization"] == "Bearer id-token"
                return FakeResponse(
                    payload={
                        "csrf_token": f"csrf-{self.session_id}",
                        "user": {"email": "clouda-smoke@example.invalid"},
                    }
                )
            if url.endswith("/auth/logout"):
                csrf = kwargs.get("headers", {}).get("X-CSRF-Token", "")
                if csrf != f"csrf-{self.session_id}":
                    return FakeResponse(status_code=403, payload={"detail": "CSRF"})
                self.logged_out = True
                return FakeResponse(payload={"status": "ok"})
            if url.endswith("/auth/logout-all"):
                assert kwargs["headers"]["X-CSRF-Token"] == f"csrf-{self.session_id}"
                for session in fake_sessions:
                    session.logout_all = True
                return FakeResponse(payload={"status": "ok"})
            raise AssertionError(url)

        def get(self, url, **kwargs):
            calls.append(("GET", url, kwargs))
            if url.endswith("/auth/me"):
                if self.logged_out or self.logout_all:
                    return FakeResponse(
                        status_code=401, payload={"detail": "Authentication required"}
                    )
                return FakeResponse(payload={"email": "clouda-smoke@example.invalid"})
            raise AssertionError(url)

    fake_sessions: list[FakeSession] = []

    def fake_session_factory():
        session = FakeSession()
        fake_sessions.append(session)
        return session

    def fake_post(url, **kwargs):
        calls.append(("POST", url, kwargs))
        assert "accounts:signInWithCustomToken" in url
        assert kwargs["json"]["token"] == "custom-token"
        return FakeResponse(payload={"idToken": "id-token"})

    real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(firebase_staging_smoke.requests, "post", fake_post)
    monkeypatch.setattr(
        firebase_staging_smoke.requests, "Session", fake_session_factory
    )
    monkeypatch.setenv("CLOUDA_ALLOW_LIVE_FIREBASE_TESTS", "1")
    monkeypatch.setenv("AUTH_MODE", "firebase_staging")
    monkeypatch.setenv("CLOUDA_ENV", "staging")
    monkeypatch.setenv("CLOUDA_FIREBASE_PROJECT_ID", "clouda-staging")
    monkeypatch.setenv("CLOUDA_STAGING_FIREBASE_PROJECT_ALLOWLIST", "clouda-staging")
    monkeypatch.setenv("FIREBASE_CLIENT_API_KEY", "public-api-key")
    monkeypatch.setenv("CLOUDA_FIREBASE_SMOKE_EMAIL", "clouda-smoke@example.invalid")

    assert firebase_staging_smoke.run_smoke() == 0
    assert [call[1].rsplit("/", 1)[-1] for call in calls] == [
        "accounts:signInWithCustomToken?key=public-api-key",
        "session",
        "me",
        "logout",
        "logout",
        "logout",
        "me",
        "session",
        "session",
        "me",
        "me",
        "logout-all",
        "me",
        "me",
    ]
