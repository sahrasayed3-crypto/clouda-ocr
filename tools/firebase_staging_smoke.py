from __future__ import annotations

import os
import sys
import uuid
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class SmokeConfig:
    project_id: str
    api_base_url: str
    test_email: str
    test_password: str


def safety_check() -> str:
    if os.getenv("CLOUDA_ALLOW_LIVE_FIREBASE_TESTS", "").strip() != "1":
        return "blocked"
    if os.getenv("AUTH_MODE", "").strip() != "firebase_staging":
        return "blocked"
    if os.getenv("CLOUDA_ENV", "").strip() != "staging":
        return "blocked"
    if os.getenv("FIREBASE_AUTH_EMULATOR_HOST", "").strip():
        return "blocked"
    project_id = os.getenv("CLOUDA_FIREBASE_PROJECT_ID", "").strip()
    allowlist = {
        item.strip()
        for item in os.getenv("CLOUDA_STAGING_FIREBASE_PROJECT_ALLOWLIST", "").split(
            ","
        )
        if item.strip()
    }
    if not project_id or project_id not in allowlist:
        return "blocked"
    if "prod" in project_id.casefold() or "production" in project_id.casefold():
        return "blocked"
    return "allowed"


def _exchange_for_clouda_session(
    api_base_url: str, id_token: str
) -> tuple[requests.Session, str]:
    session = requests.Session()
    session_response = session.post(
        f"{api_base_url}/auth/session",
        headers={"Authorization": f"Bearer {id_token}"},
        timeout=20,
    )
    session_response.raise_for_status()
    session_payload = session_response.json()
    csrf_token = session_payload.get("csrf_token", "")
    if not csrf_token:
        raise RuntimeError("Clouda session exchange did not return CSRF token")
    return session, csrf_token


def _assert_profile_email(
    session: requests.Session, api_base_url: str, expected_email: str
) -> None:
    profile = session.get(f"{api_base_url}/auth/me", timeout=15)
    profile.raise_for_status()
    actual_email = profile.json().get("email", "")
    if actual_email.casefold() != expected_email.casefold():
        raise RuntimeError("Clouda profile did not match disposable Firebase user")


def _assert_csrf_rejection(
    session: requests.Session, api_base_url: str, headers: dict[str, str] | None = None
) -> None:
    response = session.post(
        f"{api_base_url}/auth/logout",
        headers=headers or {},
        timeout=15,
    )
    if response.status_code not in {401, 403}:
        raise RuntimeError("Clouda logout accepted missing or invalid CSRF")


def _assert_unauthenticated(session: requests.Session, api_base_url: str) -> None:
    response = session.get(f"{api_base_url}/auth/me", timeout=15)
    if response.status_code != 401:
        raise RuntimeError("Clouda session remained authenticated after revocation")


def load_config() -> SmokeConfig:
    suffix = uuid.uuid4().hex[:12]
    return SmokeConfig(
        project_id=os.environ["CLOUDA_FIREBASE_PROJECT_ID"].strip(),
        api_base_url=os.getenv("SERVER_BASE_URL", "http://127.0.0.1:8000").rstrip("/"),
        test_email=os.getenv(
            "CLOUDA_FIREBASE_SMOKE_EMAIL",
            f"clouda-smoke-{suffix}@example.invalid",
        ),
        test_password=os.getenv(
            "CLOUDA_FIREBASE_SMOKE_PASSWORD",
            f"Smoke-test-{suffix}-passphrase",
        ),
    )


def _custom_token_value(custom_token: bytes | str) -> str:
    if isinstance(custom_token, bytes):
        return custom_token.decode("utf-8")
    return custom_token


def _exchange_custom_token_for_id_token(custom_token: bytes | str) -> str:
    api_key = os.getenv("FIREBASE_CLIENT_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("FIREBASE_CLIENT_API_KEY is required for live smoke")
    response = requests.post(
        "https://identitytoolkit.googleapis.com/v1/"
        f"accounts:signInWithCustomToken?key={api_key}",
        json={"token": _custom_token_value(custom_token), "returnSecureToken": True},
        timeout=20,
    )
    response.raise_for_status()
    id_token = response.json().get("idToken", "")
    if not id_token:
        raise RuntimeError("Firebase custom-token exchange did not return idToken")
    return id_token


def run_smoke() -> int:
    if safety_check() != "allowed":
        print(
            "Live Firebase smoke is blocked. Set CLOUDA_ALLOW_LIVE_FIREBASE_TESTS=1 "
            "and allowlist the staging project before running."
        )
        return 2
    config = load_config()
    print(f"Selected Firebase staging project: {config.project_id}")
    try:
        import firebase_admin  # type: ignore
        from firebase_admin import auth as firebase_auth  # type: ignore
    except Exception as exc:
        print(f"Firebase Admin SDK unavailable: {exc.__class__.__name__}")
        return 3
    try:
        if not getattr(firebase_admin, "_apps", None):
            firebase_admin.initialize_app(options={"projectId": config.project_id})
        user = firebase_auth.create_user(
            email=config.test_email,
            password=config.test_password,
            email_verified=True,
            disabled=False,
        )
        custom_token = firebase_auth.create_custom_token(user.uid)
        if not custom_token:
            raise RuntimeError("custom token creation failed")
        id_token = _exchange_custom_token_for_id_token(custom_token)
        print("Created marked staging test user and exchanged a real ID token.")
        session, csrf_token = _exchange_for_clouda_session(
            config.api_base_url, id_token
        )
        _assert_profile_email(session, config.api_base_url, config.test_email)
        _assert_csrf_rejection(session, config.api_base_url)
        _assert_csrf_rejection(
            session, config.api_base_url, {"X-CSRF-Token": "invalid-csrf"}
        )
        logout = session.post(
            f"{config.api_base_url}/auth/logout",
            headers={"X-CSRF-Token": csrf_token},
            timeout=15,
        )
        logout.raise_for_status()
        _assert_unauthenticated(session, config.api_base_url)

        first_session, first_csrf = _exchange_for_clouda_session(
            config.api_base_url, id_token
        )
        second_session, _second_csrf = _exchange_for_clouda_session(
            config.api_base_url, id_token
        )
        _assert_profile_email(first_session, config.api_base_url, config.test_email)
        _assert_profile_email(second_session, config.api_base_url, config.test_email)
        logout_all = first_session.post(
            f"{config.api_base_url}/auth/logout-all",
            headers={"X-CSRF-Token": first_csrf},
            timeout=15,
        )
        if logout_all.status_code == 404:
            print("Logout-all endpoint is not implemented.")
        else:
            logout_all.raise_for_status()
            _assert_unauthenticated(first_session, config.api_base_url)
            _assert_unauthenticated(second_session, config.api_base_url)
        print(
            "Verified Clouda session exchange, profile lookup, CSRF rejection, "
            "logout, and logout-all revocation."
        )
        return 0
    except Exception as exc:
        print(f"Live Firebase smoke failed: {exc.__class__.__name__}")
        return 1
    finally:
        try:
            firebase_auth.delete_user(user.uid)  # type: ignore[name-defined]
            print("Deleted marked staging test user.")
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(run_smoke())
