from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from typing import Protocol

from .operations import structured_log


class IdentityVerificationError(ValueError):
    pass


class InvalidIdentityError(IdentityVerificationError):
    pass


class ExpiredIdentityError(IdentityVerificationError):
    pass


class RevokedIdentityError(IdentityVerificationError):
    pass


class WrongProjectIdentityError(IdentityVerificationError):
    pass


class DisabledIdentityError(IdentityVerificationError):
    pass


@dataclass(frozen=True)
class VerifiedIdentity:
    provider_user_id: str
    email: str | None
    email_verified: bool
    provider: str
    display_name: str | None = None


class IdentityVerifier(Protocol):
    def verify_bearer_token(self, token: str) -> VerifiedIdentity: ...


VALID_AUTH_MODES = {"local", "firebase_emulator", "firebase_staging"}


def _env_false(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"0", "false", "no", "off"}


def auth_mode_from_env() -> str:
    raw_mode = os.getenv("AUTH_MODE")
    if raw_mode is not None:
        mode = raw_mode.strip().lower()
        if mode not in VALID_AUTH_MODES:
            raise RuntimeError(
                "AUTH_MODE must be local, firebase_emulator, or firebase_staging"
            )
        return mode

    legacy = os.getenv("CLOUDA_AUTH_VERIFIER", "").strip().lower()
    if legacy == "fake":
        return "local"
    if legacy == "firebase":
        return "firebase_staging"
    raise RuntimeError(
        "AUTH_MODE must be local, firebase_emulator, or firebase_staging"
    )


def _firebase_server_project_id_from_env() -> str:
    return (
        os.getenv("CLOUDA_FIREBASE_PROJECT_ID", "").strip()
        or os.getenv("FIREBASE_PROJECT_ID", "").strip()
    )


def _firebase_client_project_id_from_env() -> str:
    return (
        os.getenv("FIREBASE_CLIENT_PROJECT_ID", "").strip()
        or os.getenv("CLOUDA_FIREBASE_PROJECT_ID", "").strip()
        or os.getenv("FIREBASE_PROJECT_ID", "").strip()
    )


def _is_placeholder_project(project_id: str) -> bool:
    value = project_id.strip().casefold()
    return not value or value in {
        "local-test-project",
        "your-project-id",
        "your-firebase-project-id",
        "replace-me",
        "placeholder",
    }


def firebase_client_config_from_env() -> dict[str, str]:
    mode = auth_mode_from_env()
    if mode == "local":
        raise RuntimeError("Firebase client config is unavailable in local auth mode")
    project_id = _firebase_client_project_id_from_env()
    api_key = os.getenv("FIREBASE_CLIENT_API_KEY", "").strip()
    auth_domain = os.getenv("FIREBASE_CLIENT_AUTH_DOMAIN", "").strip()
    if not project_id or not api_key or not auth_domain:
        raise RuntimeError("Firebase client configuration is incomplete")
    config = {
        "apiKey": api_key,
        "authDomain": auth_domain,
        "projectId": project_id,
    }
    if (
        mode == "firebase_staging"
        and os.getenv("CLOUDA_ENV", "").strip().lower() == "staging"
    ):
        _validate_staging_project(project_id)
    if mode == "firebase_emulator":
        emulator_host = _firebase_emulator_origin_from_env()
        if not emulator_host:
            raise RuntimeError(
                "FIREBASE_AUTH_EMULATOR_HOST is required in firebase_emulator mode"
            )
        config["authEmulatorHost"] = emulator_host
    return config


def _firebase_emulator_origin_from_env() -> str:
    raw_host = os.getenv("FIREBASE_AUTH_EMULATOR_HOST", "").strip()
    if not raw_host:
        return ""
    if raw_host.startswith(("http://", "https://")):
        return raw_host.rstrip("/")
    return f"http://{raw_host.rstrip('/')}"


def _staging_project_allowlist() -> set[str]:
    return {
        item.strip()
        for item in os.getenv("CLOUDA_STAGING_FIREBASE_PROJECT_ALLOWLIST", "").split(
            ","
        )
        if item.strip()
    }


def _validate_staging_project(project_id: str) -> None:
    if _is_placeholder_project(project_id):
        raise RuntimeError("Firebase staging requires a non-placeholder project ID")
    if "prod" in project_id.casefold() or "production" in project_id.casefold():
        raise RuntimeError("Firebase staging project ID must not look like production")
    allowlist = _staging_project_allowlist()
    if not allowlist or project_id not in allowlist:
        raise RuntimeError(
            "Firebase staging project ID must be present in allowlist "
            "CLOUDA_STAGING_FIREBASE_PROJECT_ALLOWLIST"
        )


def _production_project_allowlist() -> set[str]:
    return {
        item.strip()
        for item in os.getenv("CLOUDA_PRODUCTION_FIREBASE_PROJECT_ALLOWLIST", "").split(
            ","
        )
        if item.strip()
    }


def _validate_production_project(project_id: str) -> None:
    if _is_placeholder_project(project_id):
        raise RuntimeError("Firebase production requires a non-placeholder project ID")
    allowlist = _production_project_allowlist()
    if not allowlist or project_id not in allowlist:
        raise RuntimeError(
            "Firebase production project ID must be present in allowlist "
            "CLOUDA_PRODUCTION_FIREBASE_PROJECT_ALLOWLIST"
        )


class FakeAuthEmulator:
    """In-memory development auth provider used only with CLOUDA_AUTH_VERIFIER=fake."""

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self._users: dict[str, dict] = {}
        self._reset_tokens: dict[str, str] = {}

    @staticmethod
    def _normalize_email(email: str) -> str:
        clean = email.strip().casefold()
        if "@" not in clean:
            raise InvalidIdentityError("Invalid email")
        return clean

    @staticmethod
    def _hash_password(password: str) -> str:
        if len(password) < 8:
            raise InvalidIdentityError("Password must be at least 8 characters")
        import hashlib

        return hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), b"dev_emulator_salt", 100000
        ).hex()

    def register(self, email: str, password: str) -> dict:
        clean = self._normalize_email(email)
        if clean in self._users:
            raise InvalidIdentityError("Account already exists")
        uid = f"password-{secrets.token_hex(16)}"
        self._users[clean] = {
            "uid": uid,
            "password_hash": self._hash_password(password),
            "email_verified": False,
            "disabled": False,
        }
        return {"email": clean, "email_verified": False}

    def verify_email(self, email: str) -> dict:
        clean = self._normalize_email(email)
        user = self._users.get(clean)
        if user is None:
            raise InvalidIdentityError("Account not found")
        user["email_verified"] = True
        return {"email": clean, "email_verified": True}

    def login(self, email: str, password: str) -> str:
        clean = self._normalize_email(email)
        user = self._users.get(clean)
        if user is None or user["password_hash"] != self._hash_password(password):
            raise InvalidIdentityError("Invalid credentials")
        if user["disabled"]:
            raise DisabledIdentityError("Identity user disabled")
        if not user["email_verified"]:
            raise InvalidIdentityError("Email is not verified")
        return self.token(
            provider="password",
            uid=user["uid"],
            email=clean,
            state="verified",
            display_name=clean.split("@", 1)[0],
        )

    def start_password_reset(self, email: str) -> dict:
        clean = self._normalize_email(email)
        if clean not in self._users:
            return {"status": "ok"}
        token = secrets.token_urlsafe(32)
        self._reset_tokens[token] = clean
        return {"status": "ok", "reset_token": token}

    def complete_password_reset(self, reset_token: str, new_password: str) -> dict:
        email = self._reset_tokens.pop(reset_token, "")
        if not email or email not in self._users:
            raise InvalidIdentityError("Invalid reset token")
        self._users[email]["password_hash"] = self._hash_password(new_password)
        return {"status": "ok"}

    def google_token(self, *, email: str, google_user_id: str) -> str:
        clean = self._normalize_email(email)
        return self.token(
            provider="google.com",
            uid=google_user_id,
            email=clean,
            state="verified",
            display_name=clean.split("@", 1)[0],
        )

    def token(
        self,
        *,
        provider: str,
        uid: str,
        email: str,
        state: str,
        display_name: str,
    ) -> str:
        import hmac
        import hashlib

        payload = f"{self.project_id}:{provider}:{uid}:{email}:{state}:{display_name}"
        signature = hmac.new(
            b"dev_secret_key", payload.encode(), hashlib.sha256
        ).hexdigest()
        return f"fake:{payload}:{signature}"


class FakeIdentityVerifier:
    """Strict offline verifier for tests and local emulators.

    Token format:
    fake:<project>:<provider>:<uid>:<email>:<state>:<display_name>
    """

    def __init__(self, project_id: str, *, require_verified_email: bool = True) -> None:
        self.project_id = project_id
        self.require_verified_email = require_verified_email

    def verify_bearer_token(self, token: str) -> VerifiedIdentity:
        parts = token.split(":", 7)
        if len(parts) != 8 or parts[0] != "fake":
            raise InvalidIdentityError("Invalid identity token")
        _prefix, project, provider, uid, email, state, display_name, signature = parts

        import hmac
        import hashlib

        payload = f"{project}:{provider}:{uid}:{email}:{state}:{display_name}"
        expected_signature = hmac.new(
            b"dev_secret_key", payload.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected_signature, signature):
            raise InvalidIdentityError("Invalid identity token signature")

        if project != self.project_id:
            raise WrongProjectIdentityError("Token project does not match")
        if state == "expired":
            raise ExpiredIdentityError("Identity token expired")
        if state == "revoked":
            raise RevokedIdentityError("Identity token revoked")
        if state == "disabled":
            raise DisabledIdentityError("Identity user disabled")
        if not uid or not provider or "@" not in email:
            raise InvalidIdentityError("Identity token is missing required claims")
        verified = state == "verified"
        if self.require_verified_email and not verified:
            raise InvalidIdentityError("Email is not verified")
        return VerifiedIdentity(
            provider_user_id=uid,
            email=email.casefold(),
            email_verified=verified,
            provider=provider,
            display_name=display_name or None,
        )


class FirebaseIdentityVerifier:
    def __init__(
        self,
        project_id: str,
        *,
        require_verified_email: bool = True,
        credential_path: str = "",
    ) -> None:
        if not project_id:
            raise RuntimeError("Firebase project ID is required")
        self.project_id = project_id
        self.require_verified_email = require_verified_email
        try:
            import firebase_admin  # type: ignore
            from firebase_admin import auth as firebase_auth  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency boundary
            raise RuntimeError("Firebase Admin SDK is not installed") from exc
        self._firebase_admin = firebase_admin
        self._firebase_auth = firebase_auth
        apps = getattr(firebase_admin, "_apps", None)
        if hasattr(firebase_admin, "initialize_app") and not apps:
            options = {"projectId": project_id}
            if credential_path:
                from firebase_admin import credentials  # type: ignore

                credential = credentials.Certificate(credential_path)
                firebase_admin.initialize_app(credential, options)
            else:
                firebase_admin.initialize_app(options=options)

    def verify_bearer_token(self, token: str) -> VerifiedIdentity:
        try:
            claims = self._firebase_auth.verify_id_token(
                token, check_revoked=True, clock_skew_seconds=60
            )
        except Exception as exc:  # pragma: no cover - SDK-specific mapping
            structured_log(
                "identity_verification_failed", reason=exc.__class__.__name__
            )
            raise InvalidIdentityError("Identity verification failed") from exc
        if claims.get("aud") != self.project_id:
            raise WrongProjectIdentityError("Token project does not match")
        expected_issuer = f"https://securetoken.google.com/{self.project_id}"
        if claims.get("iss") and claims.get("iss") != expected_issuer:
            raise WrongProjectIdentityError("Token issuer does not match")
        email = claims.get("email")
        if not email:
            raise InvalidIdentityError("Identity token is missing email")
        email_verified = claims.get("email_verified")
        if self.require_verified_email and email_verified is not True:
            raise InvalidIdentityError("Email is not verified")
        uid = str(claims.get("uid") or claims.get("sub") or "")
        if not uid:
            raise InvalidIdentityError("Identity token is missing subject")
        if hasattr(self._firebase_auth, "get_user"):
            try:
                firebase_user = self._firebase_auth.get_user(uid)
            except Exception as exc:  # pragma: no cover - SDK-specific mapping
                structured_log(
                    "identity_user_lookup_failed", reason=exc.__class__.__name__
                )
                raise InvalidIdentityError("Identity verification failed") from exc
            if getattr(firebase_user, "disabled", False):
                raise DisabledIdentityError("Identity user disabled")
        return VerifiedIdentity(
            provider_user_id=uid,
            email=str(email).casefold(),
            email_verified=bool(email_verified),
            provider=str(
                claims.get("firebase", {}).get("sign_in_provider") or "firebase"
            ),
            display_name=claims.get("name"),
        )


def identity_verifier_from_env() -> IdentityVerifier:
    env = os.getenv("CLOUDA_ENV", "development").strip().lower()
    if (
        env == "production"
        and os.getenv("AUTH_MODE") is None
        and not os.getenv("CLOUDA_AUTH_VERIFIER", "").strip()
    ):
        raise RuntimeError(
            "Firebase identity verifier must be configured in production"
        )
    mode = auth_mode_from_env()
    project_id = _firebase_server_project_id_from_env()
    require_verified = not _env_false("CLOUDA_REQUIRE_VERIFIED_EMAIL")
    if mode == "local":
        if env == "production":
            raise RuntimeError("Fake identity verifier is not allowed in production")
        return FakeIdentityVerifier(
            project_id or "local-test-project",
            require_verified_email=require_verified,
        )
    if mode == "firebase_emulator":
        if env == "production":
            raise RuntimeError("Firebase emulator is not allowed in production")
        if not _firebase_emulator_origin_from_env():
            raise RuntimeError(
                "FIREBASE_AUTH_EMULATOR_HOST is required in firebase_emulator mode"
            )
        return FirebaseIdentityVerifier(
            project_id or "local-test-project",
            require_verified_email=require_verified,
            credential_path=os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON_PATH", "").strip(),
        )
    if mode == "firebase_staging":
        if env not in {"staging", "production"}:
            raise RuntimeError("Firebase staging auth requires CLOUDA_ENV=staging")
        if _firebase_emulator_origin_from_env():
            raise RuntimeError(
                "Firebase auth emulator is not allowed in firebase_staging mode"
            )
        if env == "staging":
            _validate_staging_project(project_id)
        else:
            _validate_production_project(project_id)
        return FirebaseIdentityVerifier(
            project_id,
            require_verified_email=require_verified,
            credential_path=os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON_PATH", "").strip(),
        )
    raise RuntimeError(
        "AUTH_MODE must be local, firebase_emulator, or firebase_staging"
    )
