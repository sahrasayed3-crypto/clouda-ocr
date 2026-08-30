import hmac
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import zipfile
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response as FastAPIResponse,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pypdf import PdfReader

from clouda_contracts.archive_security import ArchiveLimits, validate_zip_archive
from .auth import (
    FakeAuthEmulator,
    IdentityVerificationError,
    auth_mode_from_env,
    firebase_client_config_from_env,
    identity_verifier_from_env,
)
from .constants import MODEL_ACCURATE_PRIMARY, MODEL_FAST
from .database import Database, utc_now
from .job_queue import get_distributed_queue
from .limits import limits_from_env
from .settings import load_settings, runtime_settings
from .correction_learning import rules_checksum
from .key_router.audit import sanitize_metadata
from .tenant_storage import TenantStorage
from .operations import (
    OperationsMetrics,
    RedisSecurityConfig,
    SlidingWindowRateLimiter,
    structured_log,
)

JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
KEY_ROUTER_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
WORKER_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$"
WORKER_PROVIDER_PATTERN = r"^[A-Za-z0-9_.:-]{0,40}$"
CLAIM_TOKEN_PATTERN = r"^[A-Fa-f0-9]{32}$"
ALLOWED_CLOUD_PROVIDERS = {
    "",
    "openrouter",
    "together",
    "fireworks",
    "google",
    "alibaba",
}
app = FastAPI(title="Clouda Worker API", docs_url=None, redoc_url=None)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        host.strip()
        for host in os.getenv("CLOUDA_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")
        if host.strip()
    ],
)
_rate_limiter = SlidingWindowRateLimiter(
    limit=max(1, int(os.getenv("CLOUDA_INTERNAL_RATE_LIMIT_PER_MINUTE", "1000")))
)
_operations_metrics = OperationsMetrics()
if int(os.getenv("WEB_CONCURRENCY", "1")) > 1:
    logging.getLogger(__name__).critical(
        "WEB_CONCURRENCY>1 detected: in-process rate limiter is not cross-process safe. "
        "Externalize to Redis before scaling past --workers=1."
    )


@app.middleware("http")
async def security_headers(request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    maximum_request_bytes = max(
        1024, int(os.getenv("CLOUDA_MAX_REQUEST_BYTES", str(110 * 1024 * 1024)))
    )
    content_length = request.headers.get("Content-Length")
    if content_length:
        try:
            declared_length = int(content_length)
        except ValueError:
            return JSONResponse(
                {"detail": "Invalid Content-Length", "request_id": request_id},
                status_code=400,
                headers={"X-Request-ID": request_id},
            )
        if declared_length < 0 or declared_length > maximum_request_bytes:
            return JSONResponse(
                {"detail": "Request body is too large", "request_id": request_id},
                status_code=413,
                headers={"X-Request-ID": request_id},
            )
    client = request.client.host if request.client else "unknown"
    if not _rate_limiter.allow(client):
        structured_log("rate_limit_exceeded", request_id=request_id, client=client)
        return JSONResponse(
            {"detail": "Rate limit exceeded", "request_id": request_id},
            status_code=429,
            headers={"X-Request-ID": request_id, "Retry-After": "60"},
        )
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["X-Request-ID"] = request_id
    _operations_metrics.observe_request(request.method, response.status_code)
    structured_log(
        "http_request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
    )
    return response


class WorkerMessage(BaseModel):
    worker_name: str = Field(
        min_length=1,
        max_length=120,
        pattern=WORKER_NAME_PATTERN,
    )
    claim_token: str | None = Field(
        default=None,
        min_length=32,
        max_length=32,
        pattern=CLAIM_TOKEN_PATTERN,
    )


class FailureMessage(WorkerMessage):
    error: str = Field(min_length=1, max_length=2000)


def _safe_worker_failure_message(error: str) -> str:
    sanitized = sanitize_metadata(error)
    return str(sanitized or "Worker failed")[:500]


class WorkerStatusMessage(WorkerMessage):
    cloud_available: bool
    cloud_provider: str = Field(
        default="", max_length=40, pattern=WORKER_PROVIDER_PATTERN
    )

    def model_post_init(self, __context) -> None:
        provider = self.cloud_provider.strip().lower()
        if provider not in ALLOWED_CLOUD_PROVIDERS:
            raise ValueError("Unsupported cloud provider")
        self.cloud_provider = provider


class TeacherReviewMessage(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=128, pattern=KEY_ROUTER_ID_RE)
    resolution: str = Field(pattern=r"^(GOLD|REVIEW|REJECTED)$")


class UserDocumentRequest(BaseModel):
    original_pdf_name: str = Field(min_length=1, max_length=160)
    page_count: int = Field(ge=1, le=500)
    owner_user_id: str | None = None


class RoleChangeRequest(BaseModel):
    role: str = Field(pattern=r"^(user|admin)$")


class StatusChangeRequest(BaseModel):
    status: str = Field(pattern=r"^(active|disabled|deletion_pending|deleted)$")


USER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _user_id(value: str) -> str:
    if not USER_ID_RE.fullmatch(value):
        raise HTTPException(status_code=400, detail="Invalid user ID")
    return value


class GuestClaimRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=64, pattern=JOB_ID_RE.pattern)
    claim_token: str = Field(min_length=20, max_length=200)


class DevRegisterRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=200)


class DevLoginRequest(DevRegisterRequest):
    pass


class DevEmailRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)


class DevResetCompleteRequest(BaseModel):
    reset_token: str = Field(min_length=20, max_length=200)
    new_password: str = Field(min_length=8, max_length=200)


class DevGoogleLoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    google_user_id: str = Field(min_length=1, max_length=128)


class StreamlitAuthBridgeRequest(BaseModel):
    bridge_token: str = Field(min_length=32, max_length=200)


_fake_auth_emulators: dict[str, FakeAuthEmulator] = {}
_streamlit_auth_bridges: dict[str, dict] = {}


def _create_streamlit_auth_bridge(session: dict, user: dict) -> dict:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=60)
    _streamlit_auth_bridges[token] = {
        "session_id": session["session_id"],
        "csrf_token": session["csrf_token"],
        "expires_at": expires_at,
        "user": _public_profile(user),
    }
    return {
        "bridge_token": token,
        "bridge_expires_at": expires_at.isoformat(),
    }


def _consume_streamlit_auth_bridge(bridge_token: str) -> dict | None:
    now = datetime.now(timezone.utc)
    for token, bridge in list(_streamlit_auth_bridges.items()):
        if bridge["expires_at"] <= now:
            _streamlit_auth_bridges.pop(token, None)
    bridge = _streamlit_auth_bridges.pop(bridge_token, {})
    if not bridge or bridge["expires_at"] <= now:
        return None
    return bridge


def _quality_allows_completed(values: dict) -> bool:
    score = values.get("text_quality_score")
    try:
        return score is not None and float(score) >= 90.0
    except (TypeError, ValueError):
        return False


def _database() -> Database:
    return Database(runtime_settings().database_path)


def _fake_auth_emulator() -> FakeAuthEmulator:
    if not _dev_auth_available():
        raise HTTPException(status_code=404, detail="Development auth is unavailable")
    project_id = os.getenv("CLOUDA_FIREBASE_PROJECT_ID", "local-test-project").strip()
    if not project_id:
        project_id = "local-test-project"
    emulator = _fake_auth_emulators.get(project_id)
    if emulator is None:
        emulator = FakeAuthEmulator(project_id)
        _fake_auth_emulators[project_id] = emulator
    return emulator


def _dev_auth_available() -> bool:
    if os.getenv("CLOUDA_AUTH_VERIFIER", "").strip().lower() != "fake":
        return False
    return os.getenv("CLOUDA_ENV", "").strip().lower() in {"development", "test"}


def _require_verified_email() -> bool:
    return os.getenv("CLOUDA_REQUIRE_VERIFIED_EMAIL", "true").lower() not in {
        "0",
        "false",
        "no",
    }


def _hash_signal(value: str) -> str:
    return Database.hash_secret(value)[:16] if value else ""


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _auth_cookie_settings() -> dict:
    secure = os.getenv("CLOUDA_SESSION_COOKIE_SECURE", "true").lower() not in {
        "0",
        "false",
        "no",
    }
    return {"httponly": True, "samesite": "lax", "secure": secure, "path": "/"}


def _session_lifetime_seconds() -> int:
    return max(300, int(os.getenv("CLOUDA_SESSION_LIFETIME_SECONDS", "3600")))


def _guest_lifetime_seconds() -> int:
    return max(60, int(os.getenv("CLOUDA_GUEST_SESSION_SECONDS", "1800")))


def _guest_job_ttl_seconds() -> int:
    return max(60, int(os.getenv("CLOUDA_GUEST_JOB_TTL_SECONDS", "1800")))


def _public_profile(user: dict) -> dict:
    return {
        "user_id": user["user_id"],
        "email": user["normalized_email"],
        "email_verified": bool(user["email_verified"]),
        "display_name": user.get("display_name", ""),
        "role": user["role"],
        "status": user["status"],
    }


def _require_user(
    clouda_session: str | None = Cookie(default=None),
) -> tuple[Database, dict, dict, str]:
    if not clouda_session:
        raise HTTPException(status_code=401, detail="Authentication required")
    database = _database()
    session = database.get_session_user(clouda_session)
    if session is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    user, session_row = session
    if user["status"] != "active":
        raise HTTPException(status_code=403, detail="Account is not active")
    return database, user, session_row, clouda_session


def _require_admin(context=Depends(_require_user)) -> tuple[Database, dict, dict, str]:
    database, user, session_row, session_id = context
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return database, user, session_row, session_id


def _require_csrf(
    x_csrf_token: str | None = Header(default=None),
    context=Depends(_require_user),
) -> tuple[Database, dict, dict, str]:
    database, user, session_row, session_id = context
    if not database.session_csrf_matches(session_id, x_csrf_token):
        raise HTTPException(status_code=403, detail="CSRF validation failed")
    return database, user, session_row, session_id


def _guest_context(
    clouda_guest: str | None = Cookie(default=None),
) -> tuple[Database, dict, str]:
    if not clouda_guest:
        raise HTTPException(status_code=401, detail="Guest session required")
    database = _database()
    guest = database.get_guest_session(clouda_guest)
    if guest is None:
        raise HTTPException(status_code=401, detail="Guest session expired")
    return database, guest, clouda_guest


def _require_guest_csrf(
    x_csrf_token: str | None = Header(default=None),
    context=Depends(_guest_context),
) -> tuple[Database, dict, str]:
    database, guest, guest_session_id = context
    if not database.guest_csrf_matches(guest_session_id, x_csrf_token):
        raise HTTPException(status_code=403, detail="CSRF validation failed")
    return database, guest, guest_session_id


def _consume_user_upload_quota(database: Database, user: dict) -> None:
    limit = int(os.getenv("CLOUDA_USER_UPLOADS_PER_MINUTE", "12"))
    if not database.consume_window_quota(
        scope_type="user",
        scope_id=user["user_id"],
        endpoint="user_upload",
        limit=limit,
        window_seconds=60,
    ):
        raise HTTPException(
            status_code=429,
            detail="User upload limit exceeded",
            headers={"Retry-After": "60"},
        )


def _consume_admin_action_quota(database: Database, user: dict) -> None:
    limit = int(os.getenv("CLOUDA_ADMIN_ACTIONS_PER_MINUTE", "30"))
    if not database.consume_window_quota(
        scope_type="admin",
        scope_id=user["user_id"],
        endpoint="admin_action",
        limit=limit,
        window_seconds=60,
    ):
        raise HTTPException(
            status_code=429,
            detail="Admin action limit exceeded",
            headers={"Retry-After": "60"},
        )


def _authenticate(x_worker_api_key: str | None = Header(default=None)) -> None:
    expected = runtime_settings().worker_api_key
    previous = os.getenv("CLOUDA_WORKER_API_KEY_PREVIOUS", "")
    accepted = [candidate for candidate in (expected, previous) if candidate]
    if (
        not accepted
        or not x_worker_api_key
        or not any(
            hmac.compare_digest(candidate, x_worker_api_key) for candidate in accepted
        )
    ):
        raise HTTPException(status_code=401, detail="Invalid worker credentials")


def _job_id(value: str) -> str:
    if not JOB_ID_RE.fullmatch(value):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    return value


def _key_router_id(value: str) -> str:
    if not KEY_ROUTER_ID_RE.fullmatch(value):
        raise HTTPException(status_code=400, detail="Invalid Key Router ID")
    return value


def _inside_storage(path_value: str) -> Path:
    root = runtime_settings().storage_root.resolve()
    path = Path(path_value)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(
            status_code=403, detail="Stored path is outside storage root"
        ) from exc
    return path


def _get_job(job_id: str) -> tuple[Database, dict]:
    database = _database()
    row = database.get_conversion(_job_id(job_id))
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return database, row


def _validate_docx_file(path: Path) -> None:
    limits = limits_from_env()
    if not path.is_file() or path.stat().st_size < 4 or path.read_bytes()[:2] != b"PK":
        raise HTTPException(status_code=503, detail="Result finalization is incomplete")
    try:
        with zipfile.ZipFile(path) as archive:
            validate_zip_archive(
                archive,
                limits=ArchiveLimits(
                    max_members=limits.max_archive_members,
                    max_total_uncompressed_bytes=limits.max_decompressed_bytes,
                ),
            )
    except (ValueError, zipfile.BadZipFile) as exc:
        raise HTTPException(
            status_code=503, detail="Result finalization is incomplete"
        ) from exc


def _finalizing_is_stale(row: dict) -> bool:
    raw_updated_at = row.get("updated_at") or ""
    try:
        updated_at = datetime.fromisoformat(raw_updated_at)
    except ValueError:
        return False
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    try:
        threshold = max(1, int(os.getenv("CLOUDA_FINALIZING_STALE_SECONDS", "300")))
    except ValueError:
        threshold = 300
    return datetime.now(timezone.utc) - updated_at > timedelta(seconds=threshold)


def _processing_is_stale(row: dict) -> bool:
    raw_timestamp = row.get("last_heartbeat") or row.get("updated_at") or ""
    try:
        heartbeat_at = datetime.fromisoformat(raw_timestamp)
    except ValueError:
        return False
    if heartbeat_at.tzinfo is None:
        heartbeat_at = heartbeat_at.replace(tzinfo=timezone.utc)
    try:
        threshold = max(1, int(os.getenv("CLOUDA_PROCESSING_STALE_SECONDS", "900")))
    except ValueError:
        threshold = 900
    return datetime.now(timezone.utc) - heartbeat_at > timedelta(seconds=threshold)


def _recover_stale_processing_job(database: Database, row: dict) -> dict:
    if row.get("status") != "processing" or not _processing_is_stale(row):
        return row
    recovered = database.abandon_stale_processing(
        row["job_id"], observed_updated_at=row.get("updated_at") or ""
    )
    if recovered:
        structured_log(
            "stale_processing_job_requeued",
            job_id=row["job_id"],
            worker_name=row.get("worker_name") or "",
        )
    refreshed = database.get_conversion(row["job_id"])
    if refreshed is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return refreshed


def _recover_finalizing_result(database: Database, row: dict) -> dict:
    if row.get("status") != "finalizing":
        return row
    target_status = row.get("intended_final_status") or ""
    if target_status not in {"completed", "manual_review"}:
        raise HTTPException(status_code=503, detail="Result finalization is incomplete")
    target = _inside_storage(row.get("stored_docx_path") or "")
    if not target.is_file():
        if _finalizing_is_stale(row):
            database.abandon_conversion_finalization(
                row["job_id"], observed_updated_at=row.get("updated_at") or ""
            )
            refreshed = database.get_conversion(row["job_id"])
            if refreshed is None:
                raise HTTPException(status_code=404, detail="Job not found")
            return refreshed
        return row
    _validate_docx_file(target)
    try:
        updated = database.complete_conversion_finalization(
            row["job_id"],
            target_status,
            worker_name=row.get("worker_name") or "",
            claim_token=row.get("claim_token") or None,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Result finalization is temporarily unavailable"
        ) from exc
    if updated.get("guest_scope_id"):
        database.mark_guest_job_result(
            updated["job_id"],
            status=updated["status"],
            stored_docx_path=str(target),
        )
    return updated


def _redis_client():
    from redis import Redis

    config = RedisSecurityConfig.from_env()
    return Redis.from_url(config.url, **config.client_kwargs())


def _redis_key(name: str) -> str:
    return f"{runtime_settings().redis_namespace}:{name}"


def _worker_status_payload(*, message: str = "") -> dict:
    workers = []
    try:
        redis = _redis_client()
        redis.ping()
        for key in redis.scan_iter(_redis_key("worker-status:*"), count=100):
            raw = redis.get(key)
            if raw:
                value = json.loads(raw)
                workers.append(
                    {
                        "worker_name": value.get("worker_name", ""),
                        "cloud_available": bool(value.get("cloud_available")),
                        "cloud_provider": value.get("cloud_provider", ""),
                        "reported_at": value.get("reported_at", ""),
                        "state": value.get("state", "ready"),
                    }
                )
        return {
            "status": "ok",
            "redis_available": True,
            "workers": workers,
            "cloud_available": any(worker["cloud_available"] for worker in workers),
            "message": message or "Distributed workers are available.",
        }
    except Exception:
        return {
            "status": "degraded",
            "redis_available": False,
            "workers": [],
            "cloud_available": False,
            "message": (
                "Distributed workers are unavailable; local conversion remains available."
            ),
        }


@app.get("/health")
def public_health() -> dict:
    config = runtime_settings()
    return {
        "status": "ok",
        "role": config.app_role,
        "local_processing_enabled": config.local_processing_enabled,
    }


@app.get("/public/pages")
def public_pages() -> dict:
    return {
        "pages": [
            "landing",
            "product",
            "guest_trial",
            "login",
            "signup",
            "password_reset",
            "privacy",
            "terms",
            "security",
            "contact",
        ]
    }


@app.get("/auth/firebase/config")
def firebase_public_config() -> dict:
    try:
        mode = auth_mode_from_env()
        if mode == "local":
            raise HTTPException(status_code=404, detail="Firebase auth is unavailable")
        config = firebase_client_config_from_env()
    except HTTPException:
        raise
    except RuntimeError as exc:
        structured_log(
            "firebase_client_config_unavailable", reason=exc.__class__.__name__
        )
        raise HTTPException(
            status_code=404, detail="Firebase auth is unavailable"
        ) from exc
    return {"mode": mode, "config": config}


@app.post("/auth/session")
def exchange_auth_session(
    request: Request,
    response: FastAPIResponse,
    authorization: str | None = Header(default=None),
    clouda_session: str | None = Cookie(default=None),
    x_clouda_streamlit_bridge: str | None = Header(default=None),
) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer token required")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        identity = identity_verifier_from_env().verify_bearer_token(token)
    except IdentityVerificationError as exc:
        structured_log("login_failed", reason=exc.__class__.__name__)
        raise HTTPException(status_code=401, detail="Invalid identity") from exc
    if _require_verified_email() and not identity.email_verified:
        structured_log("login_failed", reason="UnverifiedEmail")
        raise HTTPException(status_code=401, detail="Invalid identity")
    database = _database()
    try:
        user = database.upsert_auth_user(identity)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Account is not active") from exc
    if user["status"] != "active":
        raise HTTPException(status_code=403, detail="Account is not active")
    if clouda_session:
        database.revoke_session(clouda_session)
    session = database.create_session(
        user["user_id"],
        lifetime_seconds=_session_lifetime_seconds(),
        idle_seconds=max(60, int(os.getenv("CLOUDA_SESSION_IDLE_SECONDS", "900"))),
        ip_hash=_hash_signal(_client_ip(request)),
        user_agent_hash=_hash_signal(request.headers.get("user-agent", "")),
    )
    response.set_cookie(
        "clouda_session",
        session["session_id"],
        max_age=_session_lifetime_seconds(),
        **_auth_cookie_settings(),
    )
    database.record_auth_audit(
        "login_success",
        actor_user_id=user["user_id"],
        target_user_id=user["user_id"],
        result="success",
        ip_hash=_hash_signal(_client_ip(request)),
        metadata={"provider": identity.provider},
    )
    payload = {
        "user": _public_profile(user),
        "csrf_token": session["csrf_token"],
        "expires_at": session["expires_at"],
    }
    if (
        isinstance(x_clouda_streamlit_bridge, str)
        and x_clouda_streamlit_bridge.strip() == "1"
    ):
        payload.update(_create_streamlit_auth_bridge(session, user))
    return payload


@app.post("/auth/streamlit/bridge")
def consume_streamlit_auth_bridge(
    message: StreamlitAuthBridgeRequest,
    response: FastAPIResponse,
) -> dict:
    bridge = _consume_streamlit_auth_bridge(message.bridge_token)
    if bridge is None:
        raise HTTPException(status_code=401, detail="Invalid auth bridge token")
    database = _database()
    session = database.get_session_user(bridge["session_id"])
    if session is None:
        raise HTTPException(status_code=401, detail="Invalid auth bridge token")
    _user, _session_row = session
    response.set_cookie(
        "clouda_session",
        bridge["session_id"],
        max_age=_session_lifetime_seconds(),
        **_auth_cookie_settings(),
    )
    return {
        "session_id": bridge["session_id"],
        "csrf_token": bridge["csrf_token"],
        "user": bridge["user"],
    }


def _exchange_dev_token(
    request: Request,
    response: FastAPIResponse,
    token: str,
    clouda_session: str | None,
    streamlit_bridge: str | None = None,
) -> dict:
    return exchange_auth_session(
        request,
        response,
        authorization=f"Bearer {token}",
        clouda_session=clouda_session,
        x_clouda_streamlit_bridge=streamlit_bridge,
    )


@app.post("/auth/dev/register", status_code=201)
def dev_register(message: DevRegisterRequest) -> dict:
    if not _dev_auth_available():
        raise HTTPException(status_code=404, detail="Development auth is unavailable")
    try:
        return _fake_auth_emulator().register(message.email, message.password)
    except IdentityVerificationError as exc:
        raise HTTPException(status_code=400, detail="Registration failed") from exc


@app.post("/auth/dev/verify-email")
def dev_verify_email(message: DevEmailRequest) -> dict:
    if not _dev_auth_available():
        raise HTTPException(status_code=404, detail="Development auth is unavailable")
    try:
        return _fake_auth_emulator().verify_email(message.email)
    except IdentityVerificationError as exc:
        raise HTTPException(status_code=404, detail="Account not found") from exc


@app.post("/auth/dev/login")
def dev_login(
    message: DevLoginRequest,
    request: Request,
    response: FastAPIResponse,
    clouda_session: str | None = Cookie(default=None),
    x_clouda_streamlit_bridge: str | None = Header(default=None),
) -> dict:
    if not _dev_auth_available():
        raise HTTPException(status_code=404, detail="Development auth is unavailable")
    try:
        token = _fake_auth_emulator().login(message.email, message.password)
    except IdentityVerificationError as exc:
        status = 403 if "verified" in str(exc).lower() else 401
        raise HTTPException(status_code=status, detail="Login failed") from exc
    return _exchange_dev_token(
        request, response, token, clouda_session, x_clouda_streamlit_bridge
    )


@app.post("/auth/dev/google-login")
def dev_google_login(
    message: DevGoogleLoginRequest,
    request: Request,
    response: FastAPIResponse,
    clouda_session: str | None = Cookie(default=None),
    x_clouda_streamlit_bridge: str | None = Header(default=None),
) -> dict:
    if not _dev_auth_available():
        raise HTTPException(status_code=404, detail="Development auth is unavailable")
    try:
        token = _fake_auth_emulator().google_token(
            email=message.email, google_user_id=message.google_user_id
        )
    except IdentityVerificationError as exc:
        raise HTTPException(status_code=401, detail="Login failed") from exc
    return _exchange_dev_token(
        request, response, token, clouda_session, x_clouda_streamlit_bridge
    )


@app.post("/auth/dev/password-reset/start")
def dev_password_reset_start(message: DevEmailRequest) -> dict:
    if not _dev_auth_available():
        raise HTTPException(status_code=404, detail="Development auth is unavailable")
    try:
        result = _fake_auth_emulator().start_password_reset(message.email)
    except IdentityVerificationError:
        result = {"status": "ok"}
    _database().record_auth_audit(
        "password_reset_started", result="success", metadata={"provider": "fake"}
    )
    return result


@app.post("/auth/dev/password-reset/complete")
def dev_password_reset_complete(message: DevResetCompleteRequest) -> dict:
    if not _dev_auth_available():
        raise HTTPException(status_code=404, detail="Development auth is unavailable")
    try:
        result = _fake_auth_emulator().complete_password_reset(
            message.reset_token, message.new_password
        )
    except IdentityVerificationError as exc:
        raise HTTPException(status_code=403, detail="Invalid reset token") from exc
    _database().record_auth_audit(
        "password_reset_completed", result="success", metadata={"provider": "fake"}
    )
    return result


@app.get("/auth/me")
def auth_me(context=Depends(_require_user)) -> dict:
    _database_obj, user, _session_row, _session_id = context
    return _public_profile(user)


@app.post("/auth/logout")
def auth_logout(
    response: FastAPIResponse,
    context=Depends(_require_csrf),
) -> dict:
    database, user, _session_row, session_id = context
    database.revoke_session(session_id)
    database.record_auth_audit(
        "logout",
        actor_user_id=user["user_id"],
        target_user_id=user["user_id"],
        result="success",
    )
    response.delete_cookie("clouda_session", path="/")
    return {"status": "ok"}


@app.post("/auth/logout-all")
def auth_logout_all(
    response: FastAPIResponse,
    context=Depends(_require_csrf),
) -> dict:
    database, user, _session_row, _session_id = context
    database.revoke_user_sessions(user["user_id"])
    database.record_auth_audit(
        "logout_all",
        actor_user_id=user["user_id"],
        target_user_id=user["user_id"],
        result="success",
    )
    response.delete_cookie("clouda_session", path="/")
    return {"status": "ok"}


@app.post("/auth/delete-account")
def auth_delete_account(
    response: FastAPIResponse,
    context=Depends(_require_csrf),
) -> dict:
    database, user, _session_row, _session_id = context
    updated = database.set_auth_user_status(
        user["user_id"], "deletion_pending", actor_user_id=user["user_id"]
    )
    database.record_auth_audit(
        "account_deletion_requested",
        actor_user_id=user["user_id"],
        target_user_id=user["user_id"],
        result="success",
    )
    response.delete_cookie("clouda_session", path="/")
    return _public_profile(updated)


@app.post("/admin/bootstrap")
def admin_bootstrap(
    x_first_admin_token: str | None = Header(default=None),
    context=Depends(_require_csrf),
) -> dict:
    database, user, _session_row, _session_id = context
    _consume_admin_action_quota(database, user)
    expected = os.getenv("CLOUDA_FIRST_ADMIN_BOOTSTRAP_TOKEN", "")
    if (
        not expected
        or not x_first_admin_token
        or not hmac.compare_digest(expected, x_first_admin_token)
    ):
        raise HTTPException(status_code=403, detail="Bootstrap not allowed")
    if database.admin_count() > 0:
        raise HTTPException(status_code=409, detail="Admin already exists")
    with database.transaction() as connection:
        connection.execute(
            "UPDATE auth_users SET role='admin', updated_at=? WHERE user_id=?",
            (utc_now(), user["user_id"]),
        )
        database._record_auth_audit_on_connection(
            connection,
            "first_admin_bootstrap",
            actor_user_id=user["user_id"],
            target_user_id=user["user_id"],
            result="success",
        )
    database.revoke_user_sessions(user["user_id"])
    updated = database.get_auth_user(user["user_id"])
    if updated is None:
        raise HTTPException(status_code=404, detail="User not found")
    return _public_profile(updated)


@app.get("/admin/users")
def admin_users(context=Depends(_require_admin)) -> dict:
    database, user, _session_row, _session_id = context
    database.record_auth_audit(
        "admin_user_search",
        actor_user_id=user["user_id"],
        result="success",
    )
    return {"users": database.list_auth_users()}


@app.post("/admin/users/{user_id}/role")
def admin_change_role(
    user_id: str,
    message: RoleChangeRequest,
    context=Depends(_require_csrf),
) -> dict:
    _user_id(user_id)
    database, actor, _session_row, _session_id = context
    if actor["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    _consume_admin_action_quota(database, actor)
    try:
        updated = database.set_auth_user_role(
            user_id, message.role, actor_user_id=actor["user_id"]
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc
    return _public_profile(updated)


@app.post("/admin/users/{user_id}/status")
def admin_change_status(
    user_id: str,
    message: StatusChangeRequest,
    context=Depends(_require_csrf),
) -> dict:
    _user_id(user_id)
    database, actor, _session_row, _session_id = context
    if actor["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    _consume_admin_action_quota(database, actor)
    try:
        updated = database.set_auth_user_status(
            user_id, message.status, actor_user_id=actor["user_id"]
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc
    return _public_profile(updated)


def _conversion_payload(row: dict) -> dict:
    page_count = 0
    try:
        selected_pages = json.loads(row.get("page_numbers") or "[]")
    except (TypeError, json.JSONDecodeError):
        selected_pages = []
    if isinstance(selected_pages, list) and selected_pages:
        page_count = len(selected_pages)
    elif row.get("page_from") is not None and row.get("page_to") is not None:
        page_count = max(0, int(row["page_to"]) - int(row["page_from"]) + 1)
    status = str(row["status"])
    return {
        "job_id": row["job_id"],
        "owner_user_id": row.get("owner_user_id", ""),
        "original_pdf_name": row["original_pdf_name"],
        "status": status,
        "dispatch_status": row.get("dispatch_status", "not_configured"),
        "page_count": page_count,
        "can_retry": status in {"failed", "cancelled"},
        "can_cancel": status in {"pending", "processing"},
        "page_from": row.get("page_from"),
        "page_to": row.get("page_to"),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _dispatch_conversion_job(database: Database, job_id: str, *, actor: str) -> str:
    if runtime_settings().local_processing_enabled:
        raise RuntimeError(
            "LOCAL_PROCESSING_ENABLED is not a supported dispatch mode; run the "
            "distributed RQ worker (APP_ROLE=worker) instead."
        )
    try:
        rq_job = get_distributed_queue().enqueue(job_id)
    except Exception as exc:
        structured_log(
            "distributed_dispatch_deferred",
            job_id=job_id,
            actor=actor,
            error_type=type(exc).__name__,
        )
        return "dispatch_deferred"
    row = database.get_conversion(job_id)
    if row is not None:
        database.update_conversion(
            row["id"], {"rq_job_id": str(getattr(rq_job, "id", job_id))}
        )
    return "queued"


@app.post("/user/documents", status_code=201)
def create_user_document(
    message: UserDocumentRequest,
    context=Depends(_require_csrf),
) -> dict:
    database, user, _session_row, _session_id = context
    _consume_user_upload_quota(database, user)
    return _create_user_document_record(
        database,
        user,
        original_pdf_name=message.original_pdf_name,
        page_count=message.page_count,
    )


def _create_user_document_record(
    database: Database, user: dict, *, original_pdf_name: str, page_count: int
) -> dict:
    tenant = TenantStorage(runtime_settings().storage_root).user(user["user_id"])
    job_id = uuid.uuid4().hex
    pdf_path = tenant.uploads / f"{job_id}.pdf"
    docx_path = tenant.outputs / f"{job_id}.docx"
    now = utc_now()
    database.create_conversion(
        {
            "job_id": job_id,
            "username": user["normalized_email"],
            "owner_user_id": user["user_id"],
            "original_pdf_name": original_pdf_name,
            "stored_pdf_path": str(pdf_path),
            "output_docx_name": f"{job_id}.docx",
            "stored_docx_path": str(docx_path),
            "page_from": 1,
            "page_to": page_count,
            "page_numbers": json.dumps(list(range(1, page_count + 1))),
            "status": "pending",
            "visibility": "private",
            "created_at": now,
            "updated_at": now,
        }
    )
    database.record_auth_audit(
        "document_created",
        actor_user_id=user["user_id"],
        target_user_id=user["user_id"],
        target_resource_type="conversion",
        target_resource_id=job_id,
        result="success",
    )
    row = database.get_owner_conversion(job_id, user["user_id"])
    if row is None:
        raise HTTPException(status_code=500, detail="Document was not persisted")
    return _conversion_payload(row)


@app.post("/user/documents/upload", status_code=201)
def upload_user_document(
    file: UploadFile = File(...),
    context=Depends(_require_csrf),
) -> dict:
    database, user, _session_row, _session_id = context
    _consume_user_upload_quota(database, user)
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    limits = limits_from_env()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = Path(tmp.name)
        total = 0
        while chunk := file.file.read(1024 * 1024):
            total += len(chunk)
            if total > limits.max_upload_bytes:
                tmp.close()
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="PDF is too large")
            tmp.write(chunk)

    try:
        if total < 4:
            raise HTTPException(status_code=400, detail="Invalid PDF")
        with tmp_path.open("rb") as f:
            if f.read(4) != b"%PDF":
                raise HTTPException(status_code=400, detail="Invalid PDF")
            f.seek(0)
            try:
                page_count = len(PdfReader(f).pages)
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Invalid PDF") from exc

        validate_pdf_limit = getattr(limits, "max_pdf_pages", None)
        if validate_pdf_limit is not None and page_count > validate_pdf_limit:
            raise HTTPException(status_code=413, detail="PDF page limit exceeded")

        row = _create_user_document_record(
            database,
            user,
            original_pdf_name=file.filename,
            page_count=page_count,
        )
        stored_path = Path(database.get_conversion(row["job_id"])["stored_pdf_path"])
        stored_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_path), str(stored_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    row["dispatch_status"] = _dispatch_conversion_job(
        database, row["job_id"], actor=user["user_id"]
    )
    return row


@app.get("/user/documents")
def list_user_documents(context=Depends(_require_user)) -> dict:
    database, user, _session_row, _session_id = context
    return {
        "documents": [
            _conversion_payload(row)
            for row in database.list_owner_conversions(user["user_id"])
        ]
    }


@app.get("/user/documents/{job_id}")
def get_user_document(job_id: str, context=Depends(_require_user)) -> dict:
    database, user, _session_row, _session_id = context
    row = database.get_owner_conversion(_job_id(job_id), user["user_id"])
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return _conversion_payload(row)


@app.delete("/user/documents/{job_id}")
def delete_user_document(job_id: str, context=Depends(_require_csrf)) -> dict:
    database, user, _session_row, _session_id = context
    row = database.get_owner_conversion(_job_id(job_id), user["user_id"])
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if not database.hide_owner_conversion(job_id, user["user_id"]):
        raise HTTPException(status_code=404, detail="Document not found")
    database.record_auth_audit(
        "document_deleted",
        actor_user_id=user["user_id"],
        target_user_id=user["user_id"],
        target_resource_type="conversion",
        target_resource_id=job_id,
        result="success",
    )
    return {"status": "deleted"}


@app.post("/user/documents/{job_id}/retry")
def retry_user_document(job_id: str, context=Depends(_require_csrf)) -> dict:
    database, user, _session_row, _session_id = context
    row = database.get_owner_conversion(_job_id(job_id), user["user_id"])
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if row["status"] not in {"failed", "cancelled"}:
        raise HTTPException(status_code=409, detail="Document cannot be retried")
    database.transition_conversion(job_id, "pending")
    return {"status": "pending"}


@app.post("/user/documents/{job_id}/cancel")
def cancel_user_document(job_id: str, context=Depends(_require_csrf)) -> dict:
    database, user, _session_row, _session_id = context
    row = database.get_owner_conversion(_job_id(job_id), user["user_id"])
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        updated = database.transition_conversion(job_id, "cancelled")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    database.record_auth_audit(
        "document_cancelled",
        actor_user_id=user["user_id"],
        target_user_id=user["user_id"],
        target_resource_type="conversion",
        target_resource_id=job_id,
        result="success",
    )
    return {"status": updated["status"]}


@app.get("/user/documents/{job_id}/download")
def download_user_document(job_id: str, context=Depends(_require_user)):
    database, user, _session_row, _session_id = context
    row = database.get_owner_conversion(_job_id(job_id), user["user_id"])
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    path_value = row.get("stored_docx_path") or ""
    if not path_value:
        raise HTTPException(status_code=404, detail="Result not found")
    path = _inside_storage(path_value)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename="result.docx",
    )


@app.get("/admin/documents/{job_id}")
def admin_get_document(job_id: str, context=Depends(_require_admin)) -> dict:
    database, admin, _session_row, _session_id = context
    row = database.get_conversion(_job_id(job_id))
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    database.record_auth_audit(
        "admin_document_access",
        actor_user_id=admin["user_id"],
        target_user_id=row.get("owner_user_id", ""),
        target_resource_type="conversion",
        target_resource_id=job_id,
        result="success",
    )
    return _conversion_payload(row)


@app.post("/guest/session")
def create_guest_session(request: Request, response: FastAPIResponse) -> dict:
    database = _database()
    guest = database.create_guest_session(
        lifetime_seconds=_guest_lifetime_seconds(),
        ip_hash=_hash_signal(_client_ip(request)),
    )
    response.set_cookie(
        "clouda_guest",
        guest["guest_session_id"],
        max_age=_guest_lifetime_seconds(),
        **_auth_cookie_settings(),
    )
    database.record_auth_audit(
        "guest_session_created",
        result="success",
        ip_hash=_hash_signal(_client_ip(request)),
    )
    return {"expires_at": guest["expires_at"], "csrf_token": guest["csrf_token"]}


@app.post("/guest/trial", status_code=201)
def create_guest_trial(
    file: UploadFile = File(...),
    context=Depends(_require_guest_csrf),
) -> dict:
    database, guest, _guest_cookie = context
    if database.guest_has_active_job(guest["guest_scope_id"]):
        raise HTTPException(status_code=429, detail="Guest trial already active")
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    max_bytes = int(os.getenv("CLOUDA_GUEST_MAX_BYTES", str(10 * 1024 * 1024)))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = Path(tmp.name)
        total = 0
        while chunk := file.file.read(1024 * 1024):
            total += len(chunk)
            if total > max_bytes:
                tmp.close()
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Guest PDF is too large")
            tmp.write(chunk)

    try:
        if total < 4:
            raise HTTPException(status_code=400, detail="Invalid PDF")
        with tmp_path.open("rb") as f:
            if f.read(4) != b"%PDF":
                raise HTTPException(status_code=400, detail="Invalid PDF")
            f.seek(0)
            try:
                page_count = len(PdfReader(f).pages)
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Invalid PDF") from exc

        max_pages = int(os.getenv("CLOUDA_GUEST_MAX_PAGES", "5"))
        if page_count > max_pages:
            raise HTTPException(status_code=413, detail="Guest page limit exceeded")

        guest_quota_key = guest.get("ip_hash") or guest["guest_scope_id"]
        guest_budget = int(os.getenv("CLOUDA_GUEST_DAILY_BUDGET", "25"))
        if not database.consume_daily_quota(
            scope_type="guest_ip",
            scope_id=guest_quota_key,
            endpoint="guest_trial",
            limit=guest_budget,
        ):
            raise HTTPException(
                status_code=429,
                detail="Guest daily budget exceeded",
                headers={"Retry-After": "86400"},
            )

        storage = TenantStorage(runtime_settings().storage_root)
        paths = storage.guest(guest["guest_scope_id"])
        with tmp_path.open("rb") as src:
            stored = storage.write_upload(paths, file.filename, src)

        expires = datetime.now(timezone.utc) + timedelta(
            seconds=_guest_job_ttl_seconds()
        )
        row = database.create_guest_job(
            {
                "guest_scope_id": guest["guest_scope_id"],
                "original_pdf_name": file.filename,
                "stored_pdf_path": str(stored),
                "stored_docx_path": str(paths.outputs / (uuid.uuid4().hex + ".docx")),
                "page_count": page_count,
                "size_bytes": total,
                "expires_at": expires.isoformat(),
            }
        )
        now = utc_now()
        database.create_conversion(
            {
                "job_id": row["job_id"],
                "username": "guest",
                "original_pdf_name": file.filename,
                "stored_pdf_path": str(stored),
                "output_docx_name": f"{row['job_id']}.docx",
                "stored_docx_path": row["stored_docx_path"],
                "page_from": 1,
                "page_to": page_count,
                "page_numbers": json.dumps(list(range(1, page_count + 1))),
                "status": "pending",
                "guest_scope_id": guest["guest_scope_id"],
                "visibility": "guest",
                "lifecycle_state": "active",
                "expires_at": row["expires_at"],
                "created_at": now,
                "updated_at": now,
            }
        )
        database.record_auth_audit(
            "guest_trial_created",
            target_resource_type="guest_job",
            target_resource_id=row["job_id"],
            result="success",
            metadata={"page_count": page_count, "provider_policy": "local_free_only"},
        )
        dispatch_status = _dispatch_conversion_job(
            database, row["job_id"], actor=guest["guest_scope_id"]
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "dispatch_status": dispatch_status,
        "page_count": row["page_count"],
        "provider_policy": row["provider_policy"],
        "paid_provider_allowed": False,
        "expires_at": row["expires_at"],
        "result_token": row["result_token"],
    }


@app.get("/guest/jobs/{job_id}")
def get_guest_job(job_id: str, context=Depends(_guest_context)) -> dict:
    database, guest, _guest_cookie = context
    row = database.get_guest_job(
        _job_id(job_id), guest_scope_id=guest["guest_scope_id"]
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Guest job not found")
    conversion = database.get_conversion(job_id)
    if conversion is not None:
        conversion = _recover_finalizing_result(database, conversion)
    status = conversion["status"] if conversion else row["status"]
    return {
        "job_id": row["job_id"],
        "status": status,
        "page_count": row["page_count"],
        "provider_policy": row["provider_policy"],
        "paid_provider_allowed": False,
    }


@app.get("/guest/jobs/{job_id}/download")
def download_guest_result(job_id: str, token: str, context=Depends(_guest_context)):
    database, guest, _guest_cookie = context
    clean_job_id = _job_id(job_id)
    row = database.get_guest_job(clean_job_id, guest_scope_id=guest["guest_scope_id"])
    if row is None:
        raise HTTPException(status_code=404, detail="Guest result not found")
    conversion = database.get_conversion(clean_job_id)
    if conversion is not None:
        conversion = _recover_finalizing_result(database, conversion)
    if conversion is None or conversion["status"] != "completed":
        raise HTTPException(status_code=404, detail="Guest result not found")
    path = _inside_storage(
        conversion.get("stored_docx_path") or row["stored_docx_path"]
    )
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Guest result not found")
    consumed = database.consume_guest_result_token(
        clean_job_id, guest["guest_scope_id"], token
    )
    if consumed is None:
        raise HTTPException(status_code=404, detail="Guest result not found")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename="guest-result.docx",
    )


@app.post("/guest/claim-token")
def guest_claim_token(context=Depends(_require_guest_csrf)) -> dict:
    database, guest, _guest_cookie = context
    return {"claim_token": database.issue_guest_claim_token(guest["guest_scope_id"])}


@app.post("/guest/claim")
def guest_claim(
    message: GuestClaimRequest,
    user_context=Depends(_require_csrf),
    guest_context=Depends(_guest_context),
) -> dict:
    user_database, user, _session_row, _session_id = user_context
    guest_database, guest, _guest_cookie = guest_context
    if user_database.path != guest_database.path:
        raise HTTPException(status_code=500, detail="Database context mismatch")
    try:
        row = user_database.claim_guest_job(
            message.job_id,
            guest["guest_scope_id"],
            message.claim_token,
            user["user_id"],
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Invalid guest claim") from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=409, detail="Guest job cannot be claimed"
        ) from exc
    return {
        "job_id": row["job_id"],
        "owner_user_id": row["owner_user_id"],
        "status": row["status"],
    }


@app.get("/ready")
def readiness() -> dict:
    try:
        with _database().connect() as connection:
            connection.execute("SELECT 1").fetchone()
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Service is not ready") from exc


@app.get("/internal/metrics", dependencies=[Depends(_authenticate)])
def metrics() -> Response:
    queue_depth = 0
    failed_jobs = 0
    try:
        redis = _redis_client()
        queue_name = (
            f"{runtime_settings().redis_namespace}:{runtime_settings().rq_queue_name}"
        )
        queue_depth = int(redis.llen(f"rq:queue:{queue_name}"))
        failed_jobs = int(redis.zcard(f"rq:{queue_name}:failed"))
    except Exception:
        logging.getLogger(__name__).warning("Redis metrics unavailable", exc_info=True)
    return Response(
        _operations_metrics.prometheus(
            queue_depth=queue_depth, failed_jobs=failed_jobs
        ),
        media_type="text/plain; version=0.0.4",
    )


@app.get("/internal/health", dependencies=[Depends(_authenticate)])
def health() -> dict:
    database = _database()
    with database.connect() as connection:
        connection.execute("SELECT 1").fetchone()
    worker_status = _worker_status_payload()
    return {
        "status": "ok",
        "role": runtime_settings().app_role,
        "workers": worker_status["workers"],
        "cloud_available": worker_status["cloud_available"],
        "redis_available": worker_status["redis_available"],
        "local_processing_enabled": runtime_settings().local_processing_enabled,
        "worker_status": worker_status["status"],
        "message": worker_status["message"],
    }


@app.get("/internal/workers/status", dependencies=[Depends(_authenticate)])
def get_worker_status() -> dict:
    return _worker_status_payload()


@app.get("/internal/key-router/status", dependencies=[Depends(_authenticate)])
def key_router_status() -> dict:
    from .key_router.service import KeyRouter

    return KeyRouter(_database()).get_status()


@app.get("/internal/key-router/accounts", dependencies=[Depends(_authenticate)])
def key_router_accounts() -> dict:
    from .key_router.audit import sanitize_metadata
    from .key_router.service import KeyRouter

    accounts = KeyRouter(_database()).repository.list_accounts()
    payload = {
        "accounts": [
            {
                "account_id": account.account_id,
                "provider": account.provider,
                "pool_id": account.pool_id,
                "display_name": account.display_name,
                "enabled": account.enabled,
                "account_state": account.account_state.value,
                "region": account.region,
                "allows_free": account.allows_free,
                "allows_paid": account.allows_paid,
                "allows_vision": account.allows_vision,
                "allows_text": account.allows_text,
                "cooldown_until": account.cooldown_until,
            }
            for account in accounts
        ]
    }
    sanitized = sanitize_metadata(payload)
    return sanitized if isinstance(sanitized, dict) else {"accounts": []}


@app.get("/internal/key-router/capabilities", dependencies=[Depends(_authenticate)])
def key_router_capabilities() -> dict:
    from .key_router.audit import sanitize_metadata
    from .key_router.service import KeyRouter

    endpoints = KeyRouter(_database()).repository.list_endpoints()
    payload = {"capabilities": [asdict(endpoint) for endpoint in endpoints]}
    sanitized = sanitize_metadata(payload)
    return sanitized if isinstance(sanitized, dict) else {"capabilities": []}


@app.get("/internal/key-router/reservations", dependencies=[Depends(_authenticate)])
def key_router_reservations() -> dict:
    from .key_router.audit import sanitize_metadata
    from .key_router.service import KeyRouter

    reservations = KeyRouter(_database()).repository.list_reservations()
    payload = {
        "reservations": [
            {
                key: value.value if hasattr(value, "value") else value
                for key, value in asdict(reservation).items()
                if key not in {"reservation_token"}
            }
            for reservation in reservations
        ]
    }
    sanitized = sanitize_metadata(payload)
    return sanitized if isinstance(sanitized, dict) else {"reservations": []}


@app.get("/internal/key-router/audit", dependencies=[Depends(_authenticate)])
def key_router_audit() -> dict:
    from .key_router.audit import sanitize_metadata
    from .key_router.service import KeyRouter

    payload = {"events": KeyRouter(_database()).repository.recent_audit(limit=100)}
    sanitized = sanitize_metadata(payload)
    return sanitized if isinstance(sanitized, dict) else {"events": []}


def _admin_account_action(account_id: str, action: str) -> dict:
    from .key_router.audit import audit_event
    from .key_router.enums import AccountState
    from .key_router.service import KeyRouter

    clean_id = _key_router_id(account_id)
    router = KeyRouter(_database())
    actions = {
        "enable": (AccountState.ACTIVE, True),
        "disable": (AccountState.DISABLED, False),
        "quarantine": (AccountState.QUARANTINED, False),
        "restore": (AccountState.ACTIVE, True),
    }
    state, enabled = actions[action]
    try:
        account = router.repository.set_account_state(clean_id, state, enabled=enabled)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="Provider account not found"
        ) from exc
    router.repository.record_audit(
        audit_event(
            event_type="account_admin_action",
            provider=account.provider,
            pool_id=account.pool_id,
            account_id=account.account_id,
            new_state=state.value,
            decision_reason=action,
        )
    )
    return {
        "account_id": account.account_id,
        "state": account.account_state.value,
        "enabled": account.enabled,
    }


@app.post(
    "/internal/key-router/accounts/{account_id}/enable",
    dependencies=[Depends(_authenticate)],
)
def key_router_account_enable(account_id: str) -> dict:
    return _admin_account_action(account_id, "enable")


@app.post(
    "/internal/key-router/accounts/{account_id}/disable",
    dependencies=[Depends(_authenticate)],
)
def key_router_account_disable(account_id: str) -> dict:
    return _admin_account_action(account_id, "disable")


@app.post(
    "/internal/key-router/accounts/{account_id}/quarantine",
    dependencies=[Depends(_authenticate)],
)
def key_router_account_quarantine(account_id: str) -> dict:
    return _admin_account_action(account_id, "quarantine")


@app.post(
    "/internal/key-router/accounts/{account_id}/restore",
    dependencies=[Depends(_authenticate)],
)
def key_router_account_restore(account_id: str) -> dict:
    return _admin_account_action(account_id, "restore")


@app.post(
    "/internal/key-router/circuits/{circuit_id}/reset",
    dependencies=[Depends(_authenticate)],
)
def key_router_circuit_reset(circuit_id: str) -> dict:
    from .key_router.audit import audit_event
    from .key_router.service import KeyRouter

    clean_id = _key_router_id(circuit_id)
    router = KeyRouter(_database())
    try:
        circuit = router.repository.reset_circuit(clean_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Circuit not found") from exc
    router.repository.record_audit(
        audit_event(
            event_type="circuit_admin_reset",
            provider=str(circuit["provider"]),
            model=str(circuit["provider_model_id"]),
            decision_reason="manual_reset",
        )
    )
    return {
        "circuit_id": circuit["circuit_id"],
        "state": circuit["state"],
    }


def _teacher_rows(query: str, parameters: tuple = ()) -> list[dict]:
    from .key_router.audit import sanitize_metadata

    with _database().connect() as connection:
        rows = [dict(row) for row in connection.execute(query, parameters).fetchall()]
    sanitized = sanitize_metadata(rows)
    return sanitized if isinstance(sanitized, list) else []


@app.get("/internal/teacher-pipeline/status", dependencies=[Depends(_authenticate)])
def teacher_pipeline_status() -> dict:
    from .teacher_pipeline.service import safe_status

    return safe_status(_database())


@app.get(
    "/internal/teacher-pipeline/assignments", dependencies=[Depends(_authenticate)]
)
def teacher_pipeline_assignments() -> dict:
    return {"assignments": _teacher_rows("""
        SELECT assignment_id, account_id, owner_id, provider_id, quota_domain_id,
               allowed_teacher_roles_json, allowed_task_types_json,
               allowed_data_classifications_json, enabled, dispatch_mode,
               priority, daily_request_budget, daily_token_budget,
               daily_cost_budget, cooldown_until, terms_metadata_json,
               provenance_metadata_json, updated_at
        FROM teacher_role_assignments
        ORDER BY priority DESC, provider_id, account_id
        """)}


@app.get("/internal/teacher-pipeline/contracts", dependencies=[Depends(_authenticate)])
def teacher_pipeline_contracts() -> dict:
    from .teacher_pipeline.contracts import contract_catalog

    return {"contracts": contract_catalog()}


@app.get(
    "/internal/teacher-pipeline/capabilities", dependencies=[Depends(_authenticate)]
)
def teacher_pipeline_capabilities() -> dict:
    return {"capabilities": _teacher_rows("""
        SELECT endpoint_id, canonical_model_id, provider, provider_model_id,
               supports_text_output, supports_vision_declared,
               supports_vision_verified, supports_structured_json,
               supports_json_schema, max_images_per_request,
               supported_mime_types_json, max_payload_bytes,
               max_context_tokens, privacy_compatibility_json,
               region_support_json, teacher_role_support_json,
               training_output_policy_status, verified_status, enabled
        FROM capability_endpoints ORDER BY provider, provider_model_id
        """)}


@app.get("/internal/teacher-pipeline/jobs", dependencies=[Depends(_authenticate)])
def teacher_pipeline_jobs() -> dict:
    return {"jobs": _teacher_rows("""
        SELECT job_id, source_document_id, source_page_id, status,
               pipeline_mode, policy_version, reason_code, created_at, updated_at
        FROM teacher_jobs ORDER BY created_at DESC LIMIT 100
        """)}


@app.get("/internal/teacher-pipeline/outputs", dependencies=[Depends(_authenticate)])
def teacher_pipeline_outputs() -> dict:
    from .teacher_pipeline.repository import TeacherRepository

    return {"outputs": TeacherRepository(_database()).sanitized_output_summaries()}


@app.get("/internal/teacher-pipeline/decisions", dependencies=[Depends(_authenticate)])
def teacher_pipeline_decisions() -> dict:
    return {"decisions": _teacher_rows("""
        SELECT decision_id, job_id, decision, policy_version, metrics_json,
               reason_codes_json, human_verified, trusted_reference,
               created_at, updated_at
        FROM teacher_agreement_decisions ORDER BY created_at DESC LIMIT 100
        """)}


@app.get(
    "/internal/teacher-pipeline/review-queue", dependencies=[Depends(_authenticate)]
)
def teacher_pipeline_review_queue() -> dict:
    return {"review_queue": _teacher_rows("""
        SELECT review_id, candidate_id, status, reason_codes_json,
               assigned_reviewer, resolution, created_at, updated_at
        FROM human_review_queue ORDER BY created_at DESC LIMIT 100
        """)}


@app.get(
    "/internal/teacher-pipeline/rights-failures", dependencies=[Depends(_authenticate)]
)
def teacher_pipeline_rights_failures() -> dict:
    return {"rights_failures": _teacher_rows("""
        SELECT job_id, source_document_id, source_page_id, status,
               reason_code, created_at
        FROM teacher_jobs
        WHERE reason_code LIKE '%rights%' OR reason_code LIKE '%processing%'
        ORDER BY created_at DESC LIMIT 100
        """)}


@app.get(
    "/internal/teacher-pipeline/leakage-alerts", dependencies=[Depends(_authenticate)]
)
def teacher_pipeline_leakage_alerts() -> dict:
    rows = _teacher_rows("""
        SELECT source_page_id, COUNT(DISTINCT split) AS split_count
        FROM dataset_manifest_items
        GROUP BY source_page_id HAVING COUNT(DISTINCT split) > 1
        """)
    return {"leakage_alerts": rows, "status": "alert" if rows else "clear"}


@app.get(
    "/internal/teacher-pipeline/dataset-manifests",
    dependencies=[Depends(_authenticate)],
)
def teacher_pipeline_dataset_manifests() -> dict:
    return {"dataset_manifests": _teacher_rows("""
        SELECT manifest_id, dataset_version, policy_version, state,
               manifest_hash, created_at, frozen_at
        FROM dataset_manifests ORDER BY created_at DESC LIMIT 100
        """)}


@app.get(
    "/internal/teacher-pipeline/activation-readiness",
    dependencies=[Depends(_authenticate)],
)
def teacher_pipeline_activation_readiness() -> dict:
    from .teacher_pipeline.service import safe_status

    status = safe_status(_database())
    return {
        "live_dispatch_status": status["live_dispatch_status"],
        "training_status": status["training_status"],
        "activation_blockers": status["activation_blockers"],
        "training_blockers": status["training_blockers"],
    }


@app.post(
    "/internal/teacher-pipeline/review/{job_id}/decision",
    dependencies=[Depends(_authenticate)],
)
def teacher_pipeline_review_decision(
    job_id: str, message: TeacherReviewMessage
) -> dict:
    from .teacher_pipeline.enums import DatasetDecision
    from .teacher_pipeline.service import TeacherPipeline

    try:
        return TeacherPipeline(_database()).record_human_review(
            job_id=_key_router_id(job_id),
            reviewer_id=message.reviewer_id,
            resolution=DatasetDecision(message.resolution),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Candidate not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/internal/workers/status", dependencies=[Depends(_authenticate)])
def update_worker_status(message: WorkerStatusMessage) -> dict:
    payload = {
        "worker_name": message.worker_name,
        "cloud_available": message.cloud_available,
        "cloud_provider": message.cloud_provider if message.cloud_available else "",
        "reported_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        redis = _redis_client()
        redis.set(
            _redis_key(f"worker-status:{message.worker_name}"),
            json.dumps(payload, ensure_ascii=False),
            ex=90,
        )
        return {"status": "ok", "redis_available": True}
    except Exception:
        return _worker_status_payload(
            message="Worker status was not saved because Redis is unavailable."
        )


@app.get("/internal/corrections/snapshot", dependencies=[Depends(_authenticate)])
def correction_snapshot() -> dict:
    config = runtime_settings()
    if not config.correction_memory_enabled:
        rules = []
    else:
        rules = _database().active_memory_rules(
            config.correction_min_source_files,
            config.correction_auto_apply_threshold,
        )
    safe_fields = {
        "id",
        "wrong_text",
        "correct_text",
        "context_pattern",
        "scope",
        "scope_value",
        "language",
        "document_type",
        "confidence",
        "approved",
        "enabled",
        "is_sensitive",
    }
    payload = [{key: row.get(key) for key in safe_fields} for row in rules]
    for row in payload:
        row["threshold"] = config.correction_auto_apply_threshold
    checksum = rules_checksum(payload)
    return {
        "version": f"3:{checksum[:12]}",
        "checksum": checksum,
        "updated_at": max((row.get("updated_at", "") for row in rules), default=""),
        "rules": payload[: config.correction_max_rules_per_page],
    }


@app.get("/internal/jobs/{job_id}", dependencies=[Depends(_authenticate)])
def get_job(job_id: str) -> dict:
    database, row = _get_job(job_id)
    try:
        pages = json.loads(row.get("page_numbers") or "[]")
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail="Invalid stored page selection"
        ) from exc
    if not pages:
        pages = list(range(int(row["page_from"]), int(row["page_to"]) + 1))
    settings = load_settings(database)
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "claim_token": row.get("claim_token") or "",
        "page_numbers": pages,
        "output_docx_name": row["output_docx_name"],
        "fast_model": os.getenv("FAST_MODEL", MODEL_FAST),
        "accurate_model": os.getenv("ACCURATE_MODEL", MODEL_ACCURATE_PRIMARY),
        "settings": {
            key: settings[key]
            for key in (
                "acceptance_threshold",
                "free_model_attempts",
                "paid_model_attempts",
                "file_cost_limit",
                "daily_cost_limit",
                "scan_dpi",
                "max_dpi",
                "enabled_engines",
                "enabled_models",
                "batch_size",
            )
        }
        | {
            "current_document_cost": float(row.get("total_cost") or 0),
            "daily_cost_spent": database.daily_cost(),
        },
    }


@app.get("/internal/jobs/{job_id}/input", dependencies=[Depends(_authenticate)])
def download_input(job_id: str):
    _, row = _get_job(job_id)
    if row.get("hidden") or row.get("lifecycle_state") not in {"", "active"}:
        raise HTTPException(status_code=404, detail="Input PDF not found")
    if row["status"] not in {"pending", "processing"}:
        raise HTTPException(status_code=409, detail="Job is not queue eligible")
    path = _inside_storage(row["stored_pdf_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Input PDF not found")
    return FileResponse(path, media_type="application/pdf", filename="input.pdf")


@app.post("/internal/jobs/{job_id}/start", dependencies=[Depends(_authenticate)])
def start_job(job_id: str, message: WorkerMessage) -> dict:
    database, row = _get_job(job_id)
    row = _recover_finalizing_result(database, row)
    if row["status"] in {"completed", "manual_review"}:
        raise HTTPException(status_code=409, detail="Job is already final")
    if row["status"] == "finalizing":
        raise HTTPException(status_code=409, detail="Job finalization is in progress")
    if row["status"] in {"failed", "cancelled"}:
        database.transition_conversion(job_id, "pending")
    elif row["status"] == "processing":
        if database.claim_matches(row, message.worker_name, message.claim_token):
            database.heartbeat(job_id, message.worker_name)
            return get_job(job_id)
        row = _recover_stale_processing_job(database, row)
        if row["status"] != "pending":
            raise HTTPException(
                status_code=409, detail="Job is owned by another worker"
            )
    if row["status"] == "processing":
        raise HTTPException(status_code=409, detail="Job is owned by another worker")
    database.transition_conversion(
        job_id, "processing", worker_name=message.worker_name
    )
    return get_job(job_id)


@app.post("/internal/jobs/{job_id}/heartbeat", dependencies=[Depends(_authenticate)])
def heartbeat(job_id: str, message: WorkerMessage) -> dict:
    database, row = _get_job(job_id)
    if not database.claim_matches(row, message.worker_name, message.claim_token):
        raise HTTPException(status_code=409, detail="Worker does not own this job")
    try:
        database.heartbeat(job_id, message.worker_name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "ok"}


@app.post("/internal/jobs/{job_id}/result", dependencies=[Depends(_authenticate)])
def upload_result(
    job_id: str,
    worker_name: str = Form(..., min_length=1, max_length=120),
    claim_token: str | None = Form(default=None, min_length=32, max_length=32),
    metadata: str = Form(..., max_length=65536),
    result: UploadFile = File(...),
) -> dict:
    if not re.fullmatch(WORKER_NAME_PATTERN, worker_name):
        raise HTTPException(status_code=400, detail="Invalid worker name")
    if claim_token and not re.fullmatch(CLAIM_TOKEN_PATTERN, claim_token):
        raise HTTPException(status_code=400, detail="Invalid claim token")
    database, row = _get_job(job_id)
    row = _recover_finalizing_result(database, row)
    if row["status"] in {"completed", "manual_review"}:
        raise HTTPException(status_code=409, detail="Job is already final")
    if row["status"] != "processing" or not database.claim_matches(
        row, worker_name, claim_token
    ):
        raise HTTPException(
            status_code=409, detail="Worker does not own this processing job"
        )
    if not result.filename or Path(result.filename).suffix.lower() != ".docx":
        raise HTTPException(status_code=400, detail="Only DOCX results are accepted")
    try:
        values = json.loads(metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid result metadata") from exc
    target_status = values.get("status")
    if target_status not in {"completed", "manual_review"}:
        raise HTTPException(status_code=400, detail="Invalid final status")
    quality_forced_review = False
    if target_status == "completed" and not _quality_allows_completed(values):
        target_status = "manual_review"
        quality_forced_review = True
    target = _inside_storage(row["stored_docx_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=target.parent, suffix=".docx.part", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        total = 0
        too_large = False
        limits = limits_from_env()
        while chunk := result.file.read(1024 * 1024):
            total += len(chunk)
            if total > limits.max_result_bytes:
                too_large = True
                break
            temporary.write(chunk)
        if not too_large:
            temporary.flush()
            os.fsync(temporary.fileno())
    if too_large:
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="Result is too large")
    if total < 4 or temporary_path.read_bytes()[:2] != b"PK":
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Invalid DOCX content")
    try:
        with zipfile.ZipFile(temporary_path) as archive:
            validate_zip_archive(
                archive,
                limits=ArchiveLimits(
                    max_members=limits.max_archive_members,
                    max_total_uncompressed_bytes=limits.max_decompressed_bytes,
                ),
            )
    except (ValueError, zipfile.BadZipFile) as exc:
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Unsafe DOCX archive") from exc
    try:
        prepared = database.prepare_conversion_finalization(
            job_id,
            target_status,
            worker_name=worker_name,
            claim_token=claim_token,
            error_message=(
                "Cloud cost limit reached; best available result requires manual review."
                if values.get("cost_limit_reached")
                else (
                    "Result quality is below the 90% acceptance gate or is indeterminate."
                    if quality_forced_review
                    else None
                )
            ),
            extra={
                "stored_docx_path": str(target),
                "file_type": values.get("file_type"),
                "text_quality_score": values.get("text_quality_score"),
                "layout_quality_score": values.get("layout_quality_score"),
                "final_quality_score": values.get("final_quality_score"),
                "winning_engine": values.get("winning_engine"),
                "processing_time": values.get("processing_time", 0),
            },
        )
    except ValueError as exc:
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        os.replace(temporary_path, target)
    except OSError:
        database.rollback_conversion_finalization(
            job_id, worker_name=worker_name, claim_token=claim_token
        )
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=503, detail="Result finalization is temporarily unavailable"
        ) from None
    try:
        updated = database.complete_conversion_finalization(
            job_id,
            target_status,
            worker_name=worker_name,
            claim_token=claim_token,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Result finalization is temporarily unavailable"
        ) from exc
    if prepared.get("guest_scope_id"):
        database.mark_guest_job_result(
            job_id, status=updated["status"], stored_docx_path=str(target)
        )
    applications = values.get("correction_applications") or []
    if isinstance(applications, list):
        database.record_correction_applications(job_id, applications[:500])
    attempts = values.get("cloud_attempts") or []
    if isinstance(attempts, list):
        existing_attempts = len(database.list_attempts(row["id"]))
        for offset, attempt in enumerate(attempts[:100], start=1):
            if not isinstance(attempt, dict):
                continue
            database.record_attempt(
                {
                    "conversion_id": row["id"],
                    "engine_name": str(attempt.get("provider") or "openrouter")[:80],
                    "model_name": str(attempt.get("model") or "")[:200],
                    "engine_type": "cloud",
                    "attempt_number": existing_attempts + offset,
                    "quality_score": attempt.get("score"),
                    "cost": float(attempt.get("cost") or 0),
                    "cost_is_estimated": int(bool(attempt.get("cost_is_estimated"))),
                    "prompt_tokens": int(attempt.get("prompt_tokens") or 0),
                    "completion_tokens": int(attempt.get("completion_tokens") or 0),
                    "processing_time": float(attempt.get("latency_ms") or 0) / 1000.0,
                    "success": int(not bool(attempt.get("failure_reason"))),
                    "failure_reason": (
                        str(attempt.get("failure_reason"))[:2000]
                        if attempt.get("failure_reason")
                        else None
                    ),
                    "created_at": utc_now(),
                }
            )
    return {"job_id": job_id, "status": updated["status"]}


@app.post("/internal/jobs/{job_id}/failure", dependencies=[Depends(_authenticate)])
def fail_job(job_id: str, message: FailureMessage) -> dict:
    database, row = _get_job(job_id)
    if row["status"] == "failed":
        return {"job_id": job_id, "status": "failed"}
    if row["status"] in {"completed", "manual_review"}:
        return {"job_id": job_id, "status": row["status"]}
    if row["status"] != "processing" or not database.claim_matches(
        row, message.worker_name, message.claim_token
    ):
        raise HTTPException(
            status_code=409, detail="Worker does not own this processing job"
        )
    safe_error = _safe_worker_failure_message(message.error)
    database.transition_conversion(
        job_id, "failed", worker_name=message.worker_name, error_message=safe_error
    )
    return {"job_id": job_id, "status": "failed"}
