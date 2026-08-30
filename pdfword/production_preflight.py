from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PLACEHOLDER_MARKERS = {
    "",
    "placeholder",
    "replace-me",
    "change-me",
    "your-project-id",
    "your-firebase-project-id",
    "local-test-project",
}

STAGING_PROJECT_IDS = {"ocr-project-c57d8"}


@dataclass(frozen=True)
class PreflightCheck:
    category: str
    name: str
    status: str
    required_action: str


def _value(env: dict[str, str], name: str) -> str:
    return str(env.get(name) or "").strip()


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(value: str) -> bool:
    return value.isdigit() and int(value) > 0


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().casefold()
    return normalized in PLACEHOLDER_MARKERS or "placeholder" in normalized


def _check(
    checks: list[PreflightCheck],
    category: str,
    name: str,
    ok: bool,
    required_action: str,
) -> None:
    checks.append(
        PreflightCheck(
            category=category,
            name=name,
            status="PASS" if ok else "FAIL",
            required_action="" if ok else required_action,
        )
    )


def validate_production_environment(
    env: dict[str, str] | None = None, *, project_root: str | Path | None = None
) -> dict:
    values = dict(os.environ if env is None else env)
    root = Path(project_root or Path.cwd()).resolve()
    checks: list[PreflightCheck] = []

    clouda_env = _value(values, "CLOUDA_ENV").lower()
    auth_mode = _value(values, "AUTH_MODE").lower()
    project_id = _value(values, "CLOUDA_FIREBASE_PROJECT_ID") or _value(
        values, "FIREBASE_PROJECT_ID"
    )
    allowlist = {
        item.strip()
        for item in _value(
            values, "CLOUDA_PRODUCTION_FIREBASE_PROJECT_ALLOWLIST"
        ).split(",")
        if item.strip()
    }

    _check(
        checks,
        "environment",
        "CLOUDA_ENV",
        clouda_env == "production",
        "Set CLOUDA_ENV=production in the protected runtime environment.",
    )
    _check(
        checks,
        "identity",
        "AUTH_MODE",
        auth_mode == "firebase_staging",
        "Use AUTH_MODE=firebase_staging until a dedicated firebase_production mode exists.",
    )
    _check(
        checks,
        "identity",
        "CLOUDA_FIREBASE_PROJECT_ID",
        bool(project_id) and not _is_placeholder(project_id),
        "Set a non-placeholder production Firebase project ID.",
    )
    _check(
        checks,
        "identity",
        "CLOUDA_PRODUCTION_FIREBASE_PROJECT_ALLOWLIST",
        bool(allowlist) and project_id in allowlist,
        "Allowlist the exact production Firebase project ID.",
    )
    _check(
        checks,
        "identity",
        "staging_project_guard",
        project_id not in STAGING_PROJECT_IDS,
        "Do not use the staging Firebase project for production.",
    )
    _check(
        checks,
        "identity",
        "FIREBASE_AUTH_EMULATOR_HOST",
        not _value(values, "FIREBASE_AUTH_EMULATOR_HOST"),
        "Remove Firebase emulator variables from production.",
    )
    _check(
        checks,
        "identity",
        "FIREBASE_CLIENT_API_KEY",
        bool(_value(values, "FIREBASE_CLIENT_API_KEY"))
        and not _is_placeholder(_value(values, "FIREBASE_CLIENT_API_KEY")),
        "Provide the production Firebase Web API key from the Firebase console.",
    )
    _check(
        checks,
        "identity",
        "FIREBASE_CLIENT_AUTH_DOMAIN",
        bool(_value(values, "FIREBASE_CLIENT_AUTH_DOMAIN"))
        and "localhost" not in _value(values, "FIREBASE_CLIENT_AUTH_DOMAIN").lower(),
        "Configure the production Firebase auth domain.",
    )

    attached_identity = _truthy(_value(values, "CLOUDA_USE_ATTACHED_IDENTITY"))
    credential_paths = [
        _value(values, "GOOGLE_APPLICATION_CREDENTIALS"),
        _value(values, "FIREBASE_SERVICE_ACCOUNT_JSON_PATH"),
    ]
    credential_path = next((item for item in credential_paths if item), "")
    credential_outside_repo = True
    if credential_path:
        try:
            Path(credential_path).expanduser().resolve().relative_to(root)
            credential_outside_repo = False
        except ValueError:
            credential_outside_repo = True
    _check(
        checks,
        "secrets",
        "admin_credential_source",
        attached_identity or bool(credential_path),
        "Use attached runtime identity or a protected external credential path.",
    )
    _check(
        checks,
        "secrets",
        "credential_path_outside_repo",
        not credential_path or credential_outside_repo,
        "Keep credential files outside the repository.",
    )

    _check(
        checks,
        "network",
        "CLOUDA_ALLOWED_HOSTS",
        bool(_value(values, "CLOUDA_ALLOWED_HOSTS"))
        and "*" not in _value(values, "CLOUDA_ALLOWED_HOSTS")
        and "localhost" not in _value(values, "CLOUDA_ALLOWED_HOSTS").lower(),
        "Set explicit production trusted hosts only.",
    )
    _check(
        checks,
        "network",
        "CLOUDA_ALLOWED_ORIGINS",
        bool(_value(values, "CLOUDA_ALLOWED_ORIGINS"))
        and "localhost" not in _value(values, "CLOUDA_ALLOWED_ORIGINS").lower()
        and "*" not in _value(values, "CLOUDA_ALLOWED_ORIGINS")
        and all(
            item.strip().startswith("https://")
            for item in _value(values, "CLOUDA_ALLOWED_ORIGINS").split(",")
            if item.strip()
        ),
        "Set explicit production CORS/frontend origins.",
    )
    _check(
        checks,
        "network",
        "CLOUDA_TRUSTED_PROXY_CIDRS",
        bool(_value(values, "CLOUDA_TRUSTED_PROXY_CIDRS")),
        "Declare trusted reverse-proxy CIDRs.",
    )
    _check(
        checks,
        "session",
        "CLOUDA_SESSION_COOKIE_SECURE",
        _truthy(_value(values, "CLOUDA_SESSION_COOKIE_SECURE")),
        "Require secure cookies behind TLS.",
    )
    _check(
        checks,
        "redis",
        "CLOUDA_REDIS_TLS_REQUIRED",
        _truthy(_value(values, "CLOUDA_REDIS_TLS_REQUIRED")),
        "Require Redis TLS or an equivalent private managed transport.",
    )
    _check(
        checks,
        "redis",
        "REDIS_URL",
        _value(values, "REDIS_URL").startswith("rediss://"),
        "Use a rediss:// production Redis URL when TLS is required.",
    )
    redis_namespace = _value(values, "CLOUDA_REDIS_NAMESPACE")
    _check(
        checks,
        "redis",
        "CLOUDA_REDIS_NAMESPACE",
        bool(redis_namespace)
        and redis_namespace not in {"clouda", "clouda-local", "local", "test"},
        "Set a unique production Redis namespace.",
    )
    _check(
        checks,
        "redis",
        "RQ_QUEUE_NAME",
        bool(_value(values, "RQ_QUEUE_NAME"))
        and _value(values, "RQ_QUEUE_NAME") != "pdf_conversion",
        "Set a production-specific queue name instead of the default.",
    )

    for name in (
        "CLOUDA_DATABASE_PATH",
        "STORAGE_ROOT",
        "TEMP_ROOT",
        "CLOUDA_BACKUP_ROOT",
    ):
        raw = _value(values, name)
        _check(
            checks,
            "storage",
            name,
            bool(raw) and Path(raw).is_absolute(),
            f"Set {name} to an absolute production path outside the source tree.",
        )

    _check(
        checks,
        "backup",
        "CLOUDA_BACKUP_RPO_MINUTES",
        _positive_int(_value(values, "CLOUDA_BACKUP_RPO_MINUTES")),
        "Set a positive backup recovery point objective in minutes.",
    )
    _check(
        checks,
        "backup",
        "CLOUDA_BACKUP_RTO_MINUTES",
        _positive_int(_value(values, "CLOUDA_BACKUP_RTO_MINUTES")),
        "Set a positive backup recovery time objective in minutes.",
    )
    _check(
        checks,
        "monitoring",
        "CLOUDA_MONITORING_CONFIGURED",
        _truthy(_value(values, "CLOUDA_MONITORING_CONFIGURED")),
        "Confirm production monitoring is configured in protected runtime metadata.",
    )
    _check(
        checks,
        "monitoring",
        "CLOUDA_ALERT_ROUTING_CONFIGURED",
        _truthy(_value(values, "CLOUDA_ALERT_ROUTING_CONFIGURED")),
        "Confirm production alert routing is configured in protected runtime metadata.",
    )
    _check(
        checks,
        "monitoring",
        "CLOUDA_AUDIT_RETENTION_DAYS",
        _positive_int(_value(values, "CLOUDA_AUDIT_RETENTION_DAYS")),
        "Set a positive audit retention period in days.",
    )

    worker_key = _value(values, "WORKER_API_KEY")
    _check(
        checks,
        "worker",
        "WORKER_API_KEY",
        len(worker_key) >= 32 and not _is_placeholder(worker_key),
        "Provide a high-entropy worker API key through protected secret storage.",
    )
    _check(
        checks,
        "worker",
        "CLOUDA_PROCESSING_STALE_SECONDS",
        _value(values, "CLOUDA_PROCESSING_STALE_SECONDS").isdigit(),
        "Set the stale processing job threshold.",
    )
    _check(
        checks,
        "safety",
        "debug_switches",
        not _truthy(_value(values, "CLOUDA_DEBUG"))
        and not _truthy(_value(values, "CLOUDA_ALLOW_LIVE_FIREBASE_TESTS")),
        "Disable debug and live-test switches in production.",
    )
    web_concurrency = _value(values, "WEB_CONCURRENCY") or "1"
    _check(
        checks,
        "topology",
        "single_process_api",
        web_concurrency == "1",
        "In-process rate limiter requires --workers=1. Externalize to Redis before scaling.",
    )

    failures = [check for check in checks if check.status != "PASS"]
    return {
        "ready": not failures,
        "checks": [check.__dict__ for check in checks],
        "failure_count": len(failures),
    }


def main() -> int:
    report = validate_production_environment()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
