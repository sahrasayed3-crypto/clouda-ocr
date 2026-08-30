from pathlib import Path

from pdfword.production_preflight import validate_production_environment


def valid_env(tmp_path: Path) -> dict[str, str]:
    outside = tmp_path / "outside"
    outside.mkdir()
    return {
        "CLOUDA_ENV": "production",
        "AUTH_MODE": "firebase_staging",
        "CLOUDA_FIREBASE_PROJECT_ID": "clouda-prod-main",
        "FIREBASE_PROJECT_ID": "clouda-prod-main",
        "CLOUDA_PRODUCTION_FIREBASE_PROJECT_ALLOWLIST": "clouda-prod-main",
        "FIREBASE_CLIENT_API_KEY": "firebase-public-web-config-value",
        "FIREBASE_CLIENT_AUTH_DOMAIN": "clouda.example.com",
        "FIREBASE_CLIENT_PROJECT_ID": "clouda-prod-main",
        "CLOUDA_USE_ATTACHED_IDENTITY": "true",
        "CLOUDA_ALLOWED_HOSTS": "app.example.com,api.example.com",
        "CLOUDA_ALLOWED_ORIGINS": "https://app.example.com",
        "CLOUDA_TRUSTED_PROXY_CIDRS": "10.0.0.0/24",
        "CLOUDA_SESSION_COOKIE_SECURE": "true",
        "CLOUDA_REDIS_TLS_REQUIRED": "true",
        "REDIS_URL": "rediss://redis.example.com:6379/0",
        "CLOUDA_REDIS_NAMESPACE": "clouda-prod",
        "RQ_QUEUE_NAME": "pdf_conversion_prod",
        "CLOUDA_DATABASE_PATH": str(outside / "clouda.sqlite3"),
        "STORAGE_ROOT": str(outside / "storage"),
        "TEMP_ROOT": str(outside / "temp"),
        "CLOUDA_BACKUP_ROOT": str(outside / "backups"),
        "CLOUDA_BACKUP_RPO_MINUTES": "60",
        "CLOUDA_BACKUP_RTO_MINUTES": "240",
        "CLOUDA_MONITORING_CONFIGURED": "true",
        "CLOUDA_ALERT_ROUTING_CONFIGURED": "true",
        "CLOUDA_AUDIT_RETENTION_DAYS": "365",
        "WORKER_API_KEY": "x" * 32,
        "CLOUDA_PROCESSING_STALE_SECONDS": "900",
        "CLOUDA_DEBUG": "false",
        "CLOUDA_ALLOW_LIVE_FIREBASE_TESTS": "0",
    }


def failed_names(report: dict) -> set[str]:
    return {item["name"] for item in report["checks"] if item["status"] == "FAIL"}


def test_production_preflight_accepts_complete_sanitized_configuration(tmp_path):
    report = validate_production_environment(valid_env(tmp_path), project_root=tmp_path)

    assert report["ready"] is True
    assert report["failure_count"] == 0


def test_production_preflight_rejects_missing_placeholder_and_emulator_values(tmp_path):
    env = valid_env(tmp_path)
    env.update(
        {
            "CLOUDA_FIREBASE_PROJECT_ID": "local-test-project",
            "CLOUDA_PRODUCTION_FIREBASE_PROJECT_ALLOWLIST": "",
            "FIREBASE_AUTH_EMULATOR_HOST": "127.0.0.1:9099",
            "FIREBASE_CLIENT_API_KEY": "placeholder-public-web-api-key",
            "CLOUDA_ALLOWED_HOSTS": "localhost",
            "CLOUDA_ALLOWED_ORIGINS": "http://localhost:8501,*",
            "CLOUDA_SESSION_COOKIE_SECURE": "false",
            "REDIS_URL": "redis://127.0.0.1:6379/0",
            "CLOUDA_REDIS_NAMESPACE": "clouda-local",
            "RQ_QUEUE_NAME": "pdf_conversion",
            "CLOUDA_BACKUP_RPO_MINUTES": "0",
            "CLOUDA_BACKUP_RTO_MINUTES": "",
            "CLOUDA_MONITORING_CONFIGURED": "false",
            "CLOUDA_ALERT_ROUTING_CONFIGURED": "0",
            "CLOUDA_AUDIT_RETENTION_DAYS": "0",
            "CLOUDA_DEBUG": "true",
        }
    )

    report = validate_production_environment(env, project_root=tmp_path)

    assert report["ready"] is False
    assert {
        "CLOUDA_FIREBASE_PROJECT_ID",
        "CLOUDA_PRODUCTION_FIREBASE_PROJECT_ALLOWLIST",
        "FIREBASE_AUTH_EMULATOR_HOST",
        "FIREBASE_CLIENT_API_KEY",
        "CLOUDA_ALLOWED_HOSTS",
        "CLOUDA_ALLOWED_ORIGINS",
        "CLOUDA_SESSION_COOKIE_SECURE",
        "REDIS_URL",
        "CLOUDA_REDIS_NAMESPACE",
        "RQ_QUEUE_NAME",
        "CLOUDA_BACKUP_RPO_MINUTES",
        "CLOUDA_BACKUP_RTO_MINUTES",
        "CLOUDA_MONITORING_CONFIGURED",
        "CLOUDA_ALERT_ROUTING_CONFIGURED",
        "CLOUDA_AUDIT_RETENTION_DAYS",
        "debug_switches",
    } <= failed_names(report)


def test_production_preflight_rejects_multi_worker_topology(tmp_path):
    env = valid_env(tmp_path)
    env["WEB_CONCURRENCY"] = "4"

    report = validate_production_environment(env, project_root=tmp_path)

    assert "single_process_api" in failed_names(report)


def test_production_preflight_rejects_staging_project_and_repo_credential(tmp_path):
    env = valid_env(tmp_path)
    env.update(
        {
            "CLOUDA_FIREBASE_PROJECT_ID": "ocr-project-c57d8",
            "CLOUDA_PRODUCTION_FIREBASE_PROJECT_ALLOWLIST": "ocr-project-c57d8",
            "CLOUDA_USE_ATTACHED_IDENTITY": "false",
            "GOOGLE_APPLICATION_CREDENTIALS": str(tmp_path / "service-account.json"),
        }
    )

    report = validate_production_environment(env, project_root=tmp_path)

    assert report["ready"] is False
    assert {"staging_project_guard", "credential_path_outside_repo"} <= failed_names(
        report
    )
