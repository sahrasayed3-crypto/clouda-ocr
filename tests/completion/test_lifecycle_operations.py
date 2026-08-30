from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clouda_data.lifecycle import archive_run, cleanup, verify_archive
from pdfword.operations import (
    ExternalOIDCRequired,
    RedisSecurityConfig,
    SlidingWindowRateLimiter,
    credential_matches,
    redact,
    FileSecretSource,
    OperationsMetrics,
    structured_log,
)
from pdfword.worker_api import app


@pytest.fixture()
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "state"
    monkeypatch.setenv("CLOUDA_STATE_HOME", str(root))
    monkeypatch.setenv("CLOUDA_DATABASE_PATH", str(root / "runtime" / "db.sqlite3"))
    monkeypatch.setenv("WORKER_API_KEY", "current-test-key")
    for name in ("datasets", "artifacts", "cache", "models", "runtime"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def test_cleanup_is_dry_run_and_recoverable(state: Path) -> None:
    preview = state / "artifacts" / "previews" / "run" / "preview.png"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"preview")
    plan = cleanup("preview")
    assert plan["dry_run"]
    assert preview.exists()
    with pytest.raises(PermissionError):
        cleanup("preview", dry_run=False)
    result = cleanup(
        "preview",
        dry_run=False,
        confirmation=plan["confirmation_token"],
    )
    assert not preview.exists()
    assert Path(result["recoverable_trash"]).is_dir()


def test_archive_and_verify(state: Path) -> None:
    run = state / "datasets" / "distorted" / "run-1"
    run.mkdir(parents=True)
    (run / "asset.png").write_bytes(b"not-private-synthetic")
    plan = archive_run(run)
    assert plan["dry_run"]
    result = archive_run(
        run,
        dry_run=False,
        confirmation=plan["confirmation_token"],
    )
    report = verify_archive(result["archive"])
    assert report["passed"]
    assert (run / "asset.png").exists()


def test_rate_limiter_is_bounded() -> None:
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=10)
    assert limiter.allow("client", now=1)
    assert limiter.allow("client", now=2)
    assert not limiter.allow("client", now=3)
    assert limiter.allow("client", now=12)


def test_redaction_and_credential_rotation() -> None:
    value = redact({"api_key": "secret", "nested": {"password": "private"}, "ok": 1})
    assert value == {
        "api_key": "[REDACTED]",
        "nested": {"password": "[REDACTED]"},
        "ok": 1,
    }
    assert credential_matches("old", "new", "old")
    assert not credential_matches("wrong", "new", "old")


def test_redis_tls_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("CLOUDA_REDIS_TLS_REQUIRED", "true")
    with pytest.raises(ValueError):
        RedisSecurityConfig.from_env()


def test_oidc_boundary_does_not_fake_authentication() -> None:
    with pytest.raises(RuntimeError):
        ExternalOIDCRequired().authenticate("Bearer test")


def test_request_id_security_headers_and_metrics_auth(state: Path) -> None:
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.headers["X-Request-ID"]
    assert health.headers["X-Content-Type-Options"] == "nosniff"
    assert client.get("/internal/metrics").status_code == 401
    metrics = client.get(
        "/internal/metrics",
        headers={"X-Worker-API-Key": "current-test-key"},
    )
    assert metrics.status_code == 200
    assert "clouda_http_requests_total" in metrics.text


def test_file_secret_source_stays_inside_root(tmp_path: Path) -> None:
    root = tmp_path / "secrets"
    root.mkdir()
    (root / "WORKER_KEY").write_text("test-value", encoding="utf-8")
    source = FileSecretSource(root)
    assert source.get("WORKER_KEY") == "test-value"
    assert source.get("MISSING") is None
    with pytest.raises(ValueError):
        source.get("../escape")


def test_structured_logging_redacts(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="clouda.operations")
    structured_log(
        "test", api_key="private", safe="visible"  # pragma: allowlist secret
    )
    assert "private" not in caplog.text
    assert "visible" in caplog.text


def test_operations_metrics_prometheus() -> None:
    metrics = OperationsMetrics()
    metrics.observe_request("GET", 200)
    text = metrics.prometheus(queue_depth=3, failed_jobs=2)
    assert 'method="GET",status="200"' in text
    assert "clouda_queue_depth 3" in text
    assert "clouda_failed_jobs 2" in text
