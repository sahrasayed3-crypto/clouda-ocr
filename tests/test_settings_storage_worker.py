import io
from pathlib import Path

import pytest

from pdfword import settings, worker
from pdfword.storage import (
    atomic_write,
    atomic_write_stream,
    cleanup_temporary,
    create_job_storage,
    safe_component,
    save_job_files,
)


class FakeSettingsDatabase:
    def __init__(self) -> None:
        self.saved: dict[str, tuple[str, str]] = {}

    def get_setting_rows(self):
        return {
            "acceptance_threshold": {"value": "88.5", "type": "float"},
            "local_attempts": {"value": "2", "type": "int"},
            "enabled_models": {"value": '["a", "b"]', "type": "json"},
            "local_processing_enabled": {"value": "true", "type": "bool"},
            "bad_json": {"value": "{", "type": "json"},
        }

    def set_setting(self, key, value, value_type):
        self.saved[key] = (value, value_type)


def test_runtime_settings_env_bounds_and_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ROLE", "WORKER")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("SERVER_BASE_URL", "http://server/")
    monkeypatch.setenv("CLOUDA_REDIS_NAMESPACE", "clouda-test")
    monkeypatch.setenv("LOCAL_PROCESSING_ENABLED", "yes")
    monkeypatch.setenv("WORKER_CONCURRENCY", "0")
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("CORRECTION_AUTO_APPLY_THRESHOLD", "2.5")

    config = settings.runtime_settings()

    assert config.app_role == "worker"
    assert config.server_base_url == "http://server"
    assert config.local_processing_enabled is True
    assert config.redis_namespace == "clouda-test"
    assert config.worker_concurrency == 1
    assert config.job_timeout_seconds == 60
    assert config.correction_auto_apply_threshold == 1.0
    assert config.temp_root == Path(tmp_path / "store" / "temporary")


def test_load_validate_and_save_settings(monkeypatch):
    database = FakeSettingsDatabase()
    loaded = settings.load_settings(database)
    assert loaded["acceptance_threshold"] == 88.5
    assert loaded["local_attempts"] == 2
    assert loaded["enabled_models"] == ["a", "b"]

    monkeypatch.setattr(
        "pdfword.settings.get_engine_registry",
        lambda: type("Registry", (), {"names": lambda self: ["direct_pdf_text"]})(),
    )
    settings.save_settings(
        database,
        {
            "enabled_engines": ["direct_pdf_text"],
            "daily_cost_limit": 1.25,
            "correction_memory_enabled": True,
        },
    )
    assert database.saved["enabled_engines"][1] == "json"
    assert database.saved["daily_cost_limit"] == ("1.25", "float")
    assert database.saved["correction_memory_enabled"] == ("true", "bool")

    with pytest.raises(ValueError):
        settings.validate_setting("acceptance_threshold", 101)
    with pytest.raises(ValueError):
        settings.validate_setting("max_concurrent_jobs", 3)
    with pytest.raises(ValueError):
        settings.validate_setting("enabled_engines", ["missing"])


def test_storage_job_lifecycle_and_stream_cleanup(tmp_path):
    assert safe_component(" bad:/name?.pdf ", "fallback") == "bad_name_.pdf"
    assert safe_component(" . ", "fallback") == "fallback"

    job = create_job_storage("user:name", "input?.pdf", "out?.docx", root=tmp_path)
    assert job.temporary_dir.is_dir()
    assert "user_name" in str(job.root)

    save_job_files(job, b"%PDF", b"DOCX")
    assert job.pdf_path.read_bytes() == b"%PDF"
    assert job.docx_path.read_bytes() == b"DOCX"

    atomic_write(job.root / "nested" / "file.bin", b"abc")
    assert (job.root / "nested" / "file.bin").read_bytes() == b"abc"

    class ExplodingStream(io.BytesIO):
        def read(self, size=-1):
            raise RuntimeError("stream failed")

    with pytest.raises(RuntimeError):
        atomic_write_stream(job.root / "broken.bin", ExplodingStream(b"abc"))
    assert not (job.root / "broken.bin").exists()

    cleanup_temporary(job)
    assert not job.temporary_dir.exists()


def test_worker_health_and_startup_guards(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("APP_ROLE", "server")
    assert worker.main() == 2
    assert "APP_ROLE=worker" in capsys.readouterr().err

    monkeypatch.setenv("APP_ROLE", "worker")
    monkeypatch.setenv("WORKER_CONCURRENCY", "2")
    assert worker.main() == 2

    monkeypatch.setenv("WORKER_CONCURRENCY", "1")
    monkeypatch.delenv("WORKER_API_KEY", raising=False)
    assert worker.main() == 2

    monkeypatch.setenv("TEMP_ROOT", str(tmp_path))
    monkeypatch.setenv("WORKER_NAME", "win-worker")
    monkeypatch.setattr(
        "pdfword.local_engines.available_engine_status",
        lambda: [{"engine": "direct_pdf_text", "active": True}],
    )
    health = worker.worker_health()
    assert health["worker_name"] == "win-worker"
    assert health["active_engines"] == ["direct_pdf_text"]


def test_worker_uses_redis_security_config_and_namespaced_queue(monkeypatch, tmp_path):
    events = []

    class FakeConnection:
        def ping(self):
            events.append(("ping",))

    class FakeClient:
        def health(self):
            events.append(("health",))

        def report_status(self, worker_name, cloud_available, cloud_provider):
            events.append(("status", worker_name, cloud_available, cloud_provider))

    class FakeWorker:
        def __init__(self, queues, *, connection, name):
            events.append(("worker", queues[0].name, connection, name))

        def work(self, *, with_scheduler):
            events.append(("work", with_scheduler))

    def fake_from_url(url, **kwargs):
        events.append(("redis", url, kwargs))
        return FakeConnection()

    monkeypatch.setenv("APP_ROLE", "worker")
    monkeypatch.setenv("WORKER_API_KEY", "x" * 32)
    monkeypatch.setenv("WORKER_NAME", "win-worker")
    monkeypatch.setenv("TEMP_ROOT", str(tmp_path))
    monkeypatch.setenv("REDIS_URL", "rediss://redis.example.com:6379/0")
    monkeypatch.setenv("CLOUDA_REDIS_TLS_REQUIRED", "true")
    monkeypatch.setenv("CLOUDA_REDIS_NAMESPACE", "clouda-prod")
    monkeypatch.setenv("RQ_QUEUE_NAME", "conversion")
    monkeypatch.setattr("redis.Redis.from_url", fake_from_url)
    monkeypatch.setattr("pdfword.worker_client.WorkerApiClient", FakeClient)
    monkeypatch.setattr("rq.SimpleWorker", FakeWorker)
    monkeypatch.setattr("rq.Worker", FakeWorker)

    assert worker.main() == 0
    assert events[0][0] == "redis"
    assert events[0][1] == "rediss://redis.example.com:6379/0"
    assert events[0][2]["ssl_cert_reqs"] == "required"
    assert any(event[:2] == ("worker", "clouda-prod:conversion") for event in events)
