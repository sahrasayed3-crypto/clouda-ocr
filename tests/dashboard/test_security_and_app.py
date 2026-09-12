from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _settings(tmp_path: Path):
    from clouda_lab.dashboard.settings import LabSettings

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "runs").mkdir()
    (repo / "benchmarks").mkdir()
    return LabSettings.from_repo(repo)


def test_lab_shell_and_offline_status_are_local_and_self_contained(tmp_path: Path):
    from clouda_lab.dashboard.app import create_app

    client = TestClient(create_app(_settings(tmp_path)))
    response = client.get("/lab")

    assert response.status_code == 200
    assert "Clouda Lab" in response.text
    assert "Clouda OCR Internal Control Center" in response.text
    assert "default-src 'self'" in response.headers["content-security-policy"]
    assert "https://" not in response.text

    offline = client.get("/api/lab/offline")
    assert offline.status_code == 200
    assert offline.json() == {
        "schema_version": "clouda.lab.offline.v1",
        "offline": True,
        "network_required": False,
        "automatic_model_download": False,
        "automatic_dataset_download": False,
        "remote_provider_calls": False,
    }


def test_non_loopback_clients_are_rejected(tmp_path: Path):
    from clouda_lab.dashboard.app import create_app

    client = TestClient(create_app(_settings(tmp_path)), client=("203.0.113.7", 4321))

    assert client.get("/lab").status_code == 403
    assert client.get("/api/lab/offline").status_code == 403


def test_security_helpers_reject_traversal_hide_private_paths_and_redact_secrets(
    tmp_path: Path,
):
    from clouda_lab.dashboard.security import (
        browser_safe,
        safe_identifier,
        safe_output_label,
        safe_relative_label,
        sanitize_payload,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    inside = repo / "runs" / "r1"
    outside = tmp_path / "private" / "secret.json"

    assert safe_identifier("dataset:v1.sha-256") == "dataset:v1.sha-256"
    for unsafe in ("../secret", "a/b", r"a\b", "", "."):
        try:
            safe_identifier(unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe identifier accepted: {unsafe!r}")
    assert safe_output_label("derived-v1") == "derived-v1"
    for unsafe_label in ("../derived", "dataset:version", "CON"):
        with pytest.raises(ValueError):
            safe_output_label(unsafe_label)

    assert safe_relative_label(inside, (repo,)) == "runs/r1"
    assert safe_relative_label(outside, (repo,)) is None

    payload = sanitize_payload(
        {
            "api_key": "super-secret",
            "nested": {"authorization": "Bearer abc", "ok": "visible"},
            "message": "token=plaintext-token",
        }
    )
    rendered = repr(payload)
    assert "super-secret" not in rendered
    assert "Bearer abc" not in rendered
    assert "plaintext-token" not in rendered
    assert payload["nested"]["ok"] == "visible"

    embedded = browser_safe(
        f"validation failed at {outside} with token=plaintext-token", (repo,)
    )
    assert str(outside) not in embedded
    assert "plaintext-token" not in embedded
    assert "[PRIVATE PATH]" in embedded


def test_settings_default_to_loopback_and_server_owned_roots(tmp_path: Path):
    settings = _settings(tmp_path)

    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.local_only is True
    assert settings.preview_limit == 20
    assert settings.repo_root == (tmp_path / "repo").resolve()
    assert settings.runs_root == settings.repo_root / "runs"
    assert settings.plans_root.is_relative_to(settings.repo_root)


def test_domain_api_routes_use_real_services_and_protect_actions(tmp_path: Path):
    from tests.dashboard.test_catalog import _write_dataset

    from clouda_lab.dashboard.app import create_app
    from clouda_lab.dashboard.settings import LabSettings

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "runs").mkdir()
    (repo / "benchmarks").mkdir()
    _write_dataset(repo, dataset_id="safe-set", version="v1")
    settings = LabSettings.from_repo(repo)
    from clouda_data.results.service import ResultsService

    result_run = ResultsService(settings.results_root).create_run(
        model_id="model-a", dataset_id="safe-set"
    )
    client = TestClient(create_app(settings))

    for route in (
        "/api/lab/overview",
        "/api/lab/datasets",
        "/api/lab/dataset-sources",
        "/api/lab/datasets/safe-set",
        "/api/lab/datasets/safe-set/preview?limit=2",
        "/api/lab/quality?dataset_id=safe-set",
        "/api/lab/models",
        "/api/lab/planner/options",
        "/api/lab/plans",
        "/api/lab/runs",
        "/api/lab/results",
        "/api/lab/benchmarks",
        "/api/lab/doctor/latest",
        "/api/lab/hardware",
    ):
        response = client.get(route)
        assert response.status_code == 200, (route, response.text)

    result_detail = client.get(f"/api/lab/results/{result_run.run_id}?metric_limit=10")
    assert result_detail.status_code == 200
    assert result_detail.json()["integrity"]["valid"] is True

    body = {
        "experiment_name": "api-plan",
        "adapter_type": "hunyuanocr15_sft",
        "model_id": "Tencent-Hunyuan/HunyuanOCR-1.5",
        "dataset_id": "safe-set",
        "precision": "bf16",
        "max_steps": 2,
    }
    assert client.post("/api/lab/plans", json=body).status_code == 403
    token = client.get("/api/lab/session").json()["action_token"]
    headers = {"X-Clouda-Lab-Action": token}
    plan = client.post("/api/lab/plans", json=body, headers=headers)
    assert plan.status_code == 200
    plan_id = plan.json()["plan_id"]
    assert client.get(f"/api/lab/plans/{plan_id}").status_code == 200
    assert (
        client.post(
            "/api/lab/preflight",
            json={"plan_id": plan_id, "write_probe": False},
            headers=headers,
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/lab/doctor/run", json={"deep": False}, headers=headers
        ).status_code
        == 200
    )
