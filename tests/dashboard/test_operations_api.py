from __future__ import annotations

import shutil
import time
from pathlib import Path

from fastapi.testclient import TestClient


def _client(tmp_path: Path):
    from clouda_lab.dashboard.app import create_app
    from clouda_lab.dashboard.settings import LabSettings

    repo = tmp_path / "repo"
    repo.mkdir()
    source = Path.cwd() / "benchmarks" / "ocr_arabic"
    target = repo / "benchmarks" / "ocr_arabic"
    target.parent.mkdir(parents=True)
    shutil.copytree(source, target)
    settings = LabSettings.from_repo(repo)
    return TestClient(create_app(settings)), settings


def _wait(client: TestClient, task_id: str):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(f"/api/lab/tasks/{task_id}")
        assert response.status_code == 200
        task = response.json()
        if task["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            return task
        time.sleep(0.01)
    raise AssertionError("API operation did not finish")


def test_operational_reads_are_real_and_actions_require_token(tmp_path: Path):
    client, _settings = _client(tmp_path)
    for route in (
        "/api/lab/tasks",
        "/api/lab/downloads",
        "/api/lab/sources",
        "/api/lab/imports",
        "/api/lab/model-catalog",
        "/api/lab/model-catalog/qwen3-vl-4b-instruct",
        "/api/lab/storage",
        "/api/lab/benchmark-workspace",
        "/api/lab/benchmark-results",
        "/api/lab/training/capabilities",
    ):
        response = client.get(route)
        assert response.status_code == 200, (route, response.text)

    assert (
        client.post(
            "/api/lab/benchmark-plans",
            json={"model_ids": ["hunyuanocr-1-5"]},
        ).status_code
        == 403
    )


def test_model_asset_lifecycle_uses_managed_ids_and_two_phase_removal(tmp_path: Path):
    client, settings = _client(tmp_path)
    assets = settings.models_root / "qwen-local"
    assets.mkdir(parents=True)
    (assets / "config.json").write_text("{}", encoding="utf-8")
    token = client.get("/api/lab/session").json()["action_token"]
    headers = {"X-Clouda-Lab-Action": token}

    configured = client.post(
        "/api/lab/model-catalog/qwen3-vl-4b-instruct/assets",
        json={"asset_id": "qwen-local"},
        headers=headers,
    )
    assert configured.status_code == 200
    verified = client.post(
        "/api/lab/model-catalog/qwen3-vl-4b-instruct/verify",
        json={},
        headers=headers,
    )
    assert _wait(client, verified.json()["task_id"])["status"] == "COMPLETED"

    plan = client.post(
        "/api/lab/model-catalog/qwen3-vl-4b-instruct/removal-plan",
        json={},
        headers=headers,
    ).json()
    denied = client.post(
        "/api/lab/model-removals",
        json={"plan_id": plan["plan_id"], "confirmation": "wrong"},
        headers=headers,
    )
    assert denied.status_code == 409
    removed = client.post(
        "/api/lab/model-removals",
        json={
            "plan_id": plan["plan_id"],
            "confirmation": plan["confirmation_token"],
        },
        headers=headers,
    )
    assert _wait(client, removed.json()["task_id"])["status"] == "COMPLETED"
    assert not assets.exists()


def test_strict_operation_bodies_reject_paths_urls_commands_and_packages(
    tmp_path: Path,
):
    client, _settings = _client(tmp_path)
    token = client.get("/api/lab/session").json()["action_token"]
    headers = {"X-Clouda-Lab-Action": token}

    for body in (
        {"asset_id": "qwen-local", "path": "C:/private/model"},
        {"asset_id": "qwen-local", "url": "https://example.invalid/model"},
        {"asset_id": "qwen-local", "command": "whoami"},
        {"asset_id": "qwen-local", "package": "anything"},
    ):
        response = client.post(
            "/api/lab/model-catalog/qwen3-vl-4b-instruct/assets",
            json=body,
            headers=headers,
        )
        assert response.status_code == 422


def test_benchmark_plan_and_comparison_are_metadata_only(tmp_path: Path):
    client, _settings = _client(tmp_path)
    token = client.get("/api/lab/session").json()["action_token"]
    headers = {"X-Clouda-Lab-Action": token}
    payload = {"model_ids": ["hunyuanocr-1-5", "dots-mocr"]}
    first = client.post("/api/lab/benchmark-plans", json=payload, headers=headers)
    second = client.post("/api/lab/benchmark-plans", json=payload, headers=headers)
    assert first.status_code == 200
    assert first.json()["plan_id"] == second.json()["plan_id"]
    assert first.json()["execution"]["available"] is False

    comparison = client.get(
        "/api/lab/benchmark-comparison",
        params={"left": "hunyuanocr-1-5", "right": "dots-mocr"},
    )
    assert comparison.status_code == 200
    assert comparison.json()["winner"] is None
