import json
from pathlib import Path
from typing import BinaryIO

import requests

from .settings import runtime_settings


class WorkerApiClient:
    def __init__(self) -> None:
        config = runtime_settings()
        self.base_url = config.server_base_url
        self.headers = {"X-Worker-API-Key": config.worker_api_key}
        self.timeout = (15, 300)
        self._claim_tokens: dict[str, str] = {}

    def _request(self, method: str, path: str, **kwargs):
        response = requests.request(
            method,
            f"{self.base_url}{path}",
            headers=self.headers,
            timeout=self.timeout,
            **kwargs,
        )
        response.raise_for_status()
        if response.headers.get("content-type", "").startswith("application/json"):
            return response.json()
        return response

    def health(self) -> dict:
        return self._request("GET", "/internal/health")

    def correction_snapshot(self) -> dict:
        return self._request("GET", "/internal/corrections/snapshot")

    def report_status(
        self, worker_name: str, cloud_available: bool, cloud_provider: str = ""
    ) -> dict:
        try:
            return self._request(
                "POST",
                "/internal/workers/status",
                json={
                    "worker_name": worker_name,
                    "cloud_available": cloud_available,
                    "cloud_provider": cloud_provider,
                },
            )
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return {"status": "unsupported", "legacy_server": True}
            raise

    def get_job(self, job_id: str) -> dict:
        return self._request("GET", f"/internal/jobs/{job_id}")

    def download_input(self, job_id: str, target: Path) -> None:
        response = self._request("GET", f"/internal/jobs/{job_id}/input", stream=True)
        with target.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)

    def start(self, job_id: str, worker_name: str) -> dict:
        payload = self._request(
            "POST", f"/internal/jobs/{job_id}/start", json={"worker_name": worker_name}
        )
        token = payload.get("claim_token")
        if token:
            self._claim_tokens[job_id] = token
        return payload

    def heartbeat(self, job_id: str, worker_name: str) -> None:
        payload = {"worker_name": worker_name}
        if self._claim_tokens.get(job_id):
            payload["claim_token"] = self._claim_tokens[job_id]
        self._request(
            "POST",
            f"/internal/jobs/{job_id}/heartbeat",
            json=payload,
        )

    def upload_result(
        self, job_id: str, worker_name: str, docx: BinaryIO, metadata: dict
    ) -> dict:
        data = {"worker_name": worker_name, "metadata": json.dumps(metadata)}
        if self._claim_tokens.get(job_id):
            data["claim_token"] = self._claim_tokens[job_id]
        return self._request(
            "POST",
            f"/internal/jobs/{job_id}/result",
            data=data,
            files={
                "result": (
                    "result.docx",
                    docx,
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

    def fail(self, job_id: str, worker_name: str, error: str) -> dict:
        payload = {"worker_name": worker_name, "error": error[:2000]}
        if self._claim_tokens.get(job_id):
            payload["claim_token"] = self._claim_tokens[job_id]
        return self._request(
            "POST",
            f"/internal/jobs/{job_id}/failure",
            json=payload,
        )
