from __future__ import annotations

import io
import json
import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from clouda_lab.dashboard.document_intelligence import (
    DocumentIntelligenceService,
    MAX_PDF_BYTES,
)
from clouda_lab.dashboard.settings import LabSettings

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _settings(tmp_path: Path) -> LabSettings:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "runs").mkdir()
    (repo / "benchmarks").mkdir()
    return LabSettings.from_repo(repo)


def _blank_pdf(page_count: int) -> bytes:
    output = io.BytesIO()
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=595, height=842)
    writer.write(output)
    return output.getvalue()


def test_service_returns_categorical_sanitized_page_diagnostics(
    tmp_path: Path,
) -> None:
    service = DocumentIntelligenceService(_settings(tmp_path))

    payload = service.analyze((FIXTURES / "digital_text.pdf").read_bytes())
    serialized = json.dumps(payload).lower()

    assert payload["schema_version"] == "clouda.lab.document-intelligence.v1"
    assert payload["page_count"] == 1
    assert payload["pages"][0]["decision"] == "trusted_digital_text"
    assert payload["pages"][0]["gate_verdict"] == "trusted"
    assert payload["pages"][0]["reason_codes"]
    assert "digital pdf text" not in serialized
    assert "accuracy" not in serialized
    assert "confidence" not in serialized
    assert "quality" not in serialized
    assert "sha256" not in serialized


def test_service_performs_no_network_call_or_persistence(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path)
    before = sorted(
        path.relative_to(settings.repo_root) for path in settings.repo_root.rglob("*")
    )

    def reject_network(*_args, **_kwargs):
        raise AssertionError("document analysis attempted a network call")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    payload = DocumentIntelligenceService(settings).analyze(
        (FIXTURES / "scanned.pdf").read_bytes()
    )
    after = sorted(
        path.relative_to(settings.repo_root) for path in settings.repo_root.rglob("*")
    )

    assert payload["pages"][0]["decision"] == "ocr_required"
    assert before == after


@pytest.mark.parametrize(
    ("payload_factory", "message"),
    [
        (lambda: b"not a pdf", "PDF signature"),
        (lambda: b"%PDF-1.7\n" + (b"x" * (10 * 1024 * 1024)), "10 MiB"),
        (lambda: _blank_pdf(26), "25 pages"),
    ],
    ids=("invalid-signature", "oversized", "too-many-pages"),
)
def test_service_rejects_invalid_or_oversized_inputs(
    tmp_path: Path, payload_factory, message: str
) -> None:
    service = DocumentIntelligenceService(_settings(tmp_path))

    with pytest.raises(ValueError, match=message):
        service.analyze(payload_factory())


def test_lab_document_intelligence_requires_action_token_and_is_sanitized(
    tmp_path: Path,
) -> None:
    from clouda_lab.dashboard.app import create_app

    client = TestClient(create_app(_settings(tmp_path)))
    pdf_bytes = (FIXTURES / "digital_text.pdf").read_bytes()
    content_headers = {"Content-Type": "application/pdf"}

    assert (
        client.post(
            "/api/lab/document-intelligence/analyze",
            content=pdf_bytes,
            headers=content_headers,
        ).status_code
        == 403
    )
    token = client.get("/api/lab/session").json()["action_token"]
    response = client.post(
        "/api/lab/document-intelligence/analyze",
        headers={"X-Clouda-Lab-Action": token, **content_headers},
        content=pdf_bytes,
    )
    payload = response.json()
    serialized = json.dumps(payload).lower()

    assert response.status_code == 200
    assert payload["pages"][0]["gate_verdict"] in {
        "trusted",
        "untrusted",
        "uncertain",
    }
    assert "digital pdf text" not in serialized
    assert "accuracy" not in serialized
    assert "confidence" not in serialized
    assert "quality" not in serialized


def test_lab_document_intelligence_rejects_non_pdf_upload(tmp_path: Path) -> None:
    from clouda_lab.dashboard.app import create_app

    client = TestClient(create_app(_settings(tmp_path)))
    token = client.get("/api/lab/session").json()["action_token"]

    response = client.post(
        "/api/lab/document-intelligence/analyze",
        headers={
            "X-Clouda-Lab-Action": token,
            "Content-Type": "application/pdf",
        },
        content=b"not a pdf",
    )

    assert response.status_code == 422
    assert "PDF signature" in response.json()["detail"]


def test_lab_document_intelligence_accepts_large_pdf_without_multipart_spooling(
    tmp_path: Path,
) -> None:
    from clouda_lab.dashboard.app import create_app

    client = TestClient(create_app(_settings(tmp_path)))
    token = client.get("/api/lab/session").json()["action_token"]
    payload = _blank_pdf(1) + (b"\0" * (1024 * 1024))

    response = client.post(
        "/api/lab/document-intelligence/analyze",
        headers={
            "X-Clouda-Lab-Action": token,
            "Content-Type": "application/pdf",
        },
        content=payload,
    )

    assert response.status_code == 200
    assert response.json()["page_count"] == 1


def test_lab_document_intelligence_rejects_oversized_http_body(tmp_path: Path) -> None:
    from clouda_lab.dashboard.app import create_app

    client = TestClient(create_app(_settings(tmp_path)))
    token = client.get("/api/lab/session").json()["action_token"]

    response = client.post(
        "/api/lab/document-intelligence/analyze",
        headers={
            "X-Clouda-Lab-Action": token,
            "Content-Type": "application/pdf",
        },
        content=b"%PDF-1.7\n" + (b"x" * MAX_PDF_BYTES),
    )

    assert response.status_code == 422
    assert "10 MiB" in response.json()["detail"]
