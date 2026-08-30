from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clouda_contracts.archive_security import ArchiveLimits, validate_zip_archive
from clouda_contracts.security import (
    may_use_user_document_for_training,
    redact_mapping,
)
from clouda_data.ingestion.file_inspection import inspect_file
from pdfword.worker_api import app


def _archive(name: str, content: bytes) -> zipfile.ZipFile:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, content)
    output.seek(0)
    return zipfile.ZipFile(output)


def test_zip_traversal_and_decompression_ratio_are_rejected() -> None:
    with _archive("../escape", b"x") as archive:
        with pytest.raises(ValueError, match="unsafe"):
            validate_zip_archive(archive)
    with _archive("large.txt", b"A" * 100_000) as archive:
        with pytest.raises(ValueError, match="compression-ratio"):
            validate_zip_archive(
                archive,
                limits=ArchiveLimits(max_compression_ratio=2),
            )


def test_external_entities_are_rejected(tmp_path: Path) -> None:
    xml = tmp_path / "entity.xml"
    xml.write_text(
        '<!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]><x>&e;</x>',
        encoding="utf-8",
    )
    result = inspect_file(xml, "page_xml")
    assert result.ok is False
    assert "invalid_xml" in result.issues


def test_sensitive_logging_fields_are_redacted() -> None:
    value = redact_mapping(
        {"job_id": "safe", "api_key": "value", "nested": {"password": "value"}}
    )
    assert value == {
        "job_id": "safe",
        "api_key": "[REDACTED]",
        "nested": {"password": "[REDACTED]"},
    }


def test_user_documents_are_not_training_data_without_both_approvals() -> None:
    assert (
        may_use_user_document_for_training(
            explicit_document_consent=False,
            approved_consent_policy=False,
        )
        is False
    )
    assert (
        may_use_user_document_for_training(
            explicit_document_consent=True,
            approved_consent_policy=False,
        )
        is False
    )


def test_worker_api_sets_security_headers() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
