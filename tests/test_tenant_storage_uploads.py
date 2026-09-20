"""TenantStorage upload-integrity tests."""

from __future__ import annotations

import io
from pathlib import Path

from pdfword.tenant_storage import TenantStorage


def test_write_upload_does_not_overwrite_same_scope_filename(tmp_path: Path) -> None:
    """Two jobs in one scope uploading the same filename must not let the
    later upload silently replace the earlier job's stored input."""

    storage = TenantStorage(tmp_path / "root")
    paths = storage.guest("0f1e2d3c4b5a69788796a5b4c3d2e1f0")

    first = storage.write_upload(paths, "scan.pdf", io.BytesIO(b"FIRST-JOB-BYTES"))
    second = storage.write_upload(paths, "scan.pdf", io.BytesIO(b"SECOND-JOB-BYTES"))

    assert first.read_bytes() == b"FIRST-JOB-BYTES"
    assert second.read_bytes() == b"SECOND-JOB-BYTES"
    assert first != second

    third = storage.write_upload(paths, "other.pdf", io.BytesIO(b"OTHER-BYTES"))
    assert third.name == "other.pdf"
