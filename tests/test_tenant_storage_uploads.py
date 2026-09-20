"""TenantStorage upload-integrity tests."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from pdfword.tenant_storage import TenantStorage, safe_filename


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


@pytest.mark.parametrize(
    "name",
    ["CON.pdf", "nul.pdf", "aux.pdf", "NUL .pdf", "COM1", "CON.txt.pdf", "lpt9.dat"],
)
def test_write_upload_neutralizes_windows_reserved_names(tmp_path: Path, name: str):
    """Reserved device names must not crash the upload with an unhandled
    OSError (WinError 183) — they are neutralized into safe components."""

    storage = TenantStorage(tmp_path / "root")
    paths = storage.guest("0f1e2d3c4b5a69788796a5b4c3d2e1f0")

    stored = storage.write_upload(paths, name, io.BytesIO(b"UPLOAD-BYTES"))

    assert stored.is_file()
    assert stored.read_bytes() == b"UPLOAD-BYTES"


def test_safe_filename_reserved_names_are_prefixed() -> None:
    for name in ("CON.pdf", "nul", "AUX.txt", "COM1", "NUL .pdf"):
        cleaned = safe_filename(name, "input.pdf")
        stem = cleaned.split(".", 1)[0].rstrip(" .").upper()
        assert stem.startswith("_"), f"{name!r} -> {cleaned!r}"
