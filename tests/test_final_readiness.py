from __future__ import annotations

import io
import os
from pathlib import Path
from typing import cast

from docx import Document
from pypdf import PdfWriter

from pdfword import auto_eval, cleanup, corrections, key_store, worker
from pdfword.database import Database as RealDatabase
from pdfword.models import PageResult


def _blank_pdf() -> bytes:
    buffer = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(buffer)
    return buffer.getvalue()


def _docx(path: Path, text: str) -> None:
    document = Document()
    document.add_paragraph(text)
    document.save(str(path))


def test_auto_eval_reference_priority_and_invalid_pdf() -> None:
    assert auto_eval.extract_pdf_reference_text(b"not a PDF", [1]) == ""
    assert auto_eval.resolve_reference_text(
        _blank_pdf(), [1], b"file reference", "manual"
    ) == (
        "file reference",
        "manual_file",
    )
    assert auto_eval.resolve_reference_text(
        _blank_pdf(), [1], None, " manual reference "
    ) == (
        "manual reference",
        "manual_text",
    )
    assert auto_eval.resolve_reference_text(_blank_pdf(), [1], None, "") == ("", "none")


def test_auto_eval_uses_only_valid_mocked_scores(monkeypatch) -> None:
    class FakeProvider:
        def __init__(self) -> None:
            self.calls = 0

        def chat_with_image(self, **_kwargs) -> str:
            self.calls += 1
            return "98" if self.calls != 2 else "not a score"

    provider = FakeProvider()
    monkeypatch.setattr(
        auto_eval, "get_provider_client", lambda *_args, **_kwargs: provider
    )
    monkeypatch.setattr(
        auto_eval, "render_pdf_page_to_png_bytes", lambda **_kwargs: b"image"
    )
    monkeypatch.setattr(auto_eval, "encode_image_to_base64", lambda _value: "encoded")
    pages = [
        PageResult(page_no=index, model_used="test", markdown="text")
        for index in range(1, 5)
    ]

    assert auto_eval.estimate_ai_fidelity_score("key", _blank_pdf(), pages) == 98.0


def test_cleanup_removes_only_expired_inactive_temporary_data(
    tmp_path: Path,
) -> None:
    from pdfword.database import Database

    inactive = tmp_path / "users" / "scope-old" / "temporary"
    active = tmp_path / "users" / "scope-active" / "temporary"
    for folder in (inactive, active):
        folder.mkdir(parents=True)
        (folder / "artifact.bin").write_bytes(b"abc")
    old_time = 1
    os.utime(inactive, (old_time, old_time))
    os.utime(active, (old_time, old_time))

    db = Database(path=str(tmp_path / "test.db"))
    with db.transaction() as conn:
        conn.execute(
            """INSERT INTO conversions
               (job_id, username, original_pdf_name, stored_pdf_path, status,
                created_at, updated_at, owner_user_id, guest_scope_id, total_cost, processing_time)
               VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'), ?, ?, 0, 0)""",
            (
                "job-1",
                "testuser",
                "test.pdf",
                "/fake.pdf",
                "processing",
                "scope-active",
                "",
            ),
        )
    result = cleanup.cleanup_temporary_directories(
        tmp_path, retention_hours=1, database=db
    )

    assert result["deleted"] == 1
    assert result["bytes_freed"] == 3
    assert not inactive.exists()
    assert active.exists()


def test_corrections_extracts_docx_text_and_proposes_only_replacements(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.docx"
    corrected = tmp_path / "corrected.docx"
    _docx(original, "old word")
    _docx(corrected, "new word")

    class FakeDatabase:
        def __init__(self) -> None:
            self.rules: list[tuple[str, str, str]] = []

        def add_correction_rule(self, before: str, after: str, kind: str) -> None:
            self.rules.append((before, after, kind))

    database = FakeDatabase()
    result = corrections.propose_corrections(
        original, corrected, cast(RealDatabase, database)
    )

    assert corrections.extract_docx_text(original) == "old word"
    assert result["replaced"] == 1
    assert database.rules == [("old", "new", "replacement")]


def test_local_key_store_round_trip_and_delete(tmp_path: Path, monkeypatch) -> None:
    secret_dir = tmp_path / ".secrets"
    api_path = secret_dir / "api.key"
    legacy_path = tmp_path / "legacy.key"
    monkeypatch.setattr(key_store, "SECRETS_DIR", str(secret_dir))
    monkeypatch.setattr(key_store, "API_KEY_FILE", str(api_path))
    monkeypatch.setattr(key_store, "LEGACY_API_KEY_FILE", str(legacy_path))

    assert key_store.save_api_key_local(" test-key ") is True
    assert key_store.load_saved_api_key() == "test-key"
    assert key_store.delete_saved_api_key() is True
    assert key_store.load_saved_api_key() == ""
    legacy_path.write_text("legacy-key", encoding="utf-8")
    assert key_store.load_saved_api_key() == "legacy-key"
    assert api_path.read_text(encoding="utf-8") == "legacy-key"


def test_worker_refuses_invalid_local_startup_configuration(
    monkeypatch, capsys
) -> None:
    class Config:
        app_role = "web"
        worker_concurrency = 1
        worker_api_key = "unused"  # pragma: allowlist secret

    monkeypatch.setattr(worker, "runtime_settings", lambda: Config())

    assert worker.main() == 2
    assert "APP_ROLE=worker" in capsys.readouterr().err
