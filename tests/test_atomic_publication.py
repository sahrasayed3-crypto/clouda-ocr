"""Regression tests for atomic publication fixes (deep session, see
docs/engineering/CODE_AUDIT_CURRENT.md C1/C3/C5).

Covers:
- pdfword.atomic helpers publish fully and leave no staging files.
- create_backup never publishes a truncated zip and does not let two
  same-second backups overwrite each other.
- save_checkpoint leaves no fixed-name .tmp residue.
- safe_component neutralizes Windows reserved device names.
"""

from __future__ import annotations

import json
from pathlib import Path

from pdfword.atomic import atomic_write_bytes, atomic_write_text
from pdfword.backup import create_backup, validate_backup
from pdfword.checkpoints import checkpoint_path, load_checkpoint, save_checkpoint
from pdfword.database import Database
from pdfword.models import PageResult
from pdfword.storage import safe_component


def test_atomic_write_publishes_content_and_cleans_staging(tmp_path: Path):
    target = tmp_path / "nested" / "out.bin"

    atomic_write_bytes(target, b"first")
    atomic_write_bytes(target, b"second-payload")

    assert target.read_bytes() == b"second-payload"
    assert list(target.parent.iterdir()) == [target]


def test_atomic_write_text_utf8(tmp_path: Path):
    target = tmp_path / "doc.json"

    atomic_write_text(target, json.dumps({"text": "مرحبا"}, ensure_ascii=False))

    assert json.loads(target.read_text(encoding="utf-8"))["text"] == "مرحبا"


def test_checkpoint_roundtrip_leaves_no_tmp_residue(tmp_path: Path):
    results = {1: PageResult(page_no=1, model_used="pypdf", markdown="نص")}
    root = tmp_path / "job"

    save_checkpoint(root, results)

    assert list(root.iterdir()) == [checkpoint_path(root)]
    loaded = load_checkpoint(root)
    assert 1 in loaded
    assert loaded[1].page_no == 1


def test_same_second_backups_do_not_overwrite_each_other(tmp_path: Path):
    database = Database(tmp_path / "db.sqlite3")
    database.initialize()
    database_path = tmp_path / "db.sqlite3"
    assert database_path.is_file()
    storage = tmp_path / "storage"
    storage.mkdir()
    backup_root = tmp_path / "backups"

    first = create_backup(database, storage_root=storage, backup_root=backup_root)
    second = create_backup(database, storage_root=storage, backup_root=backup_root)

    assert first != second, "same-second backups must not share a filename"
    assert validate_backup(first)["valid"]
    assert validate_backup(second)["valid"]
    archives = sorted(backup_root.glob("clouda_backup_*.zip"))
    assert len(archives) == 2
    # No staging residue next to the published archives.
    assert not list(backup_root.glob(".*.part"))


def test_safe_component_neutralizes_windows_reserved_devices():
    assert safe_component("CON.pdf", "input.pdf") == "_CON.pdf"
    assert safe_component("nul", "input.pdf") == "_nul"
    assert safe_component("COM1.docx", "output.docx") == "_COM1.docx"
    assert safe_component("lpt4", "output.docx") == "_lpt4"
    assert safe_component("conversation.pdf", "input.pdf") == "conversation.pdf"
    assert safe_component("report", "output.docx") == "report"
