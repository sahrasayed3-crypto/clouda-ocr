"""Integrity, manifest, and run-directory safety.

Migrated from the standalone repository (tests/test_manifest_integrity.py).
"""

from __future__ import annotations

from pathlib import Path


from clouda_data.factory.manifest import (
    read_manifest,
    write_manifest_csv,
    write_manifest_jsonl,
)
from clouda_data.factory.provenance.hashing import (
    sha256_bytes,
    sha256_file,
    sha256_text,
)
from clouda_data.factory.provenance.integrity import atomic_target


def test_sha256_helpers_agree(tmp_path):
    data = b"clouda-provenance-\xc3\xa9\x00\x01"
    p = tmp_path / "f.bin"
    p.write_bytes(data)
    assert sha256_bytes(data) == sha256_file(p)
    assert sha256_text("سلام") == sha256_bytes("سلام".encode("utf-8"))


def test_atomic_target_replaces_and_cleans_up(tmp_path):
    target = tmp_path / "out.json"
    with atomic_target(target) as tmp:
        Path(tmp).write_text("{}", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "{}"
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".out.json")]
    assert not leftovers


def test_atomic_write_via_pipeline(tmp_path):
    rows = [{"a": 1, "b": "سلام", "c": {"nested": True}}, {"a": 2, "b": "x", "c": []}]
    j = write_manifest_jsonl(rows, tmp_path / "manifest.jsonl")
    c = write_manifest_csv(rows, tmp_path / "manifest.csv")
    assert read_manifest(j) == rows
    text = c.read_text(encoding="utf-8")
    assert "سلام" in text and "nested" in text


def test_manifest_utf8_arabic_roundtrip(tmp_path):
    rows = [{"document_id": "دوكيو", "note": "نص عربي مع تشكيل: السَّلَامُ"}]
    p = write_manifest_jsonl(rows, tmp_path / "m.jsonl")
    assert read_manifest(p) == rows


def test_sha256_matches_canonical_checksum_contract(tmp_path):
    """The factory hashing kernel must agree with clouda_contracts."""
    from clouda_contracts.checksums import sha256_file as contract_sha256_file

    p = tmp_path / "f.bin"
    p.write_bytes(b"agreement-check")
    assert sha256_file(p) == contract_sha256_file(p)
