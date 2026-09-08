"""Deterministic hashing infrastructure with an incremental cache.

Hashes are computed by streaming files in fixed-size chunks so arbitrarily
large images never need to fit in memory. The cache maps source id, canonical
relative path, size, and nanosecond modification time to a SHA-256. A stat
check before and after hashing rejects files modified during the read.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

CHUNK_SIZE = 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    if file_path.is_symlink():
        raise ValueError(f"Refusing to hash a symbolic link: {file_path}")
    before = file_path.stat()
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    after = file_path.stat()
    before_identity = (before.st_size, before.st_mtime_ns)
    after_identity = (after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise OSError(f"File changed while hashing: {file_path}")
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class HashCache:
    """Append-friendly, resumable cache of file hashes.

    Rows store a NUL-delimited source/path/size/mtime key and SHA-256 in JSONL.
    Reads ignore malformed rows, and a truncated final line is separated from
    the next append so interrupted runs remain recoverable.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or "key" not in row or "sha256" not in row:
                    continue
                digest = str(row["sha256"])
                if SHA256_RE.fullmatch(digest):
                    self._entries[str(row["key"])] = digest

    @staticmethod
    def make_key(
        source_id: str, relative_path: str, size_bytes: int, mtime_ns: int
    ) -> str:
        return f"{source_id}\x00{relative_path}\x00{size_bytes}\x00{mtime_ns}"

    def get(
        self, source_id: str, relative_path: str, size_bytes: int, mtime_ns: int
    ) -> str | None:
        return self._entries.get(
            self.make_key(source_id, relative_path, size_bytes, mtime_ns)
        )

    def put(
        self,
        source_id: str,
        relative_path: str,
        size_bytes: int,
        mtime_ns: int,
        sha256: str,
    ) -> None:
        if not SHA256_RE.fullmatch(sha256):
            raise ValueError("sha256 must be a lowercase 64-character hex digest")
        key = self.make_key(source_id, relative_path, size_bytes, mtime_ns)
        if self._entries.get(key) == sha256:
            return
        self._entries[key] = sha256
        needs_separator = False
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as existing:
                existing.seek(-1, os.SEEK_END)
                needs_separator = existing.read(1) != b"\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            if needs_separator:
                handle.write("\n")
            handle.write(
                json.dumps(
                    {"key": key, "sha256": sha256}, ensure_ascii=False, sort_keys=True
                )
                + "\n"
            )

    def __len__(self) -> int:
        return len(self._entries)


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Replace text atomically using a unique same-directory temporary file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
    return target
