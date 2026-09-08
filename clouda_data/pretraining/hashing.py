"""Deterministic hashing infrastructure with an incremental cache.

Hashes are computed by streaming files in fixed-size chunks so arbitrarily
large images never need to fit in memory. The cache maps
``(relative path, size)`` to a SHA-256 so repeated indexing of huge trees is
cheap; a size change naturally invalidates the cached entry.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class HashCache:
    """Append-friendly, resumable cache of file hashes.

    Rows are ``{"key": "<relpath>\\x00<size>", "sha256": ...}`` written to a
    JSONL file. Reads are tolerant of truncated final lines (interrupted
    writes), which keeps interrupted runs recoverable.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "key" in row and "sha256" in row:
                self._entries[str(row["key"])] = str(row["sha256"])

    @staticmethod
    def make_key(relative_path: str, size_bytes: int) -> str:
        return f"{relative_path}\x00{size_bytes}"

    def get(self, relative_path: str, size_bytes: int) -> str | None:
        return self._entries.get(self.make_key(relative_path, size_bytes))

    def put(self, relative_path: str, size_bytes: int, sha256: str) -> None:
        key = self.make_key(relative_path, size_bytes)
        if self._entries.get(key) == sha256:
            return
        self._entries[key] = sha256
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {"key": key, "sha256": sha256}, ensure_ascii=False, sort_keys=True
                )
                + "\n"
            )

    def __len__(self) -> int:
        return len(self._entries)


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Write text via a temp file plus ``os.replace`` (crash-safe)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(target.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(tmp_path, target)
    return target
