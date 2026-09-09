"""Text source ingestion (System B semantics).

Reads one UTF-8 text file exactly like arabic-scan-factory.generator:
strict UTF-8 decode (invalid bytes rejected), no Unicode normalization or
rewriting of any kind, source bytes hashed and later re-verified. The
source file itself is never written to.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..provenance.hashing import sha256_file, sha256_text


class TextDecodeError(ValueError):
    """Raised when a text source is not valid UTF-8."""


@dataclass(frozen=True)
class TextSource:
    path: Path
    text: str
    byte_sha256: str
    text_sha256: str
    byte_length: int

    @property
    def document_id(self) -> str:
        return self.byte_sha256[:16]


def load_text_source(path: Path) -> TextSource:
    path = Path(path)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise TextDecodeError(f"{path} is not valid UTF-8: {exc}") from exc
    return TextSource(
        path=path,
        text=text,
        byte_sha256=sha256_file(path),
        text_sha256=sha256_text(text),
        byte_length=len(raw),
    )
