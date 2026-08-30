from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validate_sha256(value: str) -> str:
    normalized = value.lower()
    if not SHA256_PATTERN.fullmatch(normalized):
        raise ValueError("Expected a 64-character SHA-256 hexadecimal digest.")
    return normalized


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Checksum:
    algorithm: str
    value: str

    def __post_init__(self) -> None:
        if self.algorithm != "sha256":
            raise ValueError("Only sha256 checksums are supported.")
        object.__setattr__(self, "value", validate_sha256(self.value))

    def to_dict(self) -> dict[str, str]:
        return {"algorithm": self.algorithm, "value": self.value}

    @classmethod
    def from_dict(cls, value: dict[str, str] | str) -> "Checksum":
        if isinstance(value, str):
            return cls("sha256", value)
        return cls(value.get("algorithm", "sha256"), value["value"])


SourceChecksum = Checksum
