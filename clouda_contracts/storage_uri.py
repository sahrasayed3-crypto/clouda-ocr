from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlparse


@dataclass(frozen=True)
class StorageURI:
    value: str

    def __post_init__(self) -> None:
        parsed = urlparse(self.value)
        if parsed.scheme not in {"runtime", "dataset", "artifact", "model", "cache"}:
            raise ValueError("Unsupported storage URI scheme.")
        relative = "/".join(
            part for part in (parsed.netloc, parsed.path.lstrip("/")) if part
        )
        path = PurePosixPath(relative)
        if not relative or ".." in path.parts or parsed.query or parsed.fragment:
            raise ValueError("Unsafe storage URI.")

    @property
    def scheme(self) -> str:
        return urlparse(self.value).scheme

    def to_dict(self) -> dict[str, str]:
        return {"uri": self.value}

    @classmethod
    def from_dict(cls, value: dict[str, str] | str) -> "StorageURI":
        return cls(value if isinstance(value, str) else value["uri"])
