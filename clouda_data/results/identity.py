"""Deterministic, machine-independent identities for the Results Store.

Every persisted identifier is derived with SHA-256 over an explicit,
domain-tagged tuple of *logical* fields — never ``hash()``, never absolute
paths, never process state. Where an existing project id is already stable
(e.g. the benchmark manifest's ``benchmark_id``), it is preserved verbatim and
only *referenced*, not re-derived.

Schema version: ``clouda.ocr.results.v1``.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

RESULTS_SCHEMA_VERSION = "clouda.ocr.results.v1"

_ABSOLUTE_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_UNSAFE_PATH_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


def canonical_json(value: Any) -> str:
    """Stable JSON encoding: key-sorted, compact, Arabic preserved."""

    return json_dumps_sorted(value)


def json_dumps_sorted(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def domain_tagged_digest(domain: str, *parts: str) -> str:
    """SHA-256 over NUL-joined, domain-tagged logical fields."""

    digest = hashlib.sha256()
    digest.update(domain.encode("utf-8"))
    digest.update(b"\x00")
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def short_digest(domain: str, *parts: str, length: int = 20) -> str:
    return domain_tagged_digest(domain, *parts)[:length]


def dataset_identity(dataset_id: str, dataset_version: str = "1") -> str:
    """Deterministic dataset identity (logical, not path based)."""

    if not dataset_id.strip():
        raise ValueError("dataset_id cannot be blank.")
    return f"{dataset_id.strip()}@{dataset_version.strip() or '1'}"


def page_identity(
    *,
    dataset_id: str,
    split: str,
    provided_page_key: str,
) -> str:
    """Stable page id inside a dataset/split.

    ``provided_page_key`` is any stable per-page identifier the source already
    carries (benchmark ``benchmark_id``, canonical ``page_id``/``sample_id``,
    or ``document_id:page_number``). It is preserved verbatim so existing
    project identities round-trip; the dataset/split prefix scopes it.
    """

    if not provided_page_key.strip():
        raise ValueError("provided_page_key cannot be blank.")
    return f"{dataset_id.strip()}@{split.strip() or 'unassigned'}:{provided_page_key.strip()}"


def run_identity(
    *,
    model_id: str,
    model_revision: str,
    dataset_id: str,
    dataset_version: str,
    manifest_sha256: str | None,
    created_at: str,
) -> str:
    """Deterministic inference-run id.

    Content-scoped (model + revision + dataset + manifest hash) with a UTC
    timestamp for uniqueness across repeated evaluations of the same content.
    """

    return "run_" + short_digest(
        "ocr.run",
        model_id.strip(),
        model_revision.strip(),
        dataset_identity(dataset_id, dataset_version),
        (manifest_sha256 or "").strip().lower(),
        created_at.strip(),
    )


def prediction_identity(*, run_id: str, page_id: str) -> str:
    return short_digest("ocr.prediction", run_id.strip(), page_id.strip())


def evaluation_record_identity(*, run_id: str, page_id: str, metric: str) -> str:
    return short_digest("ocr.metric", run_id.strip(), page_id.strip(), metric.strip())


def artifact_identity(*, kind: str, sha256: str, role: str) -> str:
    return short_digest(
        "ocr.artifact",
        kind.strip(),
        sha256.strip().lower(),
        role.strip(),
        length=24,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_sha256(value: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError("Expected a 64-character SHA-256 hexadecimal digest.")
    return normalized


def portable_relative_path(value: str, *, allow_empty: bool = False) -> str:
    """Normalize a path to a portable POSIX-style relative path.

    Rejects absolute Windows drives, UNC prefixes, leading slashes, ``..``
    escapes, and control characters. This is deliberately lexical — callers
    that touch the filesystem must resolve it against their configured root
    (see :mod:`clouda_data.results.store`).
    """

    if not isinstance(value, str):
        raise TypeError("Paths must be strings.")
    if _UNSAFE_PATH_CHARACTERS.search(value):
        raise ValueError(f"Path contains control characters: {value!r}")
    normalized = value.replace("\\", "/")
    if normalized.startswith("//") or _ABSOLUTE_WINDOWS_PATH.match(normalized):
        raise ValueError(f"Path must be relative: {value!r}")
    if normalized.startswith("/"):
        raise ValueError(f"Path must be relative: {value!r}")
    # Lexical normalization first, then the escape check on the result.
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    resolved: list[str] = []
    for part in parts:
        if part == "..":
            if not resolved:
                raise ValueError(f"Path escapes its root: {value!r}")
            resolved.pop()
        else:
            resolved.append(part)
    portable = "/".join(resolved)
    if not portable and not allow_empty:
        raise ValueError("Path cannot be empty.")
    return portable


def private_path_note(value: str) -> dict[str, str]:
    """Quarantine a machine-local absolute path under ``source_private``.

    Canonical records must stay portable; local absolute paths are retained
    only in this explicitly non-portable side-car field so evidence is not
    silently lost during ingestion.
    """

    return {"source_private": str(value)}


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Atomically write pretty JSON (UTF-8, Arabic preserved, sorted keys)."""

    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> Path:
    """Append one canonical JSON line (UTF-8, sorted keys, LF endings)."""

    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def iter_jsonl(path: str | Path) -> Any:
    """Stream JSONL rows; raises on malformed lines with line numbers."""

    import json

    source = Path(path)
    if not source.exists():
        return
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSONL at {source}, line {line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"JSONL row at {source}, line {line_number} must be an object."
                )
            yield payload


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 of file bytes (reuse semantics across the repo)."""

    from clouda_data.ground_truth.checksums import sha256_file as _sha256_file

    return _sha256_file(path)


def sha256_text(text: str) -> str:
    from clouda_data.ground_truth.checksums import sha256_text as _sha256_text

    return _sha256_text(text)


@dataclass(frozen=True)
class ArtifactRef:
    """Portable logical reference to a stored artifact.

    ``uri`` uses the repository's StorageURI scheme (``dataset://`` or
    ``artifact://``) and is always relative; machine-local absolute paths are
    never persisted in this field.
    """

    artifact_id: str
    kind: str
    uri: str
    sha256: str
    size_bytes: int | None = None
    media_type: str | None = None
    role: str = "asset"
    source_private: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id cannot be blank.")
        if not self.kind.strip():
            raise ValueError("Artifact kind cannot be blank.")
        object.__setattr__(self, "sha256", validate_sha256(self.sha256))
        if "://" not in self.uri:
            raise ValueError("Artifact uri must use a storage scheme, e.g. dataset://")
        scheme = self.uri.split("://", 1)[0]
        if scheme not in {"dataset", "artifact", "model", "runtime", "cache"}:
            raise ValueError(f"Unsupported storage URI scheme: {scheme}")
        payload = self.uri.split("://", 1)[1].replace("\\", "/")
        path = PurePosixPath(payload)
        if path.is_absolute() or not payload or ".." in path.parts:
            raise ValueError(f"Artifact uri must be relative and safe: {self.uri}")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "kind": self.kind,
            "uri": self.uri,
            "sha256": self.sha256,
            "role": self.role,
        }
        if self.size_bytes is not None:
            payload["size_bytes"] = self.size_bytes
        if self.media_type is not None:
            payload["media_type"] = self.media_type
        if self.source_private:
            payload["source_private"] = dict(self.source_private)
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ArtifactRef":
        return cls(
            artifact_id=str(value["artifact_id"]),
            kind=str(value.get("kind", "asset")),
            uri=str(value["uri"]),
            sha256=validate_sha256(str(value["sha256"])),
            size_bytes=(
                int(value["size_bytes"])
                if value.get("size_bytes") is not None
                else None
            ),
            media_type=value.get("media_type"),
            role=str(value.get("role", "asset")),
            source_private=(
                dict(value["source_private"]) if value.get("source_private") else None
            ),
        )
