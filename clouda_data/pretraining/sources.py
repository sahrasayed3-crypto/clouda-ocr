"""Source registry for pre-training dataset preparation.

Sources are declared as data, not code: registering a new dataset never
requires touching ingestion logic. The registry is a line-oriented JSONL
file so it stays append-friendly for very large future catalogs. Nothing
here downloads anything.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

SOURCE_REGISTRY_VERSION = "clouda.pretraining.sources.v1"

CLASSIFICATIONS = ("training_only", "public", "private", "restricted")
REDISTRIBUTION = ("allowed", "attribution_required", "restricted", "forbidden")


class SourceRegistryError(ValueError):
    """Raised when a source definition or registry file is invalid."""


@dataclass(frozen=True)
class SourceDefinition:
    """A registrable dataset source (local today, remote-aware tomorrow)."""

    source_id: str
    name: str
    origin: str = ""
    license: str = "unknown"
    languages: tuple[str, ...] = ("ar",)
    expected_format: str = "image+text"
    local_root: str = ""
    remote_ref: str | None = None
    adapter: str = "auto"
    enabled: bool = True
    redistribution: str = "restricted"
    classification: str = "private"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["languages"] = list(self.languages)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceDefinition:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise SourceRegistryError(f"Unknown source fields: {sorted(unknown)}")
        kwargs = dict(data)
        kwargs["languages"] = tuple(kwargs.get("languages", ("ar",)))
        source = cls(**kwargs)
        _validate_source(source)
        return source


def _validate_source(source: SourceDefinition) -> None:
    if (
        not source.source_id
        or not source.source_id.replace("-", "").replace("_", "").isalnum()
    ):
        raise SourceRegistryError(
            f"Invalid source_id: {source.source_id!r} (use letters, digits, - , _)"
        )
    if not source.name:
        raise SourceRegistryError(f"Source {source.source_id} needs a name.")
    if source.classification not in CLASSIFICATIONS:
        raise SourceRegistryError(
            f"Source {source.source_id}: classification must be one of "
            f"{CLASSIFICATIONS}."
        )
    if source.redistribution not in REDISTRIBUTION:
        raise SourceRegistryError(
            f"Source {source.source_id}: redistribution must be one of "
            f"{REDISTRIBUTION}."
        )


def load_source_registry(path: str | Path) -> list[SourceDefinition]:
    registry_path = Path(path)
    if not registry_path.exists():
        return []
    sources: list[SourceDefinition] = []
    seen: set[str] = set()
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if "source_id" not in payload:
            continue  # header/metadata line
        payload.pop("_schema_version", None)
        source = SourceDefinition.from_dict(payload)
        if source.source_id in seen:
            raise SourceRegistryError(
                f"Duplicate source_id in registry: {source.source_id}"
            )
        seen.add(source.source_id)
        sources.append(source)
    return sources


def save_source_registry(path: str | Path, sources: list[SourceDefinition]) -> Path:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = registry_path.with_name(registry_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
        header = {
            "_schema_version": SOURCE_REGISTRY_VERSION,
            "_row_count": len(sources),
        }
        handle.write(json.dumps(header, ensure_ascii=False, sort_keys=True) + "\n")
        for source in sorted(sources, key=lambda item: item.source_id):
            row = source.to_dict()
            row["_schema_version"] = SOURCE_REGISTRY_VERSION
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp_path, registry_path)
    return registry_path


def register_source(
    path: str | Path, source: SourceDefinition, *, replace: bool = False
) -> Path:
    existing = {item.source_id: item for item in load_source_registry(path)}
    if source.source_id in existing and not replace:
        raise SourceRegistryError(
            f"Source already registered: {source.source_id} (use replace=True)"
        )
    existing[source.source_id] = source
    return save_source_registry(path, list(existing.values()))


def get_source_definition(path: str | Path, source_id: str) -> SourceDefinition:
    for source in load_source_registry(path):
        if source.source_id == source_id:
            return source
    raise KeyError(f"Unknown source: {source_id}")
