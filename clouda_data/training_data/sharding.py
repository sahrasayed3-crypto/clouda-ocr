"""Deterministic dataset sharding engine and canonical shard index.

The sharding engine reads a validated canonical manifest and produces:

- ``shards/shard-NNNNNN.jsonl`` — one JSONL row per sample (metadata only);
- ``shard_index.json`` — the canonical shard index (rebuildable).

Shard assignment is deterministic: the same manifest + shard config always
produces byte-identical shards and the same shard identity. Shard ids are
derived from a stable hash of (dataset identity, manifest hash, shard config
hash, zero-based shard ordinal) — never from ``hash()``.

Shard row format (JSONL, one object per line)::

    {"sample_id": ..., "shard_id": ..., "position": ..., "row": {...}}

The embedded ``row`` is the canonical manifest row (provenance included),
preserved verbatim for traceability.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from clouda_data.pretraining.hashing import atomic_write_text
from clouda_data.training_data.models import (
    ShardConfig,
    ShardStrategy,
)

SHARD_INDEX_SCHEMA_VERSION = "clouda.training_data.shard_index.v1"
SHARD_ROW_SCHEMA_VERSION = "clouda.training_data.shard_record.v1"


class ShardIndexError(ValueError):
    """Raised when a shard index is inconsistent with its shards."""


def shard_config_hash(config: ShardConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def derive_shard_id(
    dataset_id: str,
    dataset_version: str,
    manifest_sha256: str,
    config_hash: str,
    ordinal: int,
) -> str:
    key = "\x1f".join(
        (
            str(dataset_id),
            str(dataset_version),
            str(manifest_sha256),
            str(config_hash),
            str(int(ordinal)),
        )
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
    return f"shard-{digest[:16]}"


@dataclass(frozen=True)
class ShardIndexEntry:
    shard_id: str
    ordinal: int
    sample_count: int
    approx_bytes: int
    sha256: str
    path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShardIndex:
    dataset_id: str
    dataset_version: str
    source_manifest_sha256: str
    shard_config_hash: str
    total_samples: int
    shards: tuple[ShardIndexEntry, ...]
    schema_version: str = SHARD_INDEX_SCHEMA_VERSION
    created_by: str = "clouda-pdf 0.2.0"

    @property
    def total_shards(self) -> int:
        return len(self.shards)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "source_manifest_sha256": self.source_manifest_sha256,
            "shard_config_hash": self.shard_config_hash,
            "total_samples": self.total_samples,
            "total_shards": self.total_shards,
            "created_by": self.created_by,
            "shards": [entry.to_dict() for entry in self.shards],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ShardIndex":
        if payload.get("schema_version") != SHARD_INDEX_SCHEMA_VERSION:
            raise ShardIndexError(
                f"Unsupported shard index schema: {payload.get('schema_version')!r}"
            )
        dataset_id = str(payload["dataset_id"])
        dataset_version = str(payload["dataset_version"])
        manifest_hash = str(payload["source_manifest_sha256"])
        config_hash = str(payload["shard_config_hash"])
        entries = tuple(
            ShardIndexEntry(
                shard_id=str(item["shard_id"]),
                ordinal=int(item["ordinal"]),
                sample_count=int(item["sample_count"]),
                approx_bytes=int(item.get("approx_bytes", 0)),
                sha256=str(item["sha256"]),
                path=_validated_shard_path(item["path"]),
            )
            for item in payload["shards"]
        )
        if int(payload.get("total_shards", -1)) != len(entries):
            raise ShardIndexError("Shard index total_shards does not match entries")
        if [entry.ordinal for entry in entries] != list(range(len(entries))):
            raise ShardIndexError("Shard index ordinals must be contiguous and ordered")
        if len({entry.path for entry in entries}) != len(entries):
            raise ShardIndexError("Shard index contains duplicate paths")
        for entry in entries:
            expected_id = derive_shard_id(
                dataset_id,
                dataset_version,
                manifest_hash,
                config_hash,
                entry.ordinal,
            )
            if entry.shard_id != expected_id:
                raise ShardIndexError(
                    f"Shard identity mismatch at ordinal {entry.ordinal}"
                )
            if entry.sample_count < 1:
                raise ShardIndexError("Shard sample_count must be positive")
        total_samples = int(payload["total_samples"])
        if total_samples != sum(entry.sample_count for entry in entries):
            raise ShardIndexError(
                "Shard index total_samples does not match entry counts"
            )
        return cls(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            source_manifest_sha256=manifest_hash,
            shard_config_hash=config_hash,
            total_samples=total_samples,
            shards=entries,
        )


def _validated_shard_path(value: Any) -> str:
    path = str(value)
    candidate = Path(path)
    if (
        not path
        or path in {".", ".."}
        or candidate.is_absolute()
        or candidate.name != path
        or "/" in path
        or "\\" in path
    ):
        raise ShardIndexError(f"Unsafe shard path: {path!r}")
    return path


def _iter_rows_sorted(path: Path, sample_limit: int | None) -> Iterator[dict[str, Any]]:
    from clouda_data.training_data.input_contract import iter_canonical_rows

    # Canonical manifests are deterministically ordered by write_manifest.
    # Preserve that byte-identified order while keeping memory independent of
    # total dataset size.
    for index, row in enumerate(iter_canonical_rows(path)):
        if sample_limit is not None and index >= sample_limit:
            break
        yield row


def _shard_row_payload(row: dict[str, Any], shard_id: str, position: int) -> str:
    record = {
        "schema_version": SHARD_ROW_SCHEMA_VERSION,
        "sample_id": str(row.get("sample_id", "")),
        "shard_id": shard_id,
        "position": position,
        "row": row,
    }
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def _approx_row_bytes(row: dict[str, Any]) -> int:
    size = row.get("file_size")
    if isinstance(size, int) and size > 0:
        return int(size)
    return len(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def build_shards(
    manifest_path: str | Path,
    output_dir: str | Path,
    config: ShardConfig,
    *,
    dataset_id: str,
    dataset_version: str,
    manifest_sha256: str | None = None,
    sample_limit: int | None = None,
) -> ShardIndex:
    """Shard a validated canonical manifest into JSONL shards + index.

    Deterministic: same manifest + config produce identical shard bytes and
    ids. Fail-closed on empty selection.
    """

    from clouda_data.training_data.input_contract import validate_canonical_manifest

    identity = validate_canonical_manifest(
        manifest_path,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
    )
    if manifest_sha256 is not None and manifest_sha256 != identity.manifest_sha256:
        raise ValueError("Provided manifest_sha256 does not match the manifest file")
    manifest_digest = identity.manifest_sha256
    config_hash = shard_config_hash(config)

    out_dir = Path(output_dir)
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    entries: list[ShardIndexEntry] = []
    ordinal = 0
    current_shard_id: str | None = None
    current_lines: list[str] = []
    current_bytes = 0
    total_samples = 0

    def _flush() -> None:
        nonlocal ordinal, current_shard_id, current_lines, current_bytes
        if current_shard_id is None or not current_lines:
            return
        path = shard_dir / f"{current_shard_id}.jsonl"
        atomic_write_text(path, "".join(line + "\n" for line in current_lines))
        digest = hashlib.sha256(
            "".join(line + "\n" for line in current_lines).encode("utf-8")
        ).hexdigest()
        entries.append(
            ShardIndexEntry(
                shard_id=current_shard_id,
                ordinal=ordinal,
                sample_count=len(current_lines),
                approx_bytes=current_bytes,
                sha256=digest,
                path=path.name,
            )
        )
        ordinal += 1
        current_shard_id = None
        current_lines = []
        current_bytes = 0

    for row in _iter_rows_sorted(Path(manifest_path), sample_limit):
        if current_shard_id is None:
            current_shard_id = derive_shard_id(
                dataset_id, dataset_version, manifest_digest, config_hash, ordinal
            )
        row_bytes = _approx_row_bytes(row)
        if (
            config.strategy is ShardStrategy.SIZE_AWARE
            and current_lines
            and (
                len(current_lines) >= config.samples_per_shard
                or current_bytes + row_bytes > config.max_shard_bytes
            )
        ):
            _flush()
            current_shard_id = derive_shard_id(
                dataset_id, dataset_version, manifest_digest, config_hash, ordinal
            )
        payload = _shard_row_payload(row, current_shard_id, len(current_lines))
        current_lines.append(payload)
        current_bytes += row_bytes
        total_samples += 1
        if (
            config.strategy is ShardStrategy.COUNT
            and len(current_lines) >= config.samples_per_shard
        ):
            _flush()
    _flush()

    if total_samples == 0:
        raise ValueError("Manifest contains no training-eligible rows")

    index = ShardIndex(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        source_manifest_sha256=manifest_digest,
        shard_config_hash=config_hash,
        total_samples=total_samples,
        shards=tuple(entries),
    )
    atomic_write_text(
        out_dir / "shard_index.json",
        json.dumps(index.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
    )
    return index


def load_shard_index(path: str | Path) -> ShardIndex:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ShardIndex.from_dict(payload)


def verify_shards(index: ShardIndex, root: str | Path) -> dict[str, Any]:
    """Verify shard files against the index: existence, hashes, counts.

    ``root`` is the sharding output directory (containing ``shards/``).
    Raises :class:`ShardIndexError` on mismatch.
    """

    root_path = Path(root) / "shards"
    seen_ids: set[str] = set()
    total = 0
    for entry in index.shards:
        path = root_path / entry.path
        if not path.is_file():
            raise ShardIndexError(f"Shard file missing: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for payload_block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(payload_block)
        if digest.hexdigest() != entry.sha256:
            raise ShardIndexError(f"Shard hash mismatch: {path}")
        count = 0
        with path.open("rb") as handle:
            for line_number, payload_line in enumerate(handle, start=1):
                if not payload_line.strip():
                    continue
                try:
                    record = json.loads(payload_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ShardIndexError(
                        f"Malformed shard record in {path} line {line_number}"
                    ) from exc
                if record.get("shard_id") != entry.shard_id:
                    raise ShardIndexError(
                        f"Shard id mismatch in {path} line {line_number}"
                    )
                sample_id = str(record.get("sample_id", ""))
                if not sample_id:
                    raise ShardIndexError(f"Missing sample_id in {path}:{line_number}")
                if sample_id in seen_ids:
                    raise ShardIndexError(
                        f"Duplicate sample across shards: {sample_id}"
                    )
                seen_ids.add(sample_id)
                count += 1
        if count != entry.sample_count:
            raise ShardIndexError(
                f"Shard sample count mismatch for {entry.shard_id}: "
                f"index says {entry.sample_count}, found {count}"
            )
        total += count
    if total != index.total_samples:
        raise ShardIndexError(
            f"Total sample mismatch: index says {index.total_samples}, found {total}"
        )
    return {
        "verified_shards": index.total_shards,
        "verified_samples": total,
        "unique_samples": len(seen_ids),
        "ok": True,
    }
