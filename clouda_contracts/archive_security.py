from __future__ import annotations

import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .storage import StorageSecurityError, validate_relative_components


@dataclass(frozen=True)
class ArchiveLimits:
    max_members: int = 10_000
    max_total_uncompressed_bytes: int = 1024 * 1024 * 1024
    max_member_uncompressed_bytes: int = 256 * 1024 * 1024
    max_compression_ratio: float = 100.0


def validate_zip_archive(
    archive: zipfile.ZipFile,
    *,
    limits: ArchiveLimits = ArchiveLimits(),
) -> tuple[int, int]:
    members = archive.infolist()
    if len(members) > limits.max_members:
        raise ValueError("Archive contains too many members.")
    total = 0
    normalized_names: set[str] = set()
    for member in members:
        normalized = member.filename.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts or "\x00" in normalized:
            raise ValueError("Archive contains an unsafe member path.")
        try:
            validate_relative_components(Path(*path.parts))
        except StorageSecurityError as exc:
            raise ValueError("Archive contains an unsafe member name.") from exc
        collision_key = "/".join(path.parts).casefold()
        if collision_key in normalized_names:
            raise ValueError("Archive contains duplicate or case-colliding members.")
        normalized_names.add(collision_key)
        mode = (member.external_attr >> 16) & 0xFFFF
        if mode and stat.S_ISLNK(mode):
            raise ValueError("Archive symbolic links are not accepted.")
        file_type = stat.S_IFMT(mode)
        if file_type and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError("Archive special-file members are not accepted.")
        if member.flag_bits & 0x1:
            raise ValueError("Encrypted archive members are not accepted.")
        if member.file_size > limits.max_member_uncompressed_bytes:
            raise ValueError("Archive member exceeds the uncompressed byte limit.")
        total += member.file_size
        if total > limits.max_total_uncompressed_bytes:
            raise ValueError("Archive exceeds the total uncompressed byte limit.")
        if member.file_size:
            ratio = member.file_size / max(1, member.compress_size)
            if ratio > limits.max_compression_ratio:
                raise ValueError("Archive member exceeds the compression-ratio limit.")
    return len(members), total
