from __future__ import annotations

import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

SAFE_ID_RE = re.compile(r"^[a-fA-F0-9-]{16,64}$")

_WINDOWS_RESERVED_DEVICES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{n}" for n in range(1, 10)}
    | {f"LPT{n}" for n in range(1, 10)}
)


def safe_filename(value: str, fallback: str) -> str:
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", (value or "").strip())
    clean = clean.strip(" .")
    stem = clean.split(".", 1)[0].rstrip(" .").upper()
    if stem in _WINDOWS_RESERVED_DEVICES:
        # Windows refuses or reinterprets device names (CON.pdf, NUL .pdf,
        # COM1, ...) even with extensions; prefix so os.replace succeeds.
        clean = f"_{clean}"
    return clean[:120].strip(" .") or fallback


def ensure_contained(root: Path, path: Path) -> Path:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved.is_symlink():
        raise PermissionError("Storage path must not be a symlink")
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise PermissionError("Storage path escaped tenant root") from exc
    return resolved


@dataclass(frozen=True)
class TenantPaths:
    scope_id: str
    root: Path
    uploads: Path
    processing: Path
    outputs: Path
    temporary: Path


class TenantStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def user(self, user_id: str) -> TenantPaths:
        if not SAFE_ID_RE.fullmatch(user_id):
            raise ValueError("Invalid user ID")
        return self._paths("users", user_id)

    def guest(self, guest_scope_id: str) -> TenantPaths:
        if not SAFE_ID_RE.fullmatch(guest_scope_id):
            raise ValueError("Invalid guest scope ID")
        return self._paths("guests", guest_scope_id)

    def _paths(self, family: str, scope_id: str) -> TenantPaths:
        base = self.root / family / scope_id
        paths = TenantPaths(
            scope_id=scope_id,
            root=base,
            uploads=base / "uploads",
            processing=base / "processing",
            outputs=base / "outputs",
            temporary=base / "temporary",
        )
        for path in (paths.uploads, paths.processing, paths.outputs, paths.temporary):
            path.mkdir(parents=True, exist_ok=True)
            ensure_contained(self.root, path)
        return paths

    def write_upload(self, paths: TenantPaths, filename: str, source: BinaryIO) -> Path:
        # Reserve the final name atomically with exclusive creation: two
        # concurrent uploads of the same filename cannot both win the
        # O_CREAT|O_EXCL create, so exactly one keeps the plain name and the
        # loser derives a unique suffixed name. This removes the
        # exists()-check race that let a later upload overwrite an earlier
        # job's stored input (provenance/integrity).
        candidate = paths.uploads / safe_filename(filename, "input.pdf")
        ensure_contained(self.root, candidate.parent)
        reserved: Path | None = None
        published = False
        temporary_path: Path | None = None
        source.seek(0)
        try:
            while reserved is None:
                try:
                    descriptor = os.open(
                        candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666
                    )
                except FileExistsError:
                    candidate = candidate.with_name(
                        f"{candidate.stem}_{uuid.uuid4().hex[:12]}{candidate.suffix}"
                    )
                    continue
                os.close(descriptor)
                reserved = candidate
            with tempfile.NamedTemporaryFile(dir=reserved.parent, delete=False) as tmp:
                temporary_path = Path(tmp.name)
                while chunk := source.read(1024 * 1024):
                    tmp.write(chunk)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(temporary_path, reserved)
            published = True
            temporary_path = None
            return ensure_contained(self.root, reserved)
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if reserved is not None and not published:
                # Drop our name reservation; the published content of a
                # successful upload must never be removed here.
                reserved.unlink(missing_ok=True)
            raise
        finally:
            source.seek(0)
        return ensure_contained(self.root, target)
