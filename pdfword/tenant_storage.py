from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

SAFE_ID_RE = re.compile(r"^[a-fA-F0-9-]{16,64}$")


def safe_filename(value: str, fallback: str) -> str:
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", (value or "").strip())
    clean = clean.strip(" .")
    return clean[:120] or fallback


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
        target = paths.uploads / safe_filename(filename, "input.pdf")
        ensure_contained(self.root, target.parent)
        temporary_path: Path | None = None
        source.seek(0)
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
                temporary_path = Path(tmp.name)
                while chunk := source.read(1024 * 1024):
                    tmp.write(chunk)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(temporary_path, target)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        finally:
            source.seek(0)
        return ensure_contained(self.root, target)
