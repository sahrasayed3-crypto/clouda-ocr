from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clouda_contracts.archive_security import ArchiveLimits, validate_zip_archive
from clouda_contracts.storage import StorageRoots


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confirmation_token(action: str, target: Path) -> str:
    material = f"{action}:{target.resolve()}".encode()
    return "CONFIRM-" + hashlib.sha256(material).hexdigest()[:12].upper()


def cleanup(
    action: str,
    *,
    run_id: str | None = None,
    older_than_days: int = 0,
    dry_run: bool = True,
    confirmation: str | None = None,
) -> dict[str, Any]:
    roots = StorageRoots.from_env()
    mapping = {
        "preview": roots.artifact_root / "previews",
        "failed": roots.dataset_root / "distorted",
        "temp": roots.cache_root / "temporary",
    }
    if action not in mapping:
        raise ValueError("Unsupported lifecycle cleanup")
    root = mapping[action].resolve()
    if not any(
        _inside(root, allowed)
        for allowed in (roots.artifact_root, roots.dataset_root, roots.cache_root)
    ):
        raise PermissionError("Lifecycle root escaped configured state")
    candidates: list[Path] = []
    now = datetime.now(timezone.utc).timestamp()
    if root.exists():
        for path in root.rglob("*"):
            if path.is_symlink():
                raise PermissionError("Lifecycle cleanup refuses symbolic links")
            if not path.is_file():
                continue
            if not _inside(path, root):
                raise PermissionError("Lifecycle cleanup candidate escaped root")
            if run_id and run_id not in path.parts:
                continue
            if older_than_days and now - path.stat().st_mtime < older_than_days * 86400:
                continue
            if action == "failed" and path.name.endswith((".jsonl", ".json")):
                continue
            candidates.append(path)
    token = confirmation_token(f"cleanup-{action}", root)
    if not dry_run and confirmation != token:
        raise PermissionError(f"Explicit confirmation token required: {token}")
    audit = {
        "schema_version": 1,
        "action": f"cleanup-{action}",
        "dry_run": dry_run,
        "root": str(root),
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in candidates
        ],
        "confirmation_token": token,
        "performed_at": datetime.now(timezone.utc).isoformat(),
    }
    if not dry_run:
        trash = (
            roots.artifact_root
            / "lifecycle-trash"
            / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        for path in candidates:
            destination = trash / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), destination)
        audit["recoverable_trash"] = str(trash)
    report = (
        roots.artifact_root
        / "reports"
        / "lifecycle"
        / f"{audit['action']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    audit["report"] = str(report)
    return audit


def archive_run(
    run_root: str | Path,
    *,
    dry_run: bool = True,
    confirmation: str | None = None,
) -> dict[str, Any]:
    roots = StorageRoots.from_env()
    source = Path(run_root).expanduser().resolve()
    if not _inside(source, roots.dataset_root) or not source.is_dir():
        raise PermissionError("Run archive source must be a dataset-state directory")
    files: list[Path] = []
    for path in source.rglob("*"):
        if path.is_symlink():
            raise PermissionError("Run archive refuses symbolic links")
        if path.is_file():
            if not _inside(path, source):
                raise PermissionError("Run archive member escaped source root")
            files.append(path)
    manifest = [
        {
            "path": path.relative_to(source).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    token = confirmation_token("archive-run", source)
    if not dry_run and confirmation != token:
        raise PermissionError(f"Explicit confirmation token required: {token}")
    archive = roots.artifact_root / "archives" / f"{source.name}.zip"
    if not dry_run:
        archive.parent.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            raise FileExistsError(archive)
        fd, name = tempfile.mkstemp(suffix=".zip", dir=archive.parent)
        os.close(fd)
        temp = Path(name)
        try:
            with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as handle:
                for path in files:
                    handle.write(path, path.relative_to(source).as_posix())
                handle.writestr(
                    "ARCHIVE_MANIFEST.v1.json",
                    json.dumps({"schema_version": 1, "files": manifest}, indent=2),
                )
            temp.replace(archive)
        finally:
            temp.unlink(missing_ok=True)
    return {
        "schema_version": 1,
        "dry_run": dry_run,
        "source": str(source),
        "archive": str(archive),
        "files": len(files),
        "bytes": sum(item["bytes"] for item in manifest),
        "confirmation_token": token,
    }


def verify_archive(path: str | Path) -> dict[str, Any]:
    roots = StorageRoots.from_env()
    archive = Path(path).expanduser().resolve()
    if not _inside(archive, roots.artifact_root):
        raise PermissionError("Archive must be inside artifact root")
    failures: list[str] = []
    with zipfile.ZipFile(archive) as handle:
        validate_zip_archive(
            handle,
            limits=ArchiveLimits(
                max_members=100_000, max_total_uncompressed_bytes=20 * 1024**3
            ),
        )
        manifest = json.loads(handle.read("ARCHIVE_MANIFEST.v1.json"))
        for item in manifest["files"]:
            digest = hashlib.sha256(handle.read(item["path"])).hexdigest()
            if digest != item["sha256"]:
                failures.append(item["path"])
    return {
        "schema_version": 1,
        "archive": str(archive),
        "failures": failures,
        "passed": not failures,
    }
