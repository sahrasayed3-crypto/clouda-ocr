import sqlite3
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from clouda_contracts.archive_security import ArchiveLimits, validate_zip_archive

from .database import Database


def create_backup(
    database: Database,
    storage_root: str | Path = "conversions",
    backup_root: str | Path | None = None,
    retention_days: int = 14,
) -> Path:
    project_root = Path.cwd().resolve()
    destination_root = (
        Path(backup_root).expanduser().resolve()
        if backup_root
        else (project_root.parent / f"{project_root.name}_backups").resolve()
    )
    destination_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_path = destination_root / f"clouda_backup_{timestamp}.zip"
    try:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "clouda.sqlite3"
            source = sqlite3.connect(database.path)
            target = sqlite3.connect(snapshot)
            try:
                source.backup(target)
            finally:
                target.close()
                source.close()
            with zipfile.ZipFile(
                archive_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                archive.write(snapshot, "data/clouda.sqlite3")
                for folder in (Path(storage_root), Path("logs")):
                    if folder.exists():
                        for file_path in folder.rglob("*"):
                            if file_path.is_file():
                                resolved = file_path.resolve()
                                try:
                                    archive_name = resolved.relative_to(project_root)
                                except ValueError:
                                    archive_name = Path(
                                        folder.name
                                    ) / resolved.relative_to(folder.resolve())
                                archive.write(resolved, archive_name)
        database.record_backup(
            str(archive_path), "completed", archive_path.stat().st_size
        )
    except Exception as exc:
        database.record_backup(str(archive_path), "failed", 0, str(exc))
        raise

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, retention_days))
    for old_archive in destination_root.glob("clouda_backup_*.zip"):
        if old_archive == archive_path:
            continue
        modified = datetime.fromtimestamp(old_archive.stat().st_mtime, tz=timezone.utc)
        if modified < cutoff:
            old_archive.unlink(missing_ok=True)
    return archive_path


def validate_backup(archive_path: str | Path) -> dict:
    path = Path(archive_path)
    with zipfile.ZipFile(path, "r") as archive:
        validate_zip_archive(archive)
        corrupt = archive.testzip()
        names = set(archive.namelist())
    return {
        "valid": corrupt is None and "data/clouda.sqlite3" in names,
        "corrupt_member": corrupt,
        "contains_database": "data/clouda.sqlite3" in names,
        "members": len(names),
    }


def restore_backup(archive_path: str | Path, destination: str | Path) -> Path:
    archive = Path(archive_path).resolve()
    requested_target = Path(destination).expanduser()
    if requested_target.is_symlink():
        raise PermissionError("Backup destination must not be a symbolic link")
    target = requested_target.resolve()
    validation = validate_backup(archive)
    if not validation["valid"]:
        raise ValueError("ملف Backup غير صالح للاستعادة")
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("مجلد الاستعادة يجب أن يكون فارغًا")
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as bundle:
        validate_zip_archive(bundle, limits=ArchiveLimits())
        for member in bundle.infolist():
            member_target = (target / member.filename).resolve()
            if target != member_target and target not in member_target.parents:
                raise ValueError(f"مسار غير آمن داخل Backup: {member.filename}")
            if member.is_dir():
                member_target.mkdir(parents=True, exist_ok=True)
                continue
            member_target.parent.mkdir(parents=True, exist_ok=True)
            if member_target.exists() or member_target.is_symlink():
                raise FileExistsError(f"Refusing to overwrite {member.filename}")
            with bundle.open(member, "r") as source, member_target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    restored_database = target / "data" / "clouda.sqlite3"
    connection = sqlite3.connect(restored_database)
    try:
        result = connection.execute("PRAGMA integrity_check").fetchone()
        if not result or result[0] != "ok":
            raise ValueError("فشل فحص سلامة قاعدة البيانات المستعادة")
    finally:
        connection.close()
    return target


if __name__ == "__main__":
    from .settings import load_settings

    db = Database()
    config = load_settings(db)
    created = create_backup(
        db,
        config.get("storage_root", "conversions"),
        retention_days=int(config.get("backup_retention_days", 14)),
    )
    print(
        json.dumps(
            {"path": str(created), **validate_backup(created)}, ensure_ascii=False
        )
    )
