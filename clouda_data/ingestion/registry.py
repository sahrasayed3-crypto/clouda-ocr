from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from .file_inspection import inspect_file
from .schema import FileRegistration, SourceType


def canonical_name(identifier: str, original: str | Path) -> str:
    suffix = Path(original).suffix.lower()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in identifier)
    return f"{safe}{suffix}"


def role_directory(project_root: Path, role: str) -> Path:
    mapping = {
        "document": project_root / "data/raw/documents",
        "page": project_root / "data/raw/pages",
        "ground_truth": project_root / "data/raw/ground_truth",
        "layout": project_root / "data/raw/layout_annotations",
    }
    if role not in mapping:
        raise ValueError(f"Unsupported file role: {role}")
    return mapping[role]


def register_file(
    source_path: str | Path,
    *,
    project_root: Path,
    role: str,
    identifier: str,
    source_type: SourceType,
    known_checksums: dict[str, str],
    copy: bool,
) -> FileRegistration:
    inspection = inspect_file(source_path, source_type)
    if not inspection.ok or inspection.checksum is None:
        raise ValueError(
            f"Cannot register invalid file {source_path}: {', '.join(inspection.issues)}"
        )
    duplicate_of = known_checksums.get(inspection.checksum)
    destination = role_directory(project_root, role) / canonical_name(
        identifier, source_path
    )
    canonical = destination
    if copy and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
    known_checksums.setdefault(inspection.checksum, str(destination))
    return FileRegistration(
        original_path=str(Path(source_path)),
        canonical_path=str(
            canonical
            if not canonical.is_absolute()
            else canonical.relative_to(project_root)
        ),
        file_role=role,
        source_type=source_type,
        checksum=inspection.checksum,
        size_bytes=inspection.size_bytes,
        duplicate_of=duplicate_of,
    )


def write_registry(registrations: list[FileRegistration], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(
            [asdict(item) for item in registrations], ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )


def read_registry(path: str | Path) -> list[dict]:
    registry_path = Path(path)
    if not registry_path.exists():
        return []
    return json.loads(registry_path.read_text(encoding="utf-8-sig"))
