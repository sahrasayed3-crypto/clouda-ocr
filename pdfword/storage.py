import re
import shutil
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

DEFAULT_STORAGE_ROOT = Path("conversions")


def safe_component(value: str, fallback: str) -> str:
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", (value or "").strip())
    clean = clean.strip(" .")
    return clean[:120] or fallback


@dataclass(frozen=True)
class JobStorage:
    job_id: str
    root: Path
    pdf_path: Path
    docx_path: Path
    temporary_dir: Path


def create_job_storage(
    username: str,
    original_name: str,
    output_name: str,
    root: str | Path = DEFAULT_STORAGE_ROOT,
) -> JobStorage:
    job_id = uuid.uuid4().hex
    user_dir = safe_component(username, "anonymous")
    job_root = Path(root) / user_dir / job_id
    temporary_dir = job_root / "temporary"
    temporary_dir.mkdir(parents=True, exist_ok=False)
    return JobStorage(
        job_id=job_id,
        root=job_root,
        pdf_path=job_root / safe_component(original_name, "input.pdf"),
        docx_path=job_root / safe_component(output_name, "output.docx"),
        temporary_dir=temporary_dir,
    )


def ensure_disk_space(
    root: str | Path, required_bytes: int, reserve_bytes: int = 512 * 1024 * 1024
) -> None:
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free
    required = max(0, int(required_bytes)) + max(0, int(reserve_bytes))
    if free < required:
        raise OSError(
            f"مساحة القرص غير كافية: المتاح {free / (1024**2):.1f} MB، "
            f"والمطلوب {required / (1024**2):.1f} MB"
        )


def save_job_files(storage: JobStorage, pdf_bytes: bytes, docx_bytes: bytes) -> None:
    atomic_write(storage.pdf_path, pdf_bytes)
    atomic_write(storage.docx_path, docx_bytes)


def atomic_write(path: str | Path, data: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_stream(
    path: str | Path,
    source: BinaryIO,
    chunk_size: int = 1024 * 1024,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    source.seek(0)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            while chunk := source.read(chunk_size):
                temporary.write(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    finally:
        source.seek(0)


def cleanup_temporary(storage: JobStorage) -> None:
    if storage.temporary_dir.exists():
        shutil.rmtree(storage.temporary_dir)
