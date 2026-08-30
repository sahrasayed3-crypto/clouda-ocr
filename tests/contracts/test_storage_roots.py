from __future__ import annotations

from pathlib import Path

import pytest

from clouda_contracts.storage import StorageRoots, StorageSecurityError


def _environment(tmp_path: Path) -> dict[str, str]:
    return {
        "CLOUDA_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "CLOUDA_DATASET_ROOT": str(tmp_path / "datasets"),
        "CLOUDA_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "CLOUDA_MODEL_ROOT": str(tmp_path / "models"),
        "CLOUDA_CACHE_ROOT": str(tmp_path / "cache"),
    }


def test_roots_do_not_write_by_default(tmp_path: Path) -> None:
    roots = StorageRoots.from_env(_environment(tmp_path))
    assert roots.read_only
    assert not roots.runtime_root.exists()
    with pytest.raises(PermissionError):
        roots.ensure_directories()


def test_explicit_creation_stays_inside_roots(tmp_path: Path) -> None:
    roots = StorageRoots.from_env(_environment(tmp_path), read_only=False, create=True)
    assert roots.database_path.parent.is_dir()
    assert roots.dataset_manifest_database_path.parent.is_dir()
    assert (
        roots.resolve_uri("dataset://raw/صفحة.png")
        == (roots.dataset_root / "raw" / "صفحة.png").resolve()
    )


@pytest.mark.parametrize(
    "value",
    [
        "dataset://../escape.txt",
        "dataset://raw/../../escape.txt",
        "file:///tmp/escape.txt",
        "dataset://raw/file.txt?token=not-allowed",
    ],
)
def test_storage_uri_rejects_boundary_escape(tmp_path: Path, value: str) -> None:
    roots = StorageRoots.from_env(_environment(tmp_path))
    with pytest.raises(StorageSecurityError):
        roots.resolve_uri(value)


def test_database_must_remain_under_runtime_root(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    environment["CLOUDA_DATABASE_PATH"] = str(tmp_path / "outside.sqlite3")
    with pytest.raises(StorageSecurityError):
        StorageRoots.from_env(environment)
