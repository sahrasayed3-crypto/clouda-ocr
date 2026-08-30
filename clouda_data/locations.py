from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def repository_root() -> Path | None:
    configured = os.getenv("CLOUDA_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    candidate = PACKAGE_ROOT.parent
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return None


def default_foundation_registry_path() -> Path:
    root = repository_root()
    if root is not None:
        candidate = root / "dataset_catalog" / "registry" / "foundation_sources_v1.json"
        if candidate.is_file():
            return candidate
    return PACKAGE_ROOT / "resources" / "foundation_sources_v1.json"


def default_catalog_path() -> Path:
    root = repository_root()
    if root is None:
        raise FileNotFoundError(
            "The canonical dataset catalog requires CLOUDA_PROJECT_ROOT."
        )
    return root / "dataset_catalog" / "registry" / "datasets_v1.json"


def default_data_config_path() -> Path:
    root = repository_root()
    if root is not None:
        candidate = root / "configs" / "data_foundation" / "example.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "The data foundation config requires CLOUDA_PROJECT_ROOT or an explicit --config."
    )


def default_profile_dir() -> Path:
    root = repository_root()
    if root is not None:
        candidate = root / "configs" / "data_foundation" / "distortions"
        if candidate.is_dir():
            return candidate
    return PACKAGE_ROOT / "resources" / "distortions"
