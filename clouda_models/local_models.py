from __future__ import annotations

from pathlib import Path

from clouda_contracts.storage import StorageRoots

from .metadata import ModelMetadata


def resolve_local_checkpoint(
    metadata: ModelMetadata,
    *,
    roots: StorageRoots,
) -> Path:
    if metadata.checkpoint_uri is None:
        raise FileNotFoundError("Model has no configured checkpoint URI.")
    path = roots.resolve_uri(metadata.checkpoint_uri)
    if path.suffix.lower() in {".exe", ".dll", ".bat", ".cmd", ".ps1"}:
        raise ValueError("Executable model artifacts are not accepted.")
    return path
