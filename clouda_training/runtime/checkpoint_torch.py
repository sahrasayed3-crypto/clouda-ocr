"""Torch training-state serialization with integrity protection.

Persists the full execution state (model, optimizer, scheduler, RNG) beside
the framework's ``CheckpointManager`` metadata so resume validation,
retention, and best-checkpoint tracking keep working unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from clouda_contracts.checksums import sha256_file

STATE_FILE = "state.bin"


def save_torch_state(directory: Path, payload: dict[str, Any]) -> str:
    """Atomically persist ``payload`` as state.bin; returns its sha256."""
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    final = directory / STATE_FILE
    staging = directory / f".{STATE_FILE}.partial"
    if staging.exists():
        os.remove(staging)
    torch.save(payload, staging)
    # fsync for crash safety before the atomic rename
    with open(staging, "rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(staging, final)
    return sha256_file(final)


def load_torch_state(
    directory: Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    """Load state.bin, verifying integrity when a digest is provided."""
    import torch

    final = directory / STATE_FILE
    if not final.is_file():
        raise FileNotFoundError(f"Torch state file missing: {final}")
    if expected_sha256 is not None and sha256_file(final) != expected_sha256:
        raise ValueError(f"Torch state integrity check failed: {final}")
    return torch.load(final, map_location="cpu", weights_only=False)


def state_file_path(directory: Path) -> Path:
    return directory / STATE_FILE
