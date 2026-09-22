"""Atomic file publication for user-visible artifacts.

A crash, disk-full event, or antivirus lock during a direct ``write_bytes``/
``write_text`` leaves a truncated file at its final path (the DOCX a user
downloads, a checkpoint, a backup archive). These helpers write to a unique
staging file in the *same directory* (so the final publish is a same-volume
``os.replace``), fsync before publishing, and always remove the staging file
on failure. Mirrors the hardened writers already used by
``clouda_data.pretraining.manifest`` and ``worker_api``'s DOCX promotion.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_bytes(target: Path | str, data: bytes) -> None:
    final = Path(target)
    final.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, staging_name = tempfile.mkstemp(
        dir=str(final.parent), prefix=f".{final.name}.", suffix=".part"
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(handle_fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, final)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise


def atomic_write_text(
    target: Path | str, text: str, *, encoding: str = "utf-8"
) -> None:
    atomic_write_bytes(target, text.encode(encoding))
