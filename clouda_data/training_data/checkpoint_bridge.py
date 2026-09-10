"""Checkpoint integration between the loader and the Training Experiment
Framework.

The loader does not replace checkpoint management. It provides
:data:`LoaderCheckpointHook` — a small service object that a trainer adapter
hands to ``CheckpointManager.save`` metadata so every checkpoint references
the exact data cursor it was produced at, and from which iteration can be
resumed without duplicates or gaps.

Metadata contract (added to checkpoint ``metadata.json``)::

    "data_cursor": {  # ResumeCursor.to_dict()
        "schema_version": ...,
        "dataset_id": ..., "dataset_version": ...,
        "manifest_sha256": ..., "loader_config_hash": ...,
        "global_seed": ..., "epoch": ...,
        "world_size": ..., "rank": ..., "num_workers": ..., "worker_id": ...,
        "shard_position": ..., "sample_position": ..., "yielded_count": ...
    }

The existing :class:`~clouda_training.experiments.checkpoints.CheckpointManager`
continues to own run/checkpoint storage; ``state.json`` inside a checkpoint
directory simply carries the cursor alongside trainer state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clouda_data.training_data.loader import (
    LOADER_STATE_SCHEMA_VERSION,
    ResumeError,
    StreamingTrainingDataLoader,
)
from clouda_data.training_data.models import ResumeCursor


class CursorMismatchError(ResumeError):
    """Raised when a checkpoint's cursor does not match the current loader."""


@dataclass
class LoaderCheckpointHook:
    """Attach/restore loader cursors to Training Framework checkpoints."""

    loader: StreamingTrainingDataLoader

    def cursor_payload(self) -> dict[str, Any]:
        return self.loader.get_cursor().to_dict()

    def attach_to_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return a trainer state dict augmented with the data cursor."""

        payload = dict(state)
        payload["data_cursor"] = self.cursor_payload()
        payload["data_cursor_schema"] = LOADER_STATE_SCHEMA_VERSION
        return payload

    def restore_from_state(self, state: dict[str, Any]) -> ResumeCursor:
        cursor_payload = state.get("data_cursor")
        if cursor_payload is None:
            raise CursorMismatchError(
                "Checkpoint state carries no data cursor; cannot resume data "
                "iteration safely (fail-closed)."
            )
        if state.get("data_cursor_schema") != LOADER_STATE_SCHEMA_VERSION:
            raise CursorMismatchError(
                "Checkpoint data cursor schema mismatch: "
                f"{state.get('data_cursor_schema')!r}"
            )
        cursor = ResumeCursor.from_dict(cursor_payload)
        self.loader.restore(cursor)
        return cursor

    def restore_from_checkpoint(self, checkpoint_dir: str | Path) -> ResumeCursor:
        from clouda_training.experiments.io import read_json

        state = read_json(Path(checkpoint_dir) / "state.json")
        return self.restore_from_state(state)


def write_cursor_to_checkpoint(
    checkpoint_dir: str | Path, loader: StreamingTrainingDataLoader
) -> Path:
    """Write the loader cursor into an existing checkpoint's state.json.

    Composes with the CheckpointManager flow: call after ``save()`` so the
    integrity hash must be recomputed by the trainer adapter (or use
    :class:`LoaderCheckpointHook` before the manager seals the directory).
    """

    from clouda_training.experiments.io import atomic_write_json, read_json

    path = Path(checkpoint_dir) / "state.json"
    state = read_json(path)
    hook = LoaderCheckpointHook(loader)
    return atomic_write_json(path, hook.attach_to_state(state))


def read_cursor_from_checkpoint(checkpoint_dir: str | Path) -> ResumeCursor:
    from clouda_training.experiments.io import read_json

    state = read_json(Path(checkpoint_dir) / "state.json")
    payload = state.get("data_cursor")
    if payload is None:
        raise CursorMismatchError(
            f"No data cursor in checkpoint: {checkpoint_dir}"
        )
    return ResumeCursor.from_dict(payload)
