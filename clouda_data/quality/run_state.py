"""Durable resume state for the dataset quality pipeline.

Persists :class:`QualityRunState` as atomic JSON at
``<index_dir>/run_state.json`` so a partially completed quality-gate scan can
resume exactly where it stopped. Resume is refused (fail-closed) whenever the
manifest, quality-gate config, or any algorithm version changed between runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clouda_data.pretraining.hashing import atomic_write_text

__all__ = [
    "CorruptStateError",
    "QualityRunState",
    "StaleResumeError",
    "checkpoint",
    "load_state",
    "save_state",
    "start_or_resume",
]

RUN_STATE_FILENAME = "run_state.json"

# Identity fields: a mismatch on any of these means the on-disk state belongs
# to a different run configuration and must never be resumed.
_IDENTITY_FIELDS = ("manifest_sha256", "config_identity", "algorithm_versions")


class StaleResumeError(RuntimeError):
    """Raised when persisted run state does not match the expected identity."""


class CorruptStateError(RuntimeError):
    """Raised when the persisted run state file exists but is unreadable."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class QualityRunState:
    """Immutable snapshot of a quality-gate run's resume point."""

    manifest_sha256: str
    row_count: int
    config_identity: str
    algorithm_versions: dict[str, str] = field(default_factory=dict)
    stage: str = ""
    stage_cursor: dict[str, Any] = field(default_factory=dict)
    processed_count: int = 0
    run_id: str = ""
    updated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-serializable representation."""

        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "QualityRunState":
        """Build a state from a mapping, rejecting unknown fields."""

        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise CorruptStateError(f"Unknown run-state fields: {sorted(unknown)}")
        missing = known - set(data)
        if missing:
            raise CorruptStateError(f"Missing run-state fields: {sorted(missing)}")
        return cls(**data)


def _state_path(index_dir: str | Path) -> Path:
    return Path(index_dir) / RUN_STATE_FILENAME


def _dump(state: QualityRunState) -> str:
    return json.dumps(state.to_dict(), ensure_ascii=False, indent=2)


def save_state(state: QualityRunState, index_dir: str | Path) -> Path:
    """Atomically persist *state* to ``<index_dir>/run_state.json``."""

    return atomic_write_text(_state_path(index_dir), _dump(state))


def load_state(index_dir: str | Path) -> QualityRunState | None:
    """Load persisted run state; ``None`` when no state file exists.

    Raises :class:`CorruptStateError` when the file exists but cannot be
    parsed as a valid run-state document.
    """

    path = _state_path(index_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("state document is not a JSON object")
        return QualityRunState.from_mapping(payload)
    except (ValueError, TypeError, KeyError) as exc:
        raise CorruptStateError(f"Corrupt run-state file at {path}: {exc}") from exc


def _identity_mismatches(
    loaded: QualityRunState, expected: QualityRunState
) -> dict[str, tuple[Any, Any]]:
    mismatches: dict[str, tuple[Any, Any]] = {}
    for name in _IDENTITY_FIELDS:
        expected_value = getattr(expected, name)
        found_value = getattr(loaded, name)
        if expected_value != found_value:
            mismatches[name] = (expected_value, found_value)
    return mismatches


def start_or_resume(
    index_dir: str | Path, expected: QualityRunState
) -> tuple[QualityRunState, bool]:
    """Return ``(state, resumed)`` for a run against *index_dir*.

    - No persisted state: save *expected* and return ``(expected, False)``.
    - Persisted state matches on manifest_sha256, config_identity and
      algorithm_versions: return ``(loaded, True)``.
    - Any identity mismatch: raise :class:`StaleResumeError` naming every
      mismatched field with the expected vs. found values.
    """

    loaded = load_state(index_dir)
    if loaded is None:
        save_state(expected, index_dir)
        return expected, False

    mismatches = _identity_mismatches(loaded, expected)
    if mismatches:
        lines = [
            f"{name}: expected={expected_value!r} found={found_value!r}"
            for name, (expected_value, found_value) in mismatches.items()
        ]
        raise StaleResumeError(
            "Refusing to resume: run state identity mismatch (" + "; ".join(lines) + ")"
        )
    return loaded, True


def checkpoint(
    index_dir: str | Path,
    stage: str,
    cursor: dict[str, Any],
    processed_count: int,
) -> QualityRunState:
    """Persist a stage checkpoint, preserving run identity fields.

    Returns the updated state that was written.
    """

    loaded = load_state(index_dir)
    if loaded is None:
        raise CorruptStateError(
            f"No run state to checkpoint at {_state_path(index_dir)}"
        )
    updated = replace(
        loaded,
        stage=stage,
        stage_cursor=dict(cursor),
        processed_count=processed_count,
        updated_at=_utc_now_iso(),
    )
    save_state(updated, index_dir)
    return updated
