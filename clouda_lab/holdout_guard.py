"""Holdout safety guard for dataset selection.

Fail-closed protection checks applied before any selection is returned or
any derived manifest is written. Semantics are a **strict superset** of the
Training Experiment Framework's dataset guard
(``clouda_training.experiments.dataset.validate_training_dataset``):

- protected split names: ``holdout``, ``protected_holdout``,
  ``benchmark_holdout``, ``private_holdout`` (from the framework's
  ``PROTECTED_SPLIT_NAMES``) plus any value containing ``holdout``;
- protected roles: ``holdout``, ``protected_holdout``, ``benchmark``,
  ``evaluation_only`` (``PROTECTED_ROLES``) plus ``protected``;
- ``protected`` flag: boolean true or the strings ``true``/``yes``/``1``/
  ``protected`` (case-insensitive);
- checks the row itself and its nested ``provenance`` / ``metadata`` blocks;
- **malformed metadata fails closed**: a protection-relevant field whose
  value is neither a string nor a boolean (number, list, dict, …) marks the
  row protected, so malformed metadata can never bypass the guard;
- composite string values containing a protection marker
  (``"train+holdout"``, ``"benchmark_holdout_v2"``, …) mark the row
  protected.
"""

from __future__ import annotations

from typing import Any, Mapping

from clouda_contracts.protection import (
    PROTECTED_ROLES,
    PROTECTED_SPLIT_NAMES,
    mapping_is_protected,
    record_is_protected,
)


def row_is_protected(row: Mapping[str, Any]) -> bool:
    """Fail-closed protection check for one manifest row."""
    return record_is_protected(row)


def header_is_protected(header: Mapping[str, Any]) -> bool:
    """Protection check for a manifest header (same rules as rows)."""
    return mapping_is_protected(header)


def filter_protected_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Split rows into (safe, protected_count). Deterministic order preserved."""
    safe: list[dict[str, Any]] = []
    protected = 0
    for row in rows:
        if row_is_protected(row):
            protected += 1
        else:
            safe.append(row)
    return safe, protected


def assert_selection_safe(rows: list[dict[str, Any]]) -> None:
    """Raise ``PermissionError`` if any selected row is protected."""
    for row in rows:
        if row_is_protected(row):
            raise PermissionError(
                "Selection contains protected holdout data and cannot be used"
            )


__all__ = [
    "PROTECTED_ROLES",
    "PROTECTED_SPLIT_NAMES",
    "assert_selection_safe",
    "filter_protected_rows",
    "header_is_protected",
    "row_is_protected",
]
