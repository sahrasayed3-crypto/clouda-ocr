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

from clouda_training.experiments.dataset import (  # canonical constants
    PROTECTED_ROLES,
    PROTECTED_SPLIT_NAMES,
)

_PROTECTION_FIELDS = (
    "protected",
    "target_split",
    "split",
    "source_split",
    "dataset_role",
    "role",
    "purpose",
    "evaluation_only",
    "benchmark",
)

_TRUE_STRINGS = frozenset({"true", "yes", "1", "protected"})
_MARKER_SUBSTRINGS = ("holdout", "protected", "evaluation_only", "benchmark")


def _marker(value: Any) -> str:
    return value.strip().casefold() if isinstance(value, str) else ""


def _string_marks_protected(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().casefold()
    if normalized in _TRUE_STRINGS or normalized in PROTECTED_SPLIT_NAMES or normalized in PROTECTED_ROLES:
        return True
    return any(marker in normalized for marker in _MARKER_SUBSTRINGS)


def _mapping_is_protected_strict(mapping: Any) -> bool:
    """Framework-equivalent protection check plus malformed-type rejection."""
    if not isinstance(mapping, Mapping):
        return False
    for field in _PROTECTION_FIELDS:
        if field not in mapping:
            continue
        value = mapping[field]
        if isinstance(value, bool):
            if value is True:
                return True
            continue  # False is clean
        if isinstance(value, str):
            if _string_marks_protected(value):
                return True
            continue
        # Non-str/bool value in a protection-relevant field: malformed,
        # fail closed (numbers/lists/dicts/None-typed junk are not trusted).
        if value is not None:
            return True
    return False


def row_is_protected(row: Mapping[str, Any]) -> bool:
    """Fail-closed protection check for one manifest row."""
    if _mapping_is_protected_strict(row):
        return True
    for nested_field in ("provenance", "metadata"):
        nested = row.get(nested_field)
        if nested is None:
            continue
        if not isinstance(nested, Mapping):
            # A malformed nested block on a manifest row is treated as
            # potentially hiding protection markers: fail closed.
            return True
        if _mapping_is_protected_strict(nested):
            return True
    return False


def header_is_protected(header: Mapping[str, Any]) -> bool:
    """Protection check for a manifest header (same rules as rows)."""
    return _mapping_is_protected_strict(header)


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
