"""Canonical fail-closed protected-data and training-eligibility policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

PROTECTED_SPLIT_NAMES = frozenset(
    {"holdout", "protected_holdout", "benchmark_holdout", "private_holdout"}
)
PROTECTED_ROLES = frozenset(
    {"holdout", "protected_holdout", "benchmark", "evaluation_only", "protected"}
)
PROTECTION_FIELDS = (
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
NESTED_PROTECTION_FIELDS = ("provenance", "metadata", "protection")

_TRUE_STRINGS = frozenset({"true", "yes", "1", "protected"})
_FALSE_STRINGS = frozenset({"false", "no", "0"})
_MARKER_SUBSTRINGS = ("holdout", "protected", "evaluation_only", "benchmark")
_OPTIONAL_STRING_FIELDS = frozenset(
    {"target_split", "split", "source_split", "dataset_role", "role", "purpose"}
)


def normalize_marker(value: Any) -> str:
    return value.strip().casefold() if isinstance(value, str) else ""


def string_marks_protected(value: str) -> bool:
    normalized = normalize_marker(value)
    if normalized in _FALSE_STRINGS:
        return False
    if (
        normalized in _TRUE_STRINGS
        or normalized in PROTECTED_SPLIT_NAMES
        or normalized in PROTECTED_ROLES
    ):
        return True
    return any(marker in normalized for marker in _MARKER_SUBSTRINGS)


def _mapping_is_malformed(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return True
    for field in PROTECTION_FIELDS:
        if field not in value:
            continue
        item = value[field]
        if item is None and field in _OPTIONAL_STRING_FIELDS:
            continue
        if not isinstance(item, (bool, str)):
            return True
        if isinstance(item, str) and field in {
            "protected",
            "evaluation_only",
            "benchmark",
        }:
            normalized = normalize_marker(item)
            if normalized not in _TRUE_STRINGS and normalized not in _FALSE_STRINGS:
                return True
    return False


def mapping_is_protected(value: Any) -> bool:
    """Return True for protected markers or malformed protection metadata."""
    if _mapping_is_malformed(value):
        return True
    assert isinstance(value, Mapping)
    for field in PROTECTION_FIELDS:
        if field not in value:
            continue
        item = value[field]
        if isinstance(item, bool):
            if item:
                return True
            continue
        if isinstance(item, str):
            if string_marks_protected(item):
                return True
            continue
    return False


def record_is_protected(record: Any) -> bool:
    """Inspect a record and its canonical nested metadata blocks."""
    return _record_is_protected(record, seen=set())


def _record_is_protected(record: Any, *, seen: set[int]) -> bool:
    if not isinstance(record, Mapping):
        return True
    identity = id(record)
    if identity in seen:
        return True
    seen.add(identity)
    try:
        if mapping_is_protected(record):
            return True
        for field in NESTED_PROTECTION_FIELDS:
            if (
                field in record
                and record[field] is not None
                and not isinstance(record[field], Mapping)
            ):
                return True
        return any(
            _nested_value_is_protected(nested, seen=seen) for nested in record.values()
        )
    finally:
        seen.remove(identity)


def _nested_value_is_protected(value: Any, *, seen: set[int]) -> bool:
    if isinstance(value, Mapping):
        return _record_is_protected(value, seen=seen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        identity = id(value)
        if identity in seen:
            return True
        seen.add(identity)
        try:
            return any(_nested_value_is_protected(item, seen=seen) for item in value)
        finally:
            seen.remove(identity)
    return False


def protection_metadata_is_malformed(record: Any) -> bool:
    """Report malformed protection fields without weakening fail-closed use."""
    return _protection_metadata_is_malformed(record, seen=set())


def _protection_metadata_is_malformed(record: Any, *, seen: set[int]) -> bool:
    if _mapping_is_malformed(record):
        return True
    assert isinstance(record, Mapping)
    identity = id(record)
    if identity in seen:
        return True
    seen.add(identity)
    try:
        for field in NESTED_PROTECTION_FIELDS:
            if (
                field in record
                and record[field] is not None
                and not isinstance(record[field], Mapping)
            ):
                return True
        for nested in record.values():
            if isinstance(nested, Mapping) and _protection_metadata_is_malformed(
                nested, seen=seen
            ):
                return True
            if isinstance(nested, Sequence) and not isinstance(
                nested, (str, bytes, bytearray)
            ):
                for item in nested:
                    if isinstance(item, Mapping) and _protection_metadata_is_malformed(
                        item, seen=seen
                    ):
                        return True
        return False
    finally:
        seen.remove(identity)


def is_training_split_eligible(split: Any) -> bool:
    """Only an explicit, unambiguous training split is training eligible."""
    return isinstance(split, str) and normalize_marker(split) == "train"


__all__ = [
    "NESTED_PROTECTION_FIELDS",
    "PROTECTED_ROLES",
    "PROTECTED_SPLIT_NAMES",
    "PROTECTION_FIELDS",
    "is_training_split_eligible",
    "mapping_is_protected",
    "normalize_marker",
    "protection_metadata_is_malformed",
    "record_is_protected",
    "string_marks_protected",
]
