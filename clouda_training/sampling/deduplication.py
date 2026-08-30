from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def deduplicate_records(
    records: Iterable[dict[str, Any]],
    *,
    checksum_key: str = "image_checksum",
    perceptual_duplicate: (
        Callable[[dict[str, Any], dict[str, Any]], bool] | None
    ) = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    checksums: set[str] = set()
    for record in records:
        checksum = str(record.get(checksum_key) or "")
        exact = bool(checksum and checksum in checksums)
        similar = bool(
            perceptual_duplicate
            and any(perceptual_duplicate(record, existing) for existing in kept)
        )
        if exact or similar:
            rejected.append(record)
            continue
        if checksum:
            checksums.add(checksum)
        kept.append(record)
    return kept, rejected
