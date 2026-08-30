from __future__ import annotations

import hashlib
from typing import Iterable


def deterministic_document_split(
    document_ids: Iterable[str],
    *,
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, str]:
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to one.")
    boundaries = (ratios[0], ratios[0] + ratios[1])
    assignments: dict[str, str] = {}
    for document_id in sorted(set(document_ids)):
        digest = hashlib.sha256(f"{seed}:{document_id}".encode()).digest()
        fraction = int.from_bytes(digest[:8], "big") / (2**64 - 1)
        assignments[document_id] = (
            "train"
            if fraction < boundaries[0]
            else "validation" if fraction < boundaries[1] else "test"
        )
    return assignments


def deterministic_sample(
    record_ids: Iterable[str],
    *,
    limit: int,
    seed: int,
) -> tuple[str, ...]:
    if limit < 1:
        raise ValueError("Sample limit must be positive.")
    unique = set(record_ids)
    ranked = sorted(
        unique,
        key=lambda record_id: hashlib.sha256(f"{seed}:{record_id}".encode()).digest(),
    )
    return tuple(ranked[:limit])
