"""Content hashing and deterministic seed derivation.

Every benchmark sample's randomness is derived from a seed computed from the
tuple (global_seed, source page content hash, distortion, severity, profile,
variant). The same tuple therefore always produces byte-identical output,
which is what makes the dataset reproducible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

_CHUNK = 1 << 20
# numpy's SeedSequence / default_rng accept arbitrary ints, but we keep seeds in
# the unsigned 64-bit range so they round-trip cleanly through CSV and JSON.
_SEED_MASK = (1 << 63) - 1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Stable JSON encoding - key-sorted, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_hash(*objs: Any) -> str:
    """Hash of one or more configuration objects, order-sensitive."""
    h = hashlib.sha256()
    for obj in objs:
        h.update(canonical_json(obj).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def dataset_hash(entries: Iterable[tuple[str, str]]) -> str:
    """Hash of the whole dataset, from (sample_id, sha256) pairs.

    Sorted by sample_id so the result does not depend on generation order or
    on how many samples were regenerated incrementally.
    """
    h = hashlib.sha256()
    for sample_id, digest in sorted(entries):
        h.update(sample_id.encode("utf-8"))
        h.update(b"\x1f")
        h.update(digest.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def derive_seed(
    global_seed: int,
    source_hash: str,
    distortion: str = "",
    severity: str = "",
    profile: str = "",
    variant: int = 0,
) -> int:
    """Derive a stable 63-bit seed for one sample.

    Uses BLAKE2b over a canonical, unambiguous field-separated key. Changing
    any component changes the seed; nothing about the process depends on
    machine state, wall-clock time or iteration order.
    """
    key = "\x1f".join(
        (
            str(int(global_seed)),
            source_hash,
            distortion,
            severity,
            profile,
            str(int(variant)),
        )
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _SEED_MASK
