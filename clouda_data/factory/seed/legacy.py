"""Legacy seed derivation modes — vendored byte-identical from both systems.

Reproducing historical runs MUST go through these functions; the unified
v1 scheme in `derive` intentionally produces different values.

- `derive_seed_ocr_benchmark` is verbatim ocrbench.hashing.derive_seed
  (System A, global seed 20260825).
- `variant_seed_arabic_scan_factory` is verbatim
  arabic_scan_factory.profiles.variant_seed (System B, base seed 20260831),
  including its per-page convention variant_seed + page_index.
"""

from __future__ import annotations

import hashlib

_SEED_MASK = (1 << 63) - 1


def derive_seed_ocr_benchmark(
    global_seed: int,
    source_hash: str,
    distortion: str = "",
    severity: str = "",
    profile: str = "",
    variant: int = 0,
) -> int:
    """System A: BLAKE2b over the unambiguous 1F-separated field tuple."""
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


def variant_seed_arabic_scan_factory(
    source_sha256: str,
    variant_index: int,
    profile: str,
    base_seed: int = 0,
) -> int:
    """System B: SHA-256 of the colon-joined payload, first 8 bytes big-endian."""
    payload = f"{base_seed}:{source_sha256}:variant:{variant_index}:{profile}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def page_seed_arabic_scan_factory(
    source_sha256: str,
    variant_index: int,
    profile: str,
    page_index: int,
    base_seed: int = 0,
) -> int:
    """System B per-page convention: derived_seed + page_index."""
    return variant_seed_arabic_scan_factory(
        source_sha256, variant_index, profile, base_seed
    ) + int(page_index)
