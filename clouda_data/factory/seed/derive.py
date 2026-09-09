"""Unified deterministic seed derivation (seed_mode: v1).

Extends System A's BLAKE2b scheme (ocrbench.hashing.derive_seed) with the
fields needed for multi-page documents and multi-variant output:

    global_seed, source_sha256, document_id, page_index, variant_index,
    profile, distortion_stage, severity

Changing any field changes the seed; nothing depends on machine state,
wall-clock time or iteration order. Seeds are masked to 63 bits like the
legacy ocr_benchmark scheme. Historical runs remain reproducible through
`clouda_data_factory.seed.legacy`, never through this module.
"""

from __future__ import annotations

import hashlib

_SEED_MASK = (1 << 63) - 1
_UNIT_SEPARATOR = "\x1f"


def derive_seed(
    global_seed: int,
    source_sha256: str,
    document_id: str = "",
    page_index: int = 0,
    variant_index: int = 0,
    profile: str = "",
    distortion_stage: str = "",
    severity: str = "",
) -> int:
    """Derive a stable 63-bit seed for one transform stage of one page."""
    key = _UNIT_SEPARATOR.join(
        (
            str(int(global_seed)),
            str(source_sha256),
            str(document_id),
            str(int(page_index)),
            str(int(variant_index)),
            str(profile),
            str(distortion_stage),
            str(severity),
        )
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & _SEED_MASK


def page_seed(
    global_seed: int,
    source_sha256: str,
    document_id: str,
    page_index: int,
    variant_index: int,
    profile: str,
) -> int:
    """Seed for the composite scan-simulation backend (one call per page)."""
    return derive_seed(
        global_seed,
        source_sha256,
        document_id=document_id,
        page_index=page_index,
        variant_index=variant_index,
        profile=profile,
        distortion_stage="scan_composite",
    )
